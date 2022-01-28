use crate::{DenseWeight, ConvShape, Idx, as_idx, as_usize, w_idx, Shape, VectorFieldOne, Shape2, Shape3, from_xyz, debug_assert_approx_eq_weight, kernel_column_weight_sum, kernel_column_dropped_weights_count, VectorFieldPartialOrd, CpuSDR, from_xy, SDR, kernel_column_weight_copy, range_contains};
use std::ops::{Deref, DerefMut, Index, IndexMut, Range};
use rand::Rng;
use rand::prelude::SliceRandom;
use crate::cpu_ecc_population::CpuEccPopulation;
use ndalgebra::mat::AsShape;
use itertools::Itertools;
use std::collections::HashSet;
use serde::{Serialize, Deserialize};
use rayon::prelude::*;

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct WeightSums<D: DenseWeight> {
    sums: Vec<D>,
    shape:[Idx;3]
}
impl <D:DenseWeight> WeightSums<D>{
    pub fn shape(&self)->&[Idx;3]{
        &self.shape
    }
    pub fn as_slice(&self)->&[D]{
        &self.sums
    }
    pub fn new(shape:[Idx;3])->Self{
        Self{
            sums:vec![D::TOTAL_SUM;as_usize(shape.product())],
            shape,
        }
    }
    pub fn clear(&mut self, sdr:&CpuSDR){
        for &i in sdr.as_slice(){
            self[i] = D::TOTAL_SUM
        }
    }
    pub fn clear_all(&mut self){
        self.sums.iter_mut().for_each(|a|*a=D::TOTAL_SUM)
    }
    pub fn len(&self)->Idx{
        as_idx(self.sums.len())
    }
}

impl <'a, D:DenseWeight+Send+Sync> WeightSums<D>{
    pub fn parallel_clear(&mut self, sdr:&CpuSDR){
        let sums_len = self.sums.len();
        let sums_ptr = self.sums.as_mut_ptr() as usize;
        sdr.par_iter().for_each(|&output_idx|{
            let sums_slice = unsafe{std::slice::from_raw_parts_mut(sums_ptr as *mut D,sums_len)};
            sums_slice[as_usize(output_idx)] = D::TOTAL_SUM
        })
    }
    pub fn parallel_clear_all(&mut self){
        self.sums.par_iter_mut().for_each(|a|*a=D::TOTAL_SUM)
    }
}


impl<D:DenseWeight> Index<Idx> for WeightSums<D>{
    type Output = D;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.sums[as_usize(index)]
    }
}
impl<D:DenseWeight> IndexMut<Idx> for WeightSums<D>{
    fn index_mut (&mut self, index: Idx) -> &mut Self::Output {
        &mut self.sums[as_usize(index)]
    }
}
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ConvWeights<D: DenseWeight> {
    /**The layout is w[output_idx+input_idx_relative_to_kernel_column*output_volume]
    where kernel column has shape [kernel[0],kernel[1],in_channels]*/
    w: Vec<D>,
    shape: ConvShape,
    pub plasticity: D,
}

impl<D: DenseWeight> Deref for ConvWeights<D> {
    type Target = ConvShape;

    fn deref(&self) -> &Self::Target {
        &self.shape
    }
}

impl<D: DenseWeight> DerefMut for ConvWeights<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.shape
    }
}


impl<D: DenseWeight> ConvWeights<D> {

    pub fn weight_slice(&self) ->&[D]{
        &self.w
    }
    pub fn restore_dropped_out_weights(&mut self) {
        self.w.iter_mut().filter(|w| !w.is_valid()).for_each(|w| *w = D::ZERO)
    }
    pub fn dropout_f32(&mut self, number_of_connections_to_drop: f32, rng: &mut impl Rng) {
        self.dropout((self.out_volume() as f32 * number_of_connections_to_drop) as usize, rng)
    }
    pub fn dropout(&mut self, number_of_connections_to_drop: usize, rng: &mut impl Rng) {
        assert!(number_of_connections_to_drop <= self.w.len(), "number_of_connections_to_drop={} > number_of_connections=={}", number_of_connections_to_drop, self.w.len());
        let mut indices: Vec<Idx> = (0..as_idx(self.w.len())).collect();
        indices.shuffle(rng);
        for i in 0..number_of_connections_to_drop {
            self.w[as_usize(indices[i])] = D::IMPOSSIBLE_WEIGHT;
        }
        self.normalize_all()
    }
    pub fn dropout_per_kernel_f32(&mut self, number_of_connections_to_drop_per_kernel_column: f32, rng: &mut impl Rng) {
        self.dropout_per_kernel((self.kernel_column_volume() as f32 * number_of_connections_to_drop_per_kernel_column) as usize, rng)
    }
    pub fn dropout_per_kernel(&mut self, number_of_connections_to_drop_per_kernel_column: usize, rng: &mut impl Rng) {
        let kv = self.kernel_column_volume();
        let v = self.out_volume();
        assert!(number_of_connections_to_drop_per_kernel_column <= as_usize(kv), "number_of_connections_to_drop_per_kernel_column={} > kernel_column_volume=={}", number_of_connections_to_drop_per_kernel_column, kv);
        for out_idx in 0..v {
            let mut indices: Vec<Idx> = (0..kv).collect();
            indices.shuffle(rng);
            for i in 0..number_of_connections_to_drop_per_kernel_column {
                let idx_within_kernel = indices[i];
                let w_idx = w_idx(out_idx, idx_within_kernel, v);
                self.w[as_usize(w_idx)] = D::IMPOSSIBLE_WEIGHT;
            }
        }
        self.normalize_all()
    }
    pub fn normalize_all(&mut self) {
        for output_idx in 0..self.out_volume() {
            self.normalize(output_idx)
        }
    }
    pub fn normalize(&mut self, output_idx: Idx) {
        let kv = self.kernel_column_volume();
        let v = self.out_volume();
        D::normalize_precise(&mut self.w, output_idx, kv, v);
    }

    pub fn from_repeated_column(output: [Idx; 2], pretrained: &Self, pretrained_column_pos: [Idx; 2]) -> Self {
        let output = output.add_channels(pretrained.out_channels());
        let shape = ConvShape::new_out(pretrained.in_channels(),output,*pretrained.kernel(),*pretrained.stride());
        let kv = pretrained.kernel_column_volume();
        let new_v = output.product();
        let old_v = pretrained.out_volume();
        let mut w = vec![D::ZERO; as_usize(kv * new_v)];
        for channel in 0..output.channels() {
            let pretrained_out_idx = pretrained.out_shape().idx(pretrained_column_pos.add_channels(channel));
            for idx_within_kernel_column in 0..kv {
                let old_w_i = w_idx(pretrained_out_idx, idx_within_kernel_column, old_v);
                let old_w: D = pretrained.w[as_usize(old_w_i)];
                for x in 0..output.width() {
                    for y in 0..output.height() {
                        let pos = from_xyz(x, y, channel);
                        let output_idx = output.idx(pos);
                        let new_w_i = w_idx(output_idx, idx_within_kernel_column, new_v);
                        w[as_usize(new_w_i)] = old_w;
                    }
                }
            }
        }
        Self {
            plasticity: pretrained.plasticity,
            shape,
            w,
        }
    }
    pub fn new(shape:ConvShape, rng: &mut impl Rng) -> Self {
        let v = shape.out_volume();
        let kv = shape.kernel_column_volume();
        let wf: Vec<f32> = (0..kv * v).map(|_| rng.gen()).collect();
        let w = D::initialise_weight_matrix(kv, v, wf);
        let slf = Self {
            w,
            shape,
            plasticity: D::DEFAULT_PLASTICITY,
        };
        #[cfg(debug_assertions)] {
            for output_idx in 0..v {
                debug_assert_approx_eq_weight(slf.incoming_weight_sum(output_idx), D::TOTAL_SUM);
            }
        }
        slf
    }
    pub fn shape(&self)->&ConvShape{
        &self.shape
    }
    pub fn shape_mut(&mut self)->&mut ConvShape{
        &mut self.shape
    }
    pub fn concat<T>(layers: & [T], f: impl Fn(& T) -> & Self) -> Self {
        let shape = ConvShape::concat(layers,|l|f(l).shape());
        let new_v = shape.out_volume();
        let kv = shape.kernel_column_volume();

        let mut slf = Self {
            w: vec![D::IMPOSSIBLE_WEIGHT; as_usize(kv * new_v)],
            shape,
            plasticity: f(&layers[0]).plasticity,
        };
        #[cfg(debug_assertions)]
            let mut w_written_to = vec![false; slf.w.len()];

        let mut channel_offset = 0;
        for l in 0..layers.len() {
            let l = f(&layers[l]);
            let v = l.out_volume();
            for w in 0..l.out_width() {
                for h in 0..l.out_height() {
                    for c in 0..l.out_channels() {
                        let original_output_idx = l.out_shape().idx(from_xyz(w, h, c));
                        let new_output_idx = slf.shape().out_shape().idx(from_xyz(w, h, channel_offset + c));
                        for idx_within_kernel_column in 0..kv {
                            let original_w_idx = w_idx(original_output_idx, idx_within_kernel_column, v);
                            let new_w_idx = w_idx(new_output_idx, idx_within_kernel_column, new_v);
                            #[cfg(debug_assertions)]
                            debug_assert!(!w_written_to[as_usize(new_w_idx)]);
                            slf.w[as_usize(new_w_idx)] = l.w[as_usize(original_w_idx)];
                            #[cfg(debug_assertions)]{
                                w_written_to[as_usize(new_w_idx)] = true;
                            }
                        }
                    }
                }
            }
            channel_offset += l.out_channels();
        }
        #[cfg(debug_assertions)]
        debug_assert!(w_written_to.into_iter().all(|a|a));
        slf
    }
    pub fn get_plasticity(&self) -> D {
        self.plasticity
    }
    pub fn set_plasticity(&mut self, plasticity: D) {
        self.plasticity = plasticity
    }

    pub fn len(&self) -> usize {
        self.w.len()
    }
    pub fn w(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> D {
        self.w[as_usize(self.w_index(input_pos, output_pos))]
    }
    pub fn incoming_weight_sum_f32(&self, output_neuron_idx: Idx) -> f32 {
        D::w_to_f32(self.incoming_weight_sum(output_neuron_idx))
    }
    pub fn reset_and_store_all_incoming_weight_sums(&self, sums:&mut WeightSums<D>) {
        sums.sums.iter_mut().enumerate().for_each(|(i,w)|*w=self.incoming_weight_sum(as_idx(i)))
    }
    pub fn store_all_incoming_weight_sums(&self, sums:&mut WeightSums<D>) {
        sums.sums.iter_mut().enumerate().for_each(|(i,w)|*w+=self.incoming_weight_sum(as_idx(i)))
    }

    pub fn incoming_weight_sum(&self, output_neuron_idx: Idx) -> D {
        let kv = self.kernel_column_volume();
        let v = self.out_volume();
        kernel_column_weight_sum(kv, v, output_neuron_idx, &self.w)
    }
    pub fn incoming_weight_copy(&self, output_neuron_idx: Idx) -> Vec<D> {
        let kv = self.kernel_column_volume();
        let v = self.out_volume();
        kernel_column_weight_copy(kv, v, output_neuron_idx, &self.w)
    }
    pub fn get_weights(&self) -> &[D] {
        &self.w
    }
    pub fn get_dropped_weights_count(&self) -> usize{
        self.w.iter().filter(|&w|!w.is_valid()).count()
    }
    pub fn get_dropped_weights_of_kernel_column_count(&self,output_neuron_idx:Idx) -> usize{
        kernel_column_dropped_weights_count(self.kernel_column_volume(),self.out_volume(),output_neuron_idx, &self.w)
    }
    pub fn get_weights_mut(&mut self) -> &mut [D] {
        &mut self.w
    }

    pub fn reset_and_forward(&self, input: &CpuSDR, target: &mut CpuEccPopulation<D>){
        target.reset_sums();
        self.forward(input,target)
    }
    pub fn forward(&self, input: &CpuSDR, target: &mut CpuEccPopulation<D>) {
        assert_eq!(target.shape(), self.out_shape(), "Shapes don't match!");
        self.forward_with(input,|output_idx,w|{
            target.sums[as_usize(output_idx)] += w;
            debug_assert!(target.sums[as_usize(output_idx)].le(D::TOTAL_SUM));
        })
    }
    pub fn forward_with(&self, input: &CpuSDR, mut target:impl FnMut(Idx,D)) {
        let kernel_column = self.kernel_column();
        let v = self.out_volume();
        #[cfg(debug_assertions)] {
            let mut i = input.clone();
            i.sort();
            debug_assert!(i.is_normalized());
            for output_idx in 0..v {
                debug_assert_approx_eq_weight(self.incoming_weight_sum(output_idx), D::TOTAL_SUM);
            }
        }
        let mut used_w = HashSet::new();
        for &input_idx in input.as_slice() {
            let input_pos: [Idx; 3] = self.in_shape().pos(input_idx);
            let r = input_pos.grid().conv_out_range_clipped(self.stride(), self.kernel());
            for p0 in r.start.width()..r.end.width().min(self.out_shape().width()) {
                for p1 in r.start.height()..r.end.height().min(self.out_shape().height()) {
                    let kernel_offset = from_xy(p0, p1).conv_in_range_begin(self.stride());
                    for p2 in 0..self.out_channels() {
                        let output_pos = from_xyz(p0, p1, p2);
                        let output_idx = self.out_shape().idx(output_pos);
                        let w_index = ConvShape::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                        debug_assert_eq!(w_index, self.w_index(&input_pos, &output_pos));
                        debug_assert!(used_w.insert(w_index), "{}", w_index);
                        let w = self.w[as_usize(w_index)];
                        if w.is_valid() { // the connection is disabled
                            target(output_idx,w);
                        }
                    }
                }
            }
        }
    }
    pub fn infer_in_place(&self, input: &CpuSDR, output:&mut CpuSDR, target:&mut CpuEccPopulation<D>){
        self.reset_and_forward(input, target);
        target.determine_winners_top1_per_region(output);
    }

    // pub fn batch_infer<A,B>(&self, input: &[C], f:impl Fn(&C)->&CpuSDR, target:&mut CpuEccPopulation<D>){
    //     self.reset_and_forward(input, target);
    //     target.determine_winners_top1_per_region(output);
    // }

    pub fn infer(&self, input: &CpuSDR, target:&mut CpuEccPopulation<D>)->CpuSDR{
        let mut sdr = CpuSDR::new();
        self.infer_in_place(input,&mut sdr,target);
        sdr
    }
    pub fn run_in_place(&self, input: &CpuSDR, output:&mut CpuSDR, target:&mut CpuEccPopulation<D>){
        self.infer_in_place(input,output,target);
        target.decrement_activities(output)
    }
    pub fn run(&self, input: &CpuSDR, target:&mut CpuEccPopulation<D>)->CpuSDR{
        let mut sdr = CpuSDR::new();
        self.run_in_place(input,&mut sdr,target);
        sdr
    }
    pub fn learn_and_store_sums(&mut self, input: &CpuSDR, output: &CpuSDR, weight_sums:&mut WeightSums<D>) {
        assert_eq!(weight_sums.shape(),self.out_shape());
        self.learn_with(input,output,|w, active_inputs, output_idx, p, kv, v|
            weight_sums[output_idx] += p.mul(active_inputs)
        );
    }
    pub fn normalize_with_stored_sums(&mut self, output: &CpuSDR, weight_sums:&WeightSums<D>) {
        assert_eq!(weight_sums.shape(),self.out_shape());
        let v = self.out_volume();
        let kv = self.kernel_column_volume();
        for &output_idx in output.as_slice() {
            D::normalize(&mut self.w, weight_sums[output_idx], output_idx,  kv, v)
        }
    }
    pub fn learn(&mut self, input: &CpuSDR, output: &CpuSDR) {
        self.learn_with(input,output,|w, active_inputs, output_idx, p, kv, v|
            D::normalize_recommended(w, active_inputs, output_idx, p, kv, v)
        );
        #[cfg(debug_assertions)] {
            let v = self.out_volume();
            for output_idx in 0..v {
                debug_assert_approx_eq_weight(self.incoming_weight_sum(output_idx), D::TOTAL_SUM)
            }
            debug_assert!(self.w.iter().all(|&w| w.ge(D::ZERO)));
            debug_assert!(self.w.iter().all(|&w| w.le(D::TOTAL_SUM)));
        }
    }
    pub fn learn_with(&mut self, input: &CpuSDR, output: &CpuSDR,mut normalization:impl FnMut(&mut [D],Idx,Idx,D,Idx,Idx)) {
        debug_assert!(output.clone().sorted().is_normalized());
        let v = self.out_volume();
        let p = self.plasticity;
        let kernel_column = self.kernel_column();
        let kv = kernel_column.product();
        let input_pos: Vec<[Idx; 3]> = input.iter().map(|&i| self.in_shape().pos(i)).collect();
        for &output_idx in output.as_slice() {
            let mut active_inputs = self._increment_incoming_weights(input,&input_pos,output_idx);
            normalization(&mut self.w, active_inputs, output_idx, p, kv, v)
        }

    }
    fn _increment_incoming_weights(&mut self, input: &CpuSDR, input_pos: &Vec<[u32; 3]>, output_idx: Idx) -> u32 {
        let Self{ w, shape, plasticity } = self;
        Self::increment_incoming_weights(shape,*plasticity,input,input_pos,output_idx,w.as_mut_slice())
    }
    fn increment_incoming_weights(shape:&ConvShape,plasticity:D, input: &CpuSDR, input_pos: &Vec<[u32; 3]>, output_idx: Idx, w_slice: &mut [D]) -> u32 {
        let v = shape.out_volume();
        let kernel_column = shape.kernel_column();
        let output_pos = shape.out_shape().pos(output_idx);
        let kernel_offset = shape.kernel_offset(&output_pos);
        let input_range = output_pos.grid().conv_in_range(shape.stride(), shape.kernel());
        let mut active_inputs = 0;
        for (&input_idx, input_pos) in input.iter().zip(input_pos.iter()) {
            if input_range.start.all_le(input_pos.grid()) && input_pos.grid().all_lt(&input_range.end) {
                let w_index = ConvShape::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                debug_assert_eq!(w_index, shape.w_index(input_pos, &output_pos));
                let w = w_slice[as_usize(w_index)];
                if w.is_valid() {
                    w_slice[as_usize(w_index)] = w + plasticity;
                    active_inputs += 1
                }
            }
        }
        active_inputs
    }

}
impl <'a, D:DenseWeight+Send+Sync> ConvWeights<D>{
    pub fn parallel_reset_and_store_all_incoming_weight_sums(&self, sums:&mut WeightSums<D>) {
        sums.sums.par_iter_mut().enumerate().for_each(|(i,w)|*w+=self.incoming_weight_sum(as_idx(i)))
    }
    pub fn parallel_store_all_incoming_weight_sums(&self, sums:&mut WeightSums<D>) {
        sums.sums.par_iter_mut().enumerate().for_each(|(i,w)|*w+=self.incoming_weight_sum(as_idx(i)))
    }
    pub fn parallel_reset_and_forward(&self, input: &CpuSDR, target: &'a mut CpuEccPopulation<D>){
        self.parallel_forward_with(input, target,|r,w|*r=w);
    }
    pub fn parallel_forward(&self, input: &CpuSDR, target:&'a mut CpuEccPopulation<D>) {
        self.parallel_forward_with(input,target,|r,w|*r+=w)
    }
    pub fn parallel_forward_with(&self, input: &CpuSDR, target:&'a mut CpuEccPopulation<D>, update_weight:impl Fn(&mut D, D)+Send+Sync) {
        assert_eq!(target.shape(), self.out_shape(), "Shapes don't match!");
        let kernel_column = self.kernel_column();
        let v = self.out_volume();
        debug_assert!(input.clone().sorted().is_normalized());
        #[cfg(debug_assertions)] {
            for output_idx in 0..v {
                debug_assert_approx_eq_weight(self.incoming_weight_sum(output_idx), D::TOTAL_SUM);
            }
        }
        let input_positions:Vec<[Idx; 3] > = input.iter().map(|&input_idx|self.in_shape().pos(input_idx)).collect();
        let ranges:Vec<Range<[Idx;2]>> = input_positions.iter().map(|input_pos|input_pos.grid().conv_out_range_clipped_both_sides(self.stride(), self.kernel(),self.out_grid())).collect();
        target.sums.par_iter_mut().enumerate().for_each(|(output_idx,weight_sum)|{
            let output_idx = as_idx(output_idx);
            let output_pos = self.out_shape().pos(output_idx);
            let kernel_offset = output_pos.grid().conv_in_range_begin(self.stride());
            let mut w_sum = D::ZERO;
            for (input_pos,r) in input_positions.iter().zip(ranges.iter()){
                if range_contains(r,output_pos.grid()){
                    let w_index = ConvShape::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                    let w = self.w[as_usize(w_index)];
                    if w.is_valid() { // the connection is disabled
                        w_sum += w
                    }
                }
            }
            update_weight(weight_sum, w_sum)
        });
    }
    pub fn parallel_infer_in_place(&self, input: &CpuSDR, output:&mut CpuSDR, target:&'a mut CpuEccPopulation<D>){
        self.parallel_forward_with(input, target,|r,w|*r=w);
        target.parallel_determine_winners_top1_per_region(output);
    }
    pub fn parallel_infer(&self, input: &CpuSDR, target:&mut CpuEccPopulation<D>)->CpuSDR{
        let mut sdr = CpuSDR::new();
        self.parallel_infer_in_place(input,&mut sdr,target);
        sdr
    }
    pub fn parallel_run_in_place(&self, input: &CpuSDR, output:&mut CpuSDR, target:&mut CpuEccPopulation<D>){
        self.parallel_infer_in_place(input,output,target);
        target.decrement_activities(output)
    }
    pub fn parallel_run(&self, input: &CpuSDR, target:&mut CpuEccPopulation<D>)->CpuSDR{
        let mut sdr = CpuSDR::new();
        self.parallel_run_in_place(input,&mut sdr,target);
        sdr
    }

    pub fn parallel_learn_and_store_sums(&mut self, input: &CpuSDR, output: &CpuSDR,sums:&mut WeightSums<D>) {
        debug_assert!(output.clone().sorted().is_normalized());
        let w_len = self.w.len();
        let w_ptr = self.w.as_mut_ptr() as usize;
        let sums_len = sums.sums.len();
        let sums_ptr = sums.sums.as_mut_ptr() as usize;
        let p = self.plasticity;
        let input_pos: Vec<[Idx; 3]> = input.iter().map(|&i| self.in_shape().pos(i)).collect();
        output.par_iter().for_each(|&output_idx|{
            let w_slice = unsafe{std::slice::from_raw_parts_mut(w_ptr as *mut D,w_len)};
            let weight_sums = unsafe{std::slice::from_raw_parts_mut(sums_ptr as *mut D,sums_len)};
            let active_inputs = Self::increment_incoming_weights(self.shape(),p,input, &input_pos, output_idx, w_slice);
            weight_sums[as_usize(output_idx)] += p.mul(active_inputs);
        });
    }

    pub fn parallel_learn(&mut self, input: &CpuSDR, output: &CpuSDR) {
        debug_assert!(output.clone().sorted().is_normalized());
        let w_len = self.w.len();
        let w_ptr = self.w.as_mut_ptr() as usize;
        let kv = self.kernel_column_volume();
        let v = self.out_volume();
        let p = self.plasticity;
        let input_pos: Vec<[Idx; 3]> = input.iter().map(|&i| self.in_shape().pos(i)).collect();
        output.par_iter().for_each(|&output_idx|{
            let w_slice = unsafe{std::slice::from_raw_parts_mut(w_ptr as *mut D,w_len)};
            let active_inputs = Self::increment_incoming_weights(self.shape(),p,input, &input_pos, output_idx, w_slice);
            D::normalize_recommended(w_slice,active_inputs,output_idx,p,kv,v)
        });
    }

    pub fn parallel_normalize_with_stored_sums(&mut self, output: &CpuSDR, weight_sums:&WeightSums<D>) {
        assert_eq!(weight_sums.shape(),self.out_shape());
        let v = self.out_volume();
        let kv = self.kernel_column_volume();
        let w_len = self.w.len();
        let w_ptr = self.w.as_mut_ptr() as usize;
        output.par_iter().for_each(|&output_idx| {
            let w_slice = unsafe{std::slice::from_raw_parts_mut(w_ptr as *mut D,w_len)};
            D::normalize(w_slice, weight_sums[output_idx], output_idx,  kv, v)
        })
    }
}
impl ConvWeights<f32>{
    pub fn forward_with_multiplier(&self, input: &CpuSDR, target: &mut CpuEccPopulation<f32>, multiplier:f32) {
        assert_eq!(target.shape(), self.out_shape(), "Shapes don't match!");
        self.forward_with(input,|output_idx,w|{
            target.sums[as_usize(output_idx)] += multiplier*w;
        })
    }
}

#[cfg(test)]
mod tests{
    use crate::{DenseWeight, ConvWeights, ConvShape, CpuEccDense, EccLayer, EccLayerD, CpuSDR};
    use crate::xorshift::auto_gen_seed64;
    use rand::{SeedableRng, Rng};
    #[test]
    fn test10() -> Result<(), String> {
        test10_::<u32>()
    }

    #[test]
    fn test10f() -> Result<(), String> {
        test10_::<f32>()
    }

    fn test10_<D: DenseWeight+Send+Sync>() -> Result<(), String> {
        let s = auto_gen_seed64();
        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
        let k = 16;
        let mut a = CpuEccDense::<D>::new(ConvShape::new([2, 2], [4, 4], [1, 1], 3, 4), 1,  &mut rng);
        a.set_threshold_f32(0.2);
        let mut a2 = a.clone();
        println!("seed=={}", s);
        assert_eq!(a2.k(), 1);
        assert_eq!(a.k(),1);
        assert!(a.get_threshold().eq(a2.get_threshold()), "threshold {:?}!={:?}", a.get_threshold(), a2.get_threshold());
        for i in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            assert!(a.weight_slice().iter().zip(a2.weight_slice().iter()).all(|(a, b)| a.eq(*b)), "w {:?}!={:?}", a.weights().weight_slice(), a2.weights().weight_slice());
            assert!(a.population().sums.iter().zip(a2.population().sums.iter()).all(|(a, b)| a.eq(*b)), "sums {:?}!={:?}", a.population().sums, a2.population().sums);
            assert!(a.population().activity.iter().zip(a2.population().activity.iter()).all(|(a, b)| a.eq(*b)), "activity {:?}!={:?}", a.population().activity, a2.population().activity);
            let mut o = a.run(&input);
            let mut o2 = a2.parallel_run(&input);
            assert_eq!(o, o2, "outputs i=={}", i);
            a.learn(&input, &o);
            a2.parallel_learn(&input, &o2);
        }
        Ok(())
    }
}
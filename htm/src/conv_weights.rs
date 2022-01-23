use crate::{DenseWeight, ConvShape, Idx, as_idx, as_usize, w_idx, Shape, VectorFieldOne, Shape2, Shape3, from_xyz, debug_assert_approx_eq_weight, kernel_column_weight_sum, kernel_column_dropped_weights_count, VectorFieldPartialOrd, CpuSDR, from_xy, SDR};
use std::ops::{Deref, DerefMut};
use rand::Rng;
use rand::prelude::SliceRandom;
use crate::cpu_ecc_population::CpuEccPopulation;
use ndalgebra::mat::AsShape;
use itertools::Itertools;
use std::collections::HashSet;
use serde::{Serialize, Deserialize};

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
    pub fn incoming_weight_sum(&self, output_neuron_idx: Idx) -> D {
        let kv = self.kernel_column_volume();
        let v = self.out_volume();
        kernel_column_weight_sum(kv, v, output_neuron_idx, &self.w)
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

    pub fn forward(&self, input: &CpuSDR, target: &mut CpuEccPopulation<D>) {
        assert_eq!(target.shape(), self.out_shape(), "Shapes don't match!");
        let kernel_column = self.kernel_column();
        let v = self.out_volume();
        #[cfg(debug_assertions)] {
            let mut i = input.clone();
            i.sort();
            debug_assert!(i.iter().tuple_windows().all(|(prev, next)| prev != next), "{:?}", i);
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
                            target.sums[as_usize(output_idx)] += w;
                            debug_assert!(target.sums[as_usize(output_idx)].le(D::TOTAL_SUM), "{:?}->{:?}={}@{}<={}", input_pos, output_pos, output_idx, target.sums[as_usize(output_idx)], D::TOTAL_SUM);
                        }
                    }
                }
            }
        }
    }
    pub fn infer_in_place(&self, input: &CpuSDR, output:&mut CpuSDR, target:&mut CpuEccPopulation<D>){
        target.reset_sums();
        self.forward(input, target);
        target.determine_winners_top1_per_region(output);
    }
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
    pub fn learn(&mut self, input: &CpuSDR, output: &CpuSDR) {
        #[cfg(debug_assertions)] {
            let mut i = output.clone();
            i.sort();
            debug_assert!(i.iter().tuple_windows().all(|(prev, next)| prev != next), "{:?}", i);
        }
        let v = self.out_volume();
        let p = self.plasticity;
        let kernel_column = self.kernel_column();
        let kv = kernel_column.product();
        let input_pos: Vec<[Idx; 3]> = input.iter().map(|&i| self.in_shape().pos(i)).collect();
        for &output_idx in output.as_slice() {
            let output_pos = self.out_shape().pos(output_idx);
            let kernel_offset = self.kernel_offset(&output_pos);
            let input_range = output_pos.grid().conv_in_range(self.stride(), self.kernel());
            let mut active_inputs = 0;
            for (&input_idx, input_pos) in input.iter().zip(input_pos.iter()) {
                if input_range.start.all_le(input_pos.grid()) && input_pos.grid().all_lt(&input_range.end) {
                    let w_index = ConvShape::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                    debug_assert_eq!(w_index, self.w_index(input_pos, &output_pos));
                    let w = self.w[as_usize(w_index)];
                    if w.is_valid() {
                        self.w[as_usize(w_index)] = w+p;
                        active_inputs += 1;
                    }
                }
            }
            D::normalize_recommended(&mut self.w, active_inputs, output_idx, p, kv, v);
        }
        #[cfg(debug_assertions)] {
            for output_idx in 0..v {
                debug_assert_approx_eq_weight(self.incoming_weight_sum(output_idx), D::TOTAL_SUM)
            }
            debug_assert!(self.w.iter().all(|&w| w.ge(D::ZERO)));
            debug_assert!(self.w.iter().all(|&w| w.le(D::TOTAL_SUM)));
        }
    }
}

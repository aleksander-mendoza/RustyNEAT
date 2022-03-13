use crate::{DenseWeight, ConvShape, Idx, as_idx, as_usize, w_idx, Shape, VectorFieldOne, Shape2, Shape3, from_xyz, debug_assert_approx_eq_weight, kernel_column_dropped_weights_count, VectorFieldPartialOrd, CpuSDR, from_xy, SDR, kernel_column_weight_copy, range_contains, ConvShapeTrait, HasConvShape, HasConvShapeMut, DenseWeightL2, Metric, range_foreach2d, D, ShapedArrayTrait, HasShape};
use std::ops::{Deref, DerefMut, Index, IndexMut, Range};
use rand::Rng;
use rand::prelude::SliceRandom;
use crate::cpu_ecc_population::CpuEccPopulation;
use ndalgebra::mat::AsShape;
use itertools::Itertools;
use std::collections::HashSet;
use serde::{Serialize, Deserialize};
use rayon::prelude::*;
use std::thread::JoinHandle;
use crate::parallel::{parallel_map_vector, parallel_map_collect};
use crate::as_usize::AsUsize;
use std::marker::PhantomData;

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct ConvWeights<M: Metric<D>> {
    /**The layout is w[output_idx+input_idx_relative_to_kernel_column*output_volume]
    where kernel column has shape [kernel[0],kernel[1],in_channels]*/
    w: Vec<D>,
    shape: ConvShape,
    pub plasticity: D,
    #[serde(skip_serializing, default)]
    _d: PhantomData<M>,
}

pub trait ConvWeightVec<M: Metric<D>>: HasConvShape {
    fn weight_slice(&self) -> &[D];
    fn weight_slice_mut(&mut self) -> &mut [D];
    fn len(&self) -> usize {
        self.weight_slice().len()
    }
    fn restore_dropped_out_weights(&mut self) {
        self.weight_slice_mut().iter_mut().filter(|w| !w.is_valid()).for_each(|w| *w = D::ZERO)
    }
    fn dropout_f32(&mut self, number_of_connections_to_drop: f32, rng: &mut impl Rng) {
        self.dropout((self.out_volume() as f32 * number_of_connections_to_drop) as usize, rng)
    }
    fn dropout(&mut self, number_of_connections_to_drop: usize, rng: &mut impl Rng) {
        assert!(number_of_connections_to_drop <= self.len(), "number_of_connections_to_drop={} > number_of_connections=={}", number_of_connections_to_drop, self.len());
        let mut indices: Vec<Idx> = (0..as_idx(self.len())).collect();
        indices.shuffle(rng);
        for i in 0..number_of_connections_to_drop {
            self.weight_slice_mut()[indices[i].as_usize()] = D::IMPOSSIBLE_WEIGHT;
        }
    }
    fn dropout_and_normalize(&mut self, number_of_connections_to_drop: usize, rng: &mut impl Rng) {
        self.dropout(number_of_connections_to_drop, rng);
        self.normalize_all()
    }
    fn dropout_f32_and_normalize(&mut self, number_of_connections_to_drop: f32, rng: &mut impl Rng) {
        self.dropout_f32(number_of_connections_to_drop, rng);
        self.normalize_all()
    }
    fn dropout_per_kernel_f32(&mut self, number_of_connections_to_drop_per_kernel_column: f32, rng: &mut impl Rng) {
        self.dropout_per_kernel((self.kernel_column_volume() as f32 * number_of_connections_to_drop_per_kernel_column) as usize, rng)
    }
    fn dropout_per_kernel(&mut self, number_of_connections_to_drop_per_kernel_column: usize, rng: &mut impl Rng) {
        let kv = self.kernel_column_volume();
        let v = self.out_volume();
        assert!(number_of_connections_to_drop_per_kernel_column <= kv.as_usize(), "number_of_connections_to_drop_per_kernel_column={} > kernel_column_volume=={}", number_of_connections_to_drop_per_kernel_column, kv);
        for out_idx in 0..v {
            let mut indices: Vec<Idx> = (0..kv).collect();
            indices.shuffle(rng);
            for i in 0..number_of_connections_to_drop_per_kernel_column {
                let idx_within_kernel = indices[i];
                let w_idx = w_idx(out_idx, idx_within_kernel, v);
                self.weight_slice_mut()[w_idx.as_usize()] = D::IMPOSSIBLE_WEIGHT;
            }
        }
    }
    fn dropout_per_kernel_f32_and_normalize(&mut self, number_of_connections_to_drop_per_kernel_column: f32, rng: &mut impl Rng) {
        self.dropout_per_kernel_f32(number_of_connections_to_drop_per_kernel_column, rng);
        self.normalize_all()
    }
    fn dropout_per_kernel_and_normalize(&mut self, number_of_connections_to_drop_per_kernel_column: usize, rng: &mut impl Rng) {
        self.dropout_per_kernel(number_of_connections_to_drop_per_kernel_column, rng);
        self.normalize_all()
    }
    fn normalize_all(&mut self) {
        for output_idx in 0..self.out_volume() {
            self.normalize(output_idx)
        }
    }
    /**First computes the exact sum of weights in linear O(n) time. Then uses this sum to normalise the weights*/
    fn normalize(&mut self, output_idx: Idx) {
        let magnitude = self.kernel_column_weight_magnitude(output_idx);
        self.divide_weights_in_kernel_column(magnitude,output_idx);
        debug_assert_approx_eq_weight(D::ONE, self.kernel_column_weight_magnitude(output_idx));
    }
    fn normalize_for_sdr(&mut self, output: &CpuSDR) {
        for &output_idx in output.as_slice(){
            self.normalize(output_idx)
        }
    }
    fn kernel_column_weight_magnitude(&self, output_idx: Idx)->D{
        let kv = self.kernel_column_volume();
        let v = self.out_volume();
        M::kernel_column_weight_magnitude(kv, v, output_idx, self.weight_slice())
    }
    fn kernel_column_weight_sum(&self, output_idx: Idx)->D{
        let kv = self.kernel_column_volume();
        let v = self.out_volume();
        M::kernel_column_weight_sum(kv, v, output_idx, self.weight_slice())
    }
    fn normalize_with_stored_sums(&mut self, output: &CpuSDR, weight_sums: &[D]) {
        assert_eq!(weight_sums.len(), self.out_volume().as_usize(), "Shapes don't match!");
        for &output_idx in output.as_slice() {
            let sum = weight_sums[output_idx.as_usize()];
            let magnitude = M::root(sum);
            self.divide_weights_in_kernel_column(magnitude,output_idx)
        }
    }
    fn divide_weights_in_kernel_column(&mut self, scalar: D, output_idx: Idx) {
        let kv = self.kernel_column_volume();
        let v = self.out_volume();
        let w = self.weight_slice_mut();
        Self::divide_weights_in_kernel_column_(scalar,output_idx,kv,v,w)
    }
    fn divide_weights_in_kernel_column_(scalar: D, output_idx: Idx,kv:Idx,v:Idx,w:&mut[D]){
        for input_within_kernel_column in 0..kv {
            let w_idx = w_idx(output_idx, input_within_kernel_column, v);
            if w[w_idx.as_usize()].is_valid(){
                w[w_idx.as_usize()] /= scalar;
            }
        }
    }
    fn w(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> D {
        self.weight_slice()[self.w_index(input_pos, output_pos).as_usize()]
    }
    fn reset_and_store_all_incoming_weight_sums(&self, sums: &mut [D]) {
        sums.iter_mut().enumerate().for_each(|(i, w)| *w = self.incoming_weight_sum(as_idx(i)))
    }
    fn store_all_incoming_weight_sums(&self, sums: &mut [D]) {
        sums.iter_mut().enumerate().for_each(|(i, w)| *w += self.incoming_weight_sum(as_idx(i)))
    }
    fn incoming_weight_sum(&self, output_neuron_idx: Idx) -> D {
        let kv = self.kernel_column_volume();
        let v = self.out_volume();
        M::kernel_column_weight_sum(kv, v, output_neuron_idx, self.weight_slice())
    }
    fn incoming_weight_copy(&self, output_neuron_idx: Idx) -> Vec<D> {
        let kv = self.kernel_column_volume();
        let v = self.out_volume();
        kernel_column_weight_copy(kv, v, output_neuron_idx, self.weight_slice())
    }
    fn get_dropped_weights_count(&self) -> usize {
        self.weight_slice().iter().filter(|&w| !w.is_valid()).count()
    }
    fn get_dropped_weights_of_kernel_column_count(&self, output_neuron_idx: Idx) -> usize {
        kernel_column_dropped_weights_count(self.kernel_column_volume(), self.out_volume(), output_neuron_idx, self.weight_slice())
    }

    fn copy_repeated_column(&mut self, pretrained: &Self, pretrained_column_pos: [Idx; 2]) {
        assert_eq!(pretrained.kernel(), self.kernel(), "Kernels are different");
        assert_eq!(pretrained.in_channels(), self.in_channels(), "Input channels are different");
        let s = self.output_shape();
        copy_repeated_column(self.weight_slice_mut(), s, pretrained.weight_slice(), pretrained.cshape(), pretrained_column_pos)
    }
}

pub fn copy_repeated_column<D: Copy>(destination: &mut [D], destination_out_shape: [Idx; 3],
                                     pretrained: &[D], pretrained_shape: &ConvShape,
                                     pretrained_column_pos: [Idx; 2]) {
    assert_eq!(pretrained_shape.out_channels(), destination_out_shape.channels(), "Output channels are different");
    let kv = pretrained_shape.kernel_column_volume();
    let new_v = destination_out_shape.product();
    let old_v = pretrained_shape.out_volume();
    for channel in 0..destination_out_shape.channels() {
        let pretrained_out_idx = pretrained_shape.out_shape().idx(pretrained_column_pos.add_channels(channel));
        for idx_within_kernel_column in 0..kv {
            let old_w_i = w_idx(pretrained_out_idx, idx_within_kernel_column, old_v);
            let old_w: D = pretrained[old_w_i.as_usize()];
            for x in 0..destination_out_shape.width() {
                for y in 0..destination_out_shape.height() {
                    let pos = from_xyz(x, y, channel);
                    let output_idx = destination_out_shape.idx(pos);
                    let new_w_i = w_idx(output_idx, idx_within_kernel_column, new_v);
                    destination[new_w_i.as_usize()] = old_w;
                }
            }
        }
    }
}

impl<M: Metric<D>> ConvWeightVec<M> for ConvWeights<M> {
    fn weight_slice(&self) -> &[D] {
        self.w.as_slice()
    }
    fn weight_slice_mut(&mut self) -> &mut [D] {
        self.w.as_mut_slice()
    }
}
impl<M: Metric<D>> HasShape for ConvWeights<M> {
    fn shape(&self) -> &[u32; 3] {
        self.shape.out_shape()
    }
}
impl<M: Metric<D>> HasConvShape for ConvWeights<M> {
    fn cshape(&self) -> &ConvShape {
        &self.shape
    }
}

impl<M: Metric<D>> HasConvShapeMut for ConvWeights<M> {
    fn cshape_mut(&mut self) -> &mut ConvShape {
        &mut self.shape
    }
}

impl<M: Metric<D>> ConvWeights<M> {
    pub fn from_repeated_column(output: [Idx; 2], pretrained: &Self, pretrained_column_pos: [Idx; 2]) -> Self {
        let output = output.add_channels(pretrained.out_channels());
        let shape = ConvShape::new_out(pretrained.in_channels(), output, *pretrained.kernel(), *pretrained.stride());
        let kv = pretrained.kernel_column_volume();
        let new_v = output.product();
        let mut w = vec![D::ZERO; (kv * new_v).as_usize()];
        let mut slf = Self {
            plasticity: pretrained.plasticity,
            shape,
            w,
            _d: Default::default(),
        };
        slf.copy_repeated_column(pretrained, pretrained_column_pos);
        slf
    }
    pub fn new(shape: ConvShape, rng: &mut impl Rng) -> Self {
        let v = shape.out_volume();
        let kv = shape.kernel_column_volume();
        let wf: Vec<f32> = (0..kv * v).map(|_| rng.gen()).collect();
        let w = D::initialise_weight_matrix::<M>(kv, v, wf);
        let slf = Self {
            w,
            shape,
            plasticity: D::DEFAULT_PLASTICITY,
            _d: Default::default(),
        };
        #[cfg(debug_assertions)] {
            for output_idx in 0..v {
                debug_assert_approx_eq_weight(slf.incoming_weight_sum(output_idx), D::ONE);
            }
        }
        slf
    }

    pub fn get_plasticity(&self) -> D {
        self.plasticity
    }
    pub fn set_plasticity(&mut self, plasticity: D) {
        self.plasticity = plasticity
    }


    pub fn reset_and_forward(&self, input: &CpuSDR, target: &mut CpuEccPopulation<M>) {
        target.reset_sums();
        self.forward(input, target)
    }
    pub fn forward(&self, input: &CpuSDR, target: &mut impl ShapedArrayTrait<D>) {
        assert_eq!(target.shape(), self.out_shape(), "Shapes don't match");
        self.forward_with(input, |output_idx, w| {
            target.excite(output_idx,w);
            debug_assert!(*target.state(output_idx) <= M::max_inner_product(input), "{}", target.state(output_idx));
        })
    }
    pub fn inhibit(&self, input: &CpuSDR, target: &mut impl ShapedArrayTrait<D>) {
        assert_eq!(target.shape(), self.out_shape(), "Shapes don't match");
        self.forward_with(input, |output_idx, w| {
            target.inhibit(output_idx,w);
            debug_assert!(*target.state(output_idx) <= D::ONE);
        })
    }
    pub fn forward_with(&self, input: &CpuSDR, mut target: impl FnMut(Idx, D)) {
        let kernel_column = self.kernel_column();
        let v = self.out_volume();
        #[cfg(debug_assertions)] {
            let mut i = input.clone();
            i.sort();
            debug_assert!(i.is_normalized());
            for output_idx in 0..v {
                debug_assert_approx_eq_weight(self.incoming_weight_sum(output_idx), D::ONE);
            }
        }
        let mut used_w = HashSet::new();
        for &input_idx in input.as_slice() {
            let input_pos: [Idx; 3] = self.in_shape().pos(input_idx);
            let r = input_pos.grid().conv_out_range_clipped_both_sides(self.stride(), self.kernel(),self.out_grid());
            range_foreach2d(&r,|output_pos|{
                let kernel_offset = output_pos.conv_in_range_begin(self.stride());
                for p2 in 0..self.out_channels() {
                    let output_pos = output_pos.add_channels(p2);
                    let output_idx = self.out_shape().idx(output_pos);
                    let w_index = ConvShape::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                    debug_assert_eq!(w_index, self.w_index(&input_pos, &output_pos));
                    debug_assert!(used_w.insert(w_index), "{}", w_index);
                    let w = self.w[w_index.as_usize()];
                    if w.is_valid() { // the connection is disabled
                        target(output_idx, w);
                    }
                }
            });
        }
    }
    pub fn infer_in_place(&self, input: &CpuSDR, output: &mut CpuSDR, target: &mut CpuEccPopulation<M>) {
        self.reset_and_forward(input, target);
        target.determine_winners_top1_per_region(output);
    }

    // pub fn batch_infer<A,B>(&self, input: &[C], f:impl Fn(&C)->&CpuSDR, target:&mut CpuEccPopulation<M>){
    //     self.reset_and_forward(input, target);
    //     target.determine_winners_top1_per_region(output);
    // }

    pub fn infer(&self, input: &CpuSDR, target: &mut CpuEccPopulation<M>) -> CpuSDR {
        let mut sdr = CpuSDR::new();
        self.infer_in_place(input, &mut sdr, target);
        sdr
    }
    pub fn run_in_place(&self, input: &CpuSDR, output: &mut CpuSDR, target: &mut CpuEccPopulation<M>) {
        self.infer_in_place(input, output, target);
        target.decrement_activities(output)
    }
    pub fn run(&self, input: &CpuSDR, target: &mut CpuEccPopulation<M>) -> CpuSDR {
        let mut sdr = CpuSDR::new();
        self.run_in_place(input, &mut sdr, target);
        sdr
    }
    pub fn learn_and_store_sums(&mut self, input: &CpuSDR, output: &CpuSDR, weight_sums: &mut [D]) {
        assert_eq!(weight_sums.len(), self.out_volume().as_usize(), "Shapes don't match!");
        self.increment_incoming_weights_for_sdr(input, output);
        for &output_idx in output.as_slice() {
            weight_sums[output_idx.as_usize()] +=self.kernel_column_weight_sum(output_idx);
        }
    }

    pub fn learn(&mut self, input: &CpuSDR, output: &CpuSDR) {
        self.increment_incoming_weights_for_sdr(input, output);
        self.normalize_for_sdr(output);
        #[cfg(debug_assertions)] {
            let v = self.out_volume();
            for output_idx in 0..v {
                debug_assert_approx_eq_weight(self.incoming_weight_sum(output_idx), D::ONE)
            }
            debug_assert!(self.w.iter().all(|&w| w >=D::ZERO));
            debug_assert!(self.w.iter().all(|&w| w <=D::ONE));
        }
    }
    pub fn increment_incoming_weights_for_sdr(&mut self, input: &CpuSDR, output: &CpuSDR) {
        debug_assert!(output.clone().sorted().is_normalized());
        let input_pos: Vec<[Idx; 3]> = input.iter().map(|&i| self.in_shape().pos(i)).collect();
        let mut w_to_increment = Vec::with_capacity(input.len());
        for &output_idx in output.as_slice() {
            self.increment_incoming_weights(input, &input_pos, output_idx, &mut w_to_increment);
        }
    }
    fn increment_incoming_weights(&mut self, input: &CpuSDR, input_pos: &Vec<[u32; 3]>, output_idx: Idx,w_to_increment:&mut Vec<Idx>) {
        let Self { w, shape, plasticity, .. } = self;
        Self::increment_incoming_weights_(shape, *plasticity, input, input_pos, output_idx, w.as_mut_slice(),w_to_increment)
    }
    /**w_to_increment is just a temporary storage for indices of weights that will be incremented.
    It is not cleared upon completion, so feel free to later use this vector for any extra post-processing.*/
    fn increment_incoming_weights_(shape: &ConvShape, plasticity: D, input: &CpuSDR, input_pos: &Vec<[u32; 3]>, output_idx: Idx, w_slice: &mut [D],w_to_increment:&mut Vec<Idx>) {
        let v = shape.out_volume();
        let kernel_column = shape.kernel_column();
        let output_pos = shape.out_shape().pos(output_idx);
        let kernel_offset = shape.kernel_offset(&output_pos);
        let input_range = output_pos.grid().conv_in_range(shape.stride(), shape.kernel());
        w_to_increment.clear();
        for (&input_idx, input_pos) in input.iter().zip(input_pos.iter()) {
            if range_contains(&input_range,input_pos.grid()){
                let w_index = ConvShape::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                debug_assert_eq!(w_index, shape.w_index(input_pos, &output_pos));
                if w_slice[w_index.as_usize()].is_valid() {
                    w_to_increment.push(w_index);
                }
            }
        }
        let plasticity = plasticity/ w_to_increment.len() as f32;
        for w_index in w_to_increment.iter().cloned(){
            w_slice[w_index.as_usize()] += plasticity;
        }
    }
}

impl<'a, M: Metric<D> + 'a> ConvWeights< M> {
    pub fn concat<T>(layers: &'a [T], f: impl Fn(&'a T) -> &'a Self) -> Self {
        let shape = ConvShape::concat(layers, |l| f(l).cshape());
        let new_v = shape.out_volume();
        let kv = shape.kernel_column_volume();

        let mut slf = Self {
            w: vec![D::IMPOSSIBLE_WEIGHT; (kv * new_v).as_usize()],
            shape,
            plasticity: f(&layers[0]).plasticity,
            _d: Default::default(),
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
                        let new_output_idx = slf.out_shape().idx(from_xyz(w, h, channel_offset + c));
                        for idx_within_kernel_column in 0..kv {
                            let original_w_idx = w_idx(original_output_idx, idx_within_kernel_column, v);
                            let new_w_idx = w_idx(new_output_idx, idx_within_kernel_column, new_v);
                            #[cfg(debug_assertions)]
                            debug_assert!(!w_written_to[new_w_idx.as_usize()]);
                            slf.w[new_w_idx.as_usize()] = l.w[original_w_idx.as_usize()];
                            #[cfg(debug_assertions)] {
                                w_written_to[new_w_idx.as_usize()] = true;
                            }
                        }
                    }
                }
            }
            channel_offset += l.out_channels();
        }
        #[cfg(debug_assertions)]
        debug_assert!(w_written_to.into_iter().all(|a| a));
        slf
    }
}

impl<'a, M: Metric<D> + Send + Sync> ConvWeights<M> {


    pub fn batch_infer<T, O: Send>(&self, input: &[T], f: impl Fn(&T) -> &CpuSDR + Send + Sync, target: CpuEccPopulation<M>, of: impl Fn(CpuSDR) -> O + Sync) -> Vec<O> {
        assert_eq!(target.shape(), self.out_shape(), "Shapes don't match");
        let mut targets: Vec<CpuEccPopulation<M>> = (1..num_cpus::get()).map(|_| target.clone()).collect();
        targets.push(target);
        parallel_map_collect(input, &mut targets, |i, t| of(self.infer(f(i), t)))
    }

    pub fn batch_infer_and_measure_s_expectation<T, O: Send>(&self, input: &[T], f: impl Fn(&T) -> &CpuSDR + Send + Sync, target: CpuEccPopulation<M>, of: impl Fn(CpuSDR) -> O + Sync) -> (Vec<O>, D, u32) {
        assert_eq!(target.shape(), self.out_shape(), "Shapes don't match");
        let mut targets: Vec<(D, u32, CpuEccPopulation<M>)> = (1..num_cpus::get()).map(|_| (D::ZERO, 0, target.clone())).collect();
        targets.push((D::ZERO, 0, target));
        let o_vec = parallel_map_collect(input, &mut targets, |i, (s, missed, t)| {
            let o_sdr = self.infer(f(i), t);
            *s += o_sdr.iter().map(|&k| t.sums[k.as_usize()]).sum::<D>();
            if o_sdr.is_empty() {
                *missed += 1
            }
            of(o_sdr)
        });
        (o_vec, targets.iter().map(|(s, _, _)| *s).sum(), targets.iter().map(|(_, s, _)| *s).sum())
    }
}


#[cfg(test)]
mod tests {
    use crate::{DenseWeight, ConvWeights, ConvShape, CpuEccDense, EccLayer, CpuSDR, ConvWeightVec, ConvShapeTrait, HasConvShape, Metric, MetricL1, MetricL2};
    use crate::xorshift::auto_gen_seed64;
    use rand::{SeedableRng, Rng};
    // #[test]
    // fn test10() -> Result<(), String> {
    //     test10_::<u32>()
    // }

    // #[test]
    // fn test10f() -> Result<(), String> {
    //     test10_::<f32, MetricL1<f32>>()
    // }
    // #[test]
    // fn test10fl2() -> Result<(), String> {
    //     test10_::<f32, MetricL2<f32>>()
    // }
    //
    // fn test10_<D: DenseWeight + Send + Sync, M: Metric<D> + Send + Sync>() -> Result<(), String> {
    //     let s = 1646318184253;//auto_gen_seed64();
    //     let mut rng = rand::rngs::StdRng::seed_from_u64(s);
    //     let k = 16;
    //     let mut a = CpuEccDense::<D, M>::new(ConvShape::new([2, 2], [4, 4], [1, 1], 3, 4), 1, &mut rng);
    //     a.set_threshold(D::f32_to_w(0.2));
    //     let mut a2 = a.clone();
    //     println!("seed=={}", s);
    //     assert_eq!(a2.k(), 1);
    //     assert_eq!(a.k(), 1);
    //     assert!(a.get_threshold().eq(a2.get_threshold()), "threshold {:?}!={:?}", a.get_threshold(), a2.get_threshold());
    //     for i in 0..1024 {
    //         let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
    //         let mut input = CpuSDR::from(input);
    //         input.normalize();
    //         assert_ne!(input.len(), 0);
    //         assert!(a.weight_slice().iter().zip(a2.weight_slice().iter()).all(|(a, b)| a.eq(*b)), "w {:?}!={:?}", a.weights().weight_slice(), a2.weights().weight_slice());
    //         assert!(a.population().sums.iter().zip(a2.population().sums.iter()).all(|(a, b)| a.eq(*b)), "sums {:?}!={:?}", a.population().sums, a2.population().sums);
    //         assert!(a.population().activity.iter().zip(a2.population().activity.iter()).all(|(a, b)| a.eq(*b)), "activity {:?}!={:?}", a.population().activity, a2.population().activity);
    //         let mut o = a.run(&input);
    //         // let mut o2 = a2.parallel_run(&input);
    //         assert_eq!(o, o2, "outputs i=={}", i);
    //         a.learn(&input, &o);
    //         a2.parallel_learn(&input, &o2);
    //     }
    //     Ok(())
    // }
}
use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, RangeBounds, Deref, DerefMut};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::cpu_sdr::CpuSDR;
use crate::ecc_program::EccProgram;
use ndalgebra::buffer::Buffer;
use crate::cpu_bitset::CpuBitset;
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use crate::{Shape, resolve_range, EncoderTarget, Synapse, top_large_k_indices, top_small_k_indices, Shape3, from_xyz, Shape2, from_xy, range_contains, SparseOrDense, EccMachine, OclEccSparse, OclEccDense, CpuEccMachine, top_small_k_by_channel, DenseWeight, w_idx, kernel_column_weight_sum, debug_assert_approx_eq_weight, EccLayerD, kernel_column_dropped_weights_count};
use std::collections::{Bound, HashSet};
use crate::vector_field::{VectorFieldOne, VectorFieldDiv, VectorFieldAdd, VectorFieldMul, ArrayCast, VectorFieldSub, VectorFieldPartialOrd};
use crate::population::Population;
use rand::{Rng, SeedableRng};
use crate::xorshift::{auto_gen_seed64, xorshift64, auto_gen_seed, xorshift, xorshift32, auto_gen_seed32};
use itertools::{Itertools, assert_equal};
use std::iter::Sum;
use ocl::core::DeviceInfo::MaxConstantArgs;
use crate::ecc::{EccLayer, as_usize, Idx, as_idx, Rand, xorshift_rand};
use crate::sdr::SDR;
use rand::prelude::SliceRandom;


#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct CpuEccDense<D: DenseWeight> {
    /**The layout is w[output_idx+input_idx_relative_to_kernel_column*output_volume]
    where kernel column has shape [kernel[0],kernel[1],in_channels]*/
    w: Vec<D>,
    // instead of f32 we use u32 but they are equivalent. Just imagine that you divide
    // the u32 value by some large constant and the obtain f32. Addition and subtraction factor out
    //during division (4u32/1000f32)+(8u32/1000f32) == (4u32+8u32)/1000f32
    input_shape: [Idx; 3],
    //[height, width, channels]
    output_shape: [Idx; 3],
    //[height, width, channels]
    kernel: [Idx; 2],
    //[height, width]
    stride: [Idx; 2],
    //[height, width]
    k: Idx,
    top1_per_region:bool,
    pub threshold: D,
    pub plasticity: D,
    activity: Vec<D>,
    pub sums: Vec<D>,
}

impl<D: DenseWeight> CpuEccDense<D> {
    pub fn into_machine(self) -> CpuEccMachine<D> {
        CpuEccMachine::new_singleton(SparseOrDense::Dense(self))
    }
    pub fn restore_dropped_out_weights(&mut self) {
        self.w.iter_mut().filter(|w|!w.is_valid()).for_each(|w|*w=D::ZERO)
    }
    pub fn dropout_f32(&mut self, number_of_connections_to_drop:f32, rng:&mut impl Rng) {
        self.dropout((self.out_volume() as f32 * number_of_connections_to_drop) as usize,rng)
    }
    pub fn dropout(&mut self, number_of_connections_to_drop:usize, rng:&mut impl Rng) {
        assert!(number_of_connections_to_drop<=self.w.len(),"number_of_connections_to_drop={} > number_of_connections=={}",number_of_connections_to_drop,self.w.len());
        let mut indices:Vec<Idx> = (0..as_idx(self.w.len())).collect();
        indices.shuffle(rng);
        for i in 0..number_of_connections_to_drop{
            self.w[as_usize(indices[i])] = D::IMPOSSIBLE_WEIGHT;
        }
        self.renormalise_all()
    }
    pub fn dropout_per_kernel_f32(&mut self, number_of_connections_to_drop_per_kernel_column:f32, rng:&mut impl Rng) {
        self.dropout_per_kernel((self.kernel_column().product() as f32 * number_of_connections_to_drop_per_kernel_column) as usize,rng)
    }
    pub fn dropout_per_kernel(&mut self, number_of_connections_to_drop_per_kernel_column:usize, rng:&mut impl Rng) {
        let kv = self.kernel_column().product();
        let v = self.out_volume();
        assert!(number_of_connections_to_drop_per_kernel_column<=as_usize(kv),"number_of_connections_to_drop_per_kernel_column={} > kernel_column_volume=={}",number_of_connections_to_drop_per_kernel_column,kv);
        for out_idx in 0..v {
            let mut indices:Vec<Idx> = (0..kv).collect();
            indices.shuffle(rng);
            for i in 0..number_of_connections_to_drop_per_kernel_column {
                let idx_within_kernel = indices[i];
                let w_idx = w_idx(out_idx,idx_within_kernel,v);
                self.w[as_usize(w_idx)] = D::IMPOSSIBLE_WEIGHT;
            }
        }
        self.renormalise_all()
    }
    pub fn renormalise_all(&mut self){
        for output_idx in 0..self.out_volume(){
            self.renormalise(output_idx)
        }
    }
    pub fn renormalise(&mut self,output_idx:Idx){
        let kv = self.kernel().product();
        let v = self.out_volume();
        D::normalize_precise(&mut self.w,output_idx,kv,v);
    }
    pub fn set_top1_per_region(&mut self, top1_per_region:bool) {
        if top1_per_region{
            assert_eq!(self.out_channels()%self.k(),0,"k=={} does not divide out_channels=={}",self.k(),self.out_channels());
        }
        self.top1_per_region = top1_per_region;
    }
    pub fn get_region_size(& self)->Idx {
        if self.top1_per_region{
            self.out_channels()/self.k()
        }else{
            self.out_channels()
        }
    }
    pub fn get_top1_per_region(& self)->bool {
        self.top1_per_region
    }
    pub fn set_stride(&mut self, new_stride: [Idx; 2]) {
        let input = self.out_grid().conv_in_size(&new_stride, self.kernel());
        let input = input.add_channels(self.in_channels());
        self.input_shape = input;
        self.stride = new_stride;
    }
    pub fn from_repeated_column(output: [Idx; 2], pretrained: &Self, pretrained_column_pos: [Idx; 2]) -> Self {
        let input = output.conv_in_size(pretrained.stride(), pretrained.kernel());
        let output = output.add_channels(pretrained.out_channels());
        let input = input.add_channels(pretrained.in_channels());
        let kv = pretrained.kernel_column().product();
        let new_v = output.product();
        let old_v = pretrained.out_volume();
        let mut w = vec![D::ZERO; as_usize(kv * new_v)];
        let mut activity = vec![D::ZERO; as_usize(kv * new_v)];
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
            for x in 0..output.width() {
                for y in 0..output.height() {
                    let pos = from_xyz(x, y, channel);
                    let output_idx = output.idx(pos);
                    activity[as_usize(output_idx)] = pretrained.activity[as_usize(pretrained_out_idx)];
                }
            }
        }
        Self {
            w,
            input_shape: input,
            output_shape: output,
            kernel: pretrained.kernel,
            stride: pretrained.stride,
            k: pretrained.k,
            top1_per_region: pretrained.top1_per_region,
            threshold: pretrained.threshold,
            plasticity: pretrained.plasticity,
            activity,
            sums: vec![D::ZERO; as_usize(new_v)],
        }
    }
    pub fn new(output: [Idx; 2], kernel: [Idx; 2], stride: [Idx; 2], in_channels: Idx, out_channels: Idx, k: Idx, rng: &mut impl Rng) -> Self {
        let input = output.conv_in_size(&stride, &kernel);
        let output = output.add_channels(out_channels);
        let v = output.product();
        let input = input.add_channels(in_channels);
        let kernel_column = kernel.add_channels(in_channels);
        let kv = kernel_column.product();
        assert!(k <= output.channels(), "k is larger than layer output");
        let wf: Vec<f32> = (0..kv * v).map(|_| rng.gen()).collect();
        let w = D::initialise_weight_matrix(kv, v, wf);
        let slf = Self {
            w,
            input_shape: input,
            output_shape: output,
            kernel,
            stride,
            k,
            top1_per_region: false,
            threshold: D::default_threshold(out_channels),
            plasticity: D::DEFAULT_PLASTICITY,
            activity: vec![D::INITIAL_ACTIVITY; as_usize(v)],
            sums: vec![D::ZERO; as_usize(v)],
        };
        #[cfg(debug_assertions)] {
            for output_idx in 0..v {
                debug_assert_approx_eq_weight(slf.incoming_weight_sum(output_idx), D::TOTAL_SUM);
            }
            debug_assert_eq!(slf.sums.len(), as_usize(slf.out_volume()));
        }
        slf
    }
    pub fn concat<T>(layers: &[T], f: impl Fn(&T) -> &Self) -> Self {
        assert_ne!(layers.len(), 0, "No layers provided!");
        let first_layer = f(&layers[0]);
        let mut out_shape = first_layer.output_shape;
        let in_shape = first_layer.input_shape;
        let kernel = first_layer.kernel;
        let stride = first_layer.stride;
        assert!(layers.iter().all(|a| f(a).input_shape.all_eq(&in_shape)), "All concatenated layers must have the same input shape!");
        assert!(layers.iter().all(|a| f(a).output_shape.grid().all_eq(out_shape.grid())), "All concatenated layers must have the same output width and height!");
        assert!(layers.iter().all(|a| f(a).stride.all_eq(&stride)), "All concatenated layers must have the same stride!");
        assert!(layers.iter().all(|a| f(a).kernel.all_eq(&kernel)), "All concatenated layers must have the same kernel!");
        let top1 = layers.iter().any(|l|f(l).get_top1_per_region());
        if top1{
            assert!(layers.iter().all(|a| f(a).get_top1_per_region()), "During concatenation, either all layers must be top1_per_region or none of them!");
            assert!(layers.iter().map(|a| f(a).get_region_size()).all_equal(), "All layers are top1_per_region but their region sizes are different");
        }
        let concatenated_sum: Idx = layers.iter().map(|a| f(a).out_channels()).sum();
        *out_shape.channels_mut() = concatenated_sum;
        let new_v = out_shape.product();
        let kernel_column = first_layer.kernel_column();
        let kv = kernel_column.product();

        let mut slf = Self {
            w: vec![D::IMPOSSIBLE_WEIGHT; as_usize(kv * new_v)],
            input_shape: in_shape,
            output_shape: out_shape,
            kernel,
            stride,
            k: first_layer.k,
            top1_per_region:top1,
            threshold: first_layer.threshold,
            plasticity: first_layer.plasticity,
            activity: vec![D::INITIAL_ACTIVITY; as_usize(new_v)],
            sums: vec![D::ZERO; as_usize(new_v)],
        };
        #[cfg(debug_assertions)]
        let mut w_written_to = vec![false; slf.w.len()];

        let mut channel_offset = 0;
        for l in 0..layers.len() {
            let l = f(&layers[l]);
            let v = l.out_volume();
            for w in 0..l.output_shape.width() {
                for h in 0..l.output_shape.height() {
                    for c in 0..l.output_shape.channels() {
                        let original_output_idx = l.output_shape.idx(from_xyz(w, h, c));
                        let new_output_idx = out_shape.idx(from_xyz(w, h, channel_offset + c));
                        slf.activity[as_usize(new_output_idx)] = l.activity[as_usize(original_output_idx)];
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

    pub fn w(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> D {
        self.w[as_usize(self.w_index(input_pos, output_pos))]
    }
    pub fn incoming_weight_sum_f32(&self, output_neuron_idx: Idx) -> f32 {
        D::w_to_f32(self.incoming_weight_sum(output_neuron_idx))
    }
    pub fn incoming_weight_sum(&self, output_neuron_idx: Idx) -> D {
        let kv = self.kernel_column().product();
        let v = self.out_volume();
        kernel_column_weight_sum(kv, v, output_neuron_idx, &self.w)
    }

    pub fn min_activity(&self) -> D {
        self.activity.iter().cloned().reduce(D::min_w).unwrap()
    }
    pub fn max_activity(&self) -> D {
        self.activity.iter().cloned().reduce(D::max_w).unwrap()
    }
    pub fn min_activity_f32(&self) -> f32 {
        D::w_to_f32(self.min_activity())
    }
    pub fn activity(&self, output_idx: usize) -> D {
        self.activity[output_idx]
    }
    pub fn get_activity(&self) -> &[D] {
        &self.activity
    }
    pub fn get_activity_mut(&mut self) -> &mut [D] {
        &mut self.activity
    }
    pub fn get_weights(&self) -> &[D] {
        &self.w
    }
    pub fn get_dropped_weights_count(&self) -> usize{
        self.w.iter().filter(|&w|!w.is_valid()).count()
    }
    pub fn get_dropped_weights_of_kernel_column_count(&self,output_neuron_idx:Idx) -> usize{
        kernel_column_dropped_weights_count(self.kernel_column().product(),self.out_volume(),output_neuron_idx, &self.w)
    }
    pub fn get_weights_mut(&mut self) -> &mut [D] {
        &mut self.w
    }
    pub fn set_initial_activity(&mut self) {
        self.activity.fill(D::INITIAL_ACTIVITY)
    }
    pub fn activity_f32(&self, output_idx: usize) -> f32 {
        D::w_to_f32(self.activity(output_idx))
    }
    fn determine_winners(&self, output: &mut CpuSDR) {
        if self.top1_per_region{
            self.determine_winners_top1_per_region(output)
        }else{
            self.determine_winners_topk(output)
        }
    }
    fn determine_winners_topk(&self, output: &mut CpuSDR) {
        let t = self.threshold;
        top_small_k_by_channel(self, |i| {
            debug_assert!(self.sums[i].le(D::TOTAL_SUM), "{}<={}", self.sums[i], D::TOTAL_SUM);
            debug_assert!(self.activity[i].le(D::INITIAL_ACTIVITY), "{}<={}", self.activity[i], D::INITIAL_ACTIVITY);
            self.sums[i] + self.activity[i]
        }, |i, v| self.sums[i].ge(t), D::gt, output);
    }
    fn determine_winners_top1_per_region(&self, output: &mut CpuSDR) {
        let t = self.threshold;
        let a = as_usize(self.out_area());
        let k = as_usize(self.k());
        assert_eq!(self.out_channels() % self.k(), 0);
        let region_size = as_usize(self.out_channels() / self.k());
        output.clear();
        for region_idx in 0..a * k {//There are k regions per column, each has region_size neurons.
            //We need to pick the top 1 winner within each region.
            //Giving us the total of k winners per output column.
            //The channels of each column are arranged contiguously. Regions are also contiguous.
            let channel_region_offset = region_size * region_idx;
            let mut top1_idx=channel_region_offset;
            let mut top1_val=self.sums[top1_idx]+self.activity[top1_idx];
            for i in channel_region_offset+1..channel_region_offset + region_size{
                let r = self.sums[i]+self.activity[i];
                if r.gt(top1_val){
                    top1_val = r;
                    top1_idx = i;
                }
            }
            if self.sums[top1_idx].ge(t){
                output.push(as_idx(top1_idx));
            }
        }
    }
}

impl OclEccDense {
    pub fn to_cpu(&self) -> CpuEccDense<u32> {
        CpuEccDense {
            w: self.w().to_vec(self.prog().queue()).unwrap(),
            input_shape: *self.in_shape(),
            output_shape: *self.out_shape(),
            kernel: *self.kernel(),
            stride: *self.stride(),
            k: self.k(),
            top1_per_region: false,
            threshold: self.get_threshold(),
            plasticity: self.plasticity,
            activity: self.activity().to_vec(self.prog().queue()).unwrap(),
            sums: vec![0; self.sums.len()],
        }
    }
}

impl CpuEccDense<f32> {
    pub fn reset_activity(&mut self) {
        let min = self.min_activity();
        self.activity.iter_mut().for_each(|a| *a -= min)
    }
}

impl CpuEccDense<u32> {
    pub fn reset_activity(&mut self) {
        let free_space = u32::INITIAL_ACTIVITY - self.max_activity();
        self.activity.iter_mut().for_each(|a| *a += free_space)
    }
    pub fn to_ocl(&self, prog: EccProgram) -> Result<OclEccDense, Error> {
        OclEccDense::new(self, prog)
    }
}
impl<D: DenseWeight> EccLayerD for CpuEccDense<D> {
    type D = D;
    fn get_threshold(&self) -> D {
        self.threshold
    }
    fn set_threshold(&mut self, threshold: D) {
        self.threshold = threshold
    }
    fn get_plasticity(&self) -> D {
        self.plasticity
    }
    fn set_plasticity(&mut self, plasticity: D) {
        self.plasticity = plasticity
    }
}
impl<D: DenseWeight> EccLayer for CpuEccDense<D> {
    type A = CpuSDR;

    fn k(&self) -> Idx { self.k }

    fn set_k(&mut self, k: Idx) {
        assert!(k <= self.out_channels(), "k is larger than layer output!");
        if self.top1_per_region{
            assert_eq!(self.out_channels()%k,0, "k=={} does not divide out_channels=={}! Disable top1_per_region first!",k,self.out_channels());
        }
        self.k = k;
    }


    fn out_shape(&self) -> &[Idx; 3] { &self.output_shape }

    fn in_shape(&self) -> &[Idx; 3] { &self.input_shape }

    fn kernel(&self) -> &[Idx; 2] {
        &self.kernel
    }

    fn stride(&self) -> &[Idx; 2] {
        &self.stride
    }

    fn learnable_parameters(&self) -> usize {
        self.w.len()
    }

    fn get_max_incoming_synapses(&self) -> Idx {
        self.kernel_column().product()
    }
    fn get_threshold_f32(&self) -> f32 {
        D::w_to_f32(self.threshold)
    }

    fn set_threshold_f32(&mut self, fractional: f32) {
        self.threshold = D::f32_to_w(fractional)
    }
    fn set_plasticity_f32(&mut self, fractional: f32) {
        self.plasticity = D::f32_to_w(fractional)
    }
    fn get_plasticity_f32(&self) -> f32 {
        D::w_to_f32(self.plasticity)
    }
    fn new_empty_sdr(&self, capacity: Idx) -> Self::A {
        CpuSDR::new()
    }

    fn infer_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR) {
        debug_assert_eq!(self.sums.len(), as_usize(self.out_volume()));
        self.sums.fill(D::ZERO);
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
            let input_pos: [Idx; 3] = self.input_shape.pos(input_idx);
            let r = input_pos.grid().conv_out_range_clipped(&self.stride, &self.kernel);
            for p0 in r.start.width()..r.end.width().min(self.output_shape.width()) {
                for p1 in r.start.height()..r.end.height().min(self.output_shape.height()) {
                    let kernel_offset = from_xy(p0, p1).conv_in_range_begin(&self.stride);
                    for p2 in 0..self.out_channels() {
                        let output_pos = from_xyz(p0, p1, p2);
                        let output_idx = self.output_shape.idx(output_pos);
                        let w_index = Self::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                        debug_assert_eq!(w_index, self.w_index(&input_pos, &output_pos));
                        debug_assert!(used_w.insert(w_index), "{}", w_index);
                        let w = self.w[as_usize(w_index)];
                        if w.is_valid() { // the connection is disabled
                            self.sums[as_usize(output_idx)] += w;
                            debug_assert!(self.sums[as_usize(output_idx)].le(D::TOTAL_SUM), "{:?}->{:?}={}@{}<={}", input_pos, output_pos, output_idx, self.sums[as_usize(output_idx)], D::TOTAL_SUM);
                        }
                    }
                }
            }
        }
        self.determine_winners(output);
    }

    fn decrement_activities(&mut self, output: &CpuSDR) {
        for &winner in output.iter() {
            self.activity[as_usize(winner)] -= D::ACTIVITY_PENALTY;
        }
    }

    fn learn(&mut self, input: &CpuSDR, output: &CpuSDR) {
        #[cfg(debug_assertions)] {
            let mut i = output.clone();
            i.sort();
            debug_assert!(i.iter().tuple_windows().all(|(prev, next)| prev != next), "{:?}", i);
        }
        let v = self.out_volume();
        let p = self.plasticity;
        let kernel_column = self.kernel_column();
        let kv = kernel_column.product();
        let input_pos: Vec<[Idx; 3]> = input.iter().map(|&i| self.input_shape.pos(i)).collect();
        for &output_idx in output.as_slice() {
            let output_pos = self.output_shape.pos(output_idx);
            let kernel_offset = self.kernel_offset(&output_pos);
            let input_range = output_pos.grid().conv_in_range(&self.stride, &self.kernel);
            let mut active_inputs = 0;
            for (&input_idx, input_pos) in input.iter().zip(input_pos.iter()) {
                if input_range.start.all_le(input_pos.grid()) && input_pos.grid().all_lt(&input_range.end) {
                    let w_index = Self::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
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
            let min_acc = self.min_activity();
            for output_idx in 0..v {
                debug_assert!(self.activity[as_usize(output_idx)].lt(min_acc + D::TOTAL_SUM), "{} @ {} < {}", output_idx, self.activity[as_usize(output_idx)], min_acc)
            }
            debug_assert!(self.w.iter().all(|&w| w.ge(D::ZERO)));
            debug_assert!(self.w.iter().all(|&w| w.le(D::TOTAL_SUM)));
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use std::cmp::Ordering::{Greater, Less};

    #[test]
    fn test2() -> Result<(), String> {
        test2_::<u32>()
    }

    #[test]
    fn test3() -> Result<(), String> {
        test3_::<u32>()
    }

    #[test]
    fn test4() -> Result<(), String> {
        test4_::<u32>()
    }

    #[test]
    fn test5() -> Result<(), String> {
        test5_::<u32>()
    }

    #[test]
    fn test6() -> Result<(), String> {
        test6_::<u32>()
    }

    #[test]
    fn test7() -> Result<(), String> {
        test7_::<u32>()
    }

    #[test]
    fn test2f() -> Result<(), String> {
        test2_::<f32>()
    }

    #[test]
    fn test3f() -> Result<(), String> {
        test3_::<f32>()
    }

    #[test]
    fn test4f() -> Result<(), String> {
        test4_::<f32>()
    }

    #[test]
    fn test5f() -> Result<(), String> {
        test5_::<f32>()
    }

    #[test]
    fn test6f() -> Result<(), String> {
        test6_::<f32>()
    }

    #[test]
    fn test7f() -> Result<(), String> {
        test7_::<f32>()
    }

    fn test2_<D: DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 8;
        let mut a = CpuEccDense::<D>::new([4, 4], [2, 2], [1, 1], 3, 4, 1, &mut rng);
        for _ in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            let mut o = a.run(&input);
            a.learn(&input, &o);
            o.sort();
            assert!(o.is_normalized(), "{:?}", o);
        }
        Ok(())
    }

    fn test3_<D: DenseWeight>() -> Result<(), String> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(634634636);//rand::thread_rng();
        let k = 8;
        let mut a = CpuEccDense::<D>::new([4, 4], [2, 2], [1, 1], 3, 4, 1, &mut rng);
        a.set_plasticity_f32(0.1);//let's see if this breaks anything
        for i in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            let mut o = a.run(&input);
            a.learn(&input, &o);
            o.sort();
            assert!(o.is_normalized(), "{:?}", o);
        }
        Ok(())
    }

    fn test4_<D: DenseWeight>() -> Result<(), String> {
        let s = auto_gen_seed64();
        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
        let k = 16;
        let mut a = CpuEccDense::<D>::new([1, 1], [4, 4], [1, 1], 3, 4, 1, &mut rng);
        a.set_threshold_f32(0.2);
        for i in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            let mut o = a.run(&input);
            a.learn(&input, &o);
            if o.len() == 0 {
                assert!(a.sums.iter().all(|&x| x.lt(a.threshold)), "{:?}", a.sums);
                // println!("(a[{}]=={} < {}) + {}",argmax,a.sums[argmax],a.threshold,a.activity[argmax]);
                // println!("{:?}",a.sums);
                // println!("{:?}",a.activity);
                // println!("{:?}",a.sums.iter().zip(a.activity.iter()).map(|(&a,&b)|a+b).collect::<Vec<D>>());
            } else {
                o.sort();
                assert!(o.is_normalized(), "{:?}", o);
            }
        }
        Ok(())
    }

    fn test5_<D: DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 16;
        let mut a = CpuEccDense::<D>::new([1, 1], [4, 4], [1, 1], 3, 4, 1, &mut rng);
        a.set_threshold_f32(0.99);//let's see if this breaks anything
        for _ in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            let mut o = a.run(&input);
            a.learn(&input, &o);
            assert_eq!(o.len(), 0);
            o.sort();
            assert!(o.is_normalized(), "{:?}", o);
        }
        Ok(())
    }

    fn test6_<D: DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 16;
        let mut a = CpuEccMachine::<D>::new_cpu([1, 1],
                                                &[[5, 5], [3, 3], [3, 3]],
                                                &[[2, 2], [1, 1], [1, 1]],
                                                &[1, 50, 20, 20],
                                                &[10, 1, 1],
                                                &[Some(4), None, None],
                                                &mut rng);
        for _ in 0..16 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            a.run(&input);
            a.learn();
            let o = a.last_output_sdr_mut();
            assert_ne!(o.len(), 0);
            o.sort();
            assert!(o.is_normalized(), "{:?}", o);
        }
        Ok(())
    }


    fn test7_<D: DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 1;
        let mut a = CpuEccMachine::<D>::new_cpu([1, 1],
                                                &[[5, 5], [3, 3], [3, 3]],
                                                &[[2, 2], [1, 1], [1, 1]],
                                                &[1, 5, 2, 2],
                                                &[1, 1, 1],
                                                &[Some(4), None, None],
                                                &mut rng);
        let mut number_of_empty_outputs = 0;
        for _ in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            a.run(&input);
            a.learn();
            let o = a.last_output_sdr_mut();
            if o.is_empty() {
                number_of_empty_outputs += 1;
            }
            o.sort();
            assert!(o.is_normalized(), "{:?}", o);
        }
        assert!(number_of_empty_outputs < 54, "{}", number_of_empty_outputs);
        Ok(())
    }

    #[test]
    fn test8() -> Result<(), String> {
        test8_::<u32>()
    }

    #[test]
    fn test8f() -> Result<(), String> {
        test8_::<f32>()
    }

    fn test8_<D: DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 1;
        let kernel = [3, 3];
        let stride = [1, 1];
        let in_channels = 4;
        let out_channels = [2, 3, 6];
        let mut a: Vec<CpuEccDense<D>> = out_channels.iter().map(|&out_channels|
            CpuEccDense::new([1, 1], kernel, stride, in_channels, out_channels, k, &mut rng)).collect();
        let mut c = CpuEccDense::concat(&a, |a| a);

        let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a[0].in_volume() as u32)).collect();
        let mut input = CpuSDR::from(input);
        input.normalize();
        assert_ne!(input.len(), 0);
        let mut output = CpuSDR::new();
        for a in a.iter_mut().rev() {
            a.run_in_place(&input, &mut output);
            output.shift(a.out_volume() as i32);
        }
        output.sort();
        let mut output2 = c.run(&input);
        output2.sort();
        assert_eq!(output, output2);
        Ok(())
    }


    #[test]
    fn test9() -> Result<(), String> {
        test9_::<u32>()
    }

    #[test]
    fn test9f() -> Result<(), String> {
        test9_::<f32>()
    }

    fn test9_<D: DenseWeight>() -> Result<(), String> {
        let s = auto_gen_seed64();
        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
        let k = 16;
        let mut a = CpuEccDense::<D>::new([1, 1], [4, 4], [1, 1], 3, 4, 1, &mut rng);
        a.set_threshold_f32(0.2);
        let mut a2 = a.clone();
        a2.set_top1_per_region(true);
        println!("seed=={}",s);
        assert_eq!(a2.k(),1);
        assert_eq!(a.k(),1);
        assert!(a.threshold.eq(a2.threshold),"threshold {:?}!={:?}",a.threshold,a2.threshold);
        for i in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            assert!(a.w.iter().zip(a2.w.iter()).all(|(a,b)|a.eq(*b)),"w {:?}!={:?}",a.w,a2.w);
            assert!(a.sums.iter().zip(a2.sums.iter()).all(|(a,b)|a.eq(*b)),"sums {:?}!={:?}",a.sums,a2.sums);
            assert!(a.activity.iter().zip(a2.activity.iter()).all(|(a,b)|a.eq(*b)),"activity {:?}!={:?}",a.activity,a2.activity);
            let mut o = a.run(&input);
            let mut o2 = a2.run(&input);
            assert_eq!(o,o2,"outputs i=={}",i);
            a.learn(&input, &o);
            a2.learn(&input, &o2);
        }
        Ok(())
    }


    #[test]
    fn test10() -> Result<(), String> {
        test10_::<u32>()
    }

    #[test]
    fn test10f() -> Result<(), String> {
        test10_::<f32>()
    }

    fn test10_<D: DenseWeight>() -> Result<(), String> {
        let s = auto_gen_seed64();
        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
        let k = 16;
        let mut a = CpuEccDense::<D>::new([2, 2], [4, 4], [1, 1], 3, 4, 1, &mut rng);
        a.set_threshold_f32(0.2);
        let mut a2 = a.clone();
        a2.set_top1_per_region(true);
        println!("seed=={}",s);
        assert_eq!(a2.k(),1);
        assert_ieq!(a.k(),1);
        assert!(a.threshold.eq(a2.threshold),"threshold {:?}!={:?}",a.threshold,a2.threshold);
        for i in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            assert!(a.w.iter().zip(a2.w.iter()).all(|(a,b)|a.eq(*b)),"w {:?}!={:?}",a.w,a2.w);
            assert!(a.sums.iter().zip(a2.sums.iter()).all(|(a,b)|a.eq(*b)),"sums {:?}!={:?}",a.sums,a2.sums);
            assert!(a.activity.iter().zip(a2.activity.iter()).all(|(a,b)|a.eq(*b)),"activity {:?}!={:?}",a.activity,a2.activity);
            let mut o = a.run(&input);
            let mut o2 = a2.run(&input);
            assert_eq!(o,o2,"outputs i=={}",i);
            a.learn(&input, &o);
            a2.learn(&input, &o2);
        }
        Ok(())
    }


}
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
use crate::{Shape, resolve_range, EncoderTarget, Synapse, top_large_k_indices, top_small_k_indices, Shape3, from_xyz, Shape2, from_xy, range_contains};
use std::collections::{Bound, HashSet};
use crate::vector_field::{VectorFieldOne, VectorFieldDiv, VectorFieldAdd, VectorFieldMul, ArrayCast, VectorFieldSub, VectorFieldPartialOrd};
use crate::population::Population;
use rand::{Rng, SeedableRng};
use crate::xorshift::{auto_gen_seed64, xorshift64, auto_gen_seed, xorshift};
use itertools::{Itertools, assert_equal};
use std::iter::Sum;

pub trait EccLayer {
    fn k(&self) -> usize;
    fn set_k(&mut self, k: usize);
    fn out_shape(&self) -> &[usize; 3];
    fn in_shape(&self) -> &[usize; 3];
    fn kernel(&self) -> &[usize; 2];
    fn stride(&self) -> &[usize; 2];
    fn learnable_paramemters(&self) -> usize;
    fn run(&mut self, input: &CpuSDR) -> CpuSDR {
        let k = self.k();
        let a = self.out_area();
        let mut output = CpuSDR::with_capacity(a * k);
        self.run_in_place(input, &mut output);
        output
    }
    fn run_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR);
    fn learn(&mut self, input: &CpuSDR, output: &CpuSDR);
    fn in_grid(&self) -> &[usize; 2] {
        let [ref grid @ .., _] = self.in_shape();
        grid
    }
    fn out_width(&self) -> usize {
        self.out_shape()[1]
    }
    fn out_height(&self) -> usize {
        self.out_shape()[0]
    }
    fn out_channels(&self) -> usize {
        self.out_shape()[2]
    }
    fn in_width(&self) -> usize {
        self.in_shape()[1]
    }
    fn in_height(&self) -> usize {
        self.in_shape()[0]
    }
    fn in_channels(&self) -> usize {
        self.in_shape()[2]
    }
    fn out_area(&self) -> usize {
        self.out_width() * self.out_height()
    }
    fn out_volume(&self) -> usize {
        self.out_shape().product()
    }
    fn in_volume(&self) -> usize {
        self.in_shape().product()
    }
    fn top_large_k_by_channel<T>(&self, sums: &[T], candidates_per_value: &mut [usize], f: fn(&T) -> usize, threshold: impl Fn(usize) -> bool) -> CpuSDR {
        let a = self.out_area();
        let c = self.out_channels();
        let k = self.k();
        let mut top_k = CpuSDR::with_capacity(k * a);
        for column_idx in 0..a {
            let offset = c * column_idx;
            let range = offset..offset + c;
            top_large_k_indices(k, &sums[range], candidates_per_value, f, |t| if threshold(t) { top_k.push((offset + t) as u32) });
        }
        top_k
    }
    fn top_small_k_by_channel<V: Copy + Debug>(&self, f: impl Fn(usize) -> V, filter: impl Fn(usize, V) -> bool, gt: fn(V, V) -> bool, output: &mut CpuSDR) {
        let a = self.out_area();
        let c = self.out_channels();
        let k = self.k();
        output.clear();
        for column_idx in 0..a {
            let r = c * column_idx;
            for (i, v) in top_small_k_indices(k, c, |i| f(i + r), gt) {
                if filter(i, v) {
                    let e = (r + i) as u32;
                    debug_assert!(!output.as_slice().contains(&e), "{:?}<-{}={}+{}", output, e, r, i);
                    output.push(e);
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct EccSparse {
    /**connections[input_idx]==vector_of_output_indices*/
    connections: Vec<Vec<usize>>,
    max_incoming_synapses: usize,
    input_shape: [usize; 3],
    //[height, width, channels]
    output_shape: [usize; 3],
    //[height, width, channels]
    kernel: [usize; 2],
    stride: [usize; 2],
    k: usize,
    pub threshold: u16,
    pub sums: Vec<u16>,
}

impl EccSparse {
    pub fn new(output: [usize; 2], kernel: [usize; 2], stride: [usize; 2], in_channels: usize, out_channels: usize, k: usize, connections_per_output: usize, rng: &mut impl Rng) -> Self {
        let input = output.conv_in_size(&stride, &kernel);
        let output = [output[0], output[1], out_channels];
        let input = [input[0], input[1], in_channels];
        let in_size = input.product();
        let out_size = output.product();
        let mut pop = Population::new(out_size, 1);
        pop.add_2d_column_grid_with_3d_input(0..in_size, out_channels, connections_per_output, stride, kernel, input, rng);
        let slf = Self::new_from_pop(k, input, output, kernel, stride, &pop);
        debug_assert_eq!(slf.max_incoming_synapses, connections_per_output);
        slf
    }
    pub fn get_threshold_f32(&self) -> f32 {
        self.threshold as f32 / self.max_incoming_synapses as f32
    }
    pub fn set_threshold_f32(&mut self, threshold: f32) {
        assert!(threshold > 0., "Negative threshold!");
        self.threshold = (self.max_incoming_synapses as f32 * threshold).round() as u16
    }
    fn new_from_pop(k: usize, input_shape: [usize; 3], output_shape: [usize; 3], kernel: [usize; 2], stride: [usize; 2], population: &Population) -> Self {
        let mut connections: Vec<Vec<usize>> = (0..input_shape.product()).map(|_| vec![]).collect();
        let mut max_incoming_synapses = population.neurons.iter().map(|n| n.total_synapses()).max().unwrap();
        for (out_idx, neuron) in population.neurons.iter().enumerate() {
            for seg in &neuron.segments {
                for syn in &seg.synapses {
                    connections[syn.input_idx].push(out_idx);
                }
            }
        }
        assert!(k <= output_shape.channels(), "k is larger than layer output");
        Self {
            threshold: 1,
            k,
            input_shape,
            output_shape,
            connections,
            max_incoming_synapses,
            kernel,
            stride,
            sums: vec![0u16; output_shape.product()],
        }
    }
}

impl EccLayer for EccSparse {
    fn k(&self) -> usize { self.k }

    fn set_k(&mut self, k: usize) {
        assert!(k <= self.out_channels(), "k is larger than layer output!");
        self.k = k;
    }

    fn out_shape(&self) -> &[usize; 3] { &self.output_shape }

    fn in_shape(&self) -> &[usize; 3] { &self.input_shape }

    fn kernel(&self) -> &[usize; 2] { &self.kernel }

    fn stride(&self) -> &[usize; 2] { &self.stride }

    fn learnable_paramemters(&self) -> usize {
        0
    }

    fn run_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR) {
        self.sums.fill(0);
        for &input_idx in input.as_slice() {
            for &c in &self.connections[input_idx as usize] {
                self.sums[c] += 1;
            }
        }
        let t = self.threshold;
        self.top_small_k_by_channel(|i| self.sums[i], |i, v| v >= t, |a, b| a > b, output)
    }

    fn learn(&mut self, input: &CpuSDR, output: &CpuSDR) {}
}

pub const MARGIN_OF_SAFETY: u8 = 2;

pub trait DenseWeight: Add<Output=Self> + AddAssign + Sub<Output=Self> + SubAssign + Copy + Debug + Display + Sum{
    const TOTAL_SUM: Self;
    const ZERO: Self;
    const DEFAULT_PLASTICITY: Self;
    const INITIAL_ACTIVITY: Self;
    const ACTIVITY_PENALTY: Self;
    fn w_to_f32(w: Self) -> f32;
    fn gt(self,b:Self) -> bool;
    fn lt(self,b:Self) -> bool;
    fn ge(self,b:Self) -> bool;
    fn le(self,b:Self) -> bool;
    fn eq(self,b:Self) -> bool;
    fn min(self,b:Self)->Self{
        if self.lt(b){self}else{b}
    }
    fn f32_to_w(w: f32) -> Self;
    fn initialise_weight_matrix(kernel_column_volume: usize, output_volume: usize, seed: Vec<f32>) -> Vec<Self>;
    fn default_threshold(out_channels: usize) -> Self;
    fn normalize(weights: &mut [Self], active_inputs: usize, output_idx: usize, plasticity: Self, rand_seed: usize, kernel_column_volume: usize, output_volume: usize) -> usize;
}

fn debug_assert_eq_weight<D:DenseWeight>(a:D, b:D){
    debug_assert!(a.eq(b),"{}!={}",a,b)
}
impl DenseWeight for u32 {
    const TOTAL_SUM: u32 = 1 << (10 + MARGIN_OF_SAFETY);
    const ZERO: Self = 0;
    // ACTIVITY_PENALTY == 2^2
    // TOTAL_SUM == 2^12
    //notice that in f32, the activity penalty becomes
    // ACTIVITY_PENALTY/TOTAL_SUM == 2^2/2^12 == 1/1024 ~= 0.0001
    const DEFAULT_PLASTICITY: u32 = Self::ACTIVITY_PENALTY;
    const INITIAL_ACTIVITY: u32 = u32::MAX - Self::TOTAL_SUM;
    const ACTIVITY_PENALTY: u32 = 1 << MARGIN_OF_SAFETY;

    // We have 21 bits of maneuver.
    // Should be good enough for now
    fn w_to_f32(w: u32) -> f32 {
        (w as f64 / Self::TOTAL_SUM as f64) as f32
    }

    fn gt(self, b: Self) -> bool {
        self>b
    }
    fn lt(self, b: Self) -> bool {
        self<b
    }
    fn ge(self,b:Self) -> bool{
        self>=b
    }
    fn le(self,b:Self) -> bool{
        self<=b
    }
    fn eq(self, b: Self) -> bool {
        self==b
    }

    fn f32_to_w(w: f32) -> u32 {
        (w as f64 * Self::TOTAL_SUM as f64) as u32
    }
    fn initialise_weight_matrix(kernel_column_volume: usize, output_volume: usize, seed: Vec<f32>) -> Vec<u32> {
        let kv = kernel_column_volume;
        let v = output_volume;
        let wf = seed;
        assert_eq!(kv * v, wf.len());
        let mut w: Vec<u32> = vec![u32::MAX; wf.len()];
        for output_idx in 0..v {
            let w_sum = kernel_column_weight_sum(kv, v, output_idx, &wf);
            let mut min_w = u32::MAX;
            let mut min_w_position = 0;
            let mut w_new_sum = 0;
            for input_within_kernel_column in 0..kv {
                let w_idx = w_idx(output_idx, input_within_kernel_column, v);
                let w_f32 = wf[w_idx];
                debug_assert_eq!(u32::MAX, w[w_idx]);
                let w_new = Self::f32_to_w(w_f32 / w_sum);
                w[w_idx] = w_new;
                w_new_sum += w_new;
                if w_new < min_w {
                    min_w = w_new;
                    min_w_position = input_within_kernel_column;
                }
            }
            debug_assert_ne!(min_w, u32::MAX);
            debug_assert_eq!(w_new_sum, kernel_column_weight_sum(kv, v, output_idx, &w));
            let min_w_position = w_idx(output_idx, min_w_position, v);
            w[min_w_position] = w[min_w_position].wrapping_add(Self::TOTAL_SUM.wrapping_sub(w_new_sum)); // we do this step just in case if f32 limited precision
            // caused some small drifts. Safety: Addition and subtraction for both signed and unsigned types are the same operation.
            //So overflows don't bother us.
        }
        debug_assert!(!w.contains(&u32::MAX));
        w
    }
    fn default_threshold(out_channels: usize) -> u32 {
        (Self::TOTAL_SUM as f64 / out_channels as f64) as u32
    }
    fn normalize(weights: &mut [Self], active_inputs: usize, output_idx: usize, plasticity: Self, mut rand_seed: usize, kernel_column_volume: usize, output_volume: usize) -> usize {
        let mut fallback_input_idx = rand_seed % kernel_column_volume;
        for _ in 0..active_inputs {
            rand_seed = xorshift(rand_seed);
            let input_idx_within_kernel_column = rand_seed % kernel_column_volume;
            let w_index = w_idx(output_idx, input_idx_within_kernel_column, output_volume);
            if weights[w_index] >= plasticity {
                weights[w_index] -= plasticity;
            } else {
                loop {
                    let w_index = w_idx(output_idx, fallback_input_idx, output_volume);
                    fallback_input_idx += 1;
                    if fallback_input_idx == kernel_column_volume {
                        fallback_input_idx = 0
                    }
                    if weights[w_index] >= plasticity {
                        weights[w_index] -= plasticity;
                        break;
                    }
                }
            }
        }
        rand_seed
    }
}

const EPSILON:f32=0.00001;
impl DenseWeight for f32 {
    const TOTAL_SUM: f32 = 1.;
    const ZERO: Self = 0.;
    const DEFAULT_PLASTICITY: Self = Self::ACTIVITY_PENALTY;
    const INITIAL_ACTIVITY: Self = 0.;
    const ACTIVITY_PENALTY: Self = 0.0001;

    fn w_to_f32(w: Self) -> f32 {
        w
    }

    fn gt(self, b: Self) -> bool {
        self>b
    }
    fn lt(self, b: Self) -> bool {
        self<b
    }
    fn ge(self,b:Self) -> bool{
        self>=b
    }
    fn le(self,b:Self) -> bool{
        self<=b
    }
    fn eq(self, b: Self) -> bool {
        (self-b).abs()<EPSILON
    }

    fn f32_to_w(w: f32) -> Self {
        w
    }
    fn initialise_weight_matrix(kernel_column_volume: usize, output_volume: usize, seed: Vec<f32>) -> Vec<Self> {
        let kv = kernel_column_volume;
        let v = output_volume;
        let mut w = seed;
        assert_eq!(kv * v, w.len());
        for output_idx in 0..v {
            let w_sum = kernel_column_weight_sum(kv, v, output_idx, &w);
            for input_within_kernel_column in 0..kv {
                let w_idx = w_idx(output_idx, input_within_kernel_column, v);
                w[w_idx] /= w_sum;
            }
            debug_assert_eq_weight(1., kernel_column_weight_sum(kv, v, output_idx, &w));
        }
        w
    }
    fn default_threshold(out_channels: usize) -> Self {
        Self::TOTAL_SUM / out_channels as f32
    }
    fn normalize(weights: &mut [Self], active_inputs: usize, output_idx: usize, plasticity: Self, mut rand_seed: usize, kernel_column_volume: usize, output_volume: usize) -> usize {
        let kv = kernel_column_volume;
        let v = output_volume;
        let w_sum = Self::TOTAL_SUM + active_inputs as f32 * plasticity;
        debug_assert_eq_weight(w_sum,kernel_column_weight_sum(kv, v, output_idx, &weights));
        for input_within_kernel_column in 0..kv {
            let w_idx = w_idx(output_idx, input_within_kernel_column, v);
            weights[w_idx] /= w_sum;
        }
        debug_assert_eq_weight(1.,kernel_column_weight_sum(kv, v, output_idx, &weights));
        rand_seed
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct EccDense<D: DenseWeight> {
    /**The layout is w[output_idx+input_idx_relative_to_kernel_column*output_volume]
    where kernel column has shape [kernel[0],kernel[1],in_channels]*/
    w: Vec<D>,
    // instead of f32 we use u32 but they are equivalent. Just imagine that you divide
    // the u32 value by some large constant and the obtain f32. Addition and subtraction factor out
    //during division (4u32/1000f32)+(8u32/1000f32) == (4u32+8u32)/1000f32
    input_shape: [usize; 3],
    //[height, width, channels]
    output_shape: [usize; 3],
    //[height, width, channels]
    kernel: [usize; 2],
    //[height, width]
    stride: [usize; 2],
    //[height, width]
    k: usize,
    pub threshold: D,
    pub plasticity: D,
    activity: Vec<D>,
    pub rand_seed: usize,
    pub sums: Vec<D>,
}
#[inline]
fn w_idx(output_idx: usize, idx_within_kernel_column: usize, output_volume: usize) -> usize {
    debug_assert!(output_idx < output_volume);
    output_idx + idx_within_kernel_column * output_volume
}
#[inline]
fn kernel_column_weight_sum<D:Sum+Copy>(kernel_column_volume: usize, out_volume: usize, output_neuron_idx: usize, w: &[D]) -> D {
    assert!(output_neuron_idx < out_volume);
    (0..kernel_column_volume).map(|i| w[w_idx(output_neuron_idx, i, out_volume)]).sum()
}
impl<D: DenseWeight> EccDense<D> {
    pub fn new(output: [usize; 2], kernel: [usize; 2], stride: [usize; 2], in_channels: usize, out_channels: usize, k: usize, rng: &mut impl Rng) -> Self {
        let input = output.conv_in_size(&stride, &kernel);
        let output = [output[0], output[1], out_channels];
        let v = output.product();
        let input = [input[0], input[1], in_channels];
        let kernel_column = [kernel[0], kernel[1], in_channels];
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
            threshold: D::default_threshold(out_channels),
            plasticity: D::DEFAULT_PLASTICITY,
            activity: vec![D::INITIAL_ACTIVITY; output.product()],
            rand_seed: auto_gen_seed(),
            sums: vec![D::ZERO; output.product()],
        };
        #[cfg(debug_assertions)] {
            for output_idx in 0..v {
                debug_assert_eq_weight(slf.incoming_weight_sum(output_idx), D::TOTAL_SUM);
            }
        }
        slf
    }
    pub fn set_threshold(&mut self, fractional: f32) {
        self.threshold = D::f32_to_w(fractional)
    }
    pub fn set_plasticity(&mut self, fractional: f32) {
        self.plasticity = D::f32_to_w(fractional)
    }
    pub fn get_threshold(&self) -> f32 {
        D::w_to_f32(self.threshold)
    }
    pub fn get_plasticity(&self) -> f32 {
        D::w_to_f32(self.plasticity)
    }
    pub fn kernel_column(&self) -> [usize; 3] {
        self.kernel.add_channels(self.in_channels())
    }
    pub fn kernel_offset(&self, output_pos: &[usize; 3]) -> [usize; 2] {
        output_pos.grid().conv_in_range_begin(&self.stride)
    }
    pub fn pos_within_kernel(&self, input_pos: &[usize; 3], output_pos: &[usize; 3]) -> [usize; 3] {
        debug_assert!(output_pos.all_lt(&self.output_shape));
        debug_assert!(input_pos.all_lt(&self.input_shape));
        debug_assert!(range_contains(&output_pos.grid().conv_in_range(&self.stride, &self.kernel), input_pos.grid()));
        debug_assert!(range_contains(&input_pos.grid().conv_out_range_clipped(&self.stride, &self.kernel), output_pos.grid()));
        Self::sub_kernel_offset(input_pos, &self.kernel_offset(output_pos))
    }
    fn sub_kernel_offset(input_pos: &[usize; 3], offset: &[usize; 2]) -> [usize; 3] {
        from_xyz(input_pos.width() - offset.width(), input_pos.height() - offset.height(), input_pos.channels())
    }

    #[inline]
    fn w_index_(input_pos: &[usize; 3], kernel_offset: &[usize; 2], output_idx: usize, kernel_column: &[usize; 3], output_volume: usize) -> usize {
        let position_within_kernel_column = Self::sub_kernel_offset(input_pos, kernel_offset);
        w_idx(output_idx, kernel_column.idx(position_within_kernel_column), output_volume)
    }
    pub fn idx_within_kernel(&self, input_pos: &[usize; 3], output_pos: &[usize; 3]) -> usize {
        self.kernel_column().idx(self.pos_within_kernel(input_pos, output_pos))
    }
    pub fn w_index(&self, input_pos: &[usize; 3], output_pos: &[usize; 3]) -> usize {
        debug_assert!(output_pos.all_lt(&self.output_shape));
        debug_assert!(input_pos.all_lt(&self.input_shape));
        debug_assert!(range_contains(&output_pos.grid().conv_in_range(&self.stride, &self.kernel), input_pos.grid()));
        debug_assert!(range_contains(&input_pos.grid().conv_out_range_clipped(&self.stride, &self.kernel), output_pos.grid()));
        w_idx(self.out_shape().idx(*output_pos), self.idx_within_kernel(input_pos, output_pos), self.out_volume())
    }
    pub fn w(&self, input_pos: &[usize; 3], output_pos: &[usize; 3]) -> D {
        self.w[self.w_index(input_pos, output_pos)]
    }
    pub fn incoming_weight_sum_f32(&self, output_neuron_idx: usize) -> f32 {
        D::w_to_f32(self.incoming_weight_sum(output_neuron_idx))
    }
    pub fn incoming_weight_sum(&self, output_neuron_idx: usize) -> D {
        let kv = self.kernel_column().product();
        let v = self.out_volume();
        kernel_column_weight_sum(kv, v, output_neuron_idx, &self.w)
    }


    pub fn min_activity(&self) -> D {
        self.activity.iter().cloned().reduce(D::min).unwrap()
    }
    pub fn min_activity_f32(&self) -> f32 {
        D::w_to_f32(self.min_activity())
    }
    pub fn activity(&self, output_idx: usize) -> D {
        self.activity[output_idx]
    }
    pub fn activity_f32(&self, output_idx: usize) -> f32 {
        D::w_to_f32(self.activity(output_idx))
    }
}

impl<D: DenseWeight> EccLayer for EccDense<D> {
    fn k(&self) -> usize { self.k }

    fn set_k(&mut self, k: usize) {
        assert!(k <= self.out_channels(), "k is larger than layer output!");
        self.k = k;
    }

    fn out_shape(&self) -> &[usize; 3] { &self.output_shape }

    fn in_shape(&self) -> &[usize; 3] { &self.input_shape }

    fn kernel(&self) -> &[usize; 2] {
        &self.kernel
    }

    fn stride(&self) -> &[usize; 2] {
        &self.stride
    }

    fn learnable_paramemters(&self) -> usize {
        self.w.len()
    }

    fn run_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR) {
        self.sums.fill(D::ZERO);
        let kernel_column = self.kernel_column();
        let v = self.out_volume();
        #[cfg(debug_assertions)] {
            let mut i = input.clone();
            i.sort();
            debug_assert!(i.iter().tuple_windows().all(|(prev, next)| prev != next), "{:?}", i);
            for output_idx in 0..v {
                debug_assert_eq_weight(self.incoming_weight_sum(output_idx), D::TOTAL_SUM);
            }
        }
        let mut used_w = HashSet::new();
        for &input_idx in input.as_slice() {
            let input_pos: [usize; 3] = self.input_shape.pos(input_idx as usize);
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
                        let w = self.w[w_index];
                        self.sums[output_idx] += w;
                        debug_assert!(self.sums[output_idx].lt(D::TOTAL_SUM), "{:?}->{:?}={}@{}<={}", input_pos, output_pos, output_idx, self.sums[output_idx], D::TOTAL_SUM);
                    }
                }
            }
        }
        let t = self.threshold;
        self.top_small_k_by_channel(|i| {
            debug_assert!(self.sums[i].le(D::TOTAL_SUM), "{}<={}", self.sums[i], D::TOTAL_SUM);
            debug_assert!(self.activity[i].le(D::INITIAL_ACTIVITY), "{}<={}", self.activity[i], D::INITIAL_ACTIVITY);
            self.sums[i] + self.activity[i]
        }, |i, v| self.sums[i].ge(t), D::gt, output);
        for &winner in output.iter() {
            self.activity[winner as usize] -= D::ACTIVITY_PENALTY;
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
        let one_minus_p = D::TOTAL_SUM - p;
        let kernel_column = self.kernel_column();
        let kv = kernel_column.product();
        let input_pos: Vec<[usize; 3]> = input.iter().map(|&i| self.input_shape.pos(i as usize)).collect();
        let mut rand_seed = xorshift(self.rand_seed);
        for &output_idx in output.as_slice() {
            let output_idx = output_idx as usize;
            let output_pos = self.output_shape.pos(output_idx);
            let kernel_offset = self.kernel_offset(&output_pos);
            let input_range = output_pos.grid().conv_in_range(&self.stride, &self.kernel);
            let mut active_inputs = 0;
            for (&input_idx, input_pos) in input.iter().zip(input_pos.iter()) {
                if input_range.start.all_le(input_pos.grid()) && input_pos.grid().all_lt(&input_range.end) {
                    let w_index = Self::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                    debug_assert_eq!(w_index, self.w_index(input_pos, &output_pos));
                    if self.w[w_index].le(one_minus_p) {
                        self.w[w_index] += p;
                        active_inputs += 1;
                    }
                }
            }
            rand_seed = D::normalize(&mut self.w, active_inputs, output_idx, p, rand_seed, kv, v);
        }
        #[cfg(debug_assertions)] {
            for output_idx in 0..v {
                debug_assert_eq_weight(self.incoming_weight_sum(output_idx), D::TOTAL_SUM)
            }
            let min_acc = self.min_activity();
            for output_idx in 0..v {
                debug_assert!(self.activity[output_idx].lt(min_acc + D::TOTAL_SUM), "{} @ {} < {}", output_idx, self.activity[output_idx], min_acc)
            }
            debug_assert!(self.w.iter().all(|&w| w.ge(D::ZERO)));
            debug_assert!(self.w.iter().all(|&w| w.le(D::TOTAL_SUM)));
        }
        self.rand_seed = rand_seed;
    }
}


#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum SparseOrDense<D: DenseWeight> {
    Sparse(EccSparse),
    Dense(EccDense<D>),
}

impl<D: DenseWeight> EccLayer for SparseOrDense<D> {
    fn k(&self) -> usize {
        match self {
            SparseOrDense::Sparse(a) => a.k(),
            SparseOrDense::Dense(a) => a.k()
        }
    }

    fn set_k(&mut self, k: usize) {
        match self {
            SparseOrDense::Sparse(a) => a.set_k(k),
            SparseOrDense::Dense(a) => a.set_k(k)
        }
    }

    fn out_shape(&self) -> &[usize; 3] {
        match self {
            SparseOrDense::Sparse(a) => a.out_shape(),
            SparseOrDense::Dense(a) => a.out_shape()
        }
    }

    fn in_shape(&self) -> &[usize; 3] {
        match self {
            SparseOrDense::Sparse(a) => a.in_shape(),
            SparseOrDense::Dense(a) => a.in_shape()
        }
    }

    fn kernel(&self) -> &[usize; 2] {
        match self {
            SparseOrDense::Sparse(a) => a.kernel(),
            SparseOrDense::Dense(a) => a.kernel()
        }
    }

    fn stride(&self) -> &[usize; 2] {
        match self {
            SparseOrDense::Sparse(a) => a.stride(),
            SparseOrDense::Dense(a) => a.stride()
        }
    }

    fn learnable_paramemters(&self) -> usize {
        match self {
            SparseOrDense::Sparse(a) => a.learnable_paramemters(),
            SparseOrDense::Dense(a) => a.learnable_paramemters()
        }
    }

    fn run_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR) {
        match self {
            SparseOrDense::Sparse(a) => a.run_in_place(input, output),
            SparseOrDense::Dense(a) => a.run_in_place(input, output)
        }
    }

    fn learn(&mut self, input: &CpuSDR, output: &CpuSDR) {
        match self {
            SparseOrDense::Sparse(a) => a.learn(input, output),
            SparseOrDense::Dense(a) => a.learn(input, output)
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CpuEccMachine<D: DenseWeight> {
    ecc: Vec<SparseOrDense<D>>,
    inputs: Vec<CpuSDR>,
}

impl<D: DenseWeight> Deref for CpuEccMachine<D> {
    type Target = Vec<SparseOrDense<D>>;

    fn deref(&self) -> &Self::Target {
        &self.ecc
    }
}

impl<D: DenseWeight> DerefMut for CpuEccMachine<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ecc
    }
}

impl<D: DenseWeight> CpuEccMachine<D> {
    pub fn new(output: [usize; 2], kernels: &[[usize; 2]], strides: &[[usize; 2]], channels: &[usize], k: &[usize], connections_per_output: &[Option<usize>], rng: &mut impl Rng) -> Self {
        let layers = kernels.len();

        assert!(layers > 0);
        assert_eq!(layers, strides.len());
        assert_eq!(layers, k.len());
        assert_eq!(layers, connections_per_output.len());
        assert_eq!(layers + 1, channels.len());
        let mut layers_vec = Vec::<SparseOrDense<D>>::with_capacity(layers);
        let mut prev_output = output;
        for i in (0..layers).rev() {
            let in_channels = channels[i];
            let out_channels = channels[i + 1];
            let k = k[i];
            let kernel = kernels[i];
            let stride = strides[i];
            let l = if let Some(connections_per_output) = connections_per_output[i] {
                SparseOrDense::Sparse(EccSparse::new(prev_output, kernel, stride, in_channels, out_channels, k, connections_per_output, rng))
            } else {
                SparseOrDense::Dense(EccDense::new(prev_output, kernel, stride, in_channels, out_channels, k, rng))
            };
            prev_output = *l.in_shape().grid();
            layers_vec.push(l);
        }
        layers_vec.reverse();
        #[cfg(debug_assertions)] {
            let last = layers_vec.last().unwrap().out_shape();
            debug_assert!(last.grid().all_eq(&output), "{:?}=={:?}", last.grid(), output);
            debug_assert_eq!(last.channels(), *channels.last().unwrap());
            debug_assert_eq!(layers_vec[0].in_channels(), channels[0]);
            for (prev, next) in layers_vec.iter().tuple_windows() {
                debug_assert!(prev.out_shape().all_eq(next.in_shape()), "{:?}=={:?}", prev.out_shape(), next.in_shape());
            }
        }
        Self { ecc: layers_vec, inputs: (0..channels.len()).map(|_| CpuSDR::new()).collect() }
    }
    pub fn learnable_paramemters(&self) -> usize {
        self.ecc.iter().map(|w| w.learnable_paramemters()).sum()
    }
    pub fn input_sdr(&self, layer_index: usize) -> &CpuSDR {
        &self.inputs[layer_index]
    }
    pub fn input_sdr_mut(&mut self, layer_index: usize) -> &mut CpuSDR {
        &mut self.inputs[layer_index]
    }
    pub fn output_sdr(&self, layer_index: usize) -> &CpuSDR {
        &self.inputs[layer_index + 1]
    }
    pub fn output_sdr_mut(&mut self, layer_index: usize) -> &mut CpuSDR {
        &mut self.inputs[layer_index + 1]
    }
    pub fn last_output_sdr(&self) -> &CpuSDR {
        self.inputs.last().unwrap()
    }
    pub fn last_output_sdr_mut(&mut self) -> &mut CpuSDR {
        self.inputs.last_mut().unwrap()
    }
    pub fn learn(&mut self) {
        let Self { ecc, inputs } = self;
        for (i, layer) in ecc.iter_mut().enumerate() {
            let (prev, next) = inputs.as_slice().split_at(i + 1);
            layer.learn(&prev[i], &next[0]);
        }
    }
    pub fn run(&mut self, input: &CpuSDR) -> &CpuSDR {
        let Self { ecc, inputs } = self;
        inputs[0].set(input.as_slice());
        for (i, layer) in ecc.iter_mut().enumerate() {
            let (prev, next) = inputs.as_mut_slice().split_at_mut(i + 1);
            layer.run_in_place(&prev[i], &mut next[0]);
        }
        self.last_output_sdr()
    }
    pub fn in_shape(&self) -> &[usize; 3] {
        self.ecc[0].in_shape()
    }
    pub fn in_channels(&self) -> usize {
        self.ecc[0].in_channels()
    }
    pub fn in_volume(&self) -> usize {
        self.ecc[0].in_volume()
    }
    pub fn out_shape(&self) -> &[usize; 3] {
        self.ecc.last().unwrap().out_shape()
    }
    pub fn out_channels(&self) -> usize {
        self.ecc.last().unwrap().out_channels()
    }
    pub fn out_volume(&self) -> usize {
        self.ecc.last().unwrap().out_volume()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test1() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 8;
        let mut a = EccSparse::new([4, 4], [2, 2], [1, 1], 3, 4, 1, 4, &mut rng);
        for _ in 0..64 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            let o = a.run(&input);
            assert_ne!(o.len(), 0);
        }
        Ok(())
    }
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
    fn test2_<D:DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 8;
        let mut a = EccDense::<D>::new([4, 4], [2, 2], [1, 1], 3, 4, 1, &mut rng);
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

    fn test3_<D:DenseWeight>() -> Result<(), String> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(634634636);//rand::thread_rng();
        let k = 8;
        let mut a = EccDense::<D>::new([4, 4], [2, 2], [1, 1], 3, 4, 1, &mut rng);
        a.rand_seed = 34634;
        a.set_plasticity(0.1);//let's see if this breaks anything
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

    fn test4_<D:DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 16;
        let mut a = EccDense::<D>::new([1, 1], [4, 4], [1, 1], 3, 4, 1, &mut rng);
        a.set_threshold(0.2);
        for i in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            let mut o = a.run(&input);
            a.learn(&input, &o);
            assert_ne!(o.len(), 0);
            o.sort();
            assert!(o.is_normalized(), "{:?}", o);
        }
        Ok(())
    }

    fn test5_<D:DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 16;
        let mut a = EccDense::<D>::new([1, 1], [4, 4], [1, 1], 3, 4, 1, &mut rng);
        a.set_threshold(0.99);//let's see if this breaks anything
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

    fn test6_<D:DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 16;
        let mut a = CpuEccMachine::<D>::new([1, 1],
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


    fn test7_<D:DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 1;
        let mut a = CpuEccMachine::<D>::new([1, 1],
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
}
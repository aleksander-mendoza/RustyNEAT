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
            for (i, v) in top_small_k_indices(k, c, |i| {
                debug_assert!(i<c);
                f(i + r)
            }, gt) {
                let e = r + i;
                debug_assert!(r<=e);
                debug_assert!(e<r+c);
                if filter(e, v) {
                    let e = e as u32;
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
    pub fn get_max_incoming_synapses(&self)-> usize{
        self.max_incoming_synapses
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
            debug_assert_eq!(slf.sums.len(),slf.out_volume());
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
    pub fn get_activity(&self) -> &[D] {
        &self.activity
    }
    pub fn get_activity_mut(&mut self) -> &mut [D] {
        &mut self.activity
    }
    pub fn get_weights(&self) -> &[D] {
        &self.w
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

    fn determine_winners(&self, output:&mut CpuSDR){
        let t = self.threshold;
        self.top_small_k_by_channel(|i| {
            debug_assert!(self.sums[i].le(D::TOTAL_SUM), "{}<={}", self.sums[i], D::TOTAL_SUM);
            debug_assert!(self.activity[i].le(D::INITIAL_ACTIVITY), "{}<={}", self.activity[i], D::INITIAL_ACTIVITY);
            self.sums[i] + self.activity[i]
        }, |i, v| self.sums[i].ge(t), D::gt, output);
    }
}
impl EccDense<f32>{
    pub fn reset_activity(&mut self) {
        let min = self.min_activity();
        self.activity.iter_mut().for_each(|a|*a-=min)
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
        debug_assert_eq!(self.sums.len(),self.out_volume());
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
        self.determine_winners(output);
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
    use std::cmp::Ordering::{Greater, Less};

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
        let s = auto_gen_seed64();
        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
        let k = 16;
        let mut a = EccDense::<D>::new([1, 1], [4, 4], [1, 1], 3, 4, 1, &mut rng);
        a.set_threshold(0.2);
        println!("{} {}", s, a.rand_seed);
        for i in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            let mut o = a.run(&input);
            a.learn(&input, &o);
            if o.len()==0{
                assert!(a.sums.iter().all(|&x|x.lt(a.threshold)),"{:?}",a.sums);
                // println!("(a[{}]=={} < {}) + {}",argmax,a.sums[argmax],a.threshold,a.activity[argmax]);
                // println!("{:?}",a.sums);
                // println!("{:?}",a.activity);
                // println!("{:?}",a.sums.iter().zip(a.activity.iter()).map(|(&a,&b)|a+b).collect::<Vec<D>>());
            }else {
                o.sort();
                assert!(o.is_normalized(), "{:?}", o);
            }
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


    #[test]
    fn test8() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 1;
        let t = 0.05000000074505806;
        let sums = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04444211348891258, 0.12327639758586884, 0.1591585874557495, 0.37142419815063477, 0.1458722949028015, 0.18101705610752106, 0.07984745502471924, 0.09392409771680832, 0.28591883182525635, 0.25996142625808716, 0.05837327986955643, 0.07534068822860718, 0.02638549543917179, 0.03111913986504078, 0.0538882277905941, 0.11917483061552048, 0.022866543382406235, 0.535748302936554, 0.05473457649350166, 0.08664093911647797, 0.16509628295898438, 0.09676801413297653, 0.1495167762041092, 0.19059470295906067, 0.07802321761846542, 0.5128775835037231, 0.1412210762500763, 0.1276867687702179, 0.07084469497203827, 0.13323092460632324, 0.14701993763446808, 0.06561692804098129, 0.09521125257015228, 0.18714620172977448, 0.32858362793922424, 0.20590262115001678, 0.11769439280033112, 0.24150244891643524, 0.17753037810325623, 0.2823231518268585, 0.20134755969047546, 0.24498429894447327, 0.17277522385120392, 0.14624032378196716, 0.10877879709005356, 0.2363521009683609, 0.16594263911247253, 0.521333634853363, 0.2585673928260803, 0.1812845915555954, 0.3046097159385681, 0.34775063395500183, 0.11677031964063644, 0.11035807430744171, 0.34649157524108887, 0.26169756054878235, 0.21686798334121704, 0.17656058073043823, 0.24505245685577393, 0.3260481059551239, 0.2931072413921356, 0.1575961709022522, 0.2640974819660187, 0.3540113568305969, 0.17032793164253235, 0.4183821976184845, 0.30102676153182983, 0.3634154796600342, 0.2378334105014801, 0.17686475813388824, 0.15950627624988556, 0.322731077671051, 0.22932423651218414, 0.2653253376483917, 0.18282395601272583, 0.15332850813865662, 0.26601916551589966, 0.2580893635749817, 0.12949973344802856, 0.27307116985321045, 0.16349144279956818, 0.09260819852352142, 0.1735755354166031, 0.12186183035373688, 0.16634218394756317, 0.34893038868904114, 0.24570071697235107, 0.25410884618759155, 0.2843794524669647, 0.15955579280853271, 0.2067680060863495, 0.2322847545146942, 0.14676088094711304, 0.24323062598705292, 0.13399600982666016, 0.10227569192647934, 0.2570514678955078, 0.1857430636882782, 0.2561749219894409, 0.34321293234825134, 0.0941515564918518, 0.16996799409389496, 0.14021912217140198, 0.22644974291324615, 0.23902679979801178, 0.10767314583063126, 0.16901640594005585, 0.24748459458351135, 0.06830332428216934, 0.0858033299446106, 0.12574516236782074, 0.18031054735183716, 0.13639099895954132, 0.08263755589723587, 0.42127642035484314, 0.052585311233997345, 0.1441417932510376, 0.18277324736118317, 0.08247408270835876, 0.08146771043539047, 0.029344161972403526, 0.03501490131020546, 0.03230304270982742, 0.06417171657085419, 0.13618794083595276, 0.014044813811779022, 0.04813302680850029, 0.03977655619382858, 0.023667259141802788, 0.005237135104835033, 0.11192092299461365, 0.013068132102489471, 0.01397591084241867, 0.12616081535816193, 0.015744736418128014, 0.007631942164152861, 0.03654234856367111, 0.01602862775325775, 0.06239132210612297, 0.010476917959749699, 0.011086147278547287, 0.0602642223238945, 0.026684163138270378, 0.12378256022930145, 0.03404301777482033, 0.21932628750801086, 0.013873803429305553, 0.21843495965003967, 0.015179943293333054, 0.06597188115119934, 0.06220380961894989, 0.031575631350278854, 0.014075168408453465, 0.046479493379592896, 0.010161397978663445, 0.07733651250600815, 0.0651560053229332, 0.18764421343803406, 0.03955773636698723, 0.024983882904052734, 0.09644091129302979, 0.15892143547534943, 0.3707204759120941, 0.30536743998527527, 0.09708429872989655, 0.4539055824279785, 0.15204450488090515, 0.048780668526887894, 0.33666032552719116, 0.07567237317562103, 0.09255547821521759, 0.06287240236997604, 0.062330327928066254, 0.20062609016895294, 0.06531121581792831, 0.17781896889209747, 0.06069526821374893, 0.10766284912824631, 0.14116501808166504, 0.10302668809890747, 0.16583114862442017, 0.177156463265419, 0.46423840522766113, 0.32988491654396057, 0.1977335810661316, 0.32144519686698914, 0.1900670826435089, 0.1338469237089157, 0.15850616991519928, 0.26845207810401917, 0.09956073015928268, 0.10272596031427383, 0.0919032022356987, 0.16316454112529755, 0.0777478739619255, 0.18603534996509552, 0.16325563192367554, 0.3384982645511627, 0.14971540868282318, 0.1718268096446991, 0.22379781305789948, 0.1876305788755417, 0.4644263684749603, 0.2974265217781067, 0.1932307630777359, 0.2942282557487488, 0.2753716707229614, 0.34281978011131287, 0.10575363785028458, 0.30366164445877075, 0.22759437561035156, 0.1156592071056366, 0.08932986855506897, 0.15174567699432373, 0.3063415586948395, 0.15236057341098785, 0.2519376873970032, 0.4756978750228882, 0.166803240776062, 0.2362116426229477, 0.3160053789615631, 0.24389512836933136, 0.36013442277908325, 0.2252649962902069, 0.12950414419174194, 0.4058530926704407, 0.1671471744775772, 0.21901065111160278, 0.21746402978897095, 0.13075858354568481, 0.2721801698207855, 0.1945560872554779, 0.14663533866405487, 0.23577935993671417, 0.23072490096092224, 0.27453479170799255, 0.18318982422351837, 0.28129810094833374, 0.25743356347084045, 0.11638158559799194, 0.245628222823143, 0.3183741569519043, 0.1465488225221634, 0.08687009662389755, 0.18077053129673004, 0.32830512523651123, 0.16020692884922028, 0.33926209807395935, 0.12426529824733734, 0.23436768352985382, 0.171150341629982, 0.16562053561210632, 0.11931775510311127, 0.19975364208221436, 0.21477392315864563, 0.15664632618427277, 0.17087620496749878, 0.14356760680675507, 0.2292921543121338, 0.1825459897518158, 0.16401244699954987, 0.19807639718055725, 0.1556788831949234, 0.12066211551427841, 0.1649763137102127, 0.12759855389595032, 0.14028146862983704, 0.1275387555360794, 0.34190601110458374, 0.3278481662273407, 0.09837888181209564, 0.15871088206768036, 0.12081622332334518, 0.0765913650393486, 0.17038686573505402, 0.12200640141963959, 0.15290650725364685, 0.24787171185016632, 0.09340502321720123, 0.10556550323963165, 0.09589973092079163, 0.08814462274312973, 0.109145887196064, 0.12190712243318558, 0.10203958302736282, 0.07347624748945236, 0.1049770638346672, 0.0722687840461731, 0.05934770777821541, 0.047731880098581314, 0.07906974107027054, 0.1671401858329773, 0.06306370347738266, 0.08126034587621689, 0.06260815262794495, 0.3362793028354645, 0.12143594026565552, 0.05841483175754547, 0.08287452906370163, 0.07037195563316345, 0.1509200930595398, 0.04883773624897003, 0.15999667346477509, 0.06326816231012344, 0.03443196043372154, 0.057685866951942444, 0.04042448103427887, 0.03703588992357254, 0.09099095314741135, 0.05337155982851982, 0.1694362610578537, 0.03458227589726448, 0.02379200980067253, 0.11512397229671478, 0.07958433777093887, 0.2182975709438324, 0.21195928752422333, 0.25438350439071655, 0.019041579216718674, 0.10197795927524567, 0.19042959809303284, 0.05724875628948212, 0.2558080852031708, 0.0854443833231926, 0.286624938249588, 0.11819008737802505, 0.1268143504858017, 0.17442642152309418, 0.39262425899505615, 0.1744309961795807, 0.1066296398639679, 0.0871671736240387, 0.23608076572418213, 0.12262570858001709, 0.06760376691818237, 0.14574946463108063, 0.055674243718385696, 0.055777523666620255, 0.28402581810951233, 0.08626915514469147, 0.1632896214723587, 0.08963030576705933, 0.37807321548461914, 0.2114696502685547, 0.1683070957660675, 0.2614889144897461, 0.24955341219902039, 0.172780841588974, 0.2148219496011734, 0.1382674276828766, 0.22450073063373566, 0.09882986545562744, 0.11543857306241989, 0.18542371690273285, 0.09888331592082977, 0.2967163622379303, 0.19610914587974548, 0.32920899987220764, 0.11794531345367432, 0.12555313110351562, 0.23287126421928406, 0.18089735507965088, 0.2503531575202942, 0.1661348044872284, 0.18944352865219116, 0.11776366829872131, 0.15308819711208344, 0.28927865624427795, 0.3986513912677765, 0.1474108248949051, 0.10693846642971039, 0.16328275203704834, 0.3105486035346985, 0.10135474056005478, 0.3811855912208557, 0.22665855288505554, 0.11068868637084961, 0.17081937193870544, 0.13550013303756714, 0.1848316341638565, 0.1734185516834259, 0.2548463046550751, 0.29857751727104187, 0.10197148472070694, 0.24482059478759766, 0.16937005519866943, 0.2245362102985382, 0.2584066390991211, 0.1687038540840149, 0.18760165572166443, 0.12090662866830826, 0.2608185112476349, 0.24722103774547577, 0.15074700117111206, 0.26028284430503845, 0.3238866627216339, 0.23555095493793488, 0.4117514193058014, 0.18153752386569977, 0.22355061769485474, 0.2640373706817627, 0.23703232407569885, 0.24180005490779877, 0.1753281056880951, 0.2665894329547882, 0.181670144200325, 0.24426788091659546, 0.3016436994075775, 0.33242562413215637, 0.14517059922218323, 0.15228305757045746, 0.19609883427619934, 0.2163120061159134, 0.26972872018814087, 0.09671854227781296, 0.22969958186149597, 0.29909250140190125, 0.17633694410324097, 0.12080632895231247, 0.10587969422340393, 0.23322007060050964, 0.2061508297920227, 0.16837498545646667, 0.19170410931110382, 0.21464188396930695, 0.1892302930355072, 0.18613366782665253, 0.2737697660923004, 0.24690113961696625, 0.1830735206604004, 0.18550293147563934, 0.145085871219635, 0.32863837480545044, 0.12535057961940765, 0.1466149240732193, 0.19370318949222565, 0.20512589812278748, 0.18924643099308014, 0.1284664124250412, 0.10437370091676712, 0.06368351727724075, 0.12157639116048813, 0.10797294974327087, 0.09320453554391861, 0.15612462162971497, 0.22178179025650024, 0.07581352442502975, 0.09943010658025742, 0.08950033783912659, 0.3504612445831299, 0.19558022916316986, 0.18246997892856598, 0.15211202204227448, 0.229221910238266, 0.12388277053833008, 0.08018117398023605, 0.1485007405281067, 0.15688088536262512, 0.09511623531579971, 0.2329816222190857, 0.14462324976921082, 0.2506018280982971, 0.09505051374435425, 0.15867522358894348, 0.12358806282281876, 0.4350118637084961, 0.061863526701927185, 0.21798373758792877, 0.1685037761926651, 0.1271565705537796, 0.05907551199197769, 0.18390482664108276, 0.07687932997941971, 0.04443143308162689, 0.04488151893019676, 0.06061621382832527, 0.0847879946231842, 0.15383797883987427, 0.03998224809765816, 0.4492322504520416, 0.09595303237438202, 0.09357337653636932, 0.20786237716674805, 0.3777127265930176, 0.0648217648267746, 0.19297552108764648, 0.3265119194984436, 0.07722177356481552, 0.4450664222240448, 0.13404352962970734, 0.18449069559574127, 0.09954270720481873, 0.07850544899702072, 0.18605098128318787, 0.12531456351280212, 0.1991146206855774, 0.30264824628829956, 0.18669237196445465, 0.2287481278181076, 0.08829035609960556, 0.1219208687543869, 0.12483233958482742, 0.4389035701751709, 0.3106136918067932, 0.19175805151462555, 0.27285897731781006, 0.18118849396705627, 0.16319037973880768, 0.09083116799592972, 0.3564290404319763, 0.3554528057575226, 0.24281053245067596, 0.14285044372081757, 0.11221257597208023, 0.2632931172847748, 0.13131265342235565, 0.15544553101062775, 0.2576717734336853, 0.18041402101516724, 0.3015284538269043, 0.18915364146232605, 0.16000153124332428, 0.333211213350296, 0.2272290587425232, 0.17458689212799072, 0.24920105934143066, 0.12107465416193008, 0.2319757342338562, 0.2347097396850586, 0.21371014416217804, 0.2550588548183441, 0.21686041355133057, 0.2785698175430298, 0.2626708149909973, 0.31568968296051025, 0.11083689332008362, 0.1984543353319168, 0.45332247018814087, 0.11004014313220978, 0.077368825674057, 0.17699424922466278, 0.2408166080713272, 0.11549915373325348, 0.3070163130760193, 0.17784473299980164, 0.23913931846618652, 0.08200905472040176, 0.26236674189567566, 0.2656296491622925, 0.3998585641384125, 0.1055407002568245, 0.2577245235443115, 0.28708791732788086, 0.27198824286460876, 0.38200879096984863, 0.12376926839351654, 0.27234625816345215, 0.24959725141525269, 0.2573346793651581, 0.33763304352760315, 0.37630558013916016, 0.2910066843032837, 0.30405938625335693, 0.1805003583431244, 0.30403271317481995, 0.2489071488380432, 0.16641321778297424, 0.2339145988225937, 0.26024964451789856, 0.24933931231498718, 0.13896626234054565, 0.2421138733625412, 0.2173331081867218, 0.08943424373865128, 0.24192963540554047, 0.09699080139398575, 0.2745260000228882, 0.12378822267055511, 0.2663561999797821, 0.2321677953004837, 0.17383472621440887, 0.20170699059963226, 0.41055652499198914, 0.26026639342308044, 0.21455547213554382, 0.20620308816432953, 0.09226078540086746, 0.0959414541721344, 0.23569531738758087, 0.14836210012435913, 0.23014678061008453, 0.27710193395614624, 0.2972390651702881, 0.1839936077594757, 0.29237672686576843, 0.12681972980499268, 0.13368430733680725, 0.21403248608112335, 0.22427037358283997, 0.24494291841983795, 0.2507409453392029, 0.195326566696167, 0.15442945063114166, 0.2116655558347702, 0.1543070673942566, 0.1719905138015747, 0.2923150658607483, 0.3250003457069397, 0.09373465925455093, 0.20377346873283386, 0.11960472166538239, 0.2785344123840332, 0.29876843094825745, 0.14094480872154236, 0.27224671840667725, 0.4232936501502991, 0.1441885232925415, 0.26476261019706726, 0.1603420525789261, 0.17460249364376068, 0.0781221091747284, 0.05336209759116173, 0.4887411594390869, 0.14054729044437408, 0.16477563977241516, 0.10054226964712143, 0.4653795063495636, 0.05104023590683937, 0.48370790481567383, 0.29264476895332336, 0.08098608255386353, 0.07159797102212906, 0.08369499444961548, 0.17900706827640533, 0.07288683205842972, 0.14687946438789368, 0.1798611730337143, 0.1639406979084015, 0.23600399494171143, 0.04154454916715622, 0.07478735595941544, 0.10599002242088318, 0.06112531200051308, 0.15419287979602814, 0.15438835322856903, 0.14214271306991577, 0.31474748253822327, 0.2644890546798706, 0.19032055139541626, 0.08733287453651428, 0.32702547311782837, 0.21075066924095154, 0.3489258885383606, 0.17276416718959808, 0.37590351700782776, 0.13752985000610352, 0.08924178779125214, 0.1556318700313568, 0.12185856699943542, 0.306510329246521, 0.23654308915138245, 0.2262449413537979, 0.39841708540916443, 0.2721845507621765, 0.12352309376001358, 0.3896692991256714, 0.29844948649406433, 0.2875221371650696, 0.3102327883243561, 0.16621237993240356, 0.20213206112384796, 0.11341051012277603, 0.19548052549362183, 0.24303950369358063, 0.4077897071838379, 0.12949523329734802, 0.10771368443965912, 0.17416971921920776, 0.17944453656673431, 0.25238582491874695, 0.2943873107433319, 0.19078771770000458, 0.2507118880748749, 0.24926817417144775, 0.37230122089385986, 0.3624846935272217, 0.32999497652053833, 0.11465878784656525, 0.3044351041316986, 0.2655828893184662, 0.21984779834747314, 0.1701730638742447, 0.13438786566257477, 0.26506027579307556, 0.250275194644928, 0.17328914999961853, 0.26882562041282654, 0.1759398877620697, 0.2649989724159241, 0.09900757670402527, 0.22018349170684814, 0.40207940340042114, 0.37268635630607605, 0.4294525384902954, 0.25887638330459595, 0.2647453248500824, 0.27207547426223755, 0.28504306077957153, 0.16179461777210236, 0.23247139155864716, 0.2716873586177826, 0.3177655339241028, 0.27722927927970886, 0.1168464869260788, 0.24653330445289612, 0.2682812511920929, 0.20855200290679932, 0.23606407642364502, 0.11346742510795593, 0.08674386143684387, 0.1174371987581253, 0.2679765522480011, 0.409802109003067, 0.23307286202907562, 0.19505415856838226, 0.37739497423171997, 0.3584149479866028, 0.24864444136619568, 0.23834386467933655, 0.2120048999786377, 0.13162392377853394, 0.3098834156990051, 0.22581522166728973, 0.2070089727640152, 0.22503989934921265, 0.29179978370666504, 0.2706938087940216, 0.18273627758026123, 0.2148398607969284, 0.23523306846618652, 0.2721790075302124, 0.17453578114509583, 0.26227688789367676, 0.21658171713352203, 0.21405260264873505, 0.19123691320419312, 0.36686959862709045, 0.1859995424747467, 0.1831645518541336, 0.19936509430408478, 0.2931472361087799, 0.2743668854236603, 0.19013261795043945, 0.26716142892837524, 0.18547236919403076, 0.1198372021317482, 0.12631964683532715, 0.2040655016899109, 0.3114619255065918, 0.20241549611091614, 0.18618401885032654, 0.3546736538410187, 0.29312020540237427, 0.3504379689693451, 0.256489098072052, 0.15020669996738434, 0.18853135406970978, 0.30947354435920715, 0.22595949470996857, 0.2658001482486725, 0.21984714269638062, 0.18049503862857819, 0.17577886581420898, 0.23783625662326813, 0.22775231301784515, 0.19718758761882782, 0.3081861138343811, 0.1983386129140854, 0.2316494882106781, 0.15729600191116333, 0.05577154830098152, 0.16561639308929443, 0.23238743841648102, 0.08498220890760422, 0.2493281066417694, 0.04126939922571182, 0.03393847495317459, 0.16603730618953705, 0.5047601461410522, 0.11258510500192642, 0.09282200783491135, 0.11747352033853531, 0.11984167993068695, 0.353990763425827, 0.5583568215370178, 0.09030255675315857, 0.2436700463294983, 0.1859709471464157, 0.08367874473333359, 0.10346955806016922, 0.1603425145149231, 0.16768375039100647, 0.19955025613307953, 0.34698161482810974, 0.11910848319530487, 0.1873531937599182, 0.25170135498046875, 0.0739368349313736, 0.4391665756702423, 0.17696432769298553, 0.14689336717128754, 0.07273800671100616, 0.2734939455986023, 0.18783637881278992, 0.12190503627061844, 0.3511944115161896, 0.1745886504650116, 0.15576165914535522, 0.21037372946739197, 0.18384745717048645, 0.2654995024204254, 0.13161104917526245, 0.429695725440979, 0.29440680146217346, 0.2677101194858551, 0.2088615596294403, 0.30746352672576904, 0.15617424249649048, 0.1940852850675583, 0.21081668138504028, 0.30079665780067444, 0.24625907838344574, 0.4101022481918335, 0.3116888105869293, 0.17547278106212616, 0.2589742839336395, 0.2429005354642868, 0.30660584568977356, 0.2774011790752411, 0.2540507912635803, 0.17484164237976074, 0.43907877802848816, 0.3099408447742462, 0.2579432427883148, 0.1048915907740593, 0.41370251774787903, 0.3056781589984894, 0.35814768075942993, 0.2712688446044922, 0.35953283309936523, 0.23491866886615753, 0.3361124098300934, 0.23571977019309998, 0.18149779736995697, 0.14074525237083435, 0.3371359407901764, 0.32563909888267517, 0.21974727511405945, 0.16931092739105225, 0.255697101354599, 0.2238931953907013, 0.29372406005859375, 0.298592209815979, 0.10260491073131561, 0.24574610590934753, 0.3588414788246155, 0.45488083362579346, 0.26779112219810486, 0.3859989643096924, 0.11076551675796509, 0.26328152418136597, 0.30378928780555725, 0.2526303231716156, 0.37933462858200073, 0.44264665246009827, 0.24204888939857483, 0.2216799110174179, 0.16028639674186707, 0.21471725404262543, 0.23083022236824036, 0.2273396998643875, 0.4014315903186798, 0.3145427107810974, 0.1807630956172943, 0.13380497694015503, 0.201572984457016, 0.3966546654701233, 0.24425926804542542, 0.17880982160568237, 0.28536149859428406, 0.3066266179084778, 0.205952987074852, 0.2714160978794098, 0.20142702758312225, 0.33277714252471924, 0.29506734013557434, 0.29038527607917786, 0.13084962964057922, 0.3569529950618744, 0.22299674153327942, 0.29153478145599365, 0.28403544425964355, 0.33731788396835327, 0.19806024432182312, 0.21920429170131683, 0.24654407799243927, 0.33896133303642273, 0.3924431800842285, 0.2592923939228058, 0.1929844617843628, 0.1863509714603424, 0.2961062788963318, 0.24488712847232819, 0.33250871300697327, 0.2260555475950241, 0.21129412949085236, 0.20499002933502197, 0.18098004162311554, 0.1980406641960144, 0.16197055578231812, 0.3406735956668854, 0.39691808819770813, 0.25557613372802734, 0.18091486394405365, 0.29368534684181213, 0.21707072854042053, 0.21542423963546753, 0.18068397045135498, 0.29465043544769287, 0.24537448585033417, 0.22641314566135406, 0.3311178386211395, 0.28962597250938416, 0.21249710023403168, 0.37633436918258667, 0.3519110381603241, 0.20368854701519012, 0.2426302134990692, 0.23994092643260956, 0.18039213120937347, 0.10213784873485565, 0.06882602721452713, 0.5773374438285828, 0.045867759734392166, 0.11265172809362411, 0.25278177857398987, 0.12969757616519928, 0.30469873547554016, 0.22482959926128387, 0.0317133292555809, 0.1495346873998642, 0.09245158731937408, 0.07464316487312317, 0.20737867057323456, 0.19669575989246368, 0.12073016166687012, 0.037654418498277664, 0.4083108901977539, 0.059649981558322906, 0.3529844284057617, 0.19204382598400116, 0.13482969999313354, 0.3153204321861267, 0.11912764608860016, 0.45101049542427063, 0.24101518094539642, 0.14061109721660614, 0.15056520700454712, 0.2799527943134308, 0.16154956817626953, 0.25415560603141785, 0.2774010896682739, 0.24544264376163483, 0.25817614793777466, 0.12841179966926575, 0.21673117578029633, 0.19200557470321655, 0.05337869003415108, 0.13375702500343323, 0.1295291781425476, 0.2633010149002075, 0.4285200536251068, 0.30714428424835205, 0.26798197627067566, 0.24695958197116852, 0.3825887143611908, 0.29999759793281555, 0.15814274549484253, 0.36241257190704346, 0.3834748864173889, 0.3065461218357086, 0.16932275891304016, 0.20933786034584045, 0.27809545397758484, 0.19438834488391876, 0.15321022272109985, 0.37832391262054443, 0.24825537204742432, 0.25473204255104065, 0.22732199728488922, 0.23503528535366058, 0.3818628489971161, 0.29659122228622437, 0.41731277108192444, 0.239427849650383, 0.28214311599731445, 0.2514244019985199, 0.21116898953914642, 0.4011007845401764, 0.21258342266082764, 0.34983986616134644, 0.34543395042419434, 0.22107715904712677, 0.17518974840641022, 0.30067557096481323, 0.15571683645248413, 0.34876877069473267, 0.2629641890525818, 0.1732909083366394, 0.13117308914661407, 0.19044068455696106, 0.12604549527168274, 0.43968772888183594, 0.2513902187347412, 0.11426030844449997, 0.20130889117717743, 0.3817199468612671, 0.35646992921829224, 0.11282908916473389, 0.2116623818874359, 0.43537580966949463, 0.23177936673164368, 0.3602777421474457, 0.3466145396232605, 0.23287060856819153, 0.34717777371406555, 0.2447282075881958, 0.2741853892803192, 0.2507708966732025, 0.30420011281967163, 0.4360128343105316, 0.26995301246643066, 0.33317965269088745, 0.25732073187828064, 0.2708102762699127, 0.35679182410240173, 0.18155865371227264, 0.39618155360221863, 0.18606334924697876, 0.11295906454324722, 0.33293744921684265, 0.10424761474132538, 0.24593521654605865, 0.14399749040603638, 0.2748892903327942, 0.11873038858175278, 0.3743477165699005, 0.24132266640663147, 0.24143244326114655, 0.38928428292274475, 0.3852320611476898, 0.31560277938842773, 0.14657150208950043, 0.25675949454307556, 0.26095259189605713, 0.1893995702266693, 0.2963833212852478, 0.2725619077682495, 0.28845712542533875, 0.20101267099380493, 0.4055922031402588, 0.3012944757938385, 0.339568555355072, 0.3087853491306305, 0.4347134530544281, 0.15211045742034912, 0.15106460452079773, 0.23939695954322815, 0.2829170823097229, 0.3055611550807953, 0.2383582442998886, 0.20464250445365906, 0.3020972013473511, 0.4264059066772461, 0.23911473155021667, 0.295881450176239, 0.16565261781215668, 0.2697511911392212, 0.24908503890037537, 0.29164400696754456, 0.33110716938972473, 0.21418333053588867, 0.3294464647769928, 0.3313298523426056, 0.49034127593040466, 0.2062058448791504, 0.2589973509311676, 0.1397445648908615, 0.34200483560562134, 0.22970068454742432, 0.20261381566524506, 0.10546400398015976, 0.03052516095340252, 0.11177059262990952, 0.03979099541902542, 0.5209977626800537, 0.043887753039598465, 0.08610662072896957, 0.17578460276126862, 0.1021600067615509, 0.4383445084095001, 0.14514675736427307, 0.14530105888843536, 0.08508679270744324, 0.0756542831659317, 0.2575247287750244, 0.14809785783290863, 0.11464269459247589, 0.5702393054962158, 0.3167864978313446, 0.1647406667470932, 0.13084733486175537, 0.1966942697763443, 0.3453189730644226, 0.2074984312057495, 0.19945208728313446, 0.21910421550273895, 0.2530645728111267, 0.08435936272144318, 0.20386338233947754, 0.06833434849977493, 0.10770119726657867, 0.24165886640548706, 0.486935555934906, 0.32804104685783386, 0.2889505922794342, 0.2034776508808136, 0.10488719493150711, 0.3540334105491638, 0.2664608657360077, 0.38828015327453613, 0.4735041558742523, 0.49133697152137756, 0.21747706830501556, 0.28784888982772827, 0.28738918900489807, 0.11612518876791, 0.3814978003501892, 0.2720056474208832, 0.25608524680137634, 0.14222225546836853, 0.27554813027381897, 0.3479815125465393, 0.14760497212409973, 0.2969999313354492, 0.414226770401001, 0.32407400012016296, 0.2480183243751526, 0.31660905480384827, 0.3096868395805359, 0.3115305006504059, 0.2554791569709778, 0.21256102621555328, 0.3021533787250519, 0.35385939478874207, 0.21547619998455048, 0.2456488162279129, 0.43722155690193176, 0.23183687031269073, 0.14574550092220306, 0.4915284216403961, 0.11323434114456177, 0.21309928596019745, 0.46903857588768005, 0.29184281826019287, 0.30747681856155396, 0.27719807624816895, 0.4171755015850067, 0.1660178005695343, 0.3098410665988922, 0.2235504388809204, 0.33495187759399414, 0.17027650773525238, 0.40636566281318665, 0.18470892310142517, 0.19527478516101837, 0.469842791557312, 0.3275914192199707, 0.08609913289546967, 0.2486126720905304, 0.31246742606163025, 0.46477949619293213, 0.32918524742126465, 0.19280114769935608, 0.4329802393913269, 0.13511769473552704, 0.26061201095581055, 0.2642682194709778, 0.5541598796844482, 0.18564707040786743, 0.19404394924640656, 0.31126755475997925, 0.3588026165962219, 0.29945072531700134, 0.20818530023097992, 0.3744719624519348, 0.30109575390815735, 0.5223594307899475, 0.28163689374923706, 0.4446970820426941, 0.29253435134887695, 0.23436333239078522, 0.3525652289390564, 0.09976015239953995, 0.12966042757034302, 0.17922726273536682, 0.15609946846961975, 0.3961007297039032, 0.5834128260612488, 0.3539685606956482, 0.24400180578231812, 0.32741957902908325, 0.38062021136283875, 0.1500464379787445, 0.5155537724494934, 0.20521122217178345, 0.2862008213996887, 0.39657753705978394, 0.19615568220615387, 0.4901154339313507, 0.2850000858306885, 0.3190975487232208, 0.3101295530796051, 0.12253744155168533, 0.167701855301857, 0.28244003653526306, 0.2178030014038086, 0.4379172921180725, 0.48600515723228455, 0.11433862894773483, 0.1912917196750641, 0.1809825748205185, 0.5055354833602905, 0.22152495384216309, 0.37382087111473083, 0.2549841105937958, 0.10006800293922424, 0.18318918347358704, 0.34953492879867554, 0.34683045744895935, 0.3343309462070465, 0.2552351951599121, 0.1693328619003296, 0.17205871641635895, 0.13451357185840607, 0.33001935482025146, 0.5430889129638672, 0.3184156119823456, 0.5872033834457397, 0.10410872846841812];
        let activity = vec![-4.540998458862305, -4.668871879577637, -4.954258918762207, -4.86463737487793, -4.842907905578613, -5.432507514953613, -4.653450965881348, -4.806859016418457, -4.954158782958984, -4.876253128051758, -4.604184150695801, -4.618904113769531, -4.557921409606934, -4.82658576965332, -4.7545881271362305, -4.701315879821777, -4.598075866699219, -4.9045915603637695, -4.805957794189453, -4.684392929077148, -5.260173797607422, -5.3535003662109375, -5.424596786499023, -5.588719367980957, -5.502902984619141, -5.532342910766602, -5.314447402954102, -5.344688415527344, -5.522229194641113, -5.749136924743652, -5.258271217346191, -5.273792266845703, -5.292417526245117, -5.270387649536133, -5.273191452026367, -5.375430107116699, -5.319354057312012, -5.708281517028809, -5.35169792175293, -5.351998329162598, -4.802553176879883, -4.743673324584961, -4.813868522644043, -4.870244979858398, -4.7655029296875, -4.958765029907227, -4.841205596923828, -4.723946571350098, -4.729954719543457, -4.812666893005371, -4.763700485229492, -4.723445892333984, -4.775015830993652, -4.856626510620117, -4.860631942749023, -4.8311920166015625, -4.763199806213379, -4.821478843688965, -4.783327102661133, -4.737665176391602, -4.364459037780762, -4.356247901916504, -4.363557815551758, -4.36966609954834, -4.3694658279418945, -4.415828704833984, -4.377176284790039, -4.506351470947266, -4.443366050720215, -4.419734001159668, -4.426142692565918, -4.439460754394531, -4.361154556274414, -4.3828840255737305, -4.4335527420043945, -4.42884635925293, -4.400107383728027, -4.354145050048828, -4.47220516204834, -4.434253692626953, -4.865138053894043, -5.014740943908691, -4.926120758056641, -4.923517227172852, -4.853021621704102, -4.904691696166992, -4.936534881591797, -4.885465621948242, -4.987003326416016, -4.875151634216309, -4.867441177368164, -4.930927276611328, -4.952456474304199, -4.951254844665527, -4.892475128173828, -4.892675399780273, -4.980093955993652, -4.885465621948242, -4.883663177490234, -4.883563041687012, -4.503948211669922, -4.5046491622924805, -4.542200088500977, -4.662663459777832, -4.503647804260254, -4.524075508117676, -4.571139335632324, -4.582054138183594, -4.640633583068848, -4.6520490646362305, -4.539997100830078, -4.539997100830078, -4.578649520874023, -4.553515434265137, -4.652349472045898, -4.562727928161621, -4.586660385131836, -4.510156631469727, -4.544903755187988, -4.595973014831543, -4.39870548248291, -4.483019828796387, -4.418632507324219, -4.645440101623535, -4.511157989501953, -4.527980804443359, -4.476110458374023, -4.479214668273926, -4.400107383728027, -4.511258125305176, -4.426142692565918, -4.400007247924805, -4.403111457824707, -4.400407791137695, -4.505950927734375, -4.477612495422363, -4.554717063903809, -4.3989057540893555, -4.439260482788086, -4.445869445800781, -4.4079179763793945, -4.4079179763793945, -4.408018112182617, -4.468400001525879, -4.541699409484863, -4.4079179763793945, -4.519969940185547, -4.409119606018066, -4.414226531982422, -4.686495780944824, -4.594170570373535, -4.407817840576172, -4.568635940551758, -4.4079179763793945, -4.5495100021362305, -4.892575263977051, -4.4085187911987305, -4.444667816162109, -4.481417655944824, -4.51896858215332, -5.126091957092285, -5.326563835144043, -5.25747013092041, -5.393955230712891, -5.203296661376953, -5.46405029296875, -5.193182945251465, -5.670830726623535, -5.055596351623535, -5.328566551208496, -5.115878105163574, -5.111371994018555, -5.135504722595215, -5.245153427124023, -5.069114685058594, -5.508110046386719, -5.350396156311035, -5.530340194702148, -5.291015625, -5.054194450378418, -4.919411659240723, -4.995815277099609, -5.249258995056152, -5.215513229370117, -5.038573265075684, -5.266081809997559, -5.126792907714844, -4.983298301696777, -5.225927352905273, -4.990007400512695, -4.9628705978393555, -5.011436462402344, -4.968277931213379, -5.1481218338012695, -5.060402870178223, -5.104162216186523, -5.013339042663574, -4.98640251159668, -5.081831932067871, -5.094348907470703, -5.036069869995117, -5.1003570556640625, -5.201994895935059, -5.1647443771362305, -5.1807661056518555, -5.216814994812012, -5.157334327697754, -5.107466697692871, -5.136806488037109, -5.101058006286621, -5.0875396728515625, -5.074522018432617, -5.113574981689453, -5.076224327087402, -5.081231117248535, -5.096151351928711, -5.109769821166992, -5.1769609451293945, -5.1356048583984375, -5.1224870681762695, -5.155832290649414, -5.179965019226074, -5.179264068603516, -5.151125907897949, -5.153629302978516, -5.159136772155762, -5.183169364929199, -5.184270858764648, -5.1455183029174805, -5.089241981506348, -5.144016265869141, -5.137607574462891, -5.147821426391602, -5.150825500488281, -5.176560401916504, -5.162841796875, -5.1900787353515625, -5.191680908203125, -5.157134056091309, -5.164944648742676, -4.806558609008789, -4.850818634033203, -4.898883819580078, -4.888669967651367, -4.816972732543945, -4.889370918273926, -4.840704917907715, -4.889370918273926, -4.852520942687988, -4.8443098068237305, -4.8469133377075195, -4.865338325500488, -4.856626510620117, -4.846713066101074, -4.911200523376465, -4.919611930847168, -4.864336967468262, -4.8888702392578125, -4.8693437576293945, -4.833595275878906, -4.661762237548828, -4.689199447631836, -4.651247978210449, -4.636728286743164, -4.671675682067871, -4.663464546203613, -4.700014114379883, -4.690300941467285, -4.66787052154541, -4.689199447631836, -4.655754089355469, -4.662463188171387, -4.639732360839844, -4.695207595825195, -4.6805877685546875, -4.650847434997559, -4.667269706726074, -4.736663818359375, -4.683591842651367, -4.682390213012695, -4.441062927246094, -4.480316162109375, -4.413425445556641, -4.417330741882324, -4.416529655456543, -4.415528297424316, -4.45147705078125, -4.451276779174805, -4.494034767150879, -4.577047348022461, -4.448372840881348, -4.516264915466309, -4.406015396118164, -4.406215667724609, -4.420334815979004, -4.488527297973633, -4.489728927612305, -4.445468902587891, -4.406716346740723, -4.412023544311523, -4.557921409606934, -4.40361213684082, -4.499141693115234, -4.605185508728027, -4.459888458251953, -4.403712272644043, -4.412123680114746, -4.496337890625, -4.514762878417969, -4.415328025817871, -4.403311729431152, -4.407617568969727, -4.424139976501465, -4.403411865234375, -4.619304656982422, -4.492432594299316, -4.403311729431152, -4.404413223266602, -4.466096878051758, -4.5716400146484375, -4.919912338256836, -4.8635358810424805, -5.022551536560059, -4.852320671081543, -4.809462547302246, -4.844209671020508, -4.83109188079834, -4.873449325561523, -4.947049140930176, -4.888669967651367, -4.913203239440918, -4.787132263183594, -4.8411054611206055, -4.918910980224609, -4.918210029602051, -4.983198165893555, -4.7863311767578125, -5.112173080444336, -4.866339683532715, -4.911200523376465, -4.8760528564453125, -4.912001609802246, -5.072519302368164, -4.973084449768066, -5.016843795776367, -4.899484634399414, -4.95135498046875, -5.005428314208984, -5.112573623657227, -5.001523017883301, -4.896280288696289, -4.91290283203125, -4.942042350769043, -4.96417236328125, -4.912201881408691, -4.958965301513672, -4.935633659362793, -4.929525375366211, -4.882962226867676, -4.875151634216309, -5.250260353088379, -5.259973526000977, -5.3150482177734375, -5.262076377868652, -5.261275291442871, -5.303632736206055, -5.275394439697266, -5.316750526428223, -5.309040069580078, -5.331470489501953, -5.224124908447266, -5.28160285949707, -5.290114402770996, -5.253564834594727, -5.286809921264648, -5.24615478515625, -5.224225044250488, -5.332972526550293, -5.2955217361450195, -5.259372711181641, -4.745275497436523, -4.750082015991211, -4.792840003967285, -4.861933708190918, -4.799849510192871, -4.777218818664551, -4.845611572265625, -4.801051139831543, -4.818174362182617, -4.797846794128418, -4.823882102966309, -4.782125473022461, -4.720541954040527, -4.78773307800293, -4.773113250732422, -4.743172645568848, -4.779521942138672, -4.826786041259766, -4.737565040588379, -4.796645164489746, -4.602581977844238, -4.641735076904297, -4.670373916625977, -4.644538879394531, -4.672677040100098, -4.612094879150391, -4.633023262023926, -4.716035842895508, -4.641434669494629, -4.641735076904297, -4.634625434875488, -4.657756805419922, -4.5943708419799805, -4.609591484069824, -4.70782470703125, -4.599878311157227, -4.69200325012207, -4.696809768676758, -4.595171928405762, -4.653651237487793, -4.544403076171875, -4.5873613357543945, -4.585558891296387, -4.569036483764648, -4.582354545593262, -4.581052780151367, -4.592768669128418, -4.611293792724609, -4.590465545654297, -4.593870162963867, -4.5556182861328125, -4.576346397399902, -4.571039199829102, -4.581753730773926, -4.577548027038574, -4.565031051635742, -4.5649309158325195, -4.570638656616211, -4.611994743347168, -4.564730644226074, -4.570137977600098, -4.545804977416992, -4.560224533081055, -4.541499137878418, -4.556719779968262, -4.607688903808594, -4.558422088623047, -4.5334882736206055, -4.582554817199707, -4.541098594665527, -4.617301940917969, -4.674078941345215, -4.545804977416992, -4.559823989868164, -4.535390853881836, -4.5588226318359375, -4.5331878662109375, -4.579450607299805, -4.540898323059082, -4.5652313232421875, -4.6232099533081055, -4.486724853515625, -4.436957359313965, -4.49483585357666, -4.475409507751465, -4.414226531982422, -4.480816841125488, -4.431049346923828, -4.392797470092773, -4.394599914550781, -4.383284568786621, -4.383184432983398, -4.472105026245117, -4.383284568786621, -4.526679039001465, -4.49783992767334, -4.383184432983398, -4.41602897644043, -4.679786682128906, -4.41642951965332, -4.922215461730957, -4.908797264099121, -4.878456115722656, -4.859129905700684, -4.939739227294922, -5.080630302429199, -4.856326103210449, -5.01544189453125, -5.000822067260742, -4.973284721374512, -4.837100028991699, -5.037471771240234, -4.879657745361328, -4.8373003005981445, -4.84731388092041, -4.837200164794922, -4.870545387268066, -5.075423240661621, -4.841806411743164, -4.997918128967285, -4.874150276184082, -4.937736511230469, -4.967476844787598, -4.8952789306640625, -4.883462905883789, -4.983298301696777, -4.965373992919922, -4.889070510864258, -5.0173444747924805, -4.874250411987305, -4.875151634216309, -4.916007041931152, -5.011436462402344, -4.886466979980469, -4.942543029785156, -4.920012474060059, -4.886667251586914, -4.942743301391602, -4.932629585266113, -4.895078659057617, -4.901487350463867, -4.970280647277832, -4.924618721008301, -4.87935733795166, -4.859230041503906, -4.8536224365234375, -4.853522300720215, -4.889270782470703, -4.900786399841309, -4.8923749923706055, -4.878556251525879, -4.854123115539551, -4.91140079498291, -4.887167930603027, -4.877354621887207, -4.87595272064209, -4.891674041748047, -4.871246337890625, -4.881760597229004, -4.862434387207031, -4.790737152099609, -4.840204238891602, -4.85862922668457, -4.790136337280273, -4.832794189453125, -4.806158065795898, -4.847614288330078, -4.8501176834106445, -4.815570831298828, -4.896480560302734, -4.787032127380371, -4.803554534912109, -4.797145843505859, -4.8244829177856445, -4.779521942138672, -4.802853584289551, -4.818975448608398, -4.788534164428711, -4.865939140319824, -4.808561325073242, -4.586560249328613, -4.629518508911133, -4.603182792663574, -4.626514434814453, -4.609491348266602, -4.607388496398926, -4.635626792907715, -4.629718780517578, -4.607288360595703, -4.633323669433594, -4.594971656799316, -4.639031410217285, -4.615699768066406, -4.712531089782715, -4.630319595336914, -4.633423805236816, -4.577347755432129, -4.601280212402344, -4.573843002319336, -4.690000534057617, -4.315492630004883, -4.317695617675781, -4.373971939086914, -4.331814765930176, -4.325506210327148, -4.307181358337402, -4.32029914855957, -4.3310136795043945, -4.310385704040527, -4.362155914306641, -4.309584617614746, -4.332115173339844, -4.332015037536621, -4.374973297119141, -4.327909469604492, -4.355146408081055, -4.362155914306641, -4.376775741577148, -4.335920333862305, -4.326407432556152, -4.3438310623168945, -4.361454963684082, -4.353043556213379, -4.44847297668457, -4.342128753662109, -4.428345680236816, -4.442564964294434, -4.486624717712402, -4.397403717041016, -4.49974250793457, -4.36175537109375, -4.338824272155762, -4.374873161315918, -4.4047136306762695, -4.3957014083862305, -4.492332458496094, -4.369365692138672, -4.394399642944336, -4.361454963684082, -4.375073432922363, -4.4047136306762695, -4.437958717346191, -4.697610855102539, -4.5431013107299805, -4.5168657302856445, -4.420635223388672, -4.40361213684082, -4.403411865234375, -4.524075508117676, -4.523674964904785, -4.574944496154785, -4.403912544250488, -4.403411865234375, -4.470703125, -4.415828704833984, -4.40321159362793, -4.46429443359375, -4.541399002075195, -4.4626922607421875, -4.6354265213012695, -4.647943496704102, -4.647943496704102, -4.942342758178711, -4.776017189025879, -4.722945213317871, -4.648744583129883, -4.7802228927612305, -4.675881385803223, -4.975888252258301, -4.731356620788574, -4.664666175842285, -4.648143768310547, -4.648444175720215, -4.663464546203613, -4.647943496704102, -4.655153274536133, -4.744174003601074, -4.7783203125, -4.722644805908203, -4.649946212768555, -4.509756088256836, -4.681088447570801, -4.5527143478393555, -4.562227249145508, -4.510356903076172, -4.5521135330200195, -4.533688545227051, -4.532587051391602, -4.6065874099731445, -4.509956359863281, -4.5559186935424805, -4.531085014343262, -4.525076866149902, -4.552213668823242, -4.591667175292969, -4.509756088256836, -4.5107574462890625, -4.578449249267578, -4.54119873046875, -4.509756088256836, -4.719740867614746, -4.719440460205078, -4.761697769165039, -4.7219438552856445, -4.727651596069336, -4.7475786209106445, -4.720541954040527, -4.74928092956543, -4.729454040527344, -4.755990028381348, -4.7193403244018555, -4.762699127197266, -4.745175361633301, -4.739267349243164, -4.727851867675781, -4.84581184387207, -4.746677398681641, -4.719240188598633, -4.80735969543457, -4.744474411010742, -4.547607421875, -4.581853866577148, -4.595772743225098, -4.520971298217773, -4.616701126098633, -4.557320594787598, -4.520971298217773, -4.538995742797852, -4.589664459228516, -4.5684356689453125, -4.564129829406738, -4.550411224365234, -4.537193298339844, -4.592668533325195, -4.560124397277832, -4.555017471313477, -4.569637298583984, -4.565631866455078, -4.532687187194824, -4.578849792480469, -4.41943359375, -4.42734432220459, -4.471504211425781, -4.478513717651367, -4.455782890319824, -4.422137260437012, -4.47861385345459, -4.445969581604004, -4.433352470397949, -4.419533729553223, -4.433452606201172, -4.496638298034668, -4.454280853271484, -4.461690902709961, -4.429847717285156, -4.476110458374023, -4.44016170501709, -4.466297149658203, -4.456984519958496, -4.4629926681518555, -4.420835494995117, -4.4501752853393555, -4.434654235839844, -4.379179000854492, -4.401208877563477, -4.416129112243652, -4.425041198730469, -4.412524223327637, -4.40511417388916, -4.41152286529541, -4.440662384033203, -4.431349754333496, -4.366962432861328, -4.443065643310547, -4.440662384033203, -4.419734001159668, -4.392997741699219, -4.426843643188477, -4.384285926818848, -4.432351112365723, -4.356748580932617, -4.386789321899414, -4.382183074951172, -4.451777458190918, -4.378778457641602, -4.352042198181152, -4.396402359008789, -4.387990951538086, -4.470903396606445, -4.408719062805176, -4.452578544616699, -4.336521148681641, -4.346034049987793, -4.352242469787598, -4.367463111877441, -4.470102310180664, -4.516064643859863, -4.3892927169799805, -4.360954284667969, -4.336521148681641, -4.359151840209961, -4.343630790710449, -4.37457275390625, -4.343230247497559, -4.649846076965332, -4.431750297546387, -4.44016170501709, -4.548708915710449, -4.591567039489746, -4.366962432861328, -4.343230247497559, -4.384185791015625, -4.622108459472656, -4.343130111694336, -4.394700050354004, -4.496738433837891, -4.4851226806640625, -4.358651161193848, -4.343130111694336, -4.448873519897461, -4.418932914733887, -4.564229965209961, -4.582154273986816, -4.419333457946777, -4.479214668273926, -4.436056137084961, -4.419033050537109, -4.481417655944824, -4.723245620727539, -4.419033050537109, -4.427544593811035, -4.422938346862793, -4.488727569580078, -4.703919410705566, -4.6613616943359375, -4.418932914733887, -4.583856582641602, -4.518668174743652, -4.418932914733887, -4.419033050537109, -4.552814483642578, -4.552814483642578, -4.615199089050293, -4.573843002319336, -4.577247619628906, -4.570037841796875, -4.631320953369141, -4.562828063964844, -4.657556533813477, -4.5975751876831055, -4.611093521118164, -4.5527143478393555, -4.562627792358398, -4.6261138916015625, -4.588963508605957, -4.659459114074707, -4.5527143478393555, -4.655453681945801, -4.553114891052246, -4.695407867431641, -4.506551742553711, -4.559022903442383, -4.538895606994629, -4.506551742553711, -4.507152557373047, -4.531986236572266, -4.600879669189453, -4.543301582336426, -4.608489990234375, -4.514863014221191, -4.5713396072387695, -4.534189224243164, -4.5142621994018555, -4.614598274230957, -4.566132545471191, -4.554416656494141, -4.511758804321289, -4.693805694580078, -4.562527656555176, -4.516264915466309, -4.429447174072266, -4.553014755249023, -4.536492347717285, -4.524375915527344, -4.463293075561523, -4.524175643920898, -4.4848222732543945, -4.451276779174805, -4.452378273010254, -4.475709915161133, -4.456684112548828, -4.521572113037109, -4.423138618469238, -4.459688186645508, -4.50274658203125, -4.447972297668457, -4.53629207611084, -4.521071434020996, -4.443366050720215, -4.470302581787109, -4.380881309509277, -4.470002174377441, -4.473006248474121, -4.45938777923584, -4.426342964172363, -4.470302581787109, -4.483720779418945, -4.418231964111328, -4.416930198669434, -4.432551383972168, -4.43525505065918, -4.42884635925293, -4.393898963928223, -4.479214668273926, -4.449173927307129, -4.446269989013672, -4.402009963989258, -4.4437665939331055, -4.424941062927246, -4.407317161560059, -4.283449172973633, -4.349238395690918, -4.3342180252075195, -4.3056793212890625, -4.397303581237793, -4.343630790710449, -4.330412864685059, -4.371668815612793, -4.360754013061523, -4.378077507019043, -4.29917049407959, -4.326407432556152, -4.275638580322266, -4.381281852722168, -4.332015037536621, -4.304577827453613, -4.332916259765625, -4.371368408203125, -4.358250617980957, -4.325806617736816, -4.407617568969727, -4.452478408813477, -4.52687931060791, -4.387089729309082, -4.359051704406738, -4.544102668762207, -4.38248348236084, -4.487726211547852, -4.406716346740723, -4.458286285400391, -4.358951568603516, -4.375273704528809, -4.373571395874023, -4.437257766723633, -4.36175537109375, -4.4373579025268555, -4.358851432800293, -4.387590408325195, -4.503547668457031, -4.519769668579102, -4.371468544006348, -4.519168853759766, -4.385988235473633, -4.52537727355957, -4.371468544006348, -4.448573112487793, -4.392597198486328, -4.37156867980957, -4.4402618408203125, -4.508955001831055, -4.416129112243652, -4.596673965454102, -4.532486915588379, -4.6065874099731445, -4.41452693939209, -4.565932273864746, -4.371368408203125, -4.41793155670166, -4.39680290222168, -4.449274063110352, -4.473306655883789, -4.473807334899902, -4.692904472351074, -4.473306655883789, -4.4752092361450195, -4.665667533874512, -4.660961151123047, -4.974886894226074, -4.558021545410156, -4.473306655883789, -4.492032051086426, -4.473306655883789, -4.496938705444336, -4.488126754760742, -4.539496421813965, -4.511558532714844, -4.473306655883789, -4.67277717590332, -4.473206520080566, -4.666969299316406, -4.386589050292969, -4.378778457641602, -4.390193939208984, -4.419533729553223, -4.379779815673828, -4.560524940490723, -4.535190582275391, -4.441363334655762, -4.498540878295898, -4.518768310546875, -4.370667457580566, -4.373471260070801, -4.384486198425293, -4.370967864990234, -4.370967864990234, -4.370767593383789, -4.370767593383789, -4.370767593383789, -4.371068000793457, -4.370667457580566, -4.474207878112793, -4.381381988525391, -4.394700050354004, -4.387690544128418, -4.389693260192871, -4.395801544189453, -4.424740791320801, -4.406816482543945, -4.381181716918945, -4.61660099029541, -4.457184791564941, -4.4341535568237305, -4.416229248046875, -4.389693260192871, -4.424440383911133, -4.402610778808594, -4.381882667541504, -4.391195297241211, -4.3886919021606445, -4.431449890136719, -4.316794395446777, -4.3598527908325195, -4.3540449142456055, -4.354345321655273, -4.318596839904785, -4.3668622970581055, -4.32180118560791, -4.326107025146484, -4.423739433288574, -4.34743595123291, -4.33952522277832, -4.405815124511719, -4.291460037231445, -4.3505401611328125, -4.368364334106445, -4.383885383605957, -4.298870086669922, -4.349638938903809, -4.291760444641113, -4.334918975830078, -4.326407432556152, -4.353644371032715, -4.393999099731445, -4.3604536056518555, -4.416830062866211, -4.392597198486328, -4.3569488525390625, -4.368865013122559, -4.41452693939209, -4.39680290222168, -4.353944778442383, -4.304778099060059, -4.340726852416992, -4.340025901794434, -4.368464469909668, -4.358551025390625, -4.349138259887695, -4.317895889282227, -4.372269630432129, -4.35384464263916, -4.458887100219727, -4.43525505065918, -4.544302940368652, -4.467198371887207, -4.530884742736816, -4.495737075805664, -4.453279495239258, -4.522473335266113, -4.474207878112793, -4.562627792358398, -4.456784248352051, -4.492432594299316, -4.444868087768555, -4.496037483215332, -4.479915618896484, -4.525677680969238, -4.459988594055176, -4.567534446716309, -4.441563606262207, -4.487125396728516, -4.37757682800293, -4.543501853942871, -4.600078582763672, -4.398405075073242, -4.479615211486816, -4.398505210876465, -4.4533796310424805, -4.448172569274902, -4.484121322631836, -4.383384704589844, -4.393198013305664, -4.387890815734863, -4.423539161682129, -4.466998100280762, -4.429347038269043, -4.381782531738281, -4.394299507141113, -4.452178001403809, -4.38398551940918, -4.57814884185791, -4.358050346374512, -4.370867729187012, -4.542099952697754, -4.424941062927246, -4.492432594299316, -4.373371124267578, -4.357950210571289, -4.357850074768066, -4.365059852600098, -4.413825988769531, -4.451276779174805, -4.547106742858887, -4.357950210571289, -4.544302940368652, -4.360854148864746, -4.380580902099609, -4.56683349609375, -4.35835075378418, -4.483420372009277, -4.367963790893555, -4.375273704528809, -4.373170852661133, -4.3732709884643555, -4.3732709884643555, -4.37307071685791, -4.703418731689453, -4.37307071685791, -4.37307071685791, -4.592868804931641, -4.37307071685791, -4.532286643981934, -4.373471260070801, -4.373170852661133, -4.37307071685791, -4.3732709884643555, -4.46429443359375, -4.396101951599121, -4.37307071685791, -4.551813125610352, -4.407317161560059, -4.3040771484375, -4.353644371032715, -4.410221099853516, -4.277240753173828, -4.2771406173706055, -4.277040481567383, -4.27694034576416, -4.516164779663086, -4.287254333496094, -4.2768402099609375, -4.27694034576416, -4.2768402099609375, -4.289457321166992, -4.311387062072754, -4.387890815734863, -4.314891815185547, -4.277841567993164, -4.277040481567383, -4.358050346374512, -4.474708557128906, -4.483720779418945, -4.490429878234863, -4.492232322692871, -4.467198371887207, -4.586760520935059, -4.466397285461426, -4.516765594482422, -4.703619003295898, -4.503046989440918, -4.4851226806640625, -4.48201847076416, -4.489428520202637, -4.465696334838867, -4.499241828918457, -4.4658966064453125, -4.466797828674316, -4.556118965148926, -4.4950361251831055, -4.555718421936035, -4.4655961990356445, -4.383384704589844, -4.418732643127441, -4.4562835693359375, -4.454280853271484, -4.439861297607422, -4.493534088134766, -4.396001815795898, -4.4405622482299805, -4.428245544433594, -4.426743507385254, -4.439560890197754, -4.440662384033203, -4.383384704589844, -4.413325309753418, -4.420234680175781, -4.539796829223633, -4.4175310134887695, -4.446770668029785, -4.432451248168945, -4.383584976196289, -4.319297790527344, -4.348036766052246, -4.362856864929199, -4.3854875564575195, -4.404112815856934, -4.403411865234375, -4.354946136474609, -4.386589050292969, -4.3886919021606445, -4.435956001281738, -4.326006889343262, -4.375173568725586, -4.318096160888672, -4.378878593444824, -4.3790788650512695, -4.349238395690918, -4.323102951049805, -4.414226531982422, -4.3790788650512695, -4.320499420166016, -4.30067253112793, -4.327408790588379, -4.373971939086914, -4.479114532470703, -4.321000099182129, -4.325706481933594, -4.352843284606934, -4.383184432983398, -4.285752296447754, -4.369265556335449, -4.330513000488281, -4.479214668273926, -4.367563247680664, -4.337923049926758, -4.333016395568848, -4.332115173339844, -4.312188148498535, -4.370266914367676, -4.369365692138672, -4.35684871673584, -4.30898380279541, -4.478213310241699, -4.4322509765625, -4.356047630310059, -4.36515998840332, -4.3313140869140625, -4.33271598815918, -4.503647804260254, -4.306380271911621, -4.398104667663574, -4.52988338470459, -4.387990951538086, -4.344732284545898, -4.306880950927734, -4.498741149902344, -4.340426445007324, -4.306280136108398, -4.37156867980957, -4.3553466796875, -4.334117889404297, -4.371368408203125, -4.373170852661133, -4.460289001464844, -4.390093803405762, -4.554316520690918, -4.445669174194336, -4.371468544006348, -4.371368408203125, -4.579050064086914, -4.631220817565918, -4.37156867980957, -4.833495140075684, -4.371468544006348, -4.371468544006348, -4.373771667480469, -4.615699768066406, -4.440862655639648, -4.410221099853516, -4.4979400634765625, -4.371268272399902];
        let r:Vec<f32> = sums.iter().zip(activity.iter()).map(|(a,b)|a+b).collect();
        let mut a = EccDense::<f32>::new([10, 10],
                                            [3, 3],
                                             [1, 1],
                                            50,
                                            20,
                                            k,
                                            &mut rng);
        a.sums = sums;
        a.activity = activity;
        a.threshold = t;
        let k = r.iter().position_max_by(|&&a,&&b|if a<b{Less}else{Greater}).unwrap().clone();
        let mut o = CpuSDR::new();
        a.determine_winners(&mut o);
        assert_eq!(o.len(),1);
        assert_eq!(k as u32,o[0]);
        Ok(())
    }



    // fn test8() -> Result<(), String> {
    //     let mut rng = rand::thread_rng();
    //     let o = OpenOptions::new()
    //         .read(true)
    //         .open(file)?;
    //     let mut a:CpuEccMachine<f32> = ciborium::de::from_reader(&mut BufReader::new(o))?;
    //     let k = 1;
    //
    //     let mut number_of_empty_outputs = 0;
    //     for _ in 0..1024 {
    //         let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
    //         let mut input = CpuSDR::from(input);
    //         input.normalize();
    //         assert_ne!(input.len(), 0);
    //         a.run(&input);
    //         let o = a.last_output_sdr_mut();
    //         if o.is_empty() {
    //             number_of_empty_outputs += 1;
    //         }
    //         o.sort();
    //         assert!(o.is_normalized(), "{:?}", o);
    //     }
    //     assert!(number_of_empty_outputs < 54, "{}", number_of_empty_outputs);
    //     Ok(())
    // }
}
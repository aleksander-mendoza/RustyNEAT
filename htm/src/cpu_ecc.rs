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
use crate::{Shape, resolve_range, EncoderTarget, Synapse, top_large_k_indices, top_small_k_indices, Shape3, from_xyz, Shape2, from_xy, range_contains, SparseOrDense, EccMachine, OclEccSparse, OclEccDense};
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

fn top_small_k_by_channel<V: Copy + Debug>(ecc: &impl EccLayer, f: impl Fn(usize) -> V, filter: impl Fn(usize, V) -> bool, gt: fn(V, V) -> bool, output: &mut CpuSDR) {
    let a = as_usize(ecc.out_area());
    let c = as_usize(ecc.out_channels());
    let k = as_usize(ecc.k());
    output.clear();
    for column_idx in 0..a {
        let r = c * column_idx;
        for (i, v) in top_small_k_indices(k, c, |i| {
            debug_assert!(i < c);
            f(i + r)
        }, gt) {
            let e = r + i;
            debug_assert!(r <= e);
            debug_assert!(e < r + c);
            if filter(e, v) {
                let e = as_idx(e);
                debug_assert!(!output.as_slice().contains(&e), "{:?}<-{}={}+{}", output, e, r, i);
                output.push(e);
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct CpuEccSparse {
    /**connections[input_idx]==vector_of_output_indices*/
    connections: Vec<Vec<Idx>>,
    max_incoming_synapses: Idx,
    input_shape: [Idx; 3],
    //[height, width, channels]
    output_shape: [Idx; 3],
    //[height, width, channels]
    kernel: [Idx; 2],
    stride: [Idx; 2],
    k: Idx,
    pub threshold: u16,
    pub sums: Vec<u16>,
}

impl CpuEccSparse {
    pub fn to_ocl(&self, prog:EccProgram) -> Result<OclEccSparse, Error> {
        OclEccSparse::new(self,prog)
    }
    pub fn set_plasticity(&mut self, fractional: u32) {
    }
    pub fn get_plasticity(&self) -> u32 {
        0
    }
    pub fn get_threshold(&self) -> u16 {
        self.threshold
    }
    pub fn set_threshold(&mut self, threshold: u16) {
        self.threshold = threshold
    }

    pub fn connections(&self) -> &Vec<Vec<Idx>> {
        &self.connections
    }

    pub fn new(output: [Idx; 2], kernel: [Idx; 2], stride: [Idx; 2], in_channels: Idx, out_channels: Idx, k: Idx, connections_per_output: Idx, rng: &mut impl Rng) -> Self {
        let input = output.conv_in_size(&stride, &kernel);
        let output = [output[0], output[1], out_channels];
        let input = [input[0], input[1], in_channels];
        let in_size = input.product();
        let out_size = output.product();
        let mut pop = Population::new(as_usize(out_size), 1);
        pop.add_2d_column_grid_with_3d_input(0..as_usize(in_size),
                                             as_usize(out_channels),
                                             as_usize(connections_per_output),
                                             stride.map(as_usize),
                                             kernel.map(as_usize),
                                             input.map(as_usize),
                                             rng);
        let slf = Self::new_from_pop(k, input, output, kernel, stride, &pop);
        debug_assert_eq!(slf.max_incoming_synapses, connections_per_output);
        slf
    }


    fn new_from_pop(k: Idx, input_shape: [Idx; 3], output_shape: [Idx; 3], kernel: [Idx; 2], stride: [Idx; 2], population: &Population) -> Self {
        let mut connections: Vec<Vec<Idx>> = (0..input_shape.product()).map(|_| vec![]).collect();
        let mut max_incoming_synapses = as_idx(population.neurons.iter().map(|n| n.total_synapses()).max().unwrap());
        for (out_idx, neuron) in population.neurons.iter().enumerate() {
            for seg in &neuron.segments {
                for syn in &seg.synapses {
                    connections[syn.input_idx].push(as_idx(out_idx));
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
            sums: vec![0u16; as_usize(output_shape.product())],
        }
    }
}

impl EccLayer for CpuEccSparse {
    type A = CpuSDR;
    fn k(&self) -> Idx { self.k }
    fn set_k(&mut self, k: Idx) {
        assert!(k <= self.out_channels(), "k is larger than layer output!");
        self.k = k;
    }

    fn out_shape(&self) -> &[Idx; 3] { &self.output_shape }

    fn in_shape(&self) -> &[Idx; 3] { &self.input_shape }

    fn kernel(&self) -> &[Idx; 2] { &self.kernel }

    fn stride(&self) -> &[Idx; 2] { &self.stride }

    fn learnable_parameters(&self) -> usize {
        0
    }

    fn get_max_incoming_synapses(&self) -> Idx {
        self.max_incoming_synapses
    }
    fn get_threshold_f32(&self) -> f32 {
        self.threshold as f32 / self.max_incoming_synapses as f32
    }
    fn set_threshold_f32(&mut self, threshold: f32) {
        assert!(threshold > 0., "Negative threshold!");
        self.threshold = (self.max_incoming_synapses as f32 * threshold).round() as u16
    }

    fn set_plasticity_f32(&mut self, fractional: f32) {}

    fn get_plasticity_f32(&self) -> f32 { 0. }

    fn new_empty_sdr(&self, capacity: Idx) -> Self::A {
        CpuSDR::with_capacity(as_usize(capacity))
    }

    fn infer_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR) {
        self.sums.fill(0);
        for &input_idx in input.as_slice() {
            for &c in &self.connections[as_usize(input_idx)] {
                self.sums[as_usize(c)] += 1;
            }
        }
        let t = self.threshold;
        top_small_k_by_channel(self, |i| self.sums[i], |i, v| v >= t, |a, b| a > b, output)
    }

    fn decrement_activities(&mut self, output: &CpuSDR) {}

    fn learn(&mut self, input: &CpuSDR, output: &CpuSDR) {}
}


pub trait DenseWeight: Add<Output=Self> + AddAssign + Sub<Output=Self> + SubAssign + Copy + Debug + Display + Sum {
    const IMPOSSIBLE_WEIGHT: Self;
    const TOTAL_SUM: Self;
    const ZERO: Self;
    const DEFAULT_PLASTICITY: Self;
    const INITIAL_ACTIVITY: Self;
    const ACTIVITY_PENALTY: Self;
    const APPROX_EPSILON: Self;
    fn w_to_f32(w: Self) -> f32;
    fn gt(self, b: Self) -> bool;
    fn lt(self, b: Self) -> bool;
    fn ge(self, b: Self) -> bool;
    fn le(self, b: Self) -> bool;
    fn eq(self, b: Self) -> bool;
    fn approx_eq(self, b: Self) -> bool;
    fn min_w(self, b: Self) -> Self {
        if self.lt(b) { self } else { b }
    }
    fn max_w(self, b: Self) -> Self {
        if self.gt(b) { self } else { b }
    }
    fn f32_to_w(w: f32) -> Self;
    fn initialise_weight_matrix(kernel_column_volume: Idx, output_volume: Idx, seed: Vec<f32>) -> Vec<Self>;
    fn default_threshold(out_channels: Idx) -> Self;
    fn normalize_stochastic(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, rand_seed: Rand, kernel_column_volume: Idx, output_volume: Idx) -> Idx;
    fn normalize(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, kernel_column_volume: Idx, output_volume: Idx);
}

// fn debug_assert_eq_weight<D: DenseWeight>(a: D, b: D) {
//     debug_assert!(a.eq(b), "{}!={}", a, b)
// }
fn debug_assert_approx_eq_weight<D: DenseWeight>(a: D, b: D) {
    debug_assert!(a.approx_eq(b), "{}!={} +- {}", a, b, D::APPROX_EPSILON)
}

pub const MARGIN_OF_SAFETY: u8 = 10;

impl DenseWeight for u32 {
    const IMPOSSIBLE_WEIGHT: u32 = u32::MAX;
    const TOTAL_SUM: u32 = 1 << (13 + MARGIN_OF_SAFETY);
    const ZERO: Self = 0;
    // ACTIVITY_PENALTY == 2^2
    // TOTAL_SUM == 2^12
    //notice that in f32, the activity penalty becomes
    // ACTIVITY_PENALTY/TOTAL_SUM == 2^2/2^12 == 1/1024 ~= 0.0001
    const DEFAULT_PLASTICITY: u32 = Self::ACTIVITY_PENALTY;
    const INITIAL_ACTIVITY: u32 = u32::MAX - Self::TOTAL_SUM;
    const ACTIVITY_PENALTY: u32 = 1 << MARGIN_OF_SAFETY;
    const APPROX_EPSILON: u32 = 1024;

    // We have 21 bits of maneuver.
    // Should be good enough for now
    fn w_to_f32(w: u32) -> f32 {
        (w as f64 / Self::TOTAL_SUM as f64) as f32
    }

    fn gt(self, b: Self) -> bool {
        self > b
    }
    fn lt(self, b: Self) -> bool {
        self < b
    }
    fn ge(self, b: Self) -> bool {
        self >= b
    }
    fn le(self, b: Self) -> bool {
        self <= b
    }
    fn eq(self, b: Self) -> bool {
        self == b
    }

    fn approx_eq(self, b: Self) -> bool {
        if self < b { b - self < Self::APPROX_EPSILON } else { self - b < Self::APPROX_EPSILON }
    }
    fn f32_to_w(w: f32) -> u32 {
        (w as f64 * Self::TOTAL_SUM as f64) as u32
    }
    fn initialise_weight_matrix(kernel_column_volume: Idx, output_volume: Idx, seed: Vec<f32>) -> Vec<u32> {
        let kv = kernel_column_volume;
        let v = output_volume;
        let wf = seed;
        assert_eq!(as_usize(kv * v), wf.len());
        let mut w: Vec<u32> = vec![u32::MAX; wf.len()];
        for output_idx in 0..v {
            let w_sum = kernel_column_weight_sum(kv, v, output_idx, &wf);
            let mut min_w = u32::MAX;
            let mut min_w_position = 0;
            let mut w_new_sum = 0;
            for input_within_kernel_column in 0..kv {
                let w_idx = w_idx(output_idx, input_within_kernel_column, v);
                let w_f32 = wf[as_usize(w_idx)];
                debug_assert_eq!(u32::MAX, w[as_usize(w_idx)]);
                let w_new = Self::f32_to_w(w_f32 / w_sum);
                w[as_usize(w_idx)] = w_new;
                w_new_sum += w_new;
                if w_new < min_w {
                    min_w = w_new;
                    min_w_position = input_within_kernel_column;
                }
            }
            debug_assert_ne!(min_w, u32::MAX);
            debug_assert_eq!(w_new_sum, kernel_column_weight_sum(kv, v, output_idx, &w));
            let min_w_position = w_idx(output_idx, min_w_position, v);
            w[as_usize(min_w_position)] = w[as_usize(min_w_position)].wrapping_add(Self::TOTAL_SUM.wrapping_sub(w_new_sum)); // we do this step just in case if f32 limited precision
            // caused some small drifts. Safety: Addition and subtraction for both signed and unsigned types are the same operation.
            //So overflows don't bother us.
        }
        debug_assert!(!w.contains(&u32::MAX));
        w
    }
    fn default_threshold(out_channels: Idx) -> u32 {
        (Self::TOTAL_SUM as f64 / out_channels as f64) as u32
    }
    fn normalize_stochastic(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, mut rand_seed: Rand, kernel_column_volume: Idx, output_volume: Idx) -> Idx {
        let mut fallback_input_idx = rand_seed % kernel_column_volume;
        for _ in 0..active_inputs {
            rand_seed = xorshift_rand(rand_seed);
            let input_idx_within_kernel_column = rand_seed % kernel_column_volume;
            let w_index = w_idx(output_idx, input_idx_within_kernel_column, output_volume);
            if weights[as_usize(w_index)] >= plasticity {
                weights[as_usize(w_index)] -= plasticity;
            } else {
                loop {
                    let w_index = w_idx(output_idx, fallback_input_idx, output_volume);
                    fallback_input_idx += 1;
                    if fallback_input_idx == kernel_column_volume {
                        fallback_input_idx = 0
                    }
                    if weights[as_usize(w_index)] >= plasticity {
                        weights[as_usize(w_index)] -= plasticity;
                        break;
                    }
                }
            }
        }
        rand_seed
    }

    fn normalize(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, kernel_column_volume: Idx, output_volume: Idx) {
        let kv = kernel_column_volume;
        let v = output_volume;
        let sum_before = kernel_column_weight_sum(kv, v, output_idx, &weights);
        let w_factor = Self::TOTAL_SUM as f64 / sum_before as f64;
        for input_within_kernel_column in 0..kv {
            let w_idx = w_idx(output_idx, input_within_kernel_column, v);
            let w = weights[as_usize(w_idx)];
            let new_w = (w as f64 * w_factor) as u32;
            weights[as_usize(w_idx)] = new_w;
        }
        debug_assert_approx_eq_weight(Self::TOTAL_SUM, kernel_column_weight_sum(kv, v, output_idx, &weights));
    }
}

impl DenseWeight for f32 {
    const IMPOSSIBLE_WEIGHT: f32 = -1.;
    const TOTAL_SUM: f32 = 1.;
    const ZERO: Self = 0.;
    const DEFAULT_PLASTICITY: Self = Self::ACTIVITY_PENALTY;
    const INITIAL_ACTIVITY: Self = 0.;
    const ACTIVITY_PENALTY: Self = 0.0001;
    const APPROX_EPSILON: Self = 0.00001;

    fn w_to_f32(w: Self) -> f32 {
        w
    }

    fn gt(self, b: Self) -> bool {
        self > b
    }
    fn lt(self, b: Self) -> bool {
        self < b
    }
    fn ge(self, b: Self) -> bool {
        self >= b
    }
    fn le(self, b: Self) -> bool {
        self <= b
    }
    fn eq(self, b: Self) -> bool {
        self == b
    }
    fn approx_eq(self, b: Self) -> bool {
        (self - b).abs() < Self::APPROX_EPSILON
    }

    fn f32_to_w(w: f32) -> Self {
        w
    }
    fn initialise_weight_matrix(kernel_column_volume: Idx, output_volume: Idx, seed: Vec<f32>) -> Vec<Self> {
        let kv = kernel_column_volume;
        let v = output_volume;
        let mut w = seed;
        assert_eq!(as_usize(kv * v), w.len());
        for output_idx in 0..v {
            let w_sum = kernel_column_weight_sum(kv, v, output_idx, &w);
            for input_within_kernel_column in 0..kv {
                let w_idx = w_idx(output_idx, input_within_kernel_column, v);
                w[as_usize(w_idx)] /= w_sum;
            }
            debug_assert_approx_eq_weight(1., kernel_column_weight_sum(kv, v, output_idx, &w));
        }
        w
    }
    fn default_threshold(out_channels: Idx) -> Self {
        Self::TOTAL_SUM / out_channels as f32
    }
    fn normalize_stochastic(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, mut rand_seed: Rand, kernel_column_volume: Idx, output_volume: Idx) -> Idx {
        unimplemented!();
    }
    fn normalize(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, kernel_column_volume: Idx, output_volume: Idx) {
        let kv = kernel_column_volume;
        let v = output_volume;
        let w_sum = Self::TOTAL_SUM + active_inputs as f32 * plasticity;
        debug_assert_approx_eq_weight(w_sum, kernel_column_weight_sum(kv, v, output_idx, &weights));
        for input_within_kernel_column in 0..kv {
            let w_idx = w_idx(output_idx, input_within_kernel_column, v);
            weights[as_usize(w_idx)] /= w_sum;
        }
        debug_assert_approx_eq_weight(1., kernel_column_weight_sum(kv, v, output_idx, &weights));
    }
}

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
    pub threshold: D,
    pub plasticity: D,
    activity: Vec<D>,
    pub rand_seed: Idx,
    pub sums: Vec<D>,
}

#[inline]
fn w_idx(output_idx: Idx, idx_within_kernel_column: Idx, output_volume: Idx) -> Idx {
    debug_assert!(output_idx < output_volume);
    output_idx + idx_within_kernel_column * output_volume
}

#[inline]
fn kernel_column_weight_sum<D: Sum + Copy>(kernel_column_volume: Idx, out_volume: Idx, output_neuron_idx: Idx, w: &[D]) -> D {
    assert!(output_neuron_idx < out_volume);
    (0..kernel_column_volume).map(|i| w[as_usize(w_idx(output_neuron_idx, i, out_volume))]).sum()
}

impl<D: DenseWeight> CpuEccDense<D> {
    pub fn new(output: [Idx; 2], kernel: [Idx; 2], stride: [Idx; 2], in_channels: Idx, out_channels: Idx, k: Idx, rng: &mut impl Rng) -> Self {
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
            activity: vec![D::INITIAL_ACTIVITY; as_usize(v)],
            rand_seed: auto_gen_seed32(),
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
            threshold: first_layer.threshold,
            plasticity: first_layer.plasticity,
            activity: vec![D::INITIAL_ACTIVITY; as_usize(new_v)],
            rand_seed: first_layer.rand_seed,
            sums: vec![D::ZERO; as_usize(new_v)],
        };
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
                            assert!(slf.w[as_usize(new_w_idx)].eq(D::IMPOSSIBLE_WEIGHT));
                            slf.w[as_usize(new_w_idx)] = l.w[as_usize(original_w_idx)];
                        }
                    }
                }
            }
            channel_offset += l.out_channels();
        }
        debug_assert!(slf.w.iter().find(|&&x| x.eq(D::IMPOSSIBLE_WEIGHT)).is_none());
        slf
    }
    pub fn kernel_offset(&self, output_pos: &[Idx; 3]) -> [Idx; 2] {
        output_pos.grid().conv_in_range_begin(&self.stride)
    }
    pub fn pos_within_kernel(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> [Idx; 3] {
        debug_assert!(output_pos.all_lt(&self.output_shape));
        debug_assert!(input_pos.all_lt(&self.input_shape));
        debug_assert!(range_contains(&output_pos.grid().conv_in_range(&self.stride, &self.kernel), input_pos.grid()));
        debug_assert!(range_contains(&input_pos.grid().conv_out_range_clipped(&self.stride, &self.kernel), output_pos.grid()));
        Self::sub_kernel_offset(input_pos, &self.kernel_offset(output_pos))
    }
    fn sub_kernel_offset(input_pos: &[Idx; 3], offset: &[Idx; 2]) -> [Idx; 3] {
        from_xyz(input_pos.width() - offset.width(), input_pos.height() - offset.height(), input_pos.channels())
    }

    #[inline]
    fn w_index_(input_pos: &[Idx; 3], kernel_offset: &[Idx; 2], output_idx: Idx, kernel_column: &[Idx; 3], output_volume: Idx) -> Idx {
        let position_within_kernel_column = Self::sub_kernel_offset(input_pos, kernel_offset);
        w_idx(output_idx, kernel_column.idx(position_within_kernel_column), output_volume)
    }
    pub fn idx_within_kernel(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> Idx {
        self.kernel_column().idx(self.pos_within_kernel(input_pos, output_pos))
    }
    pub fn w_index(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> Idx {
        debug_assert!(output_pos.all_lt(&self.output_shape));
        debug_assert!(input_pos.all_lt(&self.input_shape));
        debug_assert!(range_contains(&output_pos.grid().conv_in_range(&self.stride, &self.kernel), input_pos.grid()));
        debug_assert!(range_contains(&input_pos.grid().conv_out_range_clipped(&self.stride, &self.kernel), output_pos.grid()));
        w_idx(self.out_shape().idx(*output_pos), self.idx_within_kernel(input_pos, output_pos), self.out_volume())
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
    pub fn get_threshold(&self) -> D {
        self.threshold
    }
    pub fn set_threshold(&mut self, threshold: D) {
        self.threshold = threshold
    }
    pub fn set_plasticity(&mut self, plasticity: D) {
        self.plasticity = plasticity
    }
    pub fn get_plasticity(&self) -> D {
        self.plasticity
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
        let t = self.threshold;
        top_small_k_by_channel(self, |i| {
            debug_assert!(self.sums[i].le(D::TOTAL_SUM), "{}<={}", self.sums[i], D::TOTAL_SUM);
            debug_assert!(self.activity[i].le(D::INITIAL_ACTIVITY), "{}<={}", self.activity[i], D::INITIAL_ACTIVITY);
            self.sums[i] + self.activity[i]
        }, |i, v| self.sums[i].ge(t), D::gt, output);
    }
}
impl OclEccSparse{
    pub fn to_cpu(&self)->CpuEccSparse{
        let connection_ranges = self.get_connection_ranges().to_vec(self.prog().queue()).unwrap();
        let connections = self.get_connections().to_vec(self.prog().queue()).unwrap();
        CpuEccSparse{
            connections: connection_ranges.into_iter().map(|range|connections[range[0] as usize..(range[0]+range[1])as usize].to_vec()).collect(),
            max_incoming_synapses: self.get_max_incoming_synapses(),
            input_shape: *self.in_shape(),
            output_shape: *self.out_shape(),
            kernel: *self.kernel(),
            stride: *self.stride(),
            k: self.k,
            threshold: self.get_threshold(),
            sums: vec![0;self.sums.len()]
        }
    }
}

impl OclEccDense{
    pub fn to_cpu(&self)->CpuEccDense<u32>{
        CpuEccDense{
            w: self.w().to_vec(self.prog().queue()).unwrap(),
            input_shape: *self.in_shape(),
            output_shape: *self.out_shape(),
            kernel: *self.kernel(),
            stride: *self.stride(),
            k: self.k(),
            threshold: self.get_threshold(),
            plasticity: self.plasticity,
            activity: self.activity().to_vec(self.prog().queue()).unwrap(),
            rand_seed: 0,
            sums: vec![0; self.sums.len()]
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
    pub fn to_ocl(&self, prog:EccProgram) -> Result<OclEccDense, Error> {
        OclEccDense::new(self,prog)
    }
}

impl<D: DenseWeight> EccLayer for CpuEccDense<D> {
    type A = CpuSDR;

    fn k(&self) -> Idx { self.k }

    fn set_k(&mut self, k: Idx) {
        assert!(k <= self.out_channels(), "k is larger than layer output!");
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
        CpuSDR::with_capacity(as_usize(capacity))
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
                        self.sums[as_usize(output_idx)] += w;
                        debug_assert!(self.sums[as_usize(output_idx)].le(D::TOTAL_SUM), "{:?}->{:?}={}@{}<={}", input_pos, output_pos, output_idx, self.sums[as_usize(output_idx)], D::TOTAL_SUM);
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
        let one_minus_p = D::TOTAL_SUM - p;
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
                    if self.w[as_usize(w_index)].le(one_minus_p) {
                        self.w[as_usize(w_index)] += p;
                        active_inputs += 1;
                    }
                }
            }
            D::normalize(&mut self.w, active_inputs, output_idx, p, kv, v);
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

pub type CpuEccMachine<D: DenseWeight> = EccMachine<CpuSDR, CpuEccSparse, CpuEccDense<D>>;

impl<D: DenseWeight> CpuEccMachine<D> {
    pub fn new_cpu(output: [Idx; 2], kernels: &[[Idx; 2]], strides: &[[Idx; 2]], channels: &[Idx], k: &[Idx], connections_per_output: &[Option<Idx>], rng: &mut impl Rng) -> Self {
        Self::new(output, kernels, strides, channels, k, connections_per_output, rng, |output, in_channels, out_channels, k, kernel, stride, conn, rng|
            if let Some(connections_per_output) = conn {
                SparseOrDense::Sparse(CpuEccSparse::new(output, kernel, stride, in_channels, out_channels, k, connections_per_output, rng))
            } else {
                SparseOrDense::Dense(CpuEccDense::new(output, kernel, stride, in_channels, out_channels, k, rng))
            })
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
        let mut a = CpuEccSparse::new([4, 4], [2, 2], [1, 1], 3, 4, 1, 4, &mut rng);
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
        a.rand_seed = 34634;
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
        println!("{} {}", s, a.rand_seed);
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
}
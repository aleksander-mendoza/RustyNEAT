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
use crate::{Shape, resolve_range, EncoderTarget, Synapse, top_large_k_indices, top_small_k_indices, Shape3, from_xyz, Shape2, from_xy, range_contains, SparseOrDense, EccMachine, OclEccSparse, OclEccDense, CpuEccMachine, top_small_k_by_channel};
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
    fn mul(self, b: Idx) -> Self;
    fn is_valid(self)->bool{
        !self.eq(Self::IMPOSSIBLE_WEIGHT)
    }
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
    fn normalize(weights: &mut [Self], weight_sum:Self, output_idx: Idx, kernel_column_volume: Idx, output_volume: Idx);
    fn normalize_quick(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, kernel_column_volume: Idx, output_volume: Idx){
        let w_sum = Self::TOTAL_SUM + plasticity.mul(active_inputs);
        Self::normalize(weights,w_sum,output_idx, kernel_column_volume, output_volume)
    }
    fn normalize_precise(weights: &mut [Self], output_idx: Idx, kernel_column_volume: Idx, output_volume: Idx){
        let w_sum = kernel_column_weight_sum(kernel_column_volume, output_volume, output_idx, weights);
        Self::normalize(weights,w_sum,output_idx,kernel_column_volume,output_volume)
    }
    fn normalize_recommended(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, kernel_column_volume: Idx, output_volume: Idx);
}

// fn debug_assert_eq_weight<D: DenseWeight>(a: D, b: D) {
//     debug_assert!(a.eq(b), "{}!={}", a, b)
// }
pub fn debug_assert_approx_eq_weight<D: DenseWeight>(a: D, b: D) {
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
    fn mul(self, b: Idx) -> Self{
        self*b
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

    fn normalize(weights: &mut [Self], sum_before:Self, output_idx: Idx, kernel_column_volume: Idx, output_volume: Idx) {
        let kv = kernel_column_volume;
        let v = output_volume;
        let w_factor = Self::TOTAL_SUM as f64 / sum_before as f64;
        for input_within_kernel_column in 0..kv {
            let w_idx = w_idx(output_idx, input_within_kernel_column, v);
            let w = weights[as_usize(w_idx)];
            if w.is_valid() {
                let new_w = (w as f64 * w_factor) as u32;
                weights[as_usize(w_idx)] = new_w;
            }
        }
        debug_assert_approx_eq_weight(Self::TOTAL_SUM, kernel_column_weight_sum(kv, v, output_idx, &weights));
    }
    fn normalize_recommended(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, kernel_column_volume: Idx, output_volume: Idx){
        Self::normalize_precise(weights,output_idx,kernel_column_volume,output_volume)
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
    fn mul(self, b: Idx) -> Self{
        self*b as f32
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
                let weight = w[as_usize(w_idx)];
                debug_assert!(0.<=weight&&weight<=1.);
                w[as_usize(w_idx)] = weight/w_sum;
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
    fn normalize(weights: &mut [Self], w_sum: Self, output_idx: Idx, kernel_column_volume: Idx, output_volume: Idx) {
        let kv = kernel_column_volume;
        let v = output_volume;
        debug_assert_approx_eq_weight(w_sum, kernel_column_weight_sum(kv, v, output_idx, &weights));
        for input_within_kernel_column in 0..kv {
            let w_idx = w_idx(output_idx, input_within_kernel_column, v);
            if weights[as_usize(w_idx)]!=Self::IMPOSSIBLE_WEIGHT {
                weights[as_usize(w_idx)] /= w_sum;
            }
        }
        debug_assert_approx_eq_weight(1., kernel_column_weight_sum(kv, v, output_idx, &weights));
    }
    fn normalize_recommended(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, kernel_column_volume: Idx, output_volume: Idx){
        Self::normalize_quick(weights,active_inputs,output_idx,plasticity,kernel_column_volume,output_volume)
    }
}

#[inline]
pub fn w_idx(output_idx: Idx, idx_within_kernel_column: Idx, output_volume: Idx) -> Idx {
    debug_assert!(output_idx < output_volume);
    output_idx + idx_within_kernel_column * output_volume
}

#[inline]
pub fn kernel_column_weight_sum<D: DenseWeight>(kernel_column_volume: Idx, out_volume: Idx, output_neuron_idx: Idx, w: &[D]) -> D {
    assert!(output_neuron_idx < out_volume);
    (0..kernel_column_volume).map(|i| w[as_usize(w_idx(output_neuron_idx, i, out_volume))]).filter(|&w|w.is_valid()).sum()
}

#[inline]
pub fn kernel_column_dropped_weights_count<D: DenseWeight>(kernel_column_volume: Idx, out_volume: Idx, output_neuron_idx: Idx, w: &[D]) -> usize {
    assert!(output_neuron_idx < out_volume);
    (0..kernel_column_volume).map(|i| w[as_usize(w_idx(output_neuron_idx, i, out_volume))]).filter(|&w|!w.is_valid()).count()
}


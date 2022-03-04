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
use crate::xorshift::{auto_gen_seed64, xorshift64, auto_gen_seed, xorshift, xorshift32, auto_gen_seed32};
use itertools::{Itertools, assert_equal};
use std::iter::Sum;
use ocl::core::DeviceInfo::MaxConstantArgs;
use crate::ecc::{EccLayer, Idx, as_idx, Rand, xorshift_rand};
use crate::sdr::SDR;
use crate::as_usize::AsUsize;
use std::marker::PhantomData;

pub trait DenseWeight: Add<Output=Self> + AddAssign + Sub<Output=Self> + SubAssign + Copy + Debug + Display + Sum + Sync + Send + Sized{
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
    fn is_valid(self) -> bool {
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
    fn initialise_weight_matrix<M: Metric<Self>>(kernel_column_volume: Idx, output_volume: Idx, seed: Vec<f32>) -> Vec<Self>;
    fn default_threshold(out_channels: Idx) -> Self;
    fn normalize_stochastic(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, rand_seed: Rand, kernel_column_volume: Idx, output_volume: Idx) -> Idx;
    fn normalize<M: Metric<Self>>(weights: &mut [Self], weight_sum: Self, output_idx: Idx, kernel_column_volume: Idx, output_volume: Idx);
    // /**Assumes that the weights were already normalized (summing up to 1) and recently several (active_inputs) weights
    // have been incremented by (plasticity) constant. Based on that we can quickly figure out the new sum of weights.
    // Then we use this sum to normalise the weights. In ideal world, this would be great but due to floating-point finite
    // precision, this calculation will slowly build up error.*/
    // fn normalize_quick<M:Metric<D>>(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, kernel_column_volume: Idx, output_volume: Idx){
    //     let w_sum = Self::TOTAL_SUM + plasticity.mul(active_inputs);
    //     Self::normalize::<M>(weights,w_sum,output_idx, kernel_column_volume, output_volume)
    // }
    /**First computes the exact sum of weights in linear O(n) time. Then uses this sum to normalise the weights*/
    fn normalize_precise<M: Metric<Self>>(weights: &mut [Self], output_idx: Idx, kernel_column_volume: Idx, output_volume: Idx) {
        let magnitude = M::kernel_column_weight_magnitude(kernel_column_volume, output_volume, output_idx, weights);
        Self::normalize::<M>(weights, magnitude, output_idx, kernel_column_volume, output_volume)
    }
}

pub trait DenseWeightL2: DenseWeight + Mul<Output=Self> {
    fn sqrt(self) -> Self;
}

// fn debug_assert_eq_weight<D: DenseWeight>(a: D, b: D) {
//     debug_assert!(a.eq(b), "{}!={}", a, b)
// }
pub fn debug_assert_approx_eq_weight<D: DenseWeight>(a: D, b: D) {
    debug_assert!(a.approx_eq(b), "{}!={} +- {}", a, b, D::APPROX_EPSILON)
}
//
// pub const MARGIN_OF_SAFETY: u8 = 10;
//
// impl DenseWeight for u32 {
//     const IMPOSSIBLE_WEIGHT: u32 = u32::MAX;
//     const TOTAL_SUM: u32 = 1 << (13 + MARGIN_OF_SAFETY);
//     const ZERO: Self = 0;
//     // ACTIVITY_PENALTY == 2^2
//     // TOTAL_SUM == 2^12
//     //notice that in f32, the activity penalty becomes
//     // ACTIVITY_PENALTY/TOTAL_SUM == 2^2/2^12 == 1/1024 ~= 0.0001
//     const DEFAULT_PLASTICITY: u32 = Self::ACTIVITY_PENALTY;
//     const INITIAL_ACTIVITY: u32 = u32::MAX - Self::TOTAL_SUM;
//     const ACTIVITY_PENALTY: u32 = 1 << MARGIN_OF_SAFETY;
//     const APPROX_EPSILON: u32 = 1024;
//
//     // We have 21 bits of maneuver.
//     // Should be good enough for now
//     fn w_to_f32(w: u32) -> f32 {
//         (w as f64 / Self::TOTAL_SUM as f64) as f32
//     }
//
//     fn gt(self, b: Self) -> bool {
//         self > b
//     }
//     fn lt(self, b: Self) -> bool {
//         self < b
//     }
//     fn ge(self, b: Self) -> bool {
//         self >= b
//     }
//     fn le(self, b: Self) -> bool {
//         self <= b
//     }
//     fn eq(self, b: Self) -> bool {
//         self == b
//     }
//     fn mul(self, b: Idx) -> Self{
//         self*b
//     }
//     fn approx_eq(self, b: Self) -> bool {
//         if self < b { b - self < Self::APPROX_EPSILON } else { self - b < Self::APPROX_EPSILON }
//     }
//     fn f32_to_w(w: f32) -> u32 {
//         (w as f64 * Self::TOTAL_SUM as f64) as u32
//     }
//     fn initialise_weight_matrix<M:Metric<f32>>(kernel_column_volume: Idx, output_volume: Idx, seed: Vec<f32>) -> Vec<u32> {
//         let kv = kernel_column_volume;
//         let v = output_volume;
//         let wf = seed;
//         assert_eq!((kv * v).as_usize(), wf.len());
//         let mut w: Vec<u32> = vec![u32::MAX; wf.len()];
//         for output_idx in 0..v {
//             let w_sum = M::kernel_column_weight_sum(kv, v, output_idx, wf.as_slice());
//             let mut min_w = u32::MAX;
//             let mut min_w_position = 0;
//             let mut w_new_sum = 0;
//             for input_within_kernel_column in 0..kv {
//                 let w_idx = w_idx(output_idx, input_within_kernel_column, v);
//                 let w_f32 = wf[w_idx.as_usize()];
//                 debug_assert_eq!(u32::MAX, w[w_idx.as_usize()]);
//                 let w_new = Self::f32_to_w(w_f32 / w_sum);
//                 w[w_idx.as_usize()] = w_new;
//                 w_new_sum += w_new;
//                 if w_new < min_w {
//                     min_w = w_new;
//                     min_w_position = input_within_kernel_column;
//                 }
//             }
//             debug_assert_ne!(min_w, u32::MAX);
//             debug_assert_eq!(w_new_sum, kernel_column_weight_sum(kv, v, output_idx, &w));
//             let min_w_position = w_idx(output_idx, min_w_position, v);
//             w[min_w_position.as_usize()] = w[min_w_position.as_usize()].wrapping_add(Self::TOTAL_SUM.wrapping_sub(w_new_sum)); // we do this step just in case if f32 limited precision
//             // caused some small drifts. Safety: Addition and subtraction for both signed and unsigned types are the same operation.
//             //So overflows don't bother us.
//         }
//         debug_assert!(!w.contains(&u32::MAX));
//         w
//     }
//     fn default_threshold(out_channels: Idx) -> u32 {
//         (Self::TOTAL_SUM as f64 / out_channels as f64) as u32
//     }
//     fn normalize_stochastic(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, mut rand_seed: Rand, kernel_column_volume: Idx, output_volume: Idx) -> Idx {
//         let mut fallback_input_idx = rand_seed % kernel_column_volume;
//         for _ in 0..active_inputs {
//             rand_seed = xorshift_rand(rand_seed);
//             let input_idx_within_kernel_column = rand_seed % kernel_column_volume;
//             let w_index = w_idx(output_idx, input_idx_within_kernel_column, output_volume);
//             if weights[w_index.as_usize()] >= plasticity {
//                 weights[w_index.as_usize()] -= plasticity;
//             } else {
//                 loop {
//                     let w_index = w_idx(output_idx, fallback_input_idx, output_volume);
//                     fallback_input_idx += 1;
//                     if fallback_input_idx == kernel_column_volume {
//                         fallback_input_idx = 0
//                     }
//                     if weights[w_index.as_usize()] >= plasticity {
//                         weights[w_index.as_usize()] -= plasticity;
//                         break;
//                     }
//                 }
//             }
//         }
//         rand_seed
//     }
//
//     fn normalize(weights: &mut [Self], sum_before:Self, output_idx: Idx, kernel_column_volume: Idx, output_volume: Idx) {
//         let kv = kernel_column_volume;
//         let v = output_volume;
//         let w_factor = Self::TOTAL_SUM as f64 / sum_before as f64;
//         for input_within_kernel_column in 0..kv {
//             let w_idx = w_idx(output_idx, input_within_kernel_column, v);
//             let w = weights[w_idx.as_usize()];
//             if w.is_valid() {
//                 let new_w = (w as f64 * w_factor) as u32;
//                 weights[w_idx.as_usize()] = new_w;
//             }
//         }
//         debug_assert_approx_eq_weight(Self::TOTAL_SUM, kernel_column_weight_sum(kv, v, output_idx, &weights));
//     }
//     fn normalize_recommended(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, kernel_column_volume: Idx, output_volume: Idx){
//         Self::normalize_precise(weights,output_idx,kernel_column_volume,output_volume)
//     }
// }

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
    fn mul(self, b: Idx) -> Self {
        self * b as f32
    }
    fn approx_eq(self, b: Self) -> bool {
        (self - b).abs() < Self::APPROX_EPSILON
    }

    fn f32_to_w(w: f32) -> Self {
        w
    }
    fn initialise_weight_matrix<M: Metric<Self>>(kernel_column_volume: Idx, output_volume: Idx, seed: Vec<f32>) -> Vec<Self> {
        let kv = kernel_column_volume;
        let v = output_volume;
        let mut w = seed;
        assert_eq!((kv * v).as_usize(), w.len());
        for output_idx in 0..v {
            let w_sum = M::kernel_column_weight_magnitude(kv, v, output_idx, &w);
            for input_within_kernel_column in 0..kv {
                let w_idx = w_idx(output_idx, input_within_kernel_column, v);
                let weight = w[w_idx.as_usize()];
                debug_assert!(0. <= weight && weight <= 1.);
                w[w_idx.as_usize()] = weight / w_sum;
            }
            debug_assert_approx_eq_weight(1., M::kernel_column_weight_sum(kv, v, output_idx, &w));
        }
        w
    }
    fn default_threshold(out_channels: Idx) -> Self {
        Self::TOTAL_SUM / out_channels as f32
    }
    fn normalize_stochastic(weights: &mut [Self], active_inputs: Idx, output_idx: Idx, plasticity: Self, mut rand_seed: Rand, kernel_column_volume: Idx, output_volume: Idx) -> Idx {
        unimplemented!();
    }
    fn normalize<M: Metric<Self>>(weights: &mut [Self], w_magnitude: Self, output_idx: Idx, kernel_column_volume: Idx, output_volume: Idx) {
        let kv = kernel_column_volume;
        let v = output_volume;
        debug_assert_approx_eq_weight(w_magnitude, M::kernel_column_weight_magnitude(kv, v, output_idx, &weights));
        for input_within_kernel_column in 0..kv {
            let w_idx = w_idx(output_idx, input_within_kernel_column, v);
            if weights[w_idx.as_usize()] != Self::IMPOSSIBLE_WEIGHT {
                weights[w_idx.as_usize()] /= w_magnitude;
            }
        }
        debug_assert_approx_eq_weight(1., M::kernel_column_weight_sum(kv, v, output_idx, &weights));
    }
}

impl DenseWeightL2 for f32 {
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

#[inline]
pub fn w_idx(output_idx: Idx, idx_within_kernel_column: Idx, output_volume: Idx) -> Idx {
    debug_assert!(output_idx < output_volume);
    output_idx + idx_within_kernel_column * output_volume
}

#[inline]
pub fn kernel_column_weight_copy<D: DenseWeight>(kernel_column_volume: Idx, out_volume: Idx, output_neuron_idx: Idx, w: &[D]) -> Vec<D> {
    assert!(output_neuron_idx < out_volume);
    (0..kernel_column_volume).map(|i| w[w_idx(output_neuron_idx, i, out_volume).as_usize()]).collect()
}

// #[inline]
// pub fn kernel_column_weight_sum<D: DenseWeight>(kernel_column_volume: Idx, out_volume: Idx, output_neuron_idx: Idx, w: &[D]) -> D {
//     kernel_column_weight_sum_ln(kernel_column_volume,out_volume,output_neuron_idx,w,|w|w)
// }
//
// #[inline]
// pub fn kernel_column_weight_sum_l2<D: DenseWeightL2>(kernel_column_volume: Idx, out_volume: Idx, output_neuron_idx: Idx, w: &[D]) -> D {
//     kernel_column_weight_sum_ln(kernel_column_volume,out_volume,output_neuron_idx,w,|w|w*w)
// }
#[inline]
fn kernel_column_weight_sum<D: DenseWeight>(kernel_column_volume: Idx, out_volume: Idx, output_neuron_idx: Idx, w: &[D], metric: impl Fn(D) -> D) -> D {
    assert!(output_neuron_idx < out_volume);
    (0..kernel_column_volume).map(|i| w[w_idx(output_neuron_idx, i, out_volume).as_usize()]).filter(|&w| w.is_valid()).map(metric).sum()
}

pub trait Metric<D>: Clone + Sync + Send{
    fn kernel_column_weight_sum(kernel_column_volume: Idx, out_volume: Idx, output_neuron_idx: Idx, w: &[D]) -> D;
    fn root(_: D) -> D;
    fn kernel_column_weight_magnitude(kernel_column_volume: Idx, out_volume: Idx, output_neuron_idx: Idx, w: &[D]) -> D{
        Self::root(Self::kernel_column_weight_sum(kernel_column_volume,out_volume,output_neuron_idx,w))
    }
    fn max_inner_product(sdr:&CpuSDR)->D;
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct MetricL2<D: DenseWeightL2> {
    _p: PhantomData<D>,
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct MetricL1<D: DenseWeight> {
    _p: PhantomData<D>,
}

impl<D: DenseWeight> Metric<D> for MetricL1<D> {
    fn kernel_column_weight_sum(kernel_column_volume: u32, out_volume: u32, output_neuron_idx: u32, w: &[D]) -> D {
        kernel_column_weight_sum(kernel_column_volume, out_volume, output_neuron_idx, w, |w| w)
    }

    fn root(w: D) -> D {
        w
    }

    fn max_inner_product(sdr: &CpuSDR) -> D {
        D::TOTAL_SUM
    }
}

impl<D: DenseWeightL2> Metric<D> for MetricL2<D> {
    fn kernel_column_weight_sum(kernel_column_volume: u32, out_volume: u32, output_neuron_idx: u32, w: &[D]) -> D {
        kernel_column_weight_sum(kernel_column_volume, out_volume, output_neuron_idx, w, |w| w * w)
    }

    fn root(w: D) -> D {
        w.sqrt()
    }

    fn max_inner_product(sdr: &CpuSDR) -> D {
        D::f32_to_w((sdr.cardinality() as f32).sqrt())
    }
}

#[inline]
pub fn kernel_column_dropped_weights_count<D: DenseWeight>(kernel_column_volume: Idx, out_volume: Idx, output_neuron_idx: Idx, w: &[D]) -> usize {
    assert!(output_neuron_idx < out_volume);
    (0..kernel_column_volume).map(|i| w[w_idx(output_neuron_idx, i, out_volume).as_usize()]).filter(|&w| !w.is_valid()).count()
}


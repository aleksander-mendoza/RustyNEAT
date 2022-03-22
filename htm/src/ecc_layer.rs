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
use crate::{Shape, resolve_range, EncoderTarget, Synapse, top_large_k_indices, top_small_k_indices, Shape3, from_xyz, Shape2, from_xy, range_contains, w_idx, ConvShape, D, Tensor, HasShape, ConvTensorTrait, HasConvShape, ConvShapeTrait, Idx, NaiveCmp, HasConvShapeMut, EccConfig, HasEccConfig, WNorm, Activity, HasEccConfigMut};
use std::collections::{Bound, HashSet};
use crate::vector_field::{VectorFieldOne, VectorFieldDiv, VectorFieldAdd, VectorFieldMul, ArrayCast, VectorFieldSub, VectorFieldPartialOrd};
use crate::population::Population;
use rand::{Rng, SeedableRng};
use crate::xorshift::{auto_gen_seed64, xorshift64, auto_gen_seed, xorshift, xorshift32, auto_gen_seed32};
use itertools::{Itertools, assert_equal};
use std::iter::Sum;
use ocl::core::DeviceInfo::MaxConstantArgs;
use crate::sdr::SDR;
use rand::prelude::SliceRandom;
use failure::Fail;
use rayon::prelude::*;
use crate::as_usize::AsUsize;
use std::marker::PhantomData;
use num_traits::{AsPrimitive, Zero, Num, NumAssign};
use crate::tensor_trait::TensorTrait;
use crate::conv_tensor::ConvTensor;
use crate::vector_field_norm::{L, Sqrt};

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct EccLayer {
    k: Idx,
    cfg: EccConfig<D>,
    pub plasticity: D,
    pub threshold: D,
    pub activity: Tensor<D>,
    pub sums: Tensor<D>,
    pub weights: ConvTensor<D>,
    pub min_activity: Tensor<D>,
}

pub trait Weight: NumAssign + Sqrt + Copy + Sum + Debug + NaiveCmp + AsUsize {
}
impl AsUsize for f32{
    fn as_usize(self) -> usize {
        self as usize
    }

    fn from_usize(u: usize) -> Self {
        u as f32
    }
}
impl Weight for f32 {

}

pub trait EccLayerTrait<D: Weight>: HasEccConfig<D> + HasConvShape {
    type CT: ConvTensorTrait<D>;
    type T: TensorTrait<D>;
    fn repeat_column(&self, column_grid: [Idx; 2], column_pos: [Idx; 2]) -> Self;
    fn unpack_mut(&mut self) -> (&mut Self::CT, &mut Self::T, &mut Self::T, &mut Self::T);
    fn k(&self) -> Idx;
    fn len(&self) -> usize {
        self.weights().len()
    }
    fn set_k(&mut self, k: Idx);
    fn weights(&self) -> &Self::CT;
    fn weights_mut(&mut self) -> &mut Self::CT;
    fn sums(&self) -> &Self::T;
    fn sums_mut(&mut self) -> &mut Self::T;
    fn activity(&self) -> &Self::T;
    fn activity_mut(&mut self) -> &mut Self::T;
    fn min_activity(&self) -> &Self::T;
    fn min_activity_mut(&mut self) -> &mut Self::T;
    fn get_threshold(&self) -> D;
    fn set_threshold(&mut self, threshold: D);
    fn set_plasticity(&mut self, plasticity: D);
    fn get_plasticity(&self) -> D;
    fn activity_column_max(&self,column_idx:usize) -> D {
        self.activity().column_max(column_idx)
    }
    fn activity_column_min(&self,column_idx:usize) -> D {
        self.activity().column_min(column_idx)
    }
    fn fill_activity(&mut self,value:D) {
        unimplemented!();
        self.activity_mut().fill(value)
    }
    fn decrement_activities(&mut self,output:&CpuSDR,epsilon:D){
        unimplemented!();
        self.activity_mut().sparse_sub_assign_scalar(output,epsilon)
    }
    fn fill_sums(&mut self,fill_value:D) {
        self.sums_mut().fill(fill_value)
    }
    fn infer_new_sdr(&mut self, input: &CpuSDR, learn: bool) -> CpuSDR {
        let mut output = CpuSDR::new();
        self.infer_push(input, &mut output, learn);
        output
    }
    fn infer(&mut self, input: &CpuSDR, output: &mut CpuSDR, learn: bool) {
        output.clear();
        self.infer_push(input, output, learn)
    }
    fn infer_push(&mut self, input: &CpuSDR, output: &mut CpuSDR, learn: bool) {
        let k = self.k();
        let t = self.get_threshold();
        let activity = self.cfg_activity();
        let (w, s, a,min_a) = self.unpack_mut();
        s.fill(D::zero());
        match activity {
            Activity::Additive => {
                w.sparse_dot_add_assign(input, s);
                s.top1_per_region_thresholded_additive(a, k, t, output)
            },
            Activity::Multiplicative => {
                w.sparse_dot_add_assign(input, s);
                s.top1_per_region_thresholded_multiplicative_with_cached_min(a, min_a,k, t, output)
            },
            Activity::Thresholded => {
                w.sparse_dot_gt_diff_add_assign_scalar(a,min_a,input, s,D::one());
                s.top1_per_region(k,output)
            }
        }
        if learn {
            self.learn(input, output)
        }
    }
    fn learn(&mut self, input: &CpuSDR, output: &CpuSDR) {
        let em = self.cfg_entropy_maximisation();
        let norm = self.cfg_w_norm();
        let epsilon = self.get_plasticity();
        let biased = self.cfg_biased();
        let (w, s, a,min_a) = self.unpack_mut();
        if biased {
            w.sparse_biased_increment(epsilon, input, output);
        } else {
            w.sparse_unbiased_increment(epsilon, input, output);
        }
        match norm {
            WNorm::None => { w.sparse_kernel_column_mul_assign(output, D::one() - epsilon) }
            WNorm::L1 => { w.sparse_norm_assign::<L<1>>(output) }
            WNorm::L2 => { w.sparse_norm_assign::<L<2>>(output) }
        }
        if em>D::zero() {
            if min_a.is_empty() {
                a.sparse_sub_assign_scalar(output, epsilon * em);
            }else{
                a.sparse_sub_assign_scalar_and_update_colum_min(min_a,output, epsilon * em);
            }
        }
    }
    fn get_region_size(&self) -> Idx {
        crate::k_reg::get_region_size(self.k(), self.shape())
    }
    fn normalise(&mut self) {
        let norm = self.cfg_w_norm();
        let w = self.weights_mut();
        match norm {
            WNorm::None => {}
            WNorm::L1 => { w.norm_assign::<L<1>>() }
            WNorm::L2 => { w.norm_assign::<L<2>>() }
        }
    }
    fn sparse_normalise(&mut self, sdr: &CpuSDR) {
        let norm = self.cfg_w_norm();
        let w = self.weights_mut();
        match norm {
            WNorm::None => {}
            WNorm::L1 => { w.sparse_norm_assign::<L<1>>(sdr) }
            WNorm::L2 => { w.sparse_norm_assign::<L<2>>(sdr) }
        }
    }
    fn kernel_column_normalise(&mut self, column_idx: Idx) {
        let norm = self.cfg_w_norm();
        let w = self.weights_mut();
        match norm {
            WNorm::None => {}
            WNorm::L1 => { w.kernel_column_norm_assign::<L<1>>(column_idx) }
            WNorm::L2 => { w.kernel_column_norm_assign::<L<2>>(column_idx) }
        }
    }
}

impl HasEccConfig<D> for EccLayer {
    fn cfg(&self) -> &EccConfig<D> {
        &self.cfg
    }
}

impl HasEccConfigMut<D> for EccLayer {
    fn cfg_mut(&mut self) -> &mut EccConfig<D> {
        &mut self.cfg
    }
}

impl EccLayerTrait<D> for EccLayer {
    type CT = ConvTensor<D>;
    type T = Tensor<D>;

    fn repeat_column(&self, column_grid: [Idx; 2], column_pos: [Idx; 2]) -> Self {
        let new_shape = column_grid.add_channels(self.shape().channels());
        Self {
            k: self.k,
            cfg: self.cfg.clone(),
            plasticity: self.plasticity,
            min_activity: if self.min_activity.is_empty(){
                Tensor::null()
            }else{
                Tensor::new(new_shape.clone(),self.min_activity.get_at2d(column_pos))
            },
            threshold: self.threshold,
            activity: self.activity.repeat_column(column_grid, column_pos),
            sums: Self::T::new(new_shape, D::zero()),
            weights: self.weights.repeat_column(column_grid, column_pos),

        }
    }

    fn unpack_mut(&mut self) -> (&mut Self::CT, &mut Self::T, &mut Self::T,&mut Self::T) {
        let Self { activity, sums, weights,min_activity, .. } = self;
        (weights, sums, activity,min_activity)
    }

    fn k(&self) -> Idx {
        self.k
    }

    fn set_k(&mut self, k: Idx) {
        crate::k_reg::assert_valid_for(k, self.shape());
        self.k = k;
    }

    fn weights(&self) -> &Self::CT {
        &self.weights
    }

    fn weights_mut(&mut self) -> &mut Self::CT {
        &mut self.weights
    }

    fn sums(&self) -> &Self::T {
        &self.sums
    }

    fn sums_mut(&mut self) -> &mut Self::T {
        &mut self.sums
    }

    fn activity(&self) -> &Self::T {
        &self.activity
    }

    fn activity_mut(&mut self) -> &mut Self::T {
        &mut self.activity
    }

    fn min_activity(&self) -> &Self::T {
        &self.min_activity
    }

    fn min_activity_mut(&mut self) -> &mut Self::T {
        &mut self.min_activity
    }

    fn get_threshold(&self) -> D {
        self.threshold
    }

    fn set_threshold(&mut self, threshold: D) {
        self.threshold = threshold
    }

    fn set_plasticity(&mut self, plasticity: D) {
        self.plasticity = plasticity
    }

    fn get_plasticity(&self) -> D {
        self.plasticity
    }
}

impl HasShape for EccLayer {
    fn shape(&self) -> &[Idx; 3] {
        self.weights.shape()
    }
}

impl HasConvShape for EccLayer {
    fn cshape(&self) -> &ConvShape {
        self.weights.cshape()
    }
}

impl HasConvShapeMut for EccLayer {
    fn cshape_mut(&mut self) -> &mut ConvShape {
        self.weights.cshape_mut()
    }
}

impl EccLayer {
    pub fn new(cfg: EccConfig<D>, shape: ConvShape, k: Idx, rng: &mut impl Rng) -> Self {
        crate::k_reg::assert_valid_for(k, shape.out_shape());
        let initial_activity = 1.;
        let mut slf = Self {
            min_activity: if cfg.activity.cache_min_column_activity(){
                Tensor::new(shape.out_grid().add_channels(1),initial_activity)
            }else{
                Tensor::null()
            },
            k,
            cfg,
            plasticity: 0.01,
            threshold: 0.,
            activity: Tensor::new(shape.output_shape(), initial_activity),
            sums: Tensor::new(shape.output_shape(), 0.),
            weights: ConvTensor::rand(shape, rng),
        };
        slf.normalise();
        slf
    }
    // pub fn concat<'a, T>(layers: &'a [T], f: impl Fn(&'a T) -> &'a Self) -> Self {
    //     assert_ne!(layers.len(), 0, "No layers provided!");
    //     let first_layer = f(&layers[0]);
    //     let mut grid = first_layer.shape().grid();
    //     assert!(layers.iter().all(|a| f(a).shape().grid().all_eq(grid)), "All concatenated layers must have the same width and height!");
    //     let k = if layers.iter().any(|a| f(a).k() > 1) {
    //         assert!(layers.iter().map(|a| f(a).get_region_size()).all_equal(), "All layers are region_size but their region sizes are different");
    //         layers.iter().map(|a| f(a).get_region_size()).sum()
    //     } else {
    //         1
    //     };
    //     let concatenated_sum: Idx = layers.iter().map(|a| f(a).shape().channels()).sum();
    //     let shape = grid.add_channels(concatenated_sum);
    //     let new_v = shape.product();
    //     let mut activity = vec![D::INITIAL_ACTIVITY; new_v.as_usize()];
    //     let mut channel_offset = 0;
    //     for l in 0..layers.len() {
    //         let l = f(&layers[l]);
    //         let v = l.shape().product();
    //         for w in 0..l.shape().width() {
    //             for h in 0..l.shape().height() {
    //                 for c in 0..l.shape().channels() {
    //                     let original_idx = l.shape().idx(from_xyz(w, h, c));
    //                     let idx = shape.idx(from_xyz(w, h, channel_offset + c));
    //                     activity[idx.as_usize()] = l.activity[original_idx.as_usize()];
    //                 }
    //             }
    //         }
    //         channel_offset += l.shape().channels();
    //     }
    //     Self {
    //         k,
    //         threshold: first_layer.threshold,
    //         activity,
    //         sums: vec![D::ZERO; new_v.as_usize()],
    //         shape,
    //         _d: Default::default(),
    //     }
    // }
}
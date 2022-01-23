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
use crate::{Shape, resolve_range, EncoderTarget, Synapse, top_large_k_indices, top_small_k_indices, Shape3, from_xyz, Shape2, from_xy, range_contains, DenseWeight, w_idx, kernel_column_weight_sum, debug_assert_approx_eq_weight, EccLayerD, kernel_column_dropped_weights_count, ConvShape};
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
use failure::Fail;


#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct CpuEccPopulation<D: DenseWeight> {
    shape: [Idx; 3],
    k: Idx,
    pub threshold: D,
    pub activity: Vec<D>,
    pub sums: Vec<D>,
}

impl <D:DenseWeight> Deref for CpuEccPopulation<D>{
    type Target = [Idx;3];

    fn deref(&self) -> &Self::Target {
        &self.shape
    }
}


impl<D: DenseWeight> CpuEccPopulation<D> {
    pub fn shape(&self)->&[Idx;3]{
        &self.shape
    }
    pub fn get_region_size(&self) -> Idx {
        self.channels() / self.k()
    }

    pub fn from_repeated_column(column_grid: [Idx; 2], pretrained: &Self, pretrained_column_pos: [Idx; 2]) -> Self {
        let shape = column_grid.add_channels(pretrained.shape().channels());
        let kv = pretrained.shape().product();
        let new_v = shape.product();
        let mut activity = vec![D::ZERO; as_usize(kv * new_v)];
        for channel in 0..shape.channels() {
            let pretrained_idx = pretrained.shape().idx(pretrained_column_pos.add_channels(channel));
            for x in 0..shape.width() {
                for y in 0..shape.height() {
                    let pos = from_xyz(x, y, channel);
                    let idx = shape.idx(pos);
                    activity[as_usize(idx)] = pretrained.activity[as_usize(pretrained_idx)];
                }
            }
        }
        Self {
            shape,
            k: pretrained.k,
            threshold: pretrained.threshold,
            activity,
            sums: vec![D::ZERO;as_usize(new_v)]
        }
    }
    pub fn new(shape: [Idx; 3], k: Idx) -> Self {
        let v = shape.product();
        let region_size = shape.channels()/k;
        Self {
            k,
            shape,
            threshold: D::default_threshold(region_size),
            activity: vec![D::INITIAL_ACTIVITY; as_usize(v)],
            sums: vec![D::ZERO; as_usize(v)],
        }
    }
    pub fn concat<T>(layers: &[T], f: impl Fn(&T) -> &Self) -> Self {
        assert_ne!(layers.len(), 0, "No layers provided!");
        let first_layer = f(&layers[0]);
        let mut grid = first_layer.shape.grid();
        assert!(layers.iter().all(|a| f(a).grid().all_eq(grid)), "All concatenated layers must have the same width and height!");
        let k = if layers.iter().any(|a| f(a).k()>1){
            assert!(layers.iter().map(|a| f(a).get_region_size()).all_equal(), "All layers are region_size but their region sizes are different");
            layers.iter().map(|a| f(a).get_region_size()).sum()
        }else{
            1
        };
        let concatenated_sum: Idx = layers.iter().map(|a| f(a).channels()).sum();
        let shape = grid.add_channels(concatenated_sum);
        let new_v = shape.product();
        let mut activity = vec![D::INITIAL_ACTIVITY; as_usize(new_v)];
        let mut channel_offset = 0;
        for l in 0..layers.len() {
            let l = f(&layers[l]);
            let v = l.shape().product();
            for w in 0..l.shape().width() {
                for h in 0..l.shape().height() {
                    for c in 0..l.shape().channels() {
                        let original_idx = l.shape().idx(from_xyz(w, h, c));
                        let idx = shape.idx(from_xyz(w, h, channel_offset + c));
                        activity[as_usize(idx)] = l.activity[as_usize(original_idx)];
                    }
                }
            }
            channel_offset += l.channels();
        }
        Self {
            shape,
            k,
            threshold: first_layer.threshold,

            activity,
            sums: vec![D::ZERO; as_usize(new_v)],
        }
    }
    pub fn get_threshold(&self) -> D {
        self.threshold
    }
    pub fn set_threshold(&mut self, threshold: D) {
        self.threshold = threshold
    }

    pub fn k(&self) -> Idx { self.k }

    pub fn set_k(&mut self, k: Idx) {
        assert!(k <= self.channels(), "k is larger than layer output!");
        assert_eq!(self.channels()%k,0, "k=={} does not divide out_channels=={}! Disable top1_per_region first!",k,self.channels());
        self.k = k;
    }
    pub fn reset_sums(&mut self){
        self.sums.fill(D::ZERO);
    }
    pub fn get_threshold_f32(&self) -> f32 {
        D::w_to_f32(self.threshold)
    }
    pub fn set_threshold_f32(&mut self, fractional: f32) {
        self.threshold = D::f32_to_w(fractional)
    }

    pub fn decrement_activities(&mut self, output:&CpuSDR) {
        for &winner in output.iter() {
            self.activity[as_usize(winner)] -= D::ACTIVITY_PENALTY;
        }
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
    pub fn set_initial_activity(&mut self) {
        self.activity.fill(D::INITIAL_ACTIVITY)
    }
    pub fn activity_f32(&self, output_idx: usize) -> f32 {
        D::w_to_f32(self.activity(output_idx))
    }
    pub fn determine_winners_topk(&self, output: &mut CpuSDR) {
        let t = self.threshold;
        let a = as_usize(self.shape().grid().product());
        let c = as_usize(self.shape().channels());
        let k = as_usize(self.k());
        output.clear();
        for column_idx in 0..a {
            let r = c * column_idx;
            for (i, v) in top_small_k_indices(k, c, |i|{
                debug_assert!(i < c);
                let i = i + r;
                debug_assert!(self.sums[i].le(D::TOTAL_SUM), "{}<={}", self.sums[i], D::TOTAL_SUM);
                debug_assert!(self.activity[i].le(D::INITIAL_ACTIVITY), "{}<={}", self.activity[i], D::INITIAL_ACTIVITY);
                self.sums[i] + self.activity[i]
            }, D::gt) {
                let e = r + i;
                debug_assert!(r <= e);
                debug_assert!(e < r + c);
                if self.sums[e].ge(t){
                    let e = as_idx(e);
                    debug_assert!(!output.as_slice().contains(&e), "{:?}<-{}={}+{}", output, e, r, i);
                    output.push(e);
                }
            }
        }

    }
    pub fn determine_winners_top1_per_region(&self, output: &mut CpuSDR) {
        let t = self.threshold;
        let a = as_usize(self.shape().grid().product());
        let k = as_usize(self.k());
        assert_eq!(self.shape().channels() % self.k(), 0);
        let region_size = as_usize(self.shape().channels() / self.k());
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

impl CpuEccPopulation<f32> {
    pub fn reset_activity(&mut self) {
        let min = self.min_activity();
        self.activity.iter_mut().for_each(|a| *a -= min)
    }
}

impl CpuEccPopulation<u32> {
    pub fn reset_activity(&mut self) {
        let free_space = u32::INITIAL_ACTIVITY - self.max_activity();
        self.activity.iter_mut().for_each(|a| *a += free_space)
    }
}


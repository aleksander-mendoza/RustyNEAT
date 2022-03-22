use crate::{ConvShape, Idx, as_idx, as_usize, Shape, VectorFieldOne, Shape2, Shape3, from_xyz, VectorFieldPartialOrd, CpuSDR, from_xy, SDR, range_contains, ConvShapeTrait, HasConvShape, HasConvShapeMut, range_foreach2d, D, HasShape, Weight};
use std::ops::{Deref, DerefMut, Index, IndexMut, Range, DivAssign, AddAssign, SubAssign, MulAssign, Div, Sub};
use rand::Rng;
use rand::prelude::SliceRandom;
use ndalgebra::mat::AsShape;
use itertools::Itertools;
use std::collections::HashSet;
use serde::{Serialize, Deserialize};
use rayon::prelude::*;
use std::thread::JoinHandle;
use crate::parallel::{parallel_map_vector, parallel_map_collect};
use crate::as_usize::AsUsize;
use std::marker::PhantomData;
use crate::vector_field_norm::{LNorm};
use std::iter::Sum;
use crate::tensor_trait::TensorTrait;
use num_traits::One;
use std::ptr::NonNull;


#[inline]
pub fn w_idx(output_idx: Idx, idx_within_kernel_column: Idx, output_volume: Idx) -> Idx {
    debug_assert!(output_idx < output_volume);
    output_idx + idx_within_kernel_column * output_volume
}

pub struct KernelColIndexIter {
    idx_within_kernel_column_times_output_volume: Idx,
    output_idx: Idx,
    /**output volume*/
    v: Idx,
}

impl Iterator for KernelColIndexIter {
    type Item = Idx;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx_within_kernel_column_times_output_volume >= self.v {
            self.idx_within_kernel_column_times_output_volume -= self.v;
            Some(self.output_idx + self.idx_within_kernel_column_times_output_volume)
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let i = self.idx_within_kernel_column_times_output_volume/self.v;
        let i = i.as_usize();
        (i,Some(i))
    }
    fn last(self)->Option<Self::Item>{
        if self.idx_within_kernel_column_times_output_volume >= self.v {
            Some(self.output_idx)
        }else{
            None
        }
    }
    // fn advance_by(&mut self, n: usize) -> Result<(), usize>{
    //     let a = self.v * as_idx(n);
    //     if self.idx_within_kernel_column_times_output_volume >= a {
    //         self.idx_within_kernel_column_times_output_volume -= a;
    //         Ok(())
    //     } else{
    //         Err((self.idx_within_kernel_column_times_output_volume/self.v).as_usize())
    //     }
    // }
    fn nth(&mut self, n: usize) -> Option<Self::Item>{
        let a = self.v * as_idx(n+1);
        if self.idx_within_kernel_column_times_output_volume >= a {
            self.idx_within_kernel_column_times_output_volume -= a;
            Some(self.output_idx + self.idx_within_kernel_column_times_output_volume)
        } else{
            self.idx_within_kernel_column_times_output_volume = 0;
            None
        }
    }
}
impl ExactSizeIterator for KernelColIndexIter{
}

// pub fn kernel_column_iter<'a,D:Copy,C:ConvTensor<D>>(c:&'a C, output_idx: Idx) -> impl Iterator<Item=D>+'a {
//     c.kernel_column_index_iter(output_idx).map(|i|c.as_slice()[i.as_usize()])
// }
// pub fn kernel_column_iter_mut<'a,D:Copy,C:ConvTensor<D>>(c:&'a mut C, output_idx: Idx) -> impl Iterator<Item=&'a mut D>+'a{
//     c.kernel_column_index_iter(output_idx).map(|i|&mut c.as_slice_mut()[i.as_usize()])
// }
pub trait ConvTensorTrait<D:Copy>: HasConvShape {
    fn as_slice(&self) -> &[D];
    fn as_slice_mut(&mut self) -> &mut [D]{
        self.unpack_mut().0
    }
    fn unpack_mut(&mut self) -> (&mut [D], &mut ConvShape);
    fn repeat_column(&self, column_grid: [Idx; 2], column_pos: [Idx; 2]) -> Self;
    fn len(&self) -> usize {
        self.as_slice().len()
    }
    fn kernel_column_iter(&self, output_idx: Idx) -> KernelColIndexIter {
        let kv = self.kernel_column_volume();
        let v = self.out_volume();
        KernelColIndexIter {
            idx_within_kernel_column_times_output_volume: kv * v,
            output_idx,
            v,
        }
    }
    fn geti(&self,i:Idx)->D{
        self.as_slice()[i.as_usize()]
    }
    fn geti_mut(&mut self,i:Idx)->&mut D{
        &mut self.as_slice_mut()[i.as_usize()]
    }
    fn kernel_column_copy(&self, output_idx: Idx) -> Vec<D> {
        let mut l = self.kernel_column_volume().as_usize();
        let mut v = Vec::with_capacity(l);
        unsafe{v.set_len(l)}
        for i in self.kernel_column_iter(output_idx){
            l-=1;
            v[l] = self.geti(i)
        }
        v
    }
    fn kernel_column_sum(&self, output_idx: Idx) -> D  where D:Sum{
        self.kernel_column_iter(output_idx).map(|i|self.geti(i)).sum()
    }
    fn kernel_column_pow_sum<N: LNorm<D>>(&self, output_idx: Idx) -> D where D:Sum {
        self.kernel_column_iter(output_idx).map(|i| N::pow(self.geti(i))).sum()
    }
    fn kernel_column_norm<N: LNorm<D>>(&self, output_idx: Idx) -> D where D:Sum {
        N::root(self.kernel_column_pow_sum::<N>(output_idx))
    }
    fn kernel_column_div_scalar(&mut self, output_idx: Idx, scalar: D) where D:DivAssign{
        self.kernel_column_iter(output_idx).for_each(|i| *self.geti_mut(i) /= scalar)
    }
    fn kernel_column_mul_scalar(&mut self, output_idx: Idx, scalar: D) where D:MulAssign{
        self.kernel_column_iter(output_idx).for_each(|i| *self.geti_mut(i) *= scalar)
    }
    fn kernel_column_div_assign(&mut self, rhs: &impl TensorTrait<D>) where D:DivAssign{
        assert_eq!(rhs.shape(), self.shape(), "Shapes don't match!");
        for output_idx in 0..self.out_volume() {
            self.kernel_column_div_scalar(output_idx,rhs.get(output_idx.as_usize()))
        }
    }
    fn sparse_kernel_column_div_assign(&mut self, sparse_mask: &CpuSDR, scalar: D) where D:DivAssign{
        for &output_idx in sparse_mask.as_slice() {
            self.kernel_column_div_scalar(output_idx,scalar)
        }
    }
    fn sparse_kernel_column_mul_assign(&mut self, sparse_mask: &CpuSDR, scalar: D) where D:MulAssign{
        for &output_idx in sparse_mask.as_slice() {
            self.kernel_column_mul_scalar(output_idx,scalar)
        }
    }
    fn sparse_kernel_column_div_scalar_assign(&mut self, sparse_mask: &CpuSDR, rhs: &impl TensorTrait<D>) where D:DivAssign{
        assert_eq!(rhs.shape(), self.shape(), "Shapes don't match!");
        for &output_idx in sparse_mask.as_slice() {
            self.kernel_column_div_scalar(output_idx,rhs.geti(output_idx))
        }
    }
    fn w(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> D {
        self.geti(self.w_index(input_pos, output_pos))
    }
    fn w_mut(&mut self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> &mut D {
        self.geti_mut(self.w_index(input_pos, output_pos))
    }
    fn sparse_dot_add_assign(&self, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>) where D:AddAssign{
        self.sparse_dot(lhs,|idx,w|*destination.geti_mut(idx)+=w)
    }
    fn sparse_dot_gt_scalar_add_assign(&self, threshold:D, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>) where D:AddAssign+PartialOrd{
        self.sparse_dot(lhs,|idx,w|if w>threshold{*destination.geti_mut(idx)+=w})
    }
    fn sparse_dot_gt_scalar_sub_assign(&self, threshold:D, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>) where D:SubAssign+PartialOrd{
        self.sparse_dot(lhs,|idx,w|if w>threshold{*destination.geti_mut(idx)-=w})
    }
    fn sparse_dot_ge_scalar_add_assign(&self, threshold:D, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>)where D:AddAssign+PartialOrd {
        self.sparse_dot(lhs,|idx,w|if w>=threshold{*destination.geti_mut(idx)+=w})
    }
    fn sparse_dot_ge_scalar_sub_assign(&self, threshold:D, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>) where D:SubAssign+PartialOrd{
        self.sparse_dot(lhs,|idx,w|if w>=threshold{*destination.geti_mut(idx)-=w})
    }
    fn sparse_dot_gt_add_assign(&self, thresholds:&impl TensorTrait<D>, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>) where D:AddAssign+PartialOrd {
        assert_eq!(self.out_shape(),thresholds.shape(),"Shapes don't match!");
        self.sparse_dot(lhs,|idx,w|if w>thresholds.geti(idx){*destination.geti_mut(idx)+=w})
    }
    fn sparse_dot_gt_sub_assign(&self, thresholds:&impl TensorTrait<D>, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>) where D:SubAssign+PartialOrd{
        assert_eq!(self.out_shape(),thresholds.shape(),"Shapes don't match!");
        self.sparse_dot(lhs,|idx,w|if w>thresholds.geti(idx){*destination.geti_mut(idx)-=w})
    }
    fn sparse_dot_ge_add_assign(&self, thresholds:&impl TensorTrait<D>, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>)where D:AddAssign+PartialOrd  {
        assert_eq!(self.out_shape(),thresholds.shape(),"Shapes don't match!");
        self.sparse_dot(lhs,|idx,w|if w>=thresholds.geti(idx){*destination.geti_mut(idx)+=w})
    }
    fn sparse_dot_ge_sub_assign(&self, thresholds:&impl TensorTrait<D>, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>) where D:SubAssign+PartialOrd{
        assert_eq!(self.out_shape(),thresholds.shape(),"Shapes don't match!");
        self.sparse_dot(lhs,|idx,w|if w>=thresholds.geti(idx){*destination.geti_mut(idx)-=w})
    }
    fn sparse_dot_gt_add_assign_scalar(&self, thresholds:&impl TensorTrait<D>, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>, scalar:D) where D:AddAssign+PartialOrd  {
        assert_eq!(self.out_shape(),thresholds.shape(),"Shapes don't match!");
        self.sparse_dot(lhs,|idx,w|if w>thresholds.geti(idx){*destination.geti_mut(idx)+=scalar})
    }
    fn sparse_dot_lt_add_assign_scalar(&self, thresholds:&impl TensorTrait<D>, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>, scalar:D) where D:AddAssign+PartialOrd  {
        assert_eq!(self.out_shape(),thresholds.shape(),"Shapes don't match!");
        self.sparse_dot(lhs,|idx,w|if w<thresholds.geti(idx){*destination.geti_mut(idx)+=scalar})
    }
    fn sparse_dot_gt_diff_add_assign_scalar(&self, thresholds:&impl TensorTrait<D>,min_thresholds:&impl TensorTrait<D>, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>, scalar:D) where D:AddAssign+PartialOrd +Sub<Output=D> {
        assert_eq!(self.out_shape(),thresholds.shape(),"Shapes don't match!");
        assert_eq!(self.out_grid(),min_thresholds.shape().grid(),"Shapes don't match!");
        assert_eq!(1,min_thresholds.shape().channels(),"min thresholds should be a matrix, not 3D tensor");
        let c = self.out_channels();
        self.sparse_dot(lhs,|idx,w|if w>thresholds.geti(idx)-min_thresholds.geti(idx/c){*destination.geti_mut(idx)+=scalar})
    }
    fn sparse_dot_gt_count(&self, thresholds:&impl TensorTrait<D>, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>) where D:AddAssign+PartialOrd+One  {
        self.sparse_dot_gt_add_assign_scalar(thresholds,lhs,destination,D::one())
    }
    fn sparse_dot_sub_assign(&self, lhs: &CpuSDR, destination: &mut impl TensorTrait<D>) where D:SubAssign  {
        self.sparse_dot(lhs,|idx,w|*destination.geti_mut(idx)-=w)
    }
    fn sparse_dot(&self, lhs: &CpuSDR, mut target: impl FnMut(Idx, D)) {
        let kernel_column = self.kernel_column();
        let v = self.out_volume();
        let mut used_w = HashSet::new();
        for &input_idx in lhs.as_slice() {
            let input_pos: [Idx; 3] = self.in_shape().pos(input_idx);
            let r = self.out_range(input_pos.grid());
            range_foreach2d(&r, |output_pos| {
                let kernel_offset = output_pos.conv_in_range_begin(self.stride());
                for p2 in 0..self.out_channels() {
                    let output_pos = output_pos.add_channels(p2);
                    let output_idx = self.out_shape().idx(output_pos);
                    let w_index = ConvShape::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                    debug_assert_eq!(w_index, self.w_index(&input_pos, &output_pos));
                    debug_assert!(used_w.insert(w_index), "{}", w_index);
                    let w = self.geti(w_index);
                    target(output_idx, w);
                }
            });
        }
    }
    fn norm_assign<N:LNorm<D>>(&mut self) where D:DivAssign+Sum{
        for output_idx in 0..self.out_volume() {
            self.kernel_column_norm_assign::<N>(output_idx)
        }
    }
    /**First computes the exact sum of weights in linear O(n) time. Then uses this sum to normalise the weights*/
    fn kernel_column_norm_assign<N:LNorm<D>>(&mut self, output_idx: Idx) where D:DivAssign+Sum{
        let norm = self.kernel_column_norm::<N>(output_idx);
        self.kernel_column_div_scalar(output_idx, norm);
    }
    fn sparse_norm_assign<N:LNorm<D>>(&mut self, sparse: &CpuSDR)where D:DivAssign+Sum {
        for &output_idx in sparse.as_slice() {
            self.kernel_column_norm_assign::<N>(output_idx)
        }
    }
    fn copy_repeated_column(&mut self, pretrained: &Self, pretrained_column_pos: [Idx; 2]) {
        assert_eq!(pretrained.kernel(), self.kernel(), "Kernels are different");
        assert_eq!(pretrained.in_channels(), self.in_channels(), "Input channels are different");
        assert_eq!(pretrained.out_channels(), self.out_channels(), "Output channels are different");
        let kv = pretrained.kernel_column_volume();
        let new_v = self.out_volume();
        let old_v = pretrained.out_volume();
        for channel in 0..self.out_channels() {
            let pretrained_out_idx = pretrained.out_shape().idx(pretrained_column_pos.add_channels(channel));
            for idx_within_kernel_column in 0..kv {
                let old_w_i = w_idx(pretrained_out_idx, idx_within_kernel_column, old_v);
                let old_w: D = pretrained.as_slice()[old_w_i.as_usize()];
                for x in 0..self.out_width() {
                    for y in 0..self.out_height() {
                        let pos = from_xyz(x, y, channel);
                        let output_idx = self.out_shape().idx(pos);
                        let new_w_i = w_idx(output_idx, idx_within_kernel_column, new_v);
                        self.as_slice_mut()[new_w_i.as_usize()] = old_w;
                    }
                }
            }
        }
    }
    fn sparse_biased_increment(&mut self,epsilon:D, input: &CpuSDR, output: &CpuSDR)where D:AddAssign{
        let (w_slice,shape) = self.unpack_mut();
        shape.sparse_biased_increment(w_slice,epsilon,input,output)
    }
    fn sparse_unbiased_increment(&mut self,epsilon:D, input: &CpuSDR, output: &CpuSDR) where D:AddAssign+AsUsize+Div<Output=D>{
        let (w_slice,shape) = self.unpack_mut();
        shape.sparse_unbiased_increment(w_slice,epsilon,input,output)
    }
}




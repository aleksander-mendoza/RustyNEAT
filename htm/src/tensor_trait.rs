use crate::{HasShape, CpuSDR, AsUsize, ConvTensorTrait, Shape3, top_small_k_indices, EncoderTarget, Shape, Shape2, from_xyz, NaiveCmp, Idx, as_idx, Weight, TopK};
use std::iter::Sum;
use std::slice::{Iter, IterMut};
use std::fmt::Debug;
use std::ops::{Range, Mul, Add, SubAssign, MulAssign, DivAssign, AddAssign, Sub};
use num_traits::One;
use std::process::Output;

pub trait TensorTrait<D:Copy>: HasShape {
    fn as_slice(&self) -> &[D];
    fn repeat_column(&self, column_grid: [Idx; 2], column_pos: [Idx; 2]) -> Self;
    fn len(&self) -> usize {
        self.as_slice().len()
    }
    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }
    fn get_at2d(&self,pos:[Idx;2])->D{
        self.geti(self.shape().grid().idx(pos))
    }
    fn get_at(&self,pos:[Idx;3])->D{
        self.geti(self.shape().idx(pos))
    }
    fn iter(&self) -> Iter<D> {
        self.as_slice().iter()
    }
    fn iter_mut(&mut self) -> IterMut<D> {
        self.as_mut_slice().iter_mut()
    }
    fn as_mut_slice(&mut self) -> &mut [D];
    fn get(&self, idx: usize) -> D {
        self.as_slice()[idx]
    }
    fn get_mut(&mut self, idx: usize) -> &mut D {
        &mut self.as_mut_slice()[idx]
    }
    fn geti(&self, idx: Idx) -> D {
        self.get(idx.as_usize())
    }
    fn geti_mut(&mut self, idx: Idx) -> &mut D {
        self.get_mut(idx.as_usize())
    }
    fn fill(&mut self, value: D) {
        self.as_mut_slice().fill(value)
    }
    fn sparse_add_assign_scalar(&mut self, sparse_mask:&CpuSDR,scalar:D)where D:AddAssign{
        for &idx in sparse_mask.iter(){
            self.as_mut_slice()[idx.as_usize()] += scalar
        }
    }
    fn sparse_sub_assign_scalar(&mut self, sparse_mask:&CpuSDR,scalar:D)where D:SubAssign{
        for &idx in sparse_mask.iter(){
            self.as_mut_slice()[idx.as_usize()] -= scalar
        }
    }
    fn sparse_mul_assign_scalar(&mut self, sparse_mask:&CpuSDR,scalar:D)where D:MulAssign{
        for &idx in sparse_mask.iter(){
            self.as_mut_slice()[idx.as_usize()] *= scalar
        }
    }
    fn sparse_div_assign_scalar(&mut self, sparse_mask:&CpuSDR,scalar:D)where D:DivAssign{
        for &idx in sparse_mask.iter(){
            self.as_mut_slice()[idx.as_usize()] /= scalar
        }
    }
    fn sparse_sub_assign_scalar_and_update_colum_min(&mut self, column_min:&mut Self, sparse_mask:&CpuSDR,scalar:D)where D:SubAssign+PartialOrd{
        debug_assert_eq!(self.shape().grid(),column_min.shape().grid());
        debug_assert_eq!(1,column_min.shape().channels());
        let c = self.shape().channels();
        for &idx in sparse_mask.iter(){
            debug_assert_eq!(idx/c,self.shape().grid().idx(*self.shape().pos(idx).grid()));
            let val = &mut self.as_mut_slice()[idx.as_usize()];
            *val -= scalar;
            let col_min = column_min.geti_mut(idx/c);
            if *val < *col_min{
                *col_min = *val;
            }
        }
    }
    fn column_range(&self, column_idx: usize) -> Range<usize> {
        let c = self.shape().channels().as_usize();
        let offset = column_idx * c;
        offset..offset + c
    }
    fn column_slice(&self, column_idx: usize) -> &[D] {
        &self.as_slice()[self.column_range(column_idx)]
    }
    fn column_iter(&self, column_idx: usize) -> Iter<D> {
        self.column_slice(column_idx).iter()
    }
    fn column_slice_mut(&mut self, column_idx: usize) -> &mut [D] {
        let r = self.column_range(column_idx);
        &mut self.as_mut_slice()[r]
    }
    fn column_iter_mut(&mut self, column_idx: usize) -> IterMut<D> {
        self.column_slice_mut(column_idx).iter_mut()
    }
    fn column_max(&self, column_idx: usize) -> D where D:NaiveCmp{
        self.column_iter(column_idx).cloned().max_by(D::cmp_naive).unwrap()
    }
    fn column_min(&self, column_idx: usize) -> D where D:NaiveCmp{
        self.column_iter(column_idx).cloned().min_by(D::cmp_naive).unwrap()
    }
    /**sum(self[i] for i in output)*/
    fn sparse_dot(&self, output: &CpuSDR) -> D where D:Sum{
        output.iter().map(|i| self.get(i.as_usize())).sum()
    }
    fn kernel_column_sum_assign(&mut self, rhs: &impl ConvTensorTrait<D>) where D:Sum{
        self.iter_mut().enumerate().for_each(|(i, w)| *w = rhs.kernel_column_sum(as_idx(i)))
    }
    fn kernel_column_sum_add_assign(&mut self, rhs: &impl ConvTensorTrait<D>) where D:Sum+AddAssign{
        self.iter_mut().enumerate().for_each(|(i, w)| *w += rhs.kernel_column_sum(as_idx(i)))
    }
    fn kernel_column_sum_sub_assign(&mut self, rhs: &impl ConvTensorTrait<D>) where D:Sum+SubAssign{
        self.iter_mut().enumerate().for_each(|(i, w)| *w -= rhs.kernel_column_sum(as_idx(i)))
    }
    fn topk(&self, k: usize, output: &mut CpuSDR) where D:PartialOrd{
        self.shape().topk_per_column(k, |_, i| self.get(i), |_, _, i| output.push(as_idx(i)))
    }
    fn topk_thresholded(&self, k: usize, threshold: D, output: &mut CpuSDR) where D:PartialOrd{
        self.shape().topk_per_column(k, |_, i| self.get(i), |v, _, i| if v > threshold {
            output.push(as_idx(i))
        })
    }
    fn top1_per_region(&self, k: Idx, output: &mut CpuSDR) where D:PartialOrd{
        self.shape().top1_per_region(k, |i| self.get(i.as_usize()), |v, i| output.push(i))
    }
    fn top1_per_region_thresholded(&self, k: Idx, threshold: D, output: &mut CpuSDR) where D:PartialOrd{
        self.shape().top1_per_region(k, |i| self.get(i.as_usize()), |v, i| if v > threshold { output.push(i) })
    }
    fn top1_per_region_thresholded_additive(&self, additive_bias: &Self, k: Idx, threshold: D, output: &mut CpuSDR) where D:PartialOrd+Add<Output=D>{
        assert_eq!(self.shape(),additive_bias.shape(),"Shapes don't match");
        self.shape().top1_per_region(k, |i| {
            self.geti(i) + additive_bias.geti(i)
        }, |v, i| if self.geti(i) > threshold { output.push(i) })
    }
    fn top1_per_region_thresholded_multiplicative(&self, multiplicative_bias: &Self, k: Idx, threshold: D, output: &mut CpuSDR) where D:NaiveCmp+Sub<Output=D>+One+Add<Output=D>{
        assert_eq!(self.shape(),multiplicative_bias.shape(),"Shapes don't match");
        self.shape().top1_per_region_per_column(k, |c| multiplicative_bias.column_min(c.as_usize()), |&min, i| {
            self.geti(i) * (multiplicative_bias.geti(i) - min + D::one())
        }, |v, i| if self.geti(i) > threshold { output.push(i) })
    }
    fn top1_per_region_thresholded_multiplicative_with_cached_min(&self, multiplicative_bias: &Self, min_multiplicative_bias:&Self, k: Idx, threshold: D, output: &mut CpuSDR) where D:NaiveCmp+Sub<Output=D>+One+Add<Output=D>{
        assert_eq!(self.shape(),multiplicative_bias.shape(),"Shapes don't match");
        assert_eq!(self.shape().grid(),min_multiplicative_bias.shape().grid(),"Shapes don't match");
        assert_eq!(1,min_multiplicative_bias.shape().channels(),"Should be a matrix, not a 3D tensor");
        self.shape().top1_per_region_per_column(k, |c| {
            debug_assert!(min_multiplicative_bias.geti(c)==multiplicative_bias.column_min(c.as_usize()),"Cached min per column is inconsistent");
            min_multiplicative_bias.geti(c)
        }, |&min, i| {
            self.geti(i) * (multiplicative_bias.geti(i) - min + D::one())
        }, |v, i| if self.geti(i) > threshold { output.push(i) })
    }
    fn copy_repeated_column(&mut self, pretrained: &Self, pretrained_column_pos: [Idx; 2]) {
        assert_eq!(pretrained.shape().channels(), self.shape().channels(), "Output channels are different");
        for channel in 0..self.shape().channels() {
            let pretrained_idx = pretrained.shape().idx(pretrained_column_pos.add_channels(channel));
            for x in 0..self.shape().width() {
                for y in 0..self.shape().height() {
                    let pos = from_xyz(x, y, channel);
                    let idx = self.shape().idx(pos);
                    self.as_mut_slice()[idx.as_usize()] = pretrained.as_slice()[pretrained_idx.as_usize()];
                }
            }
        }
    }
}
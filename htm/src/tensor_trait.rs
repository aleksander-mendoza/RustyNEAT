use crate::{HasShape, CpuSDR, AsUsize, ConvTensorTrait, Shape3, top_small_k_indices, EncoderTarget, Shape, Shape2, from_xyz, NaiveCmp, Idx, as_idx, Weight, TopK, from_xy, Tensor, LNorm};
use std::iter::Sum;
use std::slice::{Iter, IterMut};
use std::fmt::Debug;
use std::ops::{Range, Mul, Add, SubAssign, MulAssign, DivAssign, AddAssign, Sub};
use num_traits::One;
use std::process::Output;
use rand::Rng;
use rand::distributions::{Standard, Distribution};
use itertools::Itertools;

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
    fn get_at_mut(&mut self,pos:[Idx;3])->&mut D{
        self.geti_mut(self.shape().idx(pos))
    }
    fn get_at2d_mut(&mut self,pos:[Idx;2])->&mut D{
        self.geti_mut(self.shape().grid().idx(pos))
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
    fn sparse_sum(&self, output: &CpuSDR) -> D where D:Sum{
        output.iter().map(|i| self.get(i.as_usize())).sum()
    }
    /***/
    fn mat_sparse_dot_lhs_vec(&self, lhs: &CpuSDR, output:&mut impl TensorTrait<D>) where D:Sum{
        assert_eq!(self.shape().channels(),1,"Tensor should be a matrix");
        let w = self.shape().width();
        assert_eq!(output.shape(),&[w,1,1],"Output shape is invalid");
        let out = output.as_mut_slice();
        for x in 0..w{
            let s:D = lhs.iter().map(|&y|self.get_at2d(from_xy(x,y))).sum();
            out[x.as_usize()] = s;
        }
    }
    fn mat_sparse_dot_lhs_vec_add_assign(&self, lhs: &CpuSDR, output:&mut impl TensorTrait<D>) where D:Sum+AddAssign{
        assert_eq!(self.shape().channels(),1,"Tensor should be a matrix");
        let w = self.shape().width();
        assert_eq!(output.shape(),&[w,1,1],"Output shape is invalid");
        let out = output.as_mut_slice();
        for x in 0..w{
            let s:D = lhs.iter().map(|&y|self.get_at2d(from_xy(x,y))).sum();
            out[x.as_usize()] += s;
        }
    }
    fn mat_sparse_dot_lhs_vec_sub_assign(&self, lhs: &CpuSDR, output:&mut impl TensorTrait<D>) where D:Sum+SubAssign{
        assert_eq!(self.shape().channels(),1,"Tensor should be a matrix");
        let w = self.shape().width();
        assert_eq!(output.shape(),&[w,1,1],"Output shape is invalid");
        let out = output.as_mut_slice();
        for x in 0..w{
            let s:D = lhs.iter().map(|&y|self.get_at2d(from_xy(x,y))).sum();
            out[x.as_usize()] -= s;
        }
    }
    fn mat_sparse_dot_lhs_new_vec(&self, lhs: &CpuSDR) -> Tensor<D> where D:Sum{
        let mut t = unsafe{Tensor::empty(from_xyz(self.shape().width(),1,1))};
        self.mat_sparse_dot_lhs_vec(lhs, &mut t);
        t
    }
    fn sparse_add_assign_scalar_to_area(&mut self, xy_indices:&CpuSDR, channel:Idx, scalar:D)where D:AddAssign{
        let c = self.shape().channels();
        xy_indices.iter().for_each(|&xy_idx|*self.geti_mut(xy_idx*c+channel)+=scalar)
    }
    fn sparse_add_assign_scalar_to_areas(&mut self, xy_indices:&CpuSDR, channels:&CpuSDR, scalar:D)where D:AddAssign{
        let c = self.shape().channels();
        channels.iter().for_each(|&channel|xy_indices.iter().for_each(|&xy_idx|*self.geti_mut(xy_idx*c+channel)+=scalar))
    }
    fn mat_sparse_add_assign_scalar_to_column(&mut self, x:Idx, y_indices:&CpuSDR, scalar:D)where D:AddAssign{
        y_indices.iter().for_each(|&y|*self.get_at2d_mut(from_xy(x,y))+=scalar)
    }
    fn mat_sparse_add_assign_scalar_to_row(&mut self, x_indices:&CpuSDR, y:Idx, scalar:D)where D:AddAssign{
        x_indices.iter().for_each(|&x|*self.get_at2d_mut(from_xy(x,y))+=scalar)
    }
    fn mat_sparse_sub_assign_scalar_to_column(&mut self, x:Idx, y_indices:&CpuSDR, scalar:D)where D:SubAssign{
        y_indices.iter().for_each(|&y|*self.get_at2d_mut(from_xy(x,y))-=scalar)
    }
    fn mat_sparse_sub_assign_scalar_to_row(&mut self, x_indices:&CpuSDR, y:Idx, scalar:D)where D:SubAssign{
        x_indices.iter().for_each(|&x|*self.get_at2d_mut(from_xy(x,y))-=scalar)
    }
    fn mat_sparse_add_assign_scalar_to_area(&mut self, x_indices:&CpuSDR, y_indices:&CpuSDR, scalar:D)where D:AddAssign{
        y_indices.iter().for_each(|&y|x_indices.iter().for_each(|&x| *self.get_at2d_mut(from_xy(x,y))+=scalar))
    }
    fn mat_sparse_sub_assign_scalar_to_area(&mut self, x_indices:&CpuSDR, y_indices:&CpuSDR, scalar:D)where D:SubAssign{
        y_indices.iter().for_each(|&y|x_indices.iter().for_each(|&x| *self.get_at2d_mut(from_xy(x,y))-=scalar))
    }
    fn mat_div_column(&mut self, x:Idx, scalar:D) where D:DivAssign{
        (0..self.shape().height()).for_each(|y|*self.get_at2d_mut(from_xy(x,y))/=scalar)
    }
    fn mat_div_row(&mut self, y:Idx, scalar:D) where D:DivAssign{
        (0..self.shape().width()).for_each(|x|*self.get_at2d_mut(from_xy(x,y))/=scalar)
    }
    fn mat_sum_column<N:LNorm<D>>(&self, x:Idx) -> D where D:Sum{
        (0..self.shape().height()).map(|y|N::pow(self.get_at2d(from_xy(x,y)))).sum()
    }
    fn mat_sum_row<N:LNorm<D>>(&self, y:Idx) -> D where D:Sum{
        (0..self.shape().width()).map(|x|N::pow(self.get_at2d(from_xy(x,y)))).sum()
    }
    fn mat_norm_column<N:LNorm<D>>(&self, x:Idx) -> D where D:Sum{
        N::root(self.mat_sum_column::<N>(x))
    }
    fn mat_norm_row<N:LNorm<D>>(&self, y:Idx) -> D where D:Sum{
        N::root(self.mat_sum_row::<N>(y))
    }
    fn mat_norm_assign_column<N:LNorm<D>>(&mut self, x:Idx) where D:Sum+DivAssign{
        self.mat_div_column(x,self.mat_norm_column::<N>(x))
    }
    fn mat_sparse_norm_assign_column<N:LNorm<D>>(&mut self, x:&CpuSDR) where D:Sum+DivAssign{
        for &x in x.iter(){
            self.mat_norm_assign_column::<N>(x)
        }
    }
    fn mat_sparse_norm_assign_row<N:LNorm<D>>(&mut self, y:&CpuSDR) where D:Sum+DivAssign{
        for &y in y.iter(){
            self.mat_norm_assign_row::<N>(y)
        }
    }
    fn mat_norm_assign_row<N:LNorm<D>>(&mut self, y:Idx) where D:Sum+DivAssign{
        self.mat_div_row(y,self.mat_norm_row::<N>(y))
    }
    fn mat_norm_assign_columnwise<N:LNorm<D>>(&mut self) where D:Sum+DivAssign{
        for x in 0..self.shape().width(){
            self.mat_norm_assign_column::<N>(x)
        }
    }
    fn mat_norm_assign_rowwise<N:LNorm<D>>(&mut self) where D:Sum+DivAssign{
        for y in 0..self.shape().height(){
            self.mat_norm_assign_row::<N>(y)
        }
    }
    fn rand_assign(&mut self, rng:&mut impl Rng) where Standard: Distribution<D>{
        self.iter_mut().for_each(|a|*a=rng.gen())
    }
    fn add_assign(&mut self, other:&impl TensorTrait<D>) where D:AddAssign{
        assert_eq!(self.shape(),other.shape(),"Shapes don't match");
        self.iter_mut().zip(other.iter()).for_each(|(a,b)|*a+=*b)
    }
    fn sub_assign(&mut self, other:&impl TensorTrait<D>) where D:SubAssign{
        assert_eq!(self.shape(),other.shape(),"Shapes don't match");
        self.iter_mut().zip(other.iter()).for_each(|(a,b)|*a-=*b)
    }
    fn mul_assign(&mut self, other:&impl TensorTrait<D>) where D:MulAssign{
        assert_eq!(self.shape(),other.shape(),"Shapes don't match");
        self.iter_mut().zip(other.iter()).for_each(|(a,b)|*a*=*b)
    }
    fn div_assign(&mut self, other:&impl TensorTrait<D>) where D:DivAssign{
        assert_eq!(self.shape(),other.shape(),"Shapes don't match");
        self.iter_mut().zip(other.iter()).for_each(|(a,b)|*a/=*b)
    }
    fn kernel_column_sum_assign(&mut self, rhs: &impl ConvTensorTrait<D>) where D:Sum{
        assert_eq!(self.shape(),rhs.out_shape(),"Shapes don't match");
        self.iter_mut().enumerate().for_each(|(i, w)| *w = rhs.kernel_column_sum(as_idx(i)))
    }
    fn kernel_column_sum_add_assign(&mut self, rhs: &impl ConvTensorTrait<D>) where D:Sum+AddAssign{
        assert_eq!(self.shape(),rhs.out_shape(),"Shapes don't match");
        self.iter_mut().enumerate().for_each(|(i, w)| *w += rhs.kernel_column_sum(as_idx(i)))
    }
    fn kernel_column_sum_sub_assign(&mut self, rhs: &impl ConvTensorTrait<D>) where D:Sum+SubAssign{
        assert_eq!(self.shape(),rhs.out_shape(),"Shapes don't match");
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
    fn find_sparse_between(&self, min: D, max: D, output: &mut CpuSDR) where D:PartialOrd{
        self.as_slice().iter().enumerate().for_each(|(i, &w)| if min < w && w < max{ output.push(as_idx(i))})
    }
    fn find_sparse_gt(&self, min: D, output: &mut CpuSDR) where D:PartialOrd{
        self.as_slice().iter().enumerate().for_each(|(i, &w)| if min < w{ output.push(as_idx(i))})
    }
    fn find_sparse_lt(&self, max: D, output: &mut CpuSDR) where D:PartialOrd{
        self.as_slice().iter().enumerate().for_each(|(i, &w)| if w < max{ output.push(as_idx(i))})
    }
    fn find_sparse_eq(&self, scalar: D, output: &mut CpuSDR) where D:PartialOrd{
        self.as_slice().iter().enumerate().for_each(|(i, &w)| if w == scalar{ output.push(as_idx(i))})
    }
    fn find_sparse_ge(&self, min: D, output: &mut CpuSDR) where D:PartialOrd{
        self.as_slice().iter().enumerate().for_each(|(i, &w)| if min <= w { output.push(as_idx(i))})
    }
    fn find_sparse_le(&self, max: D, output: &mut CpuSDR) where D:PartialOrd{
        self.as_slice().iter().enumerate().for_each(|(i, &w)| if w <= max{ output.push(as_idx(i))})
    }
    fn argmax(&self) -> usize where D:NaiveCmp{
        self.iter().cloned().position_max_by(D::cmp_naive).unwrap()
    }
    fn mat_argmax_in_column(&self, x:Idx) -> usize where D:NaiveCmp{
        (0..self.shape().height()).map(|y|self.get_at2d(from_xy(x,y))).position_max_by(D::cmp_naive).unwrap()
    }
    fn mat_argmax_in_row(&self, y:Idx) -> usize where D:NaiveCmp{
        (0..self.shape().width()).map(|x|self.get_at2d(from_xy(x,y))).position_max_by(D::cmp_naive).unwrap()
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
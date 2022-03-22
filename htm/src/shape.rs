use std::fmt::{Debug, Formatter};

use serde::{Serialize, Deserialize};
use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, RangeBounds, Range};
use std::mem::MaybeUninit;
use crate::vector_field::{VectorField, VectorFieldNum};
use std::collections::{Bound, HashSet};
use num_traits::{Num, One, Zero};
use itertools::Itertools;
use crate::{CpuSDR, VectorFieldPartialOrd, VectorFieldSub, VectorFieldOne, Idx, k_reg, AsUsize};
use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Less};
use std::iter::{FromIterator, FlatMap};

pub trait Shape<T: Num + Copy + Debug + PartialOrd, const DIM: usize>: Eq + PartialEq + Copy + Clone + Debug + VectorFieldNum<T> {
    fn pos(&self, mut index: T) -> [T; DIM] {
        let original_index = index;
        let mut pos: [T; DIM] = unsafe { MaybeUninit::uninit().assume_init() };
        for dim in (0..DIM).rev() {
            let dim_size = self.dim(dim);
            let coord = index % dim_size;
            index = index / dim_size;
            pos[dim] = coord;
        }
        debug_assert_eq!(index, T::zero(), "Index {:?} is out of bounds for shape {:?}", original_index, self);
        pos
    }

    fn idx(&self, pos: [T; DIM]) -> T {
        let mut idx = T::zero();
        for dim in 0..DIM {
            let dim_size = self.dim(dim);
            debug_assert!(pos[dim] < dim_size, "at dim={}, position=={:?} >= shape=={:?}", dim, pos, self);
            idx = idx * dim_size + pos[dim];
        }
        idx
    }
    fn dim(&self, dim: usize) -> T;
    fn set_dim(&mut self, dim: usize, size: T);

    fn size(&self) -> T {
        self.product()
    }
    /**returns the range of inputs that connect to a specific output neuron*/
    fn conv_in_range_begin(&self, stride: &Self) -> Self {
        let out_position = self;//position of an output neuron.
        out_position.mul(stride)
    }
    /**returns the range of inputs that connect to a specific output neuron*/
    fn conv_in_range(&self, stride: &Self, kernel_size: &Self) -> Range<Self> {
        let from = self.conv_in_range_begin(stride);
        let to = from.add(kernel_size);
        from..to
    }
    /**returns the range of inputs that connect to a specific patch of output neuron.
    That output patch starts this position specified by this vector*/
    fn conv_in_range_with_custom_size(&self, output_patch_size: &Self, stride: &Self, kernel_size: &Self) -> Range<Self> {
        if output_patch_size.all_gt_scalar(T::zero()){
            let from = self.conv_in_range_begin(stride);
            let to = self.add(output_patch_size).sub_scalar(T::one()).conv_in_range_begin(stride);
            let to = to.add(kernel_size);
            from..to
        }else{
            Self::new_const(T::zero())..Self::new_const(T::zero())
        }
    }
    /**returns the range of outputs that connect to a specific input neuron*/
    fn conv_out_range(&self, stride: &Self, kernel_size: &Self) -> Range<Self> {
        let in_position = self;//position of an input neuron.
        //out_position * stride .. out_position * stride + kernel
        //out_position * stride ..= out_position * stride + kernel - 1
        //
        //in_position_from == out_position * stride
        //in_position_from / stride == out_position
        //round_down(in_position / stride) == out_position_to
        //
        //in_position_to == out_position * stride + kernel - 1
        //(in_position_to +1 - kernel)/stride == out_position
        //round_up((in_position +1 - kernel)/stride) == out_position_from
        //round_down((in_position +1 - kernel + stride - 1)/stride) == out_position_from
        //round_down((in_position - kernel + stride)/stride) == out_position_from
        //
        //(in_position - kernel + stride)/stride ..= in_position / stride
        //(in_position - kernel + stride)/stride .. in_position / stride + 1
        let to = in_position.div(stride).add_scalar(T::one());
        let from = in_position.add(stride).sub(kernel_size).div(stride);
        from..to
    }
    fn conv_out_transpose_kernel(&self, stride: &Self) -> Self {
        let kernel = self;
        // (in_position - kernel + stride)/stride .. in_position / stride + 1
        //  in_position / stride + 1 - (in_position - kernel + stride)/stride
        //  (in_position- (in_position - kernel + stride))/stride + 1
        //  (kernel - stride)/stride + 1
        debug_assert!(kernel.all_ge(stride));
        kernel.sub(stride).div(stride).add_scalar(T::one())
    }
    /**returns the range of outputs that connect to a specific input neuron.
    output range is clipped to 0, so that you don't get overflow on negative values when dealing with unsigned integers.*/
    fn conv_out_range_clipped(&self, stride: &Self, kernel_size: &Self) -> Range<Self> {
        let in_position = self;//position of an input neuron.
        let to = in_position.div(stride).add_scalar(T::one());
        let from = in_position.add(stride).max(kernel_size).sub(kernel_size).div(stride);
        from..to
    }
    fn conv_out_range_clipped_both_sides(&self, stride: &Self, kernel_size: &Self, max_bounds: &Self) -> Range<Self> {
        let mut r = self.conv_out_range_clipped(stride, kernel_size);
        r.end = r.end.min(max_bounds);
        r
    }
    fn conv_out_size(&self, stride: &Self, kernel_size: &Self) -> Self {
        let input = self;
        assert!(kernel_size.all_le(input), "Kernel size {:?} is larger than the input shape {:?} ", kernel_size, input);
        let input_sub_kernel = input.sub(kernel_size);
        assert!(input_sub_kernel.rem(stride).all_eq_scalar(T::zero()), "Convolution stride {:?} does not evenly divide the input shape {:?}-{:?}={:?} ", stride, input,kernel_size,input_sub_kernel);
        input_sub_kernel.div(stride).add_scalar(T::one())
        //(input-kernel)/stride+1 == output
    }
    fn conv_in_size(&self, stride: &Self, kernel_size: &Self) -> Self {
        let output = self;
        assert!(output.all_gt_scalar(T::zero()), "Output size {:?} contains zero", output);
        output.sub_scalar(T::one()).mul(stride).add(kernel_size)
        //input == stride*(output-1)+kernel
    }
    fn conv_stride(&self, out_size: &Self, kernel_size: &Self) -> Self {
        let input = self;
        assert!(kernel_size.all_le(input), "Kernel size {:?} is larger than the input shape {:?}", kernel_size, input);
        let input_sub_kernel = input.sub(kernel_size);
        let out_size_minus_1 = out_size.sub_scalar(T::one());
        assert!(input_sub_kernel.rem_default_zero(&out_size_minus_1, T::zero()).all_eq_scalar(T::zero()), "Output shape {:?}-1 does not evenly divide the input shape {:?}", out_size, input);
        input_sub_kernel.div_default_zero(&out_size_minus_1, T::one())
        //(input-kernel)/(output-1) == stride
    }
    fn conv_compose(&self, self_kernel: &Self, next_stride: &Self, next_kernel: &Self) -> (Self, Self) {
        //(A-kernelA)/strideA+1 == B
        //(B-kernelB)/strideB+1 == C
        //((A-kernelA)/strideA+1-kernelB)/strideB+1 == C
        //(A-kernelA+(1-kernelB)*strideA)/(strideA*strideB)+1 == C
        //(A-(kernelA-(1-kernelB)*strideA))/(strideA*strideB)+1 == C
        //(A-(kernelA+(kernelB-1)*strideA))/(strideA*strideB)+1 == C
        //    ^^^^^^^^^^^^^^^^^^^^^^^^^^^                    composed kernel
        //                                   ^^^^^^^^^^^^^^^ composed stride
        let composed_kernel = self_kernel.add(&next_kernel.sub_scalar(T::one()).mul(self));
        let composed_stride = self.mul(next_stride);
        (composed_stride, composed_kernel)
    }

}

impl<T: Num + Debug + Eq + Copy + PartialOrd, const DIM: usize> Shape<T, DIM> for [T; DIM] {
    fn dim(&self, dim: usize) -> T {
        self[dim]
    }

    fn set_dim(&mut self, dim: usize, size: T) {
        self[dim] = size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test1() {
        let s = [4, 3];
        assert_eq!(s.idx([0, 0]), 0);
        assert_eq!(s.idx([0, 1]), 1);
        assert_eq!(s.idx([0, 2]), 2);
        assert_eq!(s.pos(0), [0, 0]);
        assert_eq!(s.pos(1), [0, 1]);
        assert_eq!(s.pos(2), [0, 2]);
        assert_eq!(s.pos(3), [1, 0]);
        assert_eq!(s.pos(4), [1, 1]);
        assert_eq!(s.pos(5), [1, 2]);
        for i in 0..(3 * 4) {
            let p = s.pos(i);
            assert_eq!(s.idx(p), i, "{}=={:?}", i, p);
        }
    }

    #[test]
    fn test2() {
        let s = [3, 4];
        for x in 0..3 {
            for y in 0..4 {
                assert_eq!(s.pos(s.idx([x, y])), [x, y]);
            }
        }
    }

    #[test]
    fn test3() {
        let s = [6, 4, 3];
        assert_eq!(s.idx([2, 0, 0]), 24);
        assert_eq!(s.idx([3, 0, 1]), 37);
        assert_eq!(s.idx([4, 0, 2]), 50);
        assert_eq!(s.pos(0), [0, 0, 0]);
        assert_eq!(s.pos(1), [0, 0, 1]);
        assert_eq!(s.pos(2), [0, 0, 2]);
        assert_eq!(s.pos(3), [0, 1, 0]);
        assert_eq!(s.pos(4), [0, 1, 1]);
        assert_eq!(s.pos(5), [0, 1, 2]);
        for i in 0..s.size() {
            let p = s.pos(i);
            assert_eq!(s.idx(p), i, "{}=={:?}", i, p);
        }
    }

    #[test]
    fn test4() {
        let s = [6u32, 3, 4];
        for x in 0..s[2] {
            for y in 0..s[1] {
                for z in 0..s[0] {
                    assert_eq!(s.pos(s.idx([z, y, x])), [z, y, x]);
                }
            }
        }
    }

    #[test]
    fn test5() {
        for x in 1..3 {
            for y in 1..4 {
                for sx in 1..2 {
                    for sy in 1..2 {
                        for ix in 1..3 {
                            for iy in 1..4 {
                                let kernel = [x, y];
                                let stride = [x, y];
                                let output_size = [ix, iy];
                                let input_size = output_size.conv_in_size(&stride, &kernel);
                                assert_eq!(output_size, input_size.conv_out_size(&stride, &kernel));
                                for ((&expected, &actual), &out) in stride.iter().zip(input_size.conv_stride(&output_size, &kernel).iter()).zip(output_size.iter()) {
                                    if out != 1 {
                                        assert_eq!(expected, actual);
                                    } else {
                                        assert_eq!(1, actual);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test6() {
        for output_idx in 0..24 {
            for x in 1..5 {
                for sx in 1..5 {
                    let i = [output_idx].conv_in_range(&[sx], &[x]);
                    let i_r = i.start[0]..i.end[0];
                    for i in i_r.clone() {
                        let o = [i].conv_out_range(&[sx], &[x]);
                        let o_r = o.start[0]..o.end[0];
                        assert!(o_r.contains(&output_idx), "o_r={:?}, i_r={:?} output_idx={} sx={} x={}", o_r, i_r, output_idx, sx, x)
                    }
                }
            }
        }
    }

    #[test]
    fn test7() {
        for input_idx in 0..24 {
            for x in 1..5 {
                for sx in 1..5 {
                    let o = [input_idx].conv_out_range(&[sx], &[x]);
                    let o_r = o.start[0]..o.end[0];
                    for o in o_r.clone() {
                        let i = [o].conv_in_range(&[sx], &[x]);
                        let i_r = i.start[0]..i.end[0];
                        assert!(i_r.contains(&input_idx), "o_r={:?}, i_r={:?} input_idx={} sx={} x={}", o_r, i_r, input_idx, sx, x)
                    }
                }
            }
        }
    }

    #[test]
    fn test8() {
        let mut rng = rand::thread_rng();
        let max = 128usize;
        for _ in 0..54 {
            let k = rng.gen_range(2usize..8);
            let arr: Vec<usize> = (0..64).map(|_| rng.gen_range(0..max)).collect();
            let mut candidates = vec![0; max];
            let mut o = Vec::new();
            top_large_k_indices(k, &arr, &mut candidates, |&a| a, |t| o.push(t));
            let mut top_values1: Vec<usize> = o.iter().map(|&i| arr[i]).collect();
            let mut arr_ind: Vec<(usize, usize)> = arr.into_iter().enumerate().collect();
            arr_ind.sort_by_key(|&(_, v)| v);
            let top_values2: Vec<usize> = arr_ind[64 - k..].iter().map(|&(_, v)| v).collect();
            top_values1.sort();
            assert_eq!(top_values1, top_values2)
        }
    }

    #[test]
    fn test9() {
        let mut rng = rand::thread_rng();
        let max = 128usize;
        for _ in 0..54 {
            let k = rng.gen_range(2usize..8);
            let arr: Vec<usize> = (0..64).map(|_| rng.gen_range(0..max)).collect();
            let o = top_small_k_indices(k, arr.len(), |i| arr[i]);
            let mut top_values1: Vec<usize> = o.into_iter().map(|(i, v)| v).collect();
            let mut arr_ind: Vec<(usize, usize)> = arr.into_iter().enumerate().collect();
            arr_ind.sort_by_key(|&(_, v)| v);
            let top_values2: Vec<usize> = arr_ind[64 - k..].iter().map(|&(_, v)| v).collect();
            top_values1.sort();
            assert_eq!(top_values1, top_values2)
        }
    }

    #[test]
    fn test10() {
        let mut rng = rand::thread_rng();
        let max = 128usize;
        for _ in 0..54 {
            let arr: Vec<usize> = (0..64).map(|_| rng.gen_range(0..max)).collect();
            let o = top_small_k_indices(1, arr.len(), |i| arr[i]);
            let (top_idx, top_val) = o[0];
            assert_eq!(top_val, *arr.iter().max().unwrap());
            assert_eq!(top_idx, arr.len() - 1 - arr.iter().rev().position_max().unwrap());
        }
    }
}

pub fn range_foreach2d<T: Copy>(range: &Range<[T; 2]>, mut f:impl FnMut([T;2])) where Range<T>:Iterator<Item=T>{
    for p0 in range.start[0]..range.end[0] {
        for p1 in range.start[1]..range.end[1] {
            f([p0,p1])
        }
    }
}
pub fn range_contains<T: Copy + PartialOrd + Debug, const DIM: usize>(range: &Range<[T; DIM]>, element: &[T; DIM]) -> bool {
    range.start.all_le(element) && element.all_lt(&range.end)
}

pub fn range_translate<T: Copy + Num + std::cmp::PartialOrd + std::cmp::Eq + Debug, const DIM: usize>(range: &Range<[T; DIM]>, element: &[T; DIM]) -> Option<T> {
    if range_contains(range,element){
        let element_within_range = element.sub(&range.start);
        let range_size = range.end.sub(&range.start);
        Some(range_size.idx(element_within_range))
    }else{
        None
    }
}
pub fn resolve_range<T: Add<Output=T> + Copy + One + Zero + PartialOrd + Debug>(input_size: T, input_range: impl RangeBounds<T>) -> Range<T> {
    let b = match input_range.start_bound() {
        Bound::Included(&x) => x,
        Bound::Excluded(&x) => x + T::one(),
        Bound::Unbounded => T::zero()
    };
    let e = match input_range.end_bound() {
        Bound::Included(&x) => x + T::one(),
        Bound::Excluded(&x) => x,
        Bound::Unbounded => input_size
    };
    assert!(b <= e, "Input range {:?}..{:?} starts later than it ends", b, e);
    assert!(e <= input_size, "Input range {:?}..{:?} exceeds input size {:?}", b, e, input_size);
    b..e
}

pub fn top_small_k_indices<V: Copy + PartialOrd>(mut k: usize, n: usize, f: impl Fn(usize) -> V) -> Vec<(usize, V)> {
    debug_assert!(k <= n);
    let mut heap: Vec<(usize, V)> = (0..k).map(&f).enumerate().collect();
    heap.sort_by(|v1, v2| if v1.1>v2.1 { Greater } else { Less });
    for (idx, v) in (k..n).map(f).enumerate() {
        let idx = idx + k;
        if v> heap[0].1 {
            let mut i = 1;
            while i < k && v > heap[i].1 {
                heap[i - 1] = heap[i];
                i += 1
            }
            heap[i - 1] = (idx, v);
        }
    }
    heap
}

pub fn top_large_k_indices<T>(mut k: usize, values: &[T], candidates_per_value: &mut [usize], f: fn(&T) -> usize, mut output: impl FnMut(usize)) {
    debug_assert!(candidates_per_value.iter().all(|&e| e == 0));
    values.iter().for_each(|v| candidates_per_value[f(v)] += 1);
    let mut min_candidate_value = 0;
    for (value, candidates) in candidates_per_value.iter_mut().enumerate().rev() {
        if k <= *candidates {
            *candidates = k;
            min_candidate_value = value;
            break;
        }
        k -= *candidates;
    }
    candidates_per_value[0..min_candidate_value].fill(0);
    for (i, v) in values.iter().enumerate() {
        let v = f(v);
        if candidates_per_value[v] > 0 {
            output(i);
            candidates_per_value[v] -= 1;
        }
    }
}

pub trait Shape3<T:Num + Copy + Debug + PartialOrd> {
    fn grid(&self) -> &[T; 2];
    fn right(&self) -> &[T; 2];
    fn width(&self) -> T;
    fn height(&self) -> T;
    fn channels(&self) -> T;
    fn width_mut(&mut self) -> &mut T;
    fn height_mut(&mut self) -> &mut T;
    fn channels_mut(&mut self) -> &mut T;
    fn area(&self) -> T;
    fn volume(&self) -> T;
}
pub trait TopK:Shape3<Idx> {
    fn topk_per_column<D:PartialOrd+Copy>(&self,k:usize, value: impl Fn(usize,usize) -> D, target: impl FnMut(D,usize,usize));
    fn top1_in_range<D:PartialOrd>(&self,range:Range<Idx>, value: impl Fn(Idx) -> D) -> (D,Idx);
    fn top1_per_region<D:PartialOrd>(&self,k:Idx, value: impl Fn(Idx) -> D, target: impl FnMut(D,Idx));
    fn top1_per_region_per_column<D:PartialOrd,V>(&self,k:Idx, column_value:impl Fn(Idx)->V, value: impl Fn(&V,Idx) -> D, target: impl FnMut(D,Idx));
}
impl TopK for [Idx;3]{
    fn topk_per_column<D:PartialOrd+Copy>(&self,k:usize, value: impl Fn(usize,usize) -> D, mut target: impl FnMut(D,usize,usize)) {
        let a = self.volume().as_usize();
        let c = self.channels().as_usize();
        for column_idx in 0..a {
            let r = c * column_idx;
            for (i, v) in top_small_k_indices(k, c, |i| {
                debug_assert!(i < c);
                value(column_idx,i + r)
            }) {
                let e = r + i;
                debug_assert!(r <= e);
                debug_assert!(e < r + c);
                target(v,column_idx,e);
            }
        }
    }
    fn top1_in_range<D:PartialOrd>(&self,mut range:Range<Idx>, value: impl Fn(Idx) -> D) -> (D,Idx){
        let mut top1_idx = range.start;
        let mut top1_val = value(top1_idx);
        range.start+=1;
        for i in range{
            let val = value(i);
            if val > top1_val{
                top1_val = val;
                top1_idx = i;
            }
        }
        (top1_val,top1_idx)
    }
    fn top1_per_region<D:PartialOrd>(&self,k:Idx, value: impl Fn(Idx) -> D, mut target: impl FnMut(D,Idx)) {
        assert_eq!(self.channels() % k, 0);
        for region_idx in 0..k_reg::get_region_count(k,self) {
            //There are k regions per column, each has region_size neurons.
            //We need to pick the top 1 winner within each region.
            //Giving us the total of k winners per output column.
            //The channels of each column are arranged contiguously. Regions are also contiguous.
            let mut range = k_reg::get_region_range(k,self, region_idx);
            let (top1_val,top1_idx) = self.top1_in_range(range,&value);
            target(top1_val,top1_idx)
        }
    }
    fn top1_per_region_per_column<D:PartialOrd,V>(&self,k:Idx, column_value:impl Fn(Idx)->V, value: impl Fn(&V,Idx) -> D, mut target: impl FnMut(D,Idx)) {
        assert_eq!(self.channels() % k, 0);
        for col_idx in 0..self.area() {
            for region_idx_within_col in 0..k {
                //There are k regions per column, each has region_size neurons.
                //We need to pick the top 1 winner within each region.
                //Giving us the total of k winners per output column.
                //The channels of each column are arranged contiguously. Regions are also contiguous.
                let col_val = column_value(col_idx);
                let region_idx = k*col_idx + region_idx_within_col;
                let mut range = k_reg::get_region_range(k, self, region_idx);
                let (top1_val, top1_idx) = self.top1_in_range(range, |i|value(&col_val,i));
                target(top1_val, top1_idx)
            }
        }
    }
}
impl<T: Num + Copy + Debug + Eq + PartialOrd> Shape3<T> for [T; 3] {

    fn grid(&self) -> &[T; 2] {
        let [ref a @ .., _] = self;
        a
    }

    fn right(&self) -> &[T; 2] {
        let [_, ref a @ .. ] = self;
        a
    }

    fn width(&self) -> T {
        self[0]
    }

    fn height(&self) -> T {
        self[1]
    }

    fn channels(&self) -> T {
        self[2]
    }
    fn width_mut(&mut self) -> &mut T { &mut self[0] }
    fn height_mut(&mut self) -> &mut T { &mut self[1] }
    fn channels_mut(&mut self) -> &mut T { &mut self[2] }

    fn area(&self) -> T {
        self.grid().product()
    }
    fn volume(&self)->T{
        self.product()
    }
}

pub trait Shape2<T:Num + Copy + Debug + Eq + PartialOrd> {
    type A3: Shape3<T>;
    fn add_channels(&self, channels: T) -> Self::A3;

    fn width(&self) -> T;

    fn height(&self) -> T;
    fn area(&self) -> T;
}

impl<T: Num + Copy + Debug + Eq + PartialOrd> Shape2<T> for [T; 2] {
    type A3 = [T; 3];

    fn add_channels(&self, channels: T) -> Self::A3 {
        [self[0], self[1], channels]
    }

    fn width(&self) -> T {
        self[0]
    }

    fn height(&self) -> T {
        self[1]
    }

    fn area(&self) -> T {
        self.product()
    }
}

pub fn from_xy<T>(width: T, height: T) -> [T; 2] {
    [width, height]
}

pub fn from_xyz<T>(width: T, height: T, channels: T) -> [T; 3] {
    [width, height, channels]
}


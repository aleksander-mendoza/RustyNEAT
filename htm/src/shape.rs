use std::fmt::{Debug, Formatter};

use serde::{Serialize, Deserialize};
use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, RangeBounds, Range};
use std::mem::MaybeUninit;
use crate::vector_field::{VectorField, VectorFieldNum};
use std::collections::Bound;
use num_traits::Num;

pub trait Shape<T:Num + Copy + Debug + PartialOrd,const DIM: usize>: Eq + PartialEq + Copy + Clone + Debug + VectorFieldNum<T> {
    fn pos(&self, mut index: T) -> [T; DIM] {
        let original_index = index;
        let mut pos:[T;DIM] = unsafe{MaybeUninit::uninit().assume_init()};
        for dim in (0..DIM).rev(){
            let dim_size = self.dim(dim);
            let coord = index % dim_size;
            index = index / dim_size;
            pos[dim] = coord;
        }
        assert_eq!(index,T::zero(),"Index {:?} is out of bounds for shape {:?}",original_index,self);
        pos
    }

    fn idx(&self, pos: [T; DIM]) -> T {
        let mut idx = T::zero();
        for dim  in 0..DIM{
            let dim_size = self.dim(dim);
            assert!(pos[dim]<dim_size,"position[{:?}]=={:?} >= shape[{:?}]=={:?}",dim,pos[dim],dim,dim_size);
            idx = idx*dim_size + pos[dim];
        }
        idx
    }
    fn dim(&self, dim:usize) -> T;
    fn set_dim(&mut self, dim:usize, size:T);

    fn size(&self) -> T {
        self.product()
    }
    fn conv_out_size(&self, stride: &Self, kernel_size: &Self) -> Self {
        let input = self;
        assert!(kernel_size.all_le(input),"Kernel size {:?} is larger than the input shape {:?} ",kernel_size,input);
        let input_sub_kernel = input.sub(kernel_size);
        assert!(input_sub_kernel.rem(stride).all_eq_scalar(T::zero()),"Convolution stride {:?} does not evenly divide the output shape {:?} ",stride,input);
        input_sub_kernel.div(stride).add_scalar(T::one())
        //(input-kernel)/stride+1 == output
    }
    fn conv_stride(&self, out_size: &Self, kernel_size: &Self) -> Self {
        let input = self;
        assert!(kernel_size.all_le(input),"Kernel size {:?} is larger than the input shape {:?}",kernel_size,input);
        let input_sub_kernel = input.sub(kernel_size);
        let out_size_minus_1 = out_size.sub_scalar(T::one());
        assert!(input_sub_kernel.rem(&out_size_minus_1).all_eq_scalar(T::zero()),"Output shape {:?}-1 does not evenly divide the input shape {:?}",out_size,input);
        input_sub_kernel.div(&out_size_minus_1)
        //(input-kernel)/(output-1) == stride
    }
    fn conv_compose(&self, self_kernel:&Self, next_stride: &Self, next_kernel: &Self) -> (Self,Self) {
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
        (composed_stride,composed_kernel)
    }
}
pub trait Shape2{
    type T;
    fn width(&self)->Self::T;
    fn height(&self)->Self::T;
    fn index(&self, y: Self::T, x:Self::T) -> Self::T;
}
pub trait Shape3{
    type T;
    fn width(&self)->Self::T;
    fn height(&self)->Self::T;
    fn depth(&self)->Self::T;
    fn index(&self, z:Self::T, y: Self::T, x:Self::T) -> Self::T;
}
macro_rules! impl_shape {
    ($t:ident) => {
impl Shape2 for [$t;2]{
    type T = $t;

    fn width(&self) -> $t {
        self[1]
    }

    fn height(&self) -> $t {
        self[0]
    }

    fn index(&self, y: $t, x: $t) -> $t {
        self.idx([y,x])
    }
}

impl Shape3 for [$t;3]{
    type T = $t;

    fn width(&self) -> $t {
        self[2]
    }

    fn height(&self) ->$t {
        self[1]
    }

    fn depth(&self) -> $t {
        self[0]
    }

    fn index(&self, z:$t, y: $t, x: $t) -> $t {
        self.idx([z,y,x])
    }
}
    };
}
impl_shape!(u32);
impl_shape!(u64);
impl_shape!(usize);


impl <T:Num + Debug + Eq + Copy + PartialOrd,const DIM:usize> Shape<T,DIM> for [T;DIM]{
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
        for x in 0..s.width() {
            for y in 0..s.height() {
                for z in 0..s.depth() {
                    assert_eq!(s.pos(s.idx([z, y, x])), [z, y, x]);
                }
            }
        }
    }
}
pub fn resolve_range(input_size:u32,input_range:impl RangeBounds<u32>)->Range<u32>{
    let b = match input_range.start_bound(){
        Bound::Included(&x) => x,
        Bound::Excluded(&x) => x+1,
        Bound::Unbounded => 0
    };
    let e = match input_range.end_bound(){
        Bound::Included(&x) => x+1,
        Bound::Excluded(&x) => x,
        Bound::Unbounded => input_size
    };
    assert!(b <= e, "Input range {}..{} starts later than it ends", b,e);
    assert!(e <= input_size, "Input range {}..{} exceeds input size {}", b,e,input_size);
    b..e
}

use std::fmt::{Debug, Formatter};

use serde::{Serialize, Deserialize};
use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, RangeBounds, Range};
use std::mem::MaybeUninit;
use crate::rnd::xorshift32;
use crate::vector_field::{VectorField, VectorFieldNum};
use std::collections::Bound;

pub trait Shape<const DIM: usize>: Eq + PartialEq + Copy + Clone + Debug + VectorFieldNum<u32> {

    fn pos(&self, index: u32) -> [u32; DIM];
    fn idx(&self, pos: [u32; DIM]) -> u32;
    fn dim(&self, dim:usize) -> u32;
    fn set_dim(&mut self, dim:usize, size:u32);
    fn rand(&mut self, size:&Self, mut rand_seed:u32) -> u32{
        for i in 0..DIM{
            rand_seed = xorshift32(rand_seed);
            self.set_dim(i, rand_seed % size.dim(i));
        }
        rand_seed
    }
    fn size(&self) -> u32 {
        (0..DIM).map(|i| self.dim(i)).product()
    }
    fn conv_out_size(&self, stride: &Self, kernel_size: &Self) -> Self {
        let grid_size = self;
        assert!(kernel_size.all_le(grid_size),"Kernel size {:?} is larger than the grid {:?} of voting columns",kernel_size,grid_size);
        let out_grid_size = grid_size.sub(kernel_size);
        assert!(out_grid_size.rem(stride).all_eq_scalar(0),"Convolution stride {:?} does not evenly divide the grid {:?} of voting columns",stride,grid_size);
        out_grid_size.div(stride).add_scalar(1)
    }
}
pub trait Shape2{
    fn width(&self)->u32;
    fn height(&self)->u32;
    fn index(&self, y: u32, x:u32) -> u32;
}
pub trait Shape3{
    fn width(&self)->u32;
    fn height(&self)->u32;
    fn depth(&self)->u32;
    fn index(&self, z:u32, y: u32, x:u32) -> u32;
}
impl Shape<2> for [u32; 2] {

    fn pos(&self, index: u32) -> [u32; 2] {
        assert!(index < self.size());
        let y = index / self.width();
        let x = index % self.width();
        [y, x]
    }
    fn idx(&self, pos: [u32; 2]) -> u32 {
        let [y, x] = pos;
        assert!(y < self.height());
        assert!(x < self.width());
        y * self.width() + x
    }
    fn dim(&self, dim:usize) -> u32{
        self[dim]
    }

    fn set_dim(&mut self, dim: usize, size: u32) {
        self[dim] = size
    }
}
impl Shape2 for [u32;2]{
    fn width(&self) -> u32 {
        self[1]
    }

    fn height(&self) -> u32 {
        self[0]
    }

    fn index(&self, y: u32, x: u32) -> u32 {
        self.idx([y,x])
    }
}

impl Shape<3> for [u32; 3] {
    fn pos(&self, index: u32) -> [u32; 3] {
        assert!(index < self.size());
        let x = index % self.width();
        let zy = index / self.width();
        let y = zy % self.height();
        let z = zy / self.height();
        [z, y, x]
    }

    fn idx(&self, pos: [u32; 3]) -> u32 {
        let [z, y, x] = pos;
        assert!(y < self.height());
        assert!(x < self.width());
        assert!(z < self.depth());
        (z * self.height() + y) * self.width() + x
    }
    fn dim(&self, dim:usize) -> u32{
        self[dim]
    }
    fn set_dim(&mut self, dim: usize, size: u32) {
        self[dim] = size
    }
}
impl Shape3 for [u32;3]{
    fn width(&self) -> u32 {
        self[2]
    }

    fn height(&self) -> u32 {
        self[1]
    }

    fn depth(&self) -> u32 {
        self[0]
    }

    fn index(&self, z: u32, y: u32, x: u32) -> u32 {
        self.idx([z,y,x])
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
        let s = [6, 3, 4];
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

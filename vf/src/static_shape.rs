use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, RangeBounds, Range};
use std::mem::MaybeUninit;
use crate::vector_field::{VectorField, VectorFieldNum};
use std::collections::{Bound, HashSet};
use num_traits::{Num, One, Zero, AsPrimitive};
use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Less};
use std::iter::{FromIterator, FlatMap, Rev, Sum};
use crate::init::empty;
use std::option::Option::{None, Some};
use crate::{VectorFieldOne, VectorFieldSub, VectorFieldPartialOrd, VectorFieldSubOwned};

pub trait Shape<T, const DIM: usize> {
    fn pos(&self, index: T) -> [T; DIM];
    fn idx(&self, pos: [T; DIM]) -> T;
    fn pos_iter(&self) -> PosIter<T, DIM>;
    fn pos_iter_rev(&self) -> PosIterRev<T, DIM>;
}

impl<T: Rem + Div + Mul + Add + Copy, const DIM: usize> Shape<T, DIM> for [T; DIM] {
    fn pos(&self, index: T) -> [T; DIM] {
        pos(self, index)
    }

    fn idx(&self, pos: [T; DIM]) -> T {
        idx(self, pos)
    }

    fn pos_iter(&self) -> PosIter<T, DIM> {
        pos_iter(*self)
    }
    fn pos_iter_rev(&self) -> PosIterRev<T, DIM> {
        pos_iter_rev(*self)
    }
}

pub fn pos<T: Rem + Div + Copy, const DIM: usize>(shape: &[T; DIM], mut index: T) -> [T; DIM] {
    let original_index = index;
    let mut pos: [T; DIM] = empty();
    for dim in (0..DIM).rev() {
        let dim_size = shape[dim];
        let coord = index % dim_size;
        index = index / dim_size;
        pos[dim] = coord;
    }
    debug_assert_eq!(index, T::zero(), "Index {:?} is out of bounds for shape {:?}", original_index, shape);
    pos
}

pub fn idx<T: Mul + Add + Copy, const DIM: usize>(shape: &[T; DIM], pos: [T; DIM]) -> T {
    let mut idx = T::zero();
    for dim in 0..DIM {
        let dim_size = shape[dim];
        debug_assert!(pos[dim] < dim_size, "at dim={}, position=={:?} >= shape=={:?}", dim, pos, shape);
        idx = idx * dim_size + pos[dim];
    }
    idx
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
}


pub struct PosIter<T, const DIM: usize> {
    pos: [T; DIM],
    shape: [T; DIM],
}

impl<T: Copy + Zero + One + Add<Output=T> + Sub<Output=T> + Ord, const DIM: usize> Iterator for PosIter<T, DIM> where Range<T>: ExactSizeIterator {
    type Item = [T; DIM];

    fn next(&mut self) -> Option<Self::Item> {
        for i in 0..DIM {
            if self.pos[i] < self.shape[i] {
                let p = self.pos;
                self.pos[i] = self.pos[i] + T::one();
                return Some(p);
            } else {
                self.pos[i] = 0;
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let s = (self.pos.product()..self.shape.product()).len();
        (s, Some(s))
    }

    fn count(self) -> usize where Self: Sized, {
        (self.pos.product()..self.shape.product()).len()
    }

    fn last(self) -> Option<Self::Item> where Self: Sized, {
        if self.shape.all_gt_scalar(T::zero()) {
            Some(self.shape._sub_scalar(T::one()))
        } else {
            None
        }
    }

    fn sum<S>(self) -> S where Self: Sized, S: Sum<Self::Item>, {
        let l = self.count();
        (1 + l) * l / 2
    }
}

impl<T: Copy + Zero + One + Add<Output=T> + Sub<Output=T> + Ord, const DIM: usize> ExactSizeIterator for PosIter<T, DIM> where Range<T>: ExactSizeIterator {}

pub fn pos_iter<T: Copy + Zero + One + Add<Output=T> + Sub<Output=T> + Ord, const DIM: usize>(shape: [T; DIM]) -> PosIter<T, DIM> where Range<T>: ExactSizeIterator {
    PosIter { pos: [T::zero(); DIM], shape }
}


pub struct PosIterRev<T, const DIM: usize> {
    pos: [T; DIM],
    shape: [T; DIM],
}


impl<T: Copy + Zero + One + Add<Output=T> + Sub<Output=T> + Ord, const DIM: usize> Iterator for PosIterRev<T, DIM> where Range<T>: ExactSizeIterator {
    type Item = [T; DIM];

    fn next(&mut self) -> Option<Self::Item> {
        for i in 0..DIM {
            if self.pos[i] > T::zero() {
                self.pos[i] = self.pos[i] - T::one();
                return Some(self.pos);
            } else {
                self.pos[i] = self.shape[i];
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let s = (T::zero()..self.pos.product()).len();
        (s, Some(s))
    }

    fn count(self) -> usize where Self: Sized, {
        (T::zero()..self.pos.product()).len()
    }

    fn last(self) -> Option<Self::Item> where Self: Sized, {
        if self.shape.all_gt_scalar(T::zero()) {
            Some([T::zero(); DIM])
        } else {
            None
        }
    }

    fn sum<S>(self) -> S where Self: Sized, S: Sum<Self::Item>, {
        let l = self.count();
        (1 + l) * l / 2
    }
}

impl<T: Copy + Zero + One + Add<Output=T> + Sub<Output=T> + Ord, const DIM: usize> ExactSizeIterator for PosIterRev<T, DIM> where Range<T>: ExactSizeIterator {}

pub fn pos_iter_rev<T: Copy + Zero + One + Add<Output=T> + Sub<Output=T> + Ord, const DIM: usize>(shape: [T; DIM]) -> PosIterRev<T, DIM> where Range<T>: ExactSizeIterator {
    PosIterRev { pos: shape, shape }
}
use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, RangeBounds, Range};
use std::mem::MaybeUninit;
use crate::vector_field::{VectorField, VectorFieldNum};
use std::collections::{Bound, HashSet};
use num_traits::{Num, One, Zero};
use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Less};
use std::iter::{FromIterator, FlatMap, Sum};
use crate::init::{empty, InitEmpty, InitEmptyWithCapacity};
use crate::{VectorFieldOne, VectorFieldPartialOrd};

pub trait Shape<T> {
    type O;
    fn ndim(&self) -> usize;
    fn size(&self) -> T;
    fn pos(&self, index: T) -> Self::O;
    fn idx(&self, pos: &Self) -> T;
    /**returns true if there was a previous position, false if there was none and pos had to be rewound to the last position*/
    fn prev_pos_(&self, pos: &mut Self) -> bool;
    fn prev_pos(&self) -> Option<Self::O>;
    /**returns true if there was a next position, false if there was none and pos had to be rewound to the first position*/
    fn next_pos_(&self, pos: &mut Self) -> bool;
    fn next_pos(&self) -> Option<Self::O>;
    fn is_valid_pos(&self, pos: &Self) -> bool;
    fn first_pos_(&self, pos: &mut Self) -> bool;
    fn first_pos(&self) -> Option<Self::O>;
    fn last_pos_(&self, pos: &mut Self) -> bool;
    fn last_pos(&self) -> Option<Self::O>;
    fn rfold_pos<D>(&self, end: D, fold: impl FnMut(D, &Self) -> D) -> D {
        'a: loop {
            break;
        }
    }
    fn pos_iter_(&self, pos: &mut Self) -> PosIter<T, Self> {
        PosIter { pos, shape: self }
    }
}

impl<T: Rem + Div + Mul + Add + Copy + Zero + One> Shape<T> for [T] {
    type O = Vec<T>;

    fn ndim(&self) -> usize {
        self.len()
    }

    fn size(&self) -> T {
        self.product()
    }

    fn pos(&self, index: T) -> Vec<T> {
        let mut pos = Vec::<T>::empty(self.len());
        pos_(self, index, &mut pos);
        pos
    }

    fn idx(&self, pos: &[T]) -> T {
        idx(self, pos)
    }

    fn prev_pos_(&self, pos: &mut [T]) -> bool {
        prev_pos_(self, pos)
    }

    fn prev_pos(&self) -> Option<Self::O> {
        let mut pos = Vec::<T>::empty(self.len());
        if self.prev_pos_(&mut pos) { Some(pos) } else { None }
    }

    fn next_pos_(&self, pos: &mut [T]) -> bool {
        next_pos_(self, pos)
    }

    fn next_pos(&self) -> Option<Self::O> {
        let mut pos = Vec::<T>::empty(self.len());
        if self.next_pos_(&mut pos) { Some(pos) } else { None }
    }

    fn is_valid_pos(&self, pos: &[T]) -> bool {
        is_valid_pos(self, pos)
    }

    fn first_pos_(&self, pos: &mut [T]) -> bool {
        first_pos_(self, pos)
    }

    fn first_pos(&self) -> Option<Self::O> {
        let mut pos = Vec::<T>::empty(self.len());
        if self.first_pos_(&mut pos) { Some(pos) } else { None }
    }

    fn last_pos_(&self, pos: &mut [T]) -> bool {
        last_pos_(self, pos)
    }

    fn last_pos(&self) -> Option<Self::O> {
        let mut pos = Vec::<T>::empty(self.len());
        if self.last_pos_(&mut pos) { Some(pos) } else { None }
    }
}


impl<T: Rem + Div + Mul + Add + Copy + Zero + One, const DIM: usize> Shape<T> for [T; DIM] {
    type O = Self;

    fn ndim(&self) -> usize {
        self.len()
    }

    fn size(&self) -> T {
        self.product()
    }

    fn pos(&self, index: T) -> Self {
        let mut pos: [T; DIM] = empty();
        pos_(self, index, &mut pos);
        pos
    }

    fn idx(&self, pos: &[T]) -> T {
        idx(self, pos)
    }

    fn prev_pos_(&self, pos: &mut Self) -> bool {
        prev_pos_(self, pos)
    }

    fn prev_pos(&self) -> Option<Self::O> {
        let mut pos = Self::empty();
        if self.prev_pos_(&mut pos) { Some(pos) } else { None }
    }

    fn next_pos_(&self, pos: &mut Self) -> bool {
        next_pos_(self, pos)
    }

    fn next_pos(&self) -> Option<Self::O> {
        let mut pos = Self::empty();
        if self.next_pos_(&mut pos) { Some(pos) } else { None }
    }

    fn is_valid_pos(&self, pos: &Self) -> bool {
        is_valid_pos(self, pos)
    }

    fn first_pos_(&self, pos: &mut Self) -> bool {
        first_pos_(self, pos)
    }

    fn first_pos(&self) -> Option<Self::O> {
        let mut pos = Self::empty();
        if self.first_pos_(&mut pos) { Some(pos) } else { None }
    }

    fn last_pos_(&self, pos: &mut Self) -> bool {
        last_pos_(self, pos)
    }

    fn last_pos(&self) -> Option<Self::O> {
        let mut pos = Self::empty();
        if self.last_pos_(&mut pos) { Some(pos) } else { None }
    }
}

pub fn is_valid_pos<T: Ord + Zero>(shape: &[T], pos: &[T]) -> bool {
    shape.len() == pos.len() && pos.all_ge_scalar(T::zero()) && shape.all_gt(pos)
}

pub fn last_pos_<T: Copy + Zero + One + Ord + Sub>(shape: &[T], pos: &mut [T]) -> bool {
    if shape.all_gt_scalar(T::zero()) {
        pos.iter_mut().zip(shape.iter()).for_each(|(p, s)| *p = s - T::one());
        true
    } else {
        false
    }
}

pub fn first_pos_<T: Copy + Ord + Zero>(shape: &[T], pos: &mut [T]) -> bool {
    if shape.all_gt_scalar(T::zero()) {
        pos.fill(T::zero());
        true
    } else {
        false
    }
}

pub fn prev_pos_<T: Copy + Zero + One + Ord + Sub>(shape: &[T], pos: &mut [T]) -> bool {
    for i in 0..shape.len() {
        if pos[i] > T::zero() {
            pos[i] = pos[i] - T::one();
            return true;
        } else {
            pos[i] = shape[i] - T::one();
        }
    }
    false
}

pub fn next_pos_<T: Copy + Ord + Add + Zero + One>(shape: &[T], pos: &mut [T]) -> bool {
    for i in 0..shape.len() {
        if pos[i] < shape[i] {
            pos[i] = pos[i] + T::one();
            return true;
        } else {
            pos[i] = T::zero();
        }
    }
    false
}

pub fn pos_<T: Rem + Div + Copy>(shape: &[T], mut index: T, pos: &mut [T]) {
    let original_index = index;

    for dim in (0..shape.len()).rev() {
        let dim_size = shape[dim];
        let coord = index % dim_size;
        index = index / dim_size;
        pos[dim] = coord;
    }
    debug_assert_eq!(index, T::zero(), "Index {:?} is out of bounds for shape {:?}", original_index, shape);
}

pub fn idx<T: Mul + Add + Copy>(shape: &[T], pos: &[T]) -> T {
    let mut idx = T::zero();
    for dim in 0..shape.len() {
        let dim_size = shape[dim];
        debug_assert!(pos[dim] < dim_size, "at dim={:?}, position=={:?} >= shape=={:?}", dim, pos, shape);
        idx = idx * dim_size + pos[dim];
    }
    idx
}


pub struct PosIter<'a, T, P: Shape<T>> {
    pos: &'a mut P,
    shape: &'a P,
}

impl<'a, T, P: Shape<T>> Iterator for PosIter<'a, T, P> {
    type Item = &'a P;

    fn next(&mut self) -> Option<Self::Item> {
        if self.shape.next_pos_(self.pos) { Some(self.pos) } else { None }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let s = (self.pos.product()..self.shape.product()).len();
        (s, Some(s))
    }

    fn count(self) -> usize {
        self.shape.size() - self.shape.idx(self.pos)
    }

    fn last(self) -> Option<Self::Item> {
        if self.shape.last_pos_(self.pos) { Some(self.pos) } else { None }
    }
}

impl<'a, T, P: Shape<T>> ExactSizeIterator for PosIter<'a, T, P> {}


pub struct PosIterRev<'a, T, P: Shape<T>> {
    pos: &'a mut P,
    shape: &'a P,
}


impl<'a, T, P: Shape<T>> Iterator for PosIterRev<'a, T, P> {
    type Item = &'a P;

    fn next(&mut self) -> Option<Self::Item> {
        if self.shape.prev_pos_(self.pos) { Some(self.pos) } else { None }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let s = (self.pos.product()..self.shape.product()).len();
        (s, Some(s))
    }

    fn count(self) -> usize {
        self.shape.idx(self.pos)
    }

    fn last(self) -> Option<Self::Item> {
        if self.shape.first_pos_(self.pos) { Some(self.pos) } else { None }
    }
}

impl<'a, T, P: Shape<T>> ExactSizeIterator for PosIterRev<'a, T, P> {}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test1() {
        let s = [4, 3];
        assert_eq!(s.idx(&[0, 0]), 0);
        assert_eq!(s.idx(&[0, 1]), 1);
        assert_eq!(s.idx(&[0, 2]), 2);
        assert_eq!(s.pos(0), [0, 0]);
        assert_eq!(s.pos(1), [0, 1]);
        assert_eq!(s.pos(2), [0, 2]);
        assert_eq!(s.pos(3), [1, 0]);
        assert_eq!(s.pos(4), [1, 1]);
        assert_eq!(s.pos(5), [1, 2]);
        for i in 0..(3 * 4) {
            let p = s.pos(i);
            assert_eq!(s.idx(&p), i, "{}=={:?}", i, p);
        }
    }

    #[test]
    fn test2() {
        let s = [3, 4];
        for x in 0..3 {
            for y in 0..4 {
                assert_eq!(s.pos(s.idx(&[x, y])), [x, y]);
            }
        }
    }

    #[test]
    fn test3() {
        let s = [6, 4, 3];
        assert_eq!(s.idx(&[2, 0, 0]), 24);
        assert_eq!(s.idx(&[3, 0, 1]), 37);
        assert_eq!(s.idx(&[4, 0, 2]), 50);
        assert_eq!(s.pos(0), [0, 0, 0]);
        assert_eq!(s.pos(1), [0, 0, 1]);
        assert_eq!(s.pos(2), [0, 0, 2]);
        assert_eq!(s.pos(3), [0, 1, 0]);
        assert_eq!(s.pos(4), [0, 1, 1]);
        assert_eq!(s.pos(5), [0, 1, 2]);
        for i in 0..s.size() {
            let p = s.pos(i);
            assert_eq!(s.idx(&p), i, "{}=={:?}", i, p);
        }
    }

    #[test]
    fn test4() {
        let s = [6u32, 3, 4];
        for x in 0..s[2] {
            for y in 0..s[1] {
                for z in 0..s[0] {
                    assert_eq!(s.pos(s.idx(&[z, y, x])), [z, y, x]);
                }
            }
        }
    }
}


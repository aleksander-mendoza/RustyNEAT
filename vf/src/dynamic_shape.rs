use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, RangeBounds, Range};
use std::mem::MaybeUninit;
use crate::vector_field::{VectorField, VectorFieldNum};
use std::collections::{Bound, HashSet};
use num_traits::{Num, One, Zero};
use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Less};
use std::iter::{FromIterator, FlatMap};
use crate::init::empty;

pub trait Shape<T> {
    fn pos(&self, index: T) -> Vec<T>;
    fn idx(&self, pos: &[T]) -> T;
    fn pos_iter(&self) -> PosIter<T>;
}

impl<T: Rem + Div + Mul + Add + Copy> Shape<T> for [T] {
    fn pos(&self, index: T) -> Vec<T> {
        pos(self, index)
    }

    fn idx(&self, pos: [T; DIM]) -> T {
        idx(self, pos)
    }

    fn pos_iter(&self) -> PosIter<T> {
        pos_iter(self)
    }
}

pub fn pos<T: Rem + Div + Copy>(shape: &[T], mut index: T) -> Vec<T> {
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

pub fn idx<T: Mul + Add + Copy>(shape: &[T; DIM], pos: &[T]) -> T {
    let mut idx = T::zero();
    for dim in 0..DIM {
        let dim_size = shape[dim];
        debug_assert!(pos[dim] < dim_size, "at dim={}, position=={:?} >= shape=={:?}", dim, pos, shape);
        idx = idx * dim_size + pos[dim];
    }
    idx
}


pub struct PosIter<'a, T: Zero + One + Add + Copy + Ord> {
    pos: Vec<T>,
    shape: &'a [T],
}

impl<'a, T: Zero + One + Add + Copy> Iterator for PosIter<'a, T> {
    type Item = [T];

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
}

pub fn pos_iter<T: Zero + One + Add + Copy>(shape: &[T]) -> PosIter<T> {
    PosIter { pos: vec![T::zero(); DIM], shape }
}
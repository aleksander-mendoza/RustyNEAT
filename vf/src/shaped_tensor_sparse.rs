use std::ops::{Add, Mul};
use crate::init::{empty, InitEmptyWithCapacity};
use num_traits::{MulAdd, Zero};
use crate::static_shape::Shape;

pub fn dot1<I: num_traits::AsPrimitive<usize>, T: Add + Copy + Zero>(lhs: &[I], rhs: &[[T; X]; Z]) -> [T; X] {
    let mut o = empty();
    for x in 0..X {
        o[x] = lhs.iter().fold(T::zero(), |sum, z| sum + rhs[z.as_()][x])
    }
    o
}

pub fn dot0<I: num_traits::AsPrimitive<usize>, T: Add + Copy + Zero>(lhs: &[I], rhs: &[T]) -> T {
    lhs.iter().fold(T::zero(), |sum, z| sum + rhs[z])
}

pub fn inner_product<T: Mul + Add + Copy + Zero>(lhs: &[T], rhs: &[T]) -> T {
    dot0(lhs, rhs)
}
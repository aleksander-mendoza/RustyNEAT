use std::ops::{Add, Mul};
use crate::init::empty;
use num_traits::{MulAdd, Zero};


pub trait DotSparse<Rhs> {
    type O;
    fn dot_sparse(&self, other: &Rhs) -> Self::O;
}

impl<I: num_traits::AsPrimitive<usize>, T: Add + Copy + Zero, const Z: usize> DotSparse<[T; Z]> for [I] {
    type O = T;

    fn dot_sparse(&self, other: &[T; Z]) -> Self::O {
        dot_sparse0(self, other)
    }
}

impl<I: num_traits::AsPrimitive<usize>, T: Add + Copy + Zero, const X: usize, const Z: usize> DotSparse<[[T; X]; Z]> for [I] {
    type O = [T; X];

    fn dot_sparse(&self, other: &[T; Z]) -> Self::O {
        dot_sparse1(self, other)
    }
}

pub fn dot_sparse1<I: num_traits::AsPrimitive<usize>, T: Add + Copy + Zero, const X: usize, const Z: usize>(lhs: &[I], rhs: &[[T; X]; Z]) -> [T; X] {
    let mut o = empty();
    for x in 0..X {
        o[x] = lhs.iter().fold(T::zero(), |sum, z| sum + rhs[z.as_()][x])
    }
    o
}

pub fn dot_sparse0<I: num_traits::AsPrimitive<usize>, T: Add + Copy + Zero, const Z: usize>(lhs: &[I], rhs: &[T; Z]) -> T {
    lhs.iter().fold(T::zero(), |sum, z| sum + rhs[z])
}

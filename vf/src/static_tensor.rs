use std::ops::{Add, Mul};
use crate::init::empty;
use num_traits::{MulAdd, Zero};

pub trait Dot<Rhs> {
    type O;
    fn dot(&self, other: &Rhs) -> Self::O;
}

impl<T: Mul + Add + Copy + Zero, const X: usize, const Y: usize, const Z: usize, const W: usize> Dot<[[[T; X]; Z]; W]> for [[[T; Z]; Y]; W] {
    type O = [[[T; X]; Y]; W];

    fn dot(&self, other: &[[[T; X]; Z]; W]) -> Self::O {
        dot3(self, other)
    }
}


impl<T: Mul + Add + Copy + Zero, const X: usize, const Y: usize, const Z: usize> Dot<[[T; X]; Z]> for [[T; Z]; Y] {
    type O = [[T; X]; Y];

    fn dot(&self, other: &[[T; X]; Z]) -> Self::O {
        dot2(self, other)
    }
}


impl<T: Mul + Add + Copy + Zero, const X: usize, const Z: usize> Dot<[[T; X]; Z]> for [T; Z] {
    type O = [T; X];

    fn dot(&self, other: &[[T; X]; Z]) -> Self::O {
        dot1(self, other)
    }
}


impl<T: Mul + Add + Copy + Zero, const Z: usize> Dot<[T; Z]> for [T; Z] {
    type O = T;

    fn dot(&self, other: &[T; Z]) -> Self::O {
        dot0(self, other)
    }
}


pub fn dot3<T: Mul + Add + Copy + Zero, const X: usize, const Y: usize, const Z: usize, const W: usize>(lhs: &[[[T; Z]; Y]; W], rhs: &[[[T; X]; Z]; W]) -> [[[T; X]; Y]; W] {
    let mut o = empty();
    for w in 0..W {
        for x in 0..X {
            for y in 0..Y {
                o[w][y][x] = (0..Z).fold(T::zero(), |sum, z| sum + lhs[w][y][z] * rhs[w][z][x]);
            }
        }
    }
    o
}

pub fn dot2<T: Mul + Add + Copy + Zero, const X: usize, const Y: usize, const Z: usize>(lhs: &[[T; Z]; Y], rhs: &[[T; X]; Z]) -> [[T; X]; Y] {
    let mut o = empty();
    for x in 0..X {
        for y in 0..Y {
            o[y][x] = (0..Z).fold(T::zero(), |sum, z| sum + lhs[y][z] * rhs[z][x]);
        }
    }
    o
}

pub fn dot1<T: Mul + Add + Copy + Zero, const X: usize, const Z: usize>(lhs: &[T; Z], rhs: &[[T; X]; Z]) -> [T; X] {
    let mut o = empty();
    for x in 0..X {
        o[x] = (0..Z).fold(T::zero(), |sum, z| sum + lhs[z] * rhs[z][x]);
    }
    o
}

pub fn dot0<T: Mul + Add + Copy + Zero, const Z: usize>(lhs: &[T; Z], rhs: &[T; Z]) -> T {
    (0..Z).fold(T::zero(), |sum, z| sum + lhs[z] * rhs[z])
}

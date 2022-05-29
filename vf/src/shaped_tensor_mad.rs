use std::ops::{Add, Mul};
use crate::init::{empty, InitEmptyWithCapacity};
use num_traits::{MulAdd, Zero};
use crate::shape::Shape;

pub fn dot3<T: MulAdd + Copy + Zero>(lhs: &[T], shape_lhs: &[usize; 3], rhs: &[T], shape_rhs: &[usize; 3]) -> (Vec<T>, [usize; 3]) {
    // [Z, Y, W] == shape_lhs;
    // [X, Z, W] == shape_rhs;
    assert_eq!(shape_lhs[0], shape_rhs[1]);
    assert_eq!(shape_lhs[2], shape_rhs[2]);
    let X = shape_rhs[0];
    let Y = shape_lhs[1];
    let Z = shape_lhs[0];
    let W = shape_rhs[2];
    let mut o = Vec::empty(X * Y * W);
    for w in 0..W {
        for x in 0..X {
            for y in 0..Y {
                o[w][y][x] = (0..Z).fold(T::zero(), |sum, z| lhs[shape_lhs.idx(&[z, y, w])].mul_add(rhs[shape_rhs.idx(&[w, z, x])], sum));
            }
        }
    }
    (o, [X, Y, W])
}

pub fn dot2<T: MulAdd + Copy + Zero>(lhs: &[T], shape_lhs: &[usize; 2], rhs: &[T], shape_rhs: &[usize; 2]) -> (Vec<T>, [usize; 2]) {
    // shape_lhs == [Z, Y]
    // shape_rhs == [X, Z]
    assert_eq!(shape_lhs[0], shape_rhs[1]);
    let X = shape_rhs[0];
    let Y = shape_lhs[1];
    let Z = shape_lhs[0];
    let mut o = Vec::empty(X * Y);
    for x in 0..X {
        for y in 0..Y {
            o[y][x] = (0..Z).fold(T::zero(), |sum, z| lhs[shape_lhs.idx(&[z, y])].mul_add(rhs[shape_rhs.idx(&[x, z])], sum));
        }
    }
    (o, [X, Y])
}

pub fn dot1<T: MulAdd + Copy + Zero>(lhs: &[T], rhs: &[T], shape_rhs: &[usize; 2]) -> Vec<T> {
    assert_eq!(lhs.len(), shape_rhs[1]);
    let mut o = Vec::empty(shape_rhs[0]);
    for x in 0..o.len() {
        o[x] = lhs.iter().enumerate().fold(T::zero(), |sum, (z, l)| l.mul_add(rhs[shape_rhs.idx(&[x, z])], sum));
    }
    o
}

pub fn dot0<T: MulAdd + Copy + Zero>(lhs: &[T], rhs: &[T]) -> T {
    lhs.iter().zip(rhs.iter()).fold(T::zero(), |sum, (l, r)| l.mul_add(r, sum))
}

pub fn inner_product<T: MulAdd + Copy + Zero>(lhs: &[T], rhs: &[T]) -> T {
    dot0(lhs, rhs)
}

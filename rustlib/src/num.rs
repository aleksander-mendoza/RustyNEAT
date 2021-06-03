use rand::prelude::Distribution;
use std::fmt::Display;
use crate::activations::ActFn;

pub trait Num: num_traits::Num + Copy + Display {
    fn act_fn(f:&ActFn)->fn(Self)->Self;
    fn random() -> Self;
}

impl Num for f64 {
    fn act_fn(f: &ActFn) -> fn(Self) -> Self {
        f.fn64()
    }
    fn random() -> Self{
        rand::random()
    }
}

impl Num for f32 {
    fn act_fn(f: &ActFn) -> fn(Self) -> Self {
        f.fn32()
    }
    fn random() -> Self{
        rand::random()
    }
}


// impl Num for i64 {}
//
// impl Num for i32 {}
//
// impl Num for i16 {}
//
// impl Num for u64 {}
//
// impl Num for u32 {}
//
// impl Num for u16 {}
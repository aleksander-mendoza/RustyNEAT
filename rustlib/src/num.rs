use rand::prelude::Distribution;
use crate::activations::{ALL_F64, ALL_STR, ALL_F32};
use std::fmt::Display;

pub trait Num: num_traits::Num + Copy + Display{
    fn find_name_by_fn_ptr(fn_ptr:fn(Self)->Self) -> &'static str;
}

impl Num for f64 {
    fn find_name_by_fn_ptr(fn_ptr: fn(Self) -> Self) -> &'static str {
        ALL_F64.iter().position(|&f|f==fn_ptr).map(|p|ALL_STR[p]).unwrap_or("CUSTOM")
    }
}

impl Num for f32 {
    fn find_name_by_fn_ptr(fn_ptr: fn(Self) -> Self) -> &'static str {
        ALL_F32.iter().position(|&f|f==fn_ptr).map(|p|ALL_STR[p]).unwrap_or("CUSTOM")
    }
}
//
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
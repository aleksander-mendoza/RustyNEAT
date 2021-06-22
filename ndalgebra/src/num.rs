use std::fmt::Display;
use ocl::OclPrm;
use ocl::core::ffi::cl_half;
use std::num::ParseIntError;
use num_traits::{Zero, One};
use std::iter::Product;

pub trait Num: OclPrm + Copy + Display + std::ops::AddAssign + Zero<Output=Self> + One<Output=Self> + Product<Self>  {
    const OPENCL_TYPE_STR: &'static str;
    const IS_FLOAT: bool = false;
    const IS_INT: bool = false;
}

impl Num for u8 {
    const OPENCL_TYPE_STR: &'static str = "uchar";
}

impl Num for u16 {
    const OPENCL_TYPE_STR: &'static str = "ushort";
}

impl Num for u32 {
    const OPENCL_TYPE_STR: &'static str = "uint";
}

impl Num for u64 {
    const OPENCL_TYPE_STR: &'static str = "ulong";
}

impl Num for i8 {
    const OPENCL_TYPE_STR: &'static str = "char";
}

impl Num for i16 {
    const OPENCL_TYPE_STR: &'static str = "short";
}

impl Num for i32 {
    const OPENCL_TYPE_STR: &'static str = "int";
}

impl Num for i64 {
    const OPENCL_TYPE_STR: &'static str = "long";
}

impl Num for f32 {
    const OPENCL_TYPE_STR: &'static str = "float";
    const IS_FLOAT: bool = true;
    const IS_INT: bool = false;
}

impl Num for usize {
    const OPENCL_TYPE_STR: &'static str = "size_t";
}

impl Num for isize {
    const OPENCL_TYPE_STR: &'static str = "ptrdiff_t";
}
use std::fmt::Display;
use ocl::OclPrm;
use ocl::core::ffi::cl_half;
use std::num::ParseIntError;
use num_traits::{Zero, One};

pub trait Num: OclPrm + Copy + Display + std::ops::AddAssign + Zero<Output=Self> + One<Output=Self> {
    fn opencl_type_str()->&'static str;
    const IS_FLOAT: bool = false;
    const IS_INT: bool = false;
}

impl Num for u8{
    fn opencl_type_str() -> &'static str {
        "uchar"
    }
}

impl Num for u16{
    fn opencl_type_str() -> &'static str {
        "ushort"
    }
}

impl Num for u32{
    fn opencl_type_str() -> &'static str {
        "uint"
    }
}

impl Num for u64{
    fn opencl_type_str() -> &'static str {
        "ulong"
    }
}

impl Num for i8{
    fn opencl_type_str() -> &'static str {
        "char"
    }
}

impl Num for i16{
    fn opencl_type_str() -> &'static str {
        "short"
    }
}

impl Num for i32{
    fn opencl_type_str() -> &'static str {
        "int"
    }
}

impl Num for i64{
    fn opencl_type_str() -> &'static str {
        "long"
    }
}

impl Num for f32{
    fn opencl_type_str() -> &'static str {
        "float"
    }
    const IS_FLOAT:bool = true;
    const IS_INT:bool = false;
}

impl Num for usize{
    fn opencl_type_str() -> &'static str {
        "size_t"
    }
}

impl Num for isize{
    fn opencl_type_str() -> &'static str {
        "ptrdiff_t"
    }
}
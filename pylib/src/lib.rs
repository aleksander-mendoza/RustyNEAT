#![feature(option_result_contains)]

mod slice_box;
mod py_ndalgebra;
mod py_rustyneat;
mod py_ocl;
mod py_htm;
mod py_ecc;
mod util;

use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol};
use pyo3::PyResult;
use rusty_neat_core::{cppn, neat, gpu};
use std::collections::HashSet;
use rusty_neat_core::activations::{STR_TO_IDX, ALL_ACT_FN};
use pyo3::exceptions::PyValueError;
use rusty_neat_core::cppn::CPPN;
use std::iter::FromIterator;
use pyo3::types::PyString;
use rusty_neat_core::num::Num;
use rusty_neat_core::gpu::{FeedForwardNetOpenCL, FeedForwardNetPicbreeder, FeedForwardNetSubstrate};
use pyo3::basic::CompareOp;
use numpy::{PyReadonlyArrayDyn, PyArrayDyn, IntoPyArray, PyArray, PY_ARRAY_API, npyffi, Element,ToNpyDims};
use numpy::npyffi::{NPY_ORDER, npy_intp, NPY_ARRAY_WRITEABLE};
use std::os::raw::c_int;
use crate::py_rustyneat::{CPPN32, Neat32, FeedForwardNet32, FeedForwardNetSubstrate32, FeedForwardNetOpenCL32, FeedForwardNetPicbreeder32};
use ndalgebra::mat::MatError;


#[pymodule]
pub fn ndalgebra(py: Python, m: &PyModule) -> PyResult<()> {
    use py_ndalgebra::*;
    m.add_class::<DynMat>()?;
    m.add_class::<DType>()?;
    m.add("u8", U8)?;
    m.add("u16", U16)?;
    m.add("u32", U32)?;
    m.add("u64", U64)?;
    m.add("i8", I8)?;
    m.add("i16", I16)?;
    m.add("i32", I32)?;
    m.add("i64", I64)?;
    m.add("f32", F32)?;
    m.add("uint8", U8)?;
    m.add("uint16", U16)?;
    m.add("uint32", U32)?;
    m.add("uint64", U64)?;
    m.add("int8", I8)?;
    m.add("int16", I16)?;
    m.add("int32", I32)?;
    m.add("int64", I64)?;
    m.add("float32", F32)?;
    m.add("byte", I8)?;
    m.add("ubyte", U8)?;
    m.add("short", I16)?;
    m.add("ushort", U16)?;
    m.add("intc", I32)?;
    m.add("uintc", U32)?;
    m.add("longlong", I64)?;
    m.add("ulonglong", U64)?;
    m.add("single", F32)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(exp2, m)?)?;
    m.add_function(wrap_pyfunction!(exp10, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(log2, m)?)?;
    m.add_function(wrap_pyfunction!(log10, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    Ok(())
}

#[pymodule]
pub fn ecc(py: Python, m: &PyModule) -> PyResult<()> {
    use py_ecc::*;
    m.add_class::<CpuEccDense>()?;
    m.add_class::<CpuEccDenseUInt>()?;
    m.add_class::<CpuEccSparse>()?;
    m.add_class::<CpuEccMachine>()?;
    m.add_class::<CpuEccMachineUInt>()?;
    m.add_class::<OclEccMachine>()?;
    m.add_class::<OclEccDense>()?;
    m.add_class::<OclEccSparse>()?;
    // m.add_class::<CpuHOM>()?;
    Ok(())
}

#[pymodule]
pub fn htm(py: Python, m: &PyModule) -> PyResult<()> {
    use py_htm::*;
    m.add_function(wrap_pyfunction!(bitset_from_bools, m)?)?;
    m.add_function(wrap_pyfunction!(bitset_from_indices, m)?)?;
    m.add_function(wrap_pyfunction!(conv_out_size, m)?)?;
    m.add_function(wrap_pyfunction!(conv_stride, m)?)?;
    m.add_function(wrap_pyfunction!(conv_compose, m)?)?;
    m.add_function(wrap_pyfunction!(conv_in_size, m)?)?;
    m.add_function(wrap_pyfunction!(vote, m)?)?;
    m.add_function(wrap_pyfunction!(vote_conv2d, m)?)?;
    m.add_function(wrap_pyfunction!(vote_conv2d_transpose, m)?)?;
    m.add_class::<BitsEncoder>()?;
    m.add_class::<CpuBitset>()?;
    m.add_class::<CpuInput>()?;
    m.add_class::<OclBitset>()?;
    m.add_class::<OclInput>()?;
    m.add_class::<Population>()?;
    m.add_class::<Neuron>()?;
    m.add_class::<Segment>()?;
    m.add_class::<CpuSDR>()?;
    m.add_class::<CategoricalEncoder>()?;
    m.add_class::<EncoderBuilder>()?;
    m.add_class::<IntegerEncoder>()?;
    m.add_class::<FloatEncoder>()?;
    m.add_class::<DayOfYearEncoder>()?;
    m.add_class::<DayOfWeekEncoder>()?;
    m.add_class::<DayOfMonthEncoder>()?;
    m.add_class::<BoolEncoder>()?;
    m.add_class::<TimeOfDayEncoder>()?;
    m.add_class::<IsWeekendEncoder>()?;
    m.add_class::<CircularIntegerEncoder>()?;
    m.add_class::<OclSDR>()?;
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn rusty_neat(py: Python, m: &PyModule) -> PyResult<()> {
    use py_rustyneat::*;
    use py_ocl::*;
    m.add_function(wrap_pyfunction!(random_activation_fn, m)?)?;
    m.add_function(wrap_pyfunction!(activation_functions, m)?)?;
    m.add_function(wrap_pyfunction!(devices, m)?)?;
    m.add_function(wrap_pyfunction!(make_new_context, m)?)?;
    m.add_function(wrap_pyfunction!(make_gpu_context, m)?)?;
    m.add_function(wrap_pyfunction!(make_cpu_context, m)?)?;
    m.add_wrapped(wrap_pymodule!(ndalgebra))?;
    m.add_wrapped(wrap_pymodule!(htm))?;
    m.add_wrapped(wrap_pymodule!(ecc))?;
    m.add_class::<py_ocl::Context>()?;
    m.add_class::<CPPN32>()?;
    m.add_class::<CPPN32>()?;
    m.add_class::<Neat32>()?;
    m.add_class::<FeedForwardNet32>()?;
    m.add_class::<FeedForwardNetSubstrate32>()?;
    m.add_class::<FeedForwardNetOpenCL32>()?;
    m.add_class::<FeedForwardNetPicbreeder32>()?;
    Ok(())
}


pub fn ocl_err_to_py_ex(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat0() -> Result<(), String> {
        Ok(())
    }
}

#![feature(option_result_contains)]

mod slice_box;
mod py_ndalgebra;
mod py_rustyneat;
mod py_ocl;
mod py_envs;

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
use rusty_neat_core::envs::evol::{AGENT_ATTRIBUTES, LIDAR_ATTRIBUTES};
use crate::py_rustyneat::{CPPN32, Neat32, FeedForwardNet32, FeedForwardNetSubstrate32, FeedForwardNetOpenCL32, FeedForwardNetPicbreeder32};
use crate::py_envs::Evol;
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
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    Ok(())
}

#[pymodule]
pub fn envs(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Evol>()?;
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
    m.add_wrapped(wrap_pymodule!(envs))?;
    m.add_wrapped(wrap_pymodule!(ndalgebra))?;
    m.add_class::<py_ocl::NeatContext>()?;
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

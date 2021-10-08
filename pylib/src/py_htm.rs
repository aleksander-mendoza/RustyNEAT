
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
use numpy::{PyReadonlyArrayDyn, PyArrayDyn, IntoPyArray, PyArray, PY_ARRAY_API, npyffi, Element, ToNpyDims};
use numpy::npyffi::{NPY_ORDER, npy_intp, NPY_ARRAY_WRITEABLE};
use std::os::raw::c_int;
use crate::ocl_err_to_py_ex;
use crate::py_ndalgebra::{DynMat, try_as_dtype};
use crate::py_ocl::Context;


#[pyclass]
pub struct CpuHTM {
    sdr: htm::cpu_htm::CpuHTM,
}

#[pyclass]
pub struct CpuHTM2 {
    sdr: htm::cpu_htm2::CpuHTM2,
}

#[pyclass]
pub struct OclSDR {
    sdr: htm::ocl_sdr::OclSDR,
}

#[pyclass]
pub struct OclHTM {
    htm: htm::ocl_htm::OclHTM,
}

#[pyclass]
pub struct OclHTM2 {
    htm: htm::ocl_htm2::OclHTM2,
}




#[pymethods]
impl OclSDR {

    #[new]
    pub fn new(context:&Context,active_neurons:Vec<u32>, max_active_neurons:usize) -> PyResult<Self> {
        htm::ocl_sdr::OclSDR::from_slice(context.c.clone(),active_neurons.as_slice(),max_active_neurons).map(|sdr|OclSDR{sdr}).map_err(ocl_err_to_py_ex)
    }

}

#[pyproto]
impl PyObjectProtocol for OclSDR {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.sdr.get().map_err(ocl_err_to_py_ex)?))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

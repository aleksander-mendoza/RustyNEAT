use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol, PyIterProtocol, PySequenceProtocol, PyTypeInfo, PyDowncastError, AsPyPointer, PyNumberProtocol};
use pyo3::PyResult;
use rusty_neat_core::{cppn, neat, gpu};
use std::collections::HashSet;
use rusty_neat_core::activations::{STR_TO_IDX, ALL_ACT_FN};
use pyo3::exceptions::PyValueError;
use rusty_neat_core::cppn::CPPN;
use std::iter::FromIterator;
use pyo3::types::{PyString, PyDateTime, PyDateAccess, PyTimeAccess, PyList, PyInt, PyFloat};
use rusty_neat_core::num::Num;
use rusty_neat_core::gpu::{FeedForwardNetOpenCL, FeedForwardNetPicbreeder, FeedForwardNetSubstrate};
use pyo3::basic::CompareOp;
use numpy::{PyReadonlyArrayDyn, PyArrayDyn, IntoPyArray, PyArray, PY_ARRAY_API, npyffi, Element, ToNpyDims, DataType};
use numpy::npyffi::{NPY_ORDER, npy_intp, NPY_ARRAY_WRITEABLE};
use std::os::raw::c_int;
use crate::ocl_err_to_py_ex;
use crate::py_ndalgebra::{DynMat, try_as_dtype};
use crate::py_ocl::Context;
use htm::{VectorFieldOne, Idx, SDR, w_idx, ConvShape, Shape2, HasConvShape, ConvShapeTrait, HasConvShapeMut, HasShape, EccLayerTrait, WNorm, D};
use std::time::SystemTime;
use std::ops::Deref;
use chrono::Utc;
use std::borrow::Borrow;
use std::io::BufWriter;
use std::fs::{File, OpenOptions};
use serde::Serialize;
use crate::util::*;
use crate::py_htm::{CpuSDR, OclSDR};
use rand::SeedableRng;
use pyo3::ffi::PyFloat_Type;
use crate::py_ecc_net::EccNet;

///
/// EccLayer(output: list[int], kernel: list[int], stride: list[int], in_channels: int, out_channels: int, k: int)
///
///
#[pyclass]
pub struct EccNetSDRs {
    pub(crate) ecc: htm::EccNetSDRs,
}

impl_save_load!(EccNetSDRs,ecc);


#[pymethods]
impl EccNetSDRs {

    #[text_signature = "(index)"]
    pub fn layer(&self, index:usize)->CpuSDR{
        CpuSDR{sdr:self.ecc.layers[index].clone()}
    }
    #[getter]
    pub fn get_len(&self) -> usize{
        self.ecc.len()
    }
    #[text_signature = "()"]
    pub fn clear(&mut self)  {
        self.ecc.clear()
    }
    #[new]
    pub fn new(length:Option<usize>) -> Self {
        Self { ecc: if let Some(length) = length {
            htm::EccNetSDRs::new(length)
        }else{
            htm::EccNetSDRs::default()
        }}
    }
}




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
use htm::{VectorFieldOne, Idx, SDR, w_idx, ConvShape, Shape2, HasConvShape, ConvShapeTrait, HasConvShapeMut, HasShape, EccLayerTrait, WNorm, D, Activity};
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
pub struct EccConfig {
    pub(crate) ecc: htm::EccConfig<D>,
}

impl_save_load!(EccConfig,ecc);


#[pymethods]
impl EccConfig {
    #[getter]
    pub fn get_cfg_entropy_maximisation(&self) -> D {
        self.ecc.entropy_maximisation
    }
    #[setter]
    pub fn set_cfg_entropy_maximisation(&mut self,v:D) {
        self.ecc.entropy_maximisation = v
    }
    #[getter]
    pub fn get_cfg_biased(&self) -> bool {
        self.ecc.biased
    }
    #[setter]
    pub fn set_cfg_biased(&mut self, v:bool) {
        self.ecc.biased=v
    }
    #[getter]
    pub fn get_cfg_activity(&self) -> String {
        format!("{:?}",self.ecc.activity)
    }
    #[setter]
    pub fn set_cfg_activity(&mut self, v:String){
        self.ecc.activity=Activity::from(v.as_str())
    }
    #[getter]
    pub fn get_w_norm(&self) -> String {
        format!("{:?}",self.ecc.w_norm)
    }
    #[setter]
    pub fn set_w_norm(&mut self,v:String)  {
        self.ecc.w_norm = WNorm::from(v.as_str())
    }
    #[new]
    pub fn new() -> Self {
        Self{ecc:htm::EccConfig::default()}
    }
    #[staticmethod]
    pub fn zero_order() -> Self {
        Self{ecc:htm::EccConfig::zero_order()}
    }
    #[staticmethod]
    pub fn l1() -> Self {
        Self{ecc:htm::EccConfig::l1()}
    }
    #[staticmethod]
    pub fn l2() -> Self {
        Self{ecc:htm::EccConfig::l2()}
    }
}




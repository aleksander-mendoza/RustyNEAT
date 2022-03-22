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
use htm::{VectorFieldOne, Idx, SDR, w_idx, ConvShape, Shape3, Shape2, HasConvShape, HasShape, D, TensorTrait};
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


///
/// WeightSums(shape, initial_value)
///
///
#[pyclass]
pub struct Tensor {
    pub ecc: htm::Tensor<D>,
}

#[pymethods]
impl Tensor {
    #[text_signature = "(index)"]
    pub fn get(&self, i:usize) -> f32 {
        self.ecc.get(i)
    }
    #[text_signature = "()"]
    pub fn as_list(&self) -> Vec<f32> {
        self.ecc.as_slice().to_vec()
    }
    #[text_signature = "(index, value)"]
    pub fn set(&mut self, i:usize, v:f32) {
        *self.ecc.get_mut(i) = v
    }
    #[text_signature = "(value)"]
    pub fn fill_all(&mut self, value:f32) {
        self.ecc.fill(value)
    }
    #[text_signature = "(value,sdr,parallel)"]
    pub fn fill(&mut self,value:f32, sdr:&CpuSDR) {
        sdr.sdr.fill_into(value,self.ecc.as_mut_slice())
    }
    #[new]
    pub fn new(shape: [Idx; 3],initial_value:Option<f32>) -> Self{
        Self { ecc: if let Some(initial_value) = initial_value{
            htm::Tensor::new(shape, initial_value)
        } else{
            unsafe{htm::Tensor::empty(shape)}
        }}
    }
}

#[pyproto]
impl PySequenceProtocol for Tensor {
    fn __len__(&self) -> usize {
        self.ecc.len()
    }
    fn __getitem__(&self, idx: isize) -> f32 {
        assert!(idx>=0);
        self.ecc.get(idx as usize)
    }

    fn __setitem__(&mut self, idx: isize, value: f32) {
        assert!(idx>=0);
        *self.ecc.get_mut(idx as usize) = value;
    }
}

impl_save_load!(Tensor,ecc);


#[pyproto]
impl PyObjectProtocol for Tensor {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.ecc))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}
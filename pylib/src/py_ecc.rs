use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol, PyIterProtocol, PySequenceProtocol, PyTypeInfo, PyDowncastError, AsPyPointer, PyNumberProtocol};
use pyo3::PyResult;
use rusty_neat_core::{cppn, neat, gpu};
use std::collections::HashSet;
use rusty_neat_core::activations::{STR_TO_IDX, ALL_ACT_FN};
use pyo3::exceptions::PyValueError;
use rusty_neat_core::cppn::CPPN;
use std::iter::FromIterator;
use pyo3::types::{PyString, PyDateTime, PyDateAccess, PyTimeAccess, PyList, PyInt};
use rusty_neat_core::num::Num;
use rusty_neat_core::gpu::{FeedForwardNetOpenCL, FeedForwardNetPicbreeder, FeedForwardNetSubstrate};
use pyo3::basic::CompareOp;
use numpy::{PyReadonlyArrayDyn, PyArrayDyn, IntoPyArray, PyArray, PY_ARRAY_API, npyffi, Element, ToNpyDims, DataType};
use numpy::npyffi::{NPY_ORDER, npy_intp, NPY_ARRAY_WRITEABLE};
use std::os::raw::c_int;
use crate::ocl_err_to_py_ex;
use crate::py_ndalgebra::{DynMat, try_as_dtype};
use crate::py_ocl::Context;
use htm::{EccSparse, EccLayer};
use std::time::SystemTime;
use std::ops::Deref;
use chrono::Utc;
use std::borrow::Borrow;
use std::io::BufWriter;
use std::fs::{File, OpenOptions};
use serde_pickle::SerOptions;
use serde::Serialize;
use crate::util::*;
use crate::py_htm::CpuSDR;

#[pyclass]
pub struct CpuEccDense {
    ecc: htm::EccDense,
}

#[pyclass]
pub struct CpuEccSparse {
    ecc: htm::EccSparse,
}

#[pymethods]
impl CpuEccDense {
    #[new]
    pub fn new(output: PyObject, kernel: PyObject, stride: PyObject, in_channels: usize, out_channels: usize, k: usize, ) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let output = arr2(py, &output)?;
        let kernel = arr2(py, &kernel)?;
        let stride = arr2(py, &stride)?;
        Ok(CpuEccDense { ecc: htm::EccDense::new(output, kernel, stride, in_channels, out_channels, k, &mut rand::thread_rng()) })
    }
    #[getter]
    pub fn get_in_shape(&self) -> Vec<usize> {
        self.ecc.in_shape().to_vec()
    }
    #[getter]
    pub fn get_out_shape(&self) -> Vec<usize> {
        self.ecc.out_shape().to_vec()
    }
    #[getter]
    pub fn get_k(&self) -> usize {
        self.ecc.k
    }
    #[setter]
    pub fn set_k(&mut self, k: usize) {
        self.ecc.k = k
    }
    #[getter]
    pub fn get_threshold(&self) -> f32 {
        self.ecc.threshold
    }
    #[setter]
    pub fn set_threshold(&mut self, threshold: f32) {
        self.ecc.threshold = threshold
    }

    #[getter]
    pub fn get_plasticity(&self) -> f32 {
        self.ecc.plasticity
    }
    #[setter]
    pub fn set_plasticity(&mut self, plasticity: f32) {
        self.ecc.plasticity = plasticity
    }

    #[text_signature = "(input_sdr)"]
    pub fn run(&self, input: &CpuSDR) -> CpuSDR {
        CpuSDR { sdr: self.ecc.run(&input.sdr) }
    }
    #[text_signature = "(input_pos,output_pos)"]
    pub fn w_index(&self, input_pos:PyObject, output_pos:PyObject)->PyResult<usize>{
        let gil = Python::acquire_gil();
        let py = gil.python();
        let input_pos = arr3(py, &input_pos)?;
        let output_pos = arr3(py, &output_pos)?;
        Ok(self.ecc.w_index(&input_pos,&output_pos))
    }

    #[text_signature = "(input_sdr,output_sdr)"]
    pub fn learn(&mut self, input: &CpuSDR, output:&CpuSDR){
        self.ecc.learn(&input.sdr, &output.sdr,&mut rand::thread_rng())
    }

    #[text_signature = "(file)"]
    pub fn save(&self, file: String) -> PyResult<()> {
        pickle(&self.ecc,file)
    }
}


#[pymethods]
impl CpuEccSparse {
    #[new]
    pub fn new(output: PyObject, kernel: PyObject, stride: PyObject, in_channels: usize, out_channels: usize, k: usize, connections_per_output:usize) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let output = arr2(py, &output)?;
        let kernel = arr2(py, &kernel)?;
        let stride = arr2(py, &stride)?;
        Ok(CpuEccSparse { ecc: htm::EccSparse::new(output, kernel, stride, in_channels, out_channels, k, connections_per_output,&mut rand::thread_rng()) })
    }
    #[getter]
    pub fn get_in_shape(&self) -> Vec<usize> {
        self.ecc.in_shape().to_vec()
    }
    #[getter]
    pub fn get_out_shape(&self) -> Vec<usize> {
        self.ecc.out_shape().to_vec()
    }
    #[getter]
    pub fn get_k(&self) -> usize {
        self.ecc.k
    }
    #[setter]
    pub fn set_k(&mut self, k: usize) {
        self.ecc.k = k
    }
    #[getter]
    pub fn get_threshold(&self) -> u16 {
        self.ecc.threshold
    }
    #[setter]
    pub fn set_threshold(&mut self, threshold: u16) {
        self.ecc.threshold = threshold
    }

    #[text_signature = "(input_sdr)"]
    pub fn run(&self, input: &CpuSDR) -> CpuSDR {
        CpuSDR { sdr: self.ecc.run(&input.sdr) }
    }
    #[text_signature = "(file)"]
    pub fn save(&self, file: String) -> PyResult<()> {
        pickle(&self.ecc,file)
    }

}

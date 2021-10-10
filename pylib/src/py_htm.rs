
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
use htm::CpuSDR;


#[pyclass]
pub struct CpuHTM {
    htm: htm::CpuHTM,
}

#[pyclass]
pub struct CpuHTM2 {
    htm: htm::CpuHTM2,
}

#[pyclass]
pub struct OclSDR {
    sdr: htm::OclSDR,
}

#[pyclass]
pub struct OclHTM {
    htm: htm::OclHTM,
}

#[pyclass]
pub struct OclHTM2 {
    htm: htm::OclHTM2,
}

const DEFAULT_PERMANENCE_THRESHOLD:f32 = 0.7;
const DEFAULT_PERMANENCE_DECREMENT:f32 = -0.01;
const DEFAULT_PERMANENCE_INCREMENT:f32 = 0.02;

#[pymethods]
impl CpuHTM {

    #[new]
    pub fn new(input_size:u32, minicolumns:u32,inputs_per_minicolumn:u32, n:u32) -> Self {
        CpuHTM{htm:htm::CpuHTM::new_globally_uniform_prob(input_size,minicolumns,n,DEFAULT_PERMANENCE_THRESHOLD,DEFAULT_PERMANENCE_DECREMENT,DEFAULT_PERMANENCE_INCREMENT,inputs_per_minicolumn)}
    }

    #[getter]
    fn get_n(&self) -> u32 {
        self.htm.n
    }

    #[setter]
    fn set_n(&mut self, n:u32) {
        self.htm.n = n
    }

    #[getter]
    fn get_permanence_decrement(&self) -> f32 {
        self.htm.permanence_decrement_increment[0]
    }

    #[setter]
    fn set_permanence_decrement(&mut self, permanence_decrement:f32) {
        self.htm.permanence_decrement_increment[0] = permanence_decrement
    }

    #[getter]
    fn get_permanence_increment(&self) -> f32 {
        self.htm.permanence_decrement_increment[1]
    }

    #[setter]
    fn set_permanence_increment(&mut self, permanence_increment:f32) {
        self.htm.permanence_decrement_increment[1] = permanence_increment
    }

    #[getter]
    fn get_permanence_threshold(&self) -> f32 {
        self.htm.permanence_threshold
    }

    #[setter]
    fn set_permanence_threshold(&mut self, permanence_threshold:f32) {
        self.htm.permanence_threshold = permanence_threshold
    }

    #[getter]
    fn get_max_overlap(&self) -> u32 {
        self.htm.max_overlap
    }

    #[setter]
    fn set_max_overlap(&mut self, max_overlap:u32) {
        self.htm.max_overlap = max_overlap
    }

    #[call]
    fn __call__(&mut self, sdr: Vec<u32>, learn:bool) -> Vec<u32> {
        self.htm.infer(&CpuSDR::from(sdr),learn).to_vec()
    }

    #[text_signature = "( /)"]
    fn to_htm2(&self) -> CpuHTM2 {
        CpuHTM2{htm:htm::CpuHTM2::from(&self.htm)}
    }

}



#[pymethods]
impl CpuHTM2 {

    #[new]
    pub fn new(input_size:u32, minicolumns:u32,inputs_per_minicolumn:u32, n:u32) -> Self {
        CpuHTM2{htm:htm::CpuHTM2::new_globally_uniform_prob(input_size,minicolumns,n,DEFAULT_PERMANENCE_THRESHOLD,DEFAULT_PERMANENCE_DECREMENT,DEFAULT_PERMANENCE_INCREMENT,inputs_per_minicolumn)}
    }

    #[getter]
    fn get_n(&self) -> u32 {
        self.htm.n
    }

    #[setter]
    fn set_n(&mut self, n:u32) {
        self.htm.n = n
    }

    #[getter]
    fn get_permanence_decrement(&self) -> f32 {
        self.htm.permanence_decrement_increment[0]
    }

    #[setter]
    fn set_permanence_decrement(&mut self, permanence_decrement:f32) {
        self.htm.permanence_decrement_increment[0] = permanence_decrement
    }

    #[getter]
    fn get_permanence_increment(&self) -> f32 {
        self.htm.permanence_decrement_increment[1]
    }

    #[setter]
    fn set_permanence_increment(&mut self, permanence_increment:f32) {
        self.htm.permanence_decrement_increment[1] = permanence_increment
    }

    #[getter]
    fn get_permanence_threshold(&self) -> f32 {
        self.htm.permanence_threshold
    }

    #[setter]
    fn set_permanence_threshold(&mut self, permanence_threshold:f32) {
        self.htm.permanence_threshold = permanence_threshold
    }

    #[getter]
    fn get_max_overlap(&self) -> u32 {
        self.htm.max_overlap
    }

    #[setter]
    fn set_max_overlap(&mut self, max_overlap:u32) {
        self.htm.max_overlap = max_overlap
    }

    #[call]
    fn __call__(&mut self, sdr: Vec<u32>, learn:bool) -> Vec<u32> {
        self.htm.infer2(&CpuSDR::from(sdr),learn).to_vec()
    }
}


#[pymethods]
impl OclSDR {

    #[new]
    pub fn new(context:&Context,max_active_neurons:usize) -> PyResult<Self> {
        htm::OclSDR::new(context.c.clone(),max_active_neurons).map(|sdr|OclSDR{sdr}).map_err(ocl_err_to_py_ex)
    }

    #[getter]
    fn get_max_active_neurons(&self) -> usize {
        self.sdr.max_active_neurons()
    }

    #[setter]
    fn set_active_neurons(&mut self, neuron_indices:Vec<u32>) -> PyResult<()>{
        self.sdr.set(neuron_indices.as_slice()).map_err(ocl_err_to_py_ex)
    }

    #[getter]
    fn get_active_neurons(&self) -> PyResult<Vec<u32>> {
        self.sdr.get().map_err(ocl_err_to_py_ex)
    }
}

#[pymethods]
impl OclHTM {

    #[new]
    pub fn new(context:&mut Context, htm:&CpuHTM) -> PyResult<Self> {
        htm::OclHTM::new(&htm.htm, context.compile_htm_program()?.clone()).map(|htm|OclHTM{htm}).map_err(ocl_err_to_py_ex)
    }

    #[getter]
    fn get_n(&self) -> u32 {
        self.htm.n
    }

    #[setter]
    fn set_n(&mut self, n:u32) {
        self.htm.n = n
    }

    #[getter]
    fn get_permanence_decrement(&self) -> f32 {
        self.htm.permanence_decrement_increment[0]
    }

    #[setter]
    fn set_permanence_decrement(&mut self, permanence_decrement:f32) {
        self.htm.permanence_decrement_increment[0] = permanence_decrement
    }

    #[getter]
    fn get_permanence_increment(&self) -> f32 {
        self.htm.permanence_decrement_increment[1]
    }

    #[setter]
    fn set_permanence_increment(&mut self, permanence_increment:f32) {
        self.htm.permanence_decrement_increment[1] = permanence_increment
    }

    #[getter]
    fn get_permanence_threshold(&self) -> f32 {
        self.htm.permanence_threshold
    }

    #[setter]
    fn set_permanence_threshold(&mut self, permanence_threshold:f32) {
        self.htm.permanence_threshold = permanence_threshold
    }

    #[getter]
    fn get_max_overlap(&self) -> u32 {
        self.htm.max_overlap
    }

    #[setter]
    fn set_max_overlap(&mut self, max_overlap:u32) {
        self.htm.max_overlap = max_overlap
    }

    #[call]
    fn __call__(&mut self, sdr: &OclSDR, learn:bool) -> PyResult<OclSDR> {
        self.htm.infer(&sdr.sdr,learn).map(|sdr|OclSDR{sdr}).map_err(ocl_err_to_py_ex)
    }
}

#[pymethods]
impl OclHTM2 {

    #[new]
    pub fn new(context:&mut Context, htm:&CpuHTM2) -> PyResult<Self> {
        htm::OclHTM2::new(&htm.htm, context.compile_htm_program2()?.clone()).map(|htm|OclHTM2{htm}).map_err(ocl_err_to_py_ex)
    }

    #[getter]
    fn get_n(&self) -> u32 {
        self.htm.n
    }

    #[setter]
    fn set_n(&mut self, n:u32) {
        self.htm.n = n
    }

    #[getter]
    fn get_permanence_decrement(&self) -> f32 {
        self.htm.permanence_decrement_increment[0]
    }

    #[setter]
    fn set_permanence_decrement(&mut self, permanence_decrement:f32) {
        self.htm.permanence_decrement_increment[0] = permanence_decrement
    }

    #[getter]
    fn get_permanence_increment(&self) -> f32 {
        self.htm.permanence_decrement_increment[1]
    }

    #[setter]
    fn set_permanence_increment(&mut self, permanence_increment:f32) {
        self.htm.permanence_decrement_increment[1] = permanence_increment
    }

    #[getter]
    fn get_permanence_threshold(&self) -> f32 {
        self.htm.permanence_threshold
    }

    #[setter]
    fn set_permanence_threshold(&mut self, permanence_threshold:f32) {
        self.htm.permanence_threshold = permanence_threshold
    }

    #[getter]
    fn get_max_overlap(&self) -> u32 {
        self.htm.max_overlap
    }

    #[setter]
    fn set_max_overlap(&mut self, max_overlap:u32) {
        self.htm.max_overlap = max_overlap
    }

    #[call]
    fn __call__(&mut self, sdr: &OclSDR, learn:bool) -> PyResult<OclSDR> {
        self.htm.infer2(&sdr.sdr,learn).map(|sdr|OclSDR{sdr}).map_err(ocl_err_to_py_ex)
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

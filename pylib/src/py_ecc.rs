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
use htm::{EccSparse, EccLayer, SparseOrDense};
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
use rand::SeedableRng;

///
/// CpuEccDense(output: list[int], kernel: list[int], stride: list[int], in_channels: int, out_channels: int, k: int)
///
///
#[pyclass]
pub struct CpuEccDense {
    ecc: htm::EccDense,
}

///
/// CpuEccSparse(output: list[int], kernel: list[int], stride: list[int], in_channels: int, out_channels: int, k: int, connections_per_output: int)
///
///
#[pyclass]
pub struct CpuEccSparse {
    ecc: htm::EccSparse,
}


///
/// CpuEccMachine(output: list[int], kernels: list[list[int]], strides: list[list[int]], channels: list[int], k: list[int], connections_per_output: list[int])
///
///
#[pyclass]
pub struct CpuEccMachine {
    ecc: htm::CpuEccMachine
}

#[pymethods]
impl CpuEccMachine {
    #[new]
    pub fn new(output: PyObject, kernels: Vec<PyObject>, strides: Vec<PyObject>, channels: Vec<usize>, k: Vec<usize>, connections_per_output: Vec<Option<usize>>) -> PyResult<Self> {
        let layers = kernels.len();
        if layers != strides.len() {
            return Err(PyValueError::new_err(format!("{}==len(kernels)!=len(strides)=={}", layers, strides.len())));
        }
        if layers != k.len() {
            return Err(PyValueError::new_err(format!("{}==len(kernels)!=len(k)=={}", layers, k.len())));
        }
        if layers != connections_per_output.len() {
            return Err(PyValueError::new_err(format!("{}==len(kernels)!=len(connections_per_output)=={}", layers, connections_per_output.len())));
        }
        if layers + 1 != channels.len() {
            return Err(PyValueError::new_err(format!("{}==len(kernels)+1!=len(channels)=={}", layers + 1, channels.len())));
        }
        let gil = Python::acquire_gil();
        let py = gil.python();
        let output = arr2(py, &output)?;
        let mut rng = rand::thread_rng();
        let kernels:PyResult<Vec<[usize;2]>> = kernels.into_iter().map(|k|arr2(py, &k)).collect();
        let strides:PyResult<Vec<[usize;2]>> = strides.into_iter().map(|k|arr2(py, &k)).collect();
        Ok(Self{ecc:htm::CpuEccMachine::new(output,&kernels?,&strides?,&channels,&k,&connections_per_output,&mut rng)})
    }

    #[text_signature = "(layer)"]
    pub fn get_in_shape(&self,layer:usize) -> Vec<usize> {
        self.ecc[layer].in_shape().to_vec()
    }
    #[text_signature = "(layer)"]
    pub fn get_out_shape(&self,layer:usize) -> Vec<usize> {
        self.ecc[layer].in_shape().to_vec()
    }
    #[text_signature = "(layer)"]
    pub fn get_kernel(&self,layer:usize) -> Vec<usize> {
        self.ecc[layer].kernel().to_vec()
    }
    #[text_signature = "(layer)"]
    pub fn get_in_channels(&self,layer:usize) -> usize {
        self.ecc[layer].in_channels()
    }
    #[text_signature = "(layer)"]
    pub fn get_out_channels(&self,layer:usize) -> usize {
        self.ecc[layer].out_channels()
    }
    #[text_signature = "(layer)"]
    pub fn get_in_volume(&self,layer:usize) -> usize {
        self.ecc[layer].in_volume()
    }
    #[text_signature = "(layer)"]
    pub fn get_out_volume(&self,layer:usize) -> usize {
        self.ecc[layer].out_volume()
    }
    #[text_signature = "(layer)"]
    pub fn get_stride(&self,layer:usize) -> Vec<usize> {
        self.ecc[layer].stride().to_vec()
    }
    #[text_signature = "(layer)"]
    pub fn get_sums(&self,py:Python,layer:usize) -> PyObject {
        match &self.ecc[layer]{
            SparseOrDense::Sparse(a) => a.sums.to_object(py),
            SparseOrDense::Dense(a) => a.sums.to_object(py)
        }
    }
    #[text_signature = "(layer)"]
    pub fn get_k(&self,layer:usize) -> usize {
        self.ecc[layer].k()
    }
    #[text_signature = "(layer, k)"]
    pub fn set_k(&mut self, layer:usize,k: usize) {
        self.ecc[layer].set_k(k)
    }
    #[text_signature = "(layer)"]
    pub fn get_threshold(&self,py: Python, layer:usize) -> PyObject {
        match &self.ecc[layer]{
            SparseOrDense::Sparse(a) => a.threshold.to_object(py),
            SparseOrDense::Dense(a) => a.threshold.to_object(py),
        }
    }
    #[text_signature = "(layer, threshold)"]
    pub fn set_threshold(&mut self, py: Python, layer:usize, threshold: PyObject) -> PyResult<()>{
        match &mut self.ecc[layer]{
            SparseOrDense::Sparse(a) => a.threshold = threshold.extract(py)?,
            SparseOrDense::Dense(a) => a.threshold = threshold.extract(py)?,
        }
        Ok(())
    }
    #[text_signature = "(layer)"]
    pub fn get_threshold_f32(&self, layer:usize) -> f32 {
        match &self.ecc[layer]{
            SparseOrDense::Sparse(a) => a.get_threshold_f32(),
            SparseOrDense::Dense(a) => a.get_threshold(),
        }
    }
    #[text_signature = "(layer, threshold)"]
    pub fn set_threshold_f32(&mut self, layer:usize, threshold: f32) -> PyResult<()>{
        match &mut self.ecc[layer]{
            SparseOrDense::Sparse(a) => a.set_threshold_f32(threshold),
            SparseOrDense::Dense(a) => a.set_threshold(threshold),
        }
        Ok(())
    }
    #[text_signature = "(input_sdr)"]
    pub fn learnable_paramemters(&self) -> usize {
        self.ecc.learnable_paramemters()
    }
    #[text_signature = "(input_sdr)"]
    pub fn run(&mut self, input: &CpuSDR){
        self.ecc.run(&input.sdr);
    }
    #[text_signature = "()"]
    pub fn learn(&mut self){
        self.ecc.learn();
    }
    #[text_signature = "()"]
    pub fn last_output_sdr(&self)->CpuSDR{
        CpuSDR{sdr:self.ecc.last_output_sdr().clone()}
    }
    #[text_signature = "()"]
    pub fn last_output_shape(&self)->Vec<usize>{
        self.ecc.last().unwrap().out_shape().to_vec()
    }
    #[text_signature = "()"]
    pub fn last_output_channels(&self)->usize{
        self.ecc.last().unwrap().out_channels()
    }
    #[text_signature = "()"]
    pub fn item(&self)->Option<u32>{
        let o = self.ecc.last_output_sdr();
        if o.is_empty(){None}else{Some(o.item())}
    }
    #[text_signature = "(layer_index)"]
    pub fn output_sdr(&self, layer_index:usize)->CpuSDR{
        CpuSDR{sdr:self.ecc.output_sdr(layer_index).clone()}
    }
    #[text_signature = "(layer_index)"]
    pub fn input_sdr(&self, layer_index:usize)->CpuSDR{
        CpuSDR{sdr:self.ecc.output_sdr(layer_index).clone()}
    }
    #[text_signature = "(file)"]
    pub fn save(&self, file: String) -> PyResult<()> {
        pickle(&self.ecc, file)
    }
}

#[pymethods]
impl CpuEccDense {
    #[new]
    pub fn new(output: PyObject, kernel: PyObject, stride: PyObject, in_channels: usize, out_channels: usize, k: usize) -> PyResult<Self> {
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
        self.ecc.k()
    }
    #[setter]
    pub fn set_k(&mut self, k: usize) {
        self.ecc.set_k(k)
    }
    #[getter]
    pub fn get_threshold(&self) -> f32 {
        self.ecc.get_threshold()
    }
    #[setter]
    pub fn set_threshold(&mut self, threshold: f32) {
        self.ecc.set_threshold(threshold)
    }

    #[getter]
    pub fn get_plasticity(&self) -> f32 {
        self.ecc.get_plasticity()
    }
    #[setter]
    pub fn set_plasticity(&mut self, plasticity: f32) {
        self.ecc.set_plasticity( plasticity)
    }

    #[getter]
    pub fn get_rand_seed(&self) -> usize {
        self.ecc.rand_seed
    }
    #[setter]
    pub fn set_rand_seed(&mut self, rand_seed: usize) {
        self.ecc.rand_seed = rand_seed
    }
    #[text_signature = "(input_sdr,output_sdr)"]
    pub fn run_in_place(&mut self, input: &CpuSDR, output:&mut CpuSDR){
        self.ecc.run_in_place(&input.sdr,&mut output.sdr)
    }
    #[text_signature = "(input_sdr)"]
    pub fn run(&mut self, input: &CpuSDR) -> CpuSDR {
        CpuSDR { sdr: self.ecc.run(&input.sdr) }
    }
    #[text_signature = "()"]
    pub fn min_activity(&self) -> u32 {
        self.ecc.min_activity()
    }
    #[text_signature = "()"]
    pub fn min_activity_f32(&self) -> f32 {
        self.ecc.min_activity_f32()
    }
    #[text_signature = "(output_idx)"]
    pub fn activity(&self, output_idx: usize) -> u32 {
        self.ecc.activity(output_idx)
    }
    #[text_signature = "(output_idx)"]
    pub fn activity_f32(&self, output_idx: usize) -> f32 {
        self.ecc.activity_f32(output_idx)
    }
    #[text_signature = "(input_pos,output_pos)"]
    pub fn w_index(&self, input_pos: PyObject, output_pos: PyObject) -> PyResult<usize> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let input_pos = arr3(py, &input_pos)?;
        let output_pos = arr3(py, &output_pos)?;
        Ok(self.ecc.w_index(&input_pos, &output_pos))
    }
    #[getter]
    pub fn get_sums(&self) -> Vec<u32> {
        self.ecc.sums.clone()
    }

    #[text_signature = "(input_sdr,output_sdr,rand_seed)"]
    pub fn learn(&mut self, input: &CpuSDR, output: &CpuSDR) {
        self.ecc.learn(&input.sdr, &output.sdr)
    }

    #[text_signature = "(file)"]
    pub fn save(&self, file: String) -> PyResult<()> {
        pickle(&self.ecc, file)
    }
    #[text_signature = "(output_neuron_idx)"]
    pub fn incoming_weight_sum(&self, output_neuron_idx: usize) -> u32 {
        self.ecc.incoming_weight_sum(output_neuron_idx)
    }
    #[text_signature = "(output_neuron_idx)"]
    pub fn incoming_weight_sum_f32(&self, output_neuron_idx: usize) -> f32 {
        self.ecc.incoming_weight_sum_f32(output_neuron_idx)
    }
}


#[pymethods]
impl CpuEccSparse {
    #[new]
    pub fn new(output: PyObject, kernel: PyObject, stride: PyObject, in_channels: usize, out_channels: usize, k: usize, connections_per_output: usize) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let output = arr2(py, &output)?;
        let kernel = arr2(py, &kernel)?;
        let stride = arr2(py, &stride)?;
        Ok(CpuEccSparse { ecc: htm::EccSparse::new(output, kernel, stride, in_channels, out_channels, k, connections_per_output, &mut rand::thread_rng()) })
    }
    #[getter]
    pub fn get_sums(&self) -> Vec<u16> {
        self.ecc.sums.clone()
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
        self.ecc.k()
    }
    #[setter]
    pub fn set_k(&mut self, k: usize) {
        self.ecc.set_k(k)
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
    pub fn run(&mut self, input: &CpuSDR) -> CpuSDR {
        CpuSDR { sdr: self.ecc.run(&input.sdr) }
    }
    #[text_signature = "(input_sdr,output_sdr)"]
    pub fn run_in_place(&mut self, input: &CpuSDR, output:&mut CpuSDR){
        self.ecc.run_in_place(&input.sdr,&mut output.sdr)
    }
    #[text_signature = "(file)"]
    pub fn save(&self, file: String) -> PyResult<()> {
        pickle(&self.ecc, file)
    }
}

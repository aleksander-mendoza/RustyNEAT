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
use htm::{VectorFieldOne, Idx, SDR, w_idx, ConvShape, Shape2, HasConvShape, ConvShapeTrait, HasConvShapeMut, HasShape, EccLayerTrait, D,TensorTrait, ConvTensorTrait, L};
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
use crate::py_ecc_config::EccConfig;
use crate::py_ecc_tensor::Tensor;

///
/// EccLayer(output: list[int], kernel: list[int], stride: list[int], in_channels: int, out_channels: int, k: int)
///
///
#[pyclass]
pub struct EccLayer {
    pub(crate) ecc: htm::EccLayer,
}


#[pyproto]
impl PySequenceProtocol for EccLayer {
    fn __len__(&self) -> usize {
        self.ecc.len()
    }
}


impl_save_load!(EccLayer,ecc);


#[pymethods]
impl EccLayer {
    // #[text_signature = "(list_of_input_sdrs)"]
    // pub fn batch_infer(&self, input: Vec<PyRef<CpuSDR>>) -> Vec<CpuSDR> {
    //     self.ecc.batch_infer(&input, |s| &s.sdr, |o| CpuSDR { sdr: o })
    // }
    #[text_signature = "(fill_value)"]
    pub fn fill_sums(&mut self,fill_value:D) {
        self.ecc.fill_sums(fill_value)
    }
    #[text_signature = "(output,epsilon)"]
    pub fn decrement_activities(&mut self, output: &mut CpuSDR,epsilon:D) {
        self.ecc.decrement_activities(&output.sdr,epsilon)
    }
    #[getter]
    pub fn get_in_shape(&self) -> Vec<Idx> {
        self.ecc.in_shape().to_vec()
    }
    #[getter]
    pub fn get_stride(&self) -> Vec<Idx> {
        self.ecc.stride().to_vec()
    }
    #[getter]
    pub fn get_kernel(&self) -> Vec<Idx> {
        self.ecc.kernel().to_vec()
    }
    #[getter]
    pub fn get_in_volume(&self) -> Idx {
        self.ecc.in_volume()
    }
    #[getter]
    pub fn get_in_area(&self) -> Idx {
        self.ecc.in_area()
    }
    #[getter]
    pub fn get_out_volume(&self) -> Idx {
        self.ecc.out_volume()
    }
    #[getter]
    pub fn get_out_area(&self) -> Idx {
        self.ecc.out_area()
    }
    #[getter]
    pub fn get_out_channels(&self) -> Idx {
        self.ecc.out_channels()
    }
    #[getter]
    pub fn get_in_channels(&self) -> Idx {
        self.ecc.in_channels()
    }
    #[getter]
    pub fn get_out_grid(&self) -> Vec<Idx> {
        self.ecc.out_grid().to_vec()
    }
    #[getter]
    pub fn get_in_grid(&self) -> Vec<Idx> {
        self.ecc.in_grid().to_vec()
    }
    #[getter]
    pub fn get_out_shape(&self) -> Vec<Idx> {
        self.ecc.out_shape().to_vec()
    }
    #[getter]
    pub fn get_k(&self) -> Idx {
        self.ecc.k()
    }
    #[setter]
    pub fn set_k(&mut self, k: Idx) {
        self.ecc.set_k(k)
    }
    #[getter]
    fn get_threshold(&self) -> f32 {
        self.ecc.get_threshold()
    }
    #[setter]
    fn set_threshold(&mut self, threshold: f32) {
        self.ecc.set_threshold(threshold)
    }
    #[getter]
    fn get_plasticity(&self) -> f32 {
        self.ecc.get_plasticity()
    }
    #[setter]
    fn set_plasticity(&mut self, plasticity: f32) {
        self.ecc.set_plasticity(plasticity)
    }

    #[text_signature = "(input_sdr, output_sdr, learn)"]
    pub fn infer(&mut self, input: &CpuSDR, output:&mut CpuSDR, learn: Option<bool>){
        self.ecc.infer(&input.sdr, &mut output.sdr,learn.unwrap_or(false))
    }
    #[text_signature = "(input_sdr, output_sdr, learn)"]
    pub fn infer_new_sdr(&mut self, input: &CpuSDR, learn: Option<bool>) -> CpuSDR{
        CpuSDR{sdr:self.ecc.infer_new_sdr(&input.sdr,learn.unwrap_or(false))}
    }
    #[text_signature = "(input_sdr, output_sdr, learn)"]
    pub fn infer_push(&mut self, input: &CpuSDR, output:&mut CpuSDR, learn: Option<bool>){
        self.ecc.infer_push(&input.sdr, &mut output.sdr,learn.unwrap_or(false))
    }
    #[text_signature = "(input_sdr,output_sdr,stored_sums)"]
    pub fn learn(&mut self, input: &CpuSDR, output: &CpuSDR) {
        self.ecc.learn(&input.sdr, &output.sdr)
    }
    #[text_signature = "()"]
    pub fn to_net(&self) -> EccNet {
        EccNet { ecc: htm::EccNet::from(self.ecc.clone()) }
    }

    #[text_signature = "()"]
    pub fn normalise(&mut self) {
        self.ecc.normalise()
    }
    #[text_signature = "(sdr)"]
    pub fn sparse_normalise(&mut self,sdr:&CpuSDR) {
        self.ecc.sparse_normalise(&sdr.sdr)
    }
    #[text_signature = "(idx)"]
    pub fn kernel_column_normalise(&mut self,idx:Idx) {
        self.ecc.kernel_column_normalise(idx)
    }
    #[getter]
    pub fn get_region_size(&self) -> Idx {
        self.ecc.get_region_size()
    }
    #[new]
    pub fn new(cfg:&EccConfig,output: PyObject, kernel: PyObject, stride: PyObject, in_channels: Idx, out_channels: Idx, k: Idx) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let output = arr2(py, &output)?;
        let kernel = arr2(py, &kernel)?;
        let stride = arr2(py, &stride)?;
        let shape= ConvShape::new(output, kernel, stride, in_channels, out_channels);
        Ok(EccLayer { ecc: htm::EccLayer::new(cfg.ecc.clone(), shape, k, &mut rand::thread_rng()) })
    }
    #[staticmethod]
    #[text_signature = "(input, kernel, stride,in_channels, out_channels)"]
    pub fn new_in(cfg:&EccConfig,input: PyObject, kernel: PyObject, stride: PyObject, in_channels: Idx, out_channels: Idx, k: Idx) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let input = arr2(py, &input)?;
        let kernel = arr2(py, &kernel)?;
        let stride = arr2(py, &stride)?;
        let shape = ConvShape::new_in(input.add_channels(in_channels), out_channels, kernel, stride);
        Ok(EccLayer { ecc: htm::EccLayer::new(cfg.ecc.clone(),shape, k, &mut rand::thread_rng()) })
    }
    #[text_signature = "(new_stride)"]
    pub fn set_stride(&mut self, new_stride: PyObject) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let new_stride = arr2(py, &new_stride)?;
        self.ecc.set_stride(new_stride);
        Ok(())
    }
    #[text_signature = "(output_size, column_pos)"]
    pub fn repeat_column(&self, output: PyObject, pretrained_column_pos: Option<PyObject>) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let output = arr2(py, &output)?;
        let pretrained_column_pos = if let Some(pretrained_column_pos) = pretrained_column_pos {
            arr2(py, &pretrained_column_pos)?
        } else {
            [0, 0]
        };
        Ok(Self { ecc: self.ecc.repeat_column(output, pretrained_column_pos) })
    }
    #[staticmethod]
    #[text_signature = "(input,output,k)"]
    pub fn new_linear(cfg:&EccConfig,input: Idx, output: Idx, k: Option<Idx>) -> Self {
        let k = k.unwrap_or(1);
        Self { ecc: htm::EccLayer::new(cfg.ecc.clone(),ConvShape::new_linear(input, output), k, &mut rand::thread_rng()) }
    }
    #[staticmethod]
    #[text_signature = "(shape,k)"]
    pub fn new_identity(py: Python,cfg:&EccConfig, shape: PyObject, k: Option<Idx>) -> PyResult<Self> {
        let k = k.unwrap_or(1);
        let shape = arrX(py, &shape, 1, 1, 1)?;
        Ok(Self { ecc: htm::EccLayer::new(cfg.ecc.clone(),ConvShape::new_identity(shape), k, &mut rand::thread_rng()) })
    }
    // #[staticmethod]
    // #[text_signature = "(layers)"]
    // pub fn concat(layers: Vec<PyRef<Self>>) -> Self {
    //     Self { ecc: htm::EccLayer::concat(&layers, |s| &s.ecc) }
    // }
    #[text_signature = "(column_idx)"]
    pub fn min_activity(&self) -> Tensor {
        Tensor{ecc:self.ecc.min_activity().clone()}
    }
    #[text_signature = "(column_idx)"]
    pub fn activity_column_min(&self,column_idx:usize) -> D {
        self.ecc.activity_column_min(column_idx)
    }
    #[text_signature = "(column_idx)"]
    pub fn activity_column_max(&self,column_idx:usize) -> D {
        self.ecc.activity_column_max(column_idx)
    }
    #[text_signature = "(output_idx)"]
    pub fn activity(&self, output_idx: Option<usize>) -> Vec<D> {
        if let Some(output_idx)= output_idx {
            self.ecc.activity.column_slice(output_idx).to_vec()
        }else{
            self.ecc.activity.as_slice().to_vec()
        }
    }
    #[text_signature = "(input_pos,output_pos)"]
    pub fn w_index(&self, input_pos: PyObject, output_pos: PyObject) -> PyResult<Idx> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let input_pos = arr3(py, &input_pos)?;
        let output_pos = arr3(py, &output_pos)?;
        Ok(self.ecc.w_index(&input_pos, &output_pos))
    }
    #[text_signature = "(value)"]
    pub fn fill_activity(&mut self,value:D) {
        self.ecc.fill_activity(value)
    }
    #[getter]
    pub fn get_sums(&self) -> Tensor {
        Tensor{ecc: self.ecc.sums.clone()}
    }
    #[text_signature = "(output_neuron_idx)"]
    pub fn kernel_column_sum(&self, output_neuron_idx: Idx) -> f32 {
        self.ecc.weights.kernel_column_sum(output_neuron_idx)
    }
    #[text_signature = "(output_neuron_idx)"]
    pub fn kernel_column_square_sum(&self, output_neuron_idx: Idx) -> f32 {
        self.ecc.weights.kernel_column_pow_sum::<L<2>>(output_neuron_idx)
    }
    #[text_signature = "(sdr)"]
    pub fn sums_for_sdr(&self, sdr: &CpuSDR) -> f32 {
        self.ecc.sums().sparse_sum(&sdr.sdr)
    }
    #[text_signature = "(output_neuron_idx)"]
    pub fn get_weights(&self, output_neuron_idx: Option<Idx>) -> Vec<f32> {
        if let Some(output_neuron_idx) = output_neuron_idx {
            self.ecc.weights().kernel_column_copy(output_neuron_idx)
        } else {
            self.ecc.weights().as_slice().to_vec()
        }
    }
}




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
use numpy::{PyReadonlyArrayDyn, PyArrayDyn, IntoPyArray, PyArray, PY_ARRAY_API, npyffi, Element, ToNpyDims, DataType, PyArray3};
use numpy::npyffi::{NPY_ORDER, npy_intp, NPY_ARRAY_WRITEABLE};
use std::os::raw::c_int;
use crate::ocl_err_to_py_ex;
use crate::py_ndalgebra::{DynMat, try_as_dtype};
use crate::py_ocl::Context;
use htm::{VectorFieldOne, Idx, SDR, w_idx, ConvShape, Shape3, Shape2, HasConvShape, HasShape, D, TensorTrait, from_xyz, from_xy, L, AsUsize};
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
use crate::py_ecc_conv_tensor::ConvTensor;


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
    pub fn get(&self, i: usize) -> f32 {
        self.ecc.get(i)
    }
    #[text_signature = "(x,y,z)"]
    pub fn get_at(&self, x: Idx, y: Idx, z: Idx) -> D {
        self.ecc.get_at(from_xyz(x, y, z))
    }
    #[text_signature = "(x,y)"]
    pub fn get_at2d(&self, x: Idx, y: Idx) -> D {
        self.ecc.get_at2d(from_xy(x, y))
    }
    #[text_signature = "()"]
    pub fn as_list(&self) -> Vec<f32> {
        self.ecc.as_slice().to_vec()
    }
    #[text_signature = "(index, value)"]
    pub fn set(&mut self, i: usize, v: f32) {
        *self.ecc.get_mut(i) = v
    }
    #[text_signature = "(index, value)"]
    pub fn add(&mut self, i: usize, v: f32) {
        *self.ecc.get_mut(i) += v
    }
    #[text_signature = "(index, value)"]
    pub fn sub(&mut self, i: usize, v: f32) {
        *self.ecc.get_mut(i) -= v
    }
    #[text_signature = "(index, value)"]
    pub fn mul(&mut self, i: usize, v: f32) {
        *self.ecc.get_mut(i) *= v
    }
    #[text_signature = "(index, value)"]
    pub fn div(&mut self, i: usize, v: f32) {
        *self.ecc.get_mut(i) /= v
    }
    #[text_signature = "(x,y,z,value)"]
    pub fn set_at(&mut self, x: Idx, y: Idx, z: Idx, value: D) {
        *self.ecc.get_at_mut(from_xyz(x, y, z)) = value
    }
    #[text_signature = "(x,y,z,value)"]
    pub fn set_at2d(&mut self, x: Idx, y: Idx, value: D) {
        *self.ecc.get_at2d_mut(from_xy(x, y)) = value
    }
    #[text_signature = "(value)"]
    pub fn fill_all(&mut self, value: D) {
        self.ecc.fill(value)
    }
    #[text_signature = "(value,sdr,parallel)"]
    pub fn fill(&mut self, value: D, sdr: &CpuSDR) {
        sdr.sdr.fill_into(value, self.ecc.as_mut_slice())
    }
    #[getter]
    pub fn is_empty(&self) -> bool {
        self.ecc.is_empty()
    }
    #[getter]
    pub fn shape(&self) -> Vec<Idx> {
        self.ecc.shape().to_vec()
    }
    #[getter]
    pub fn width(&self) -> Idx {
        self.ecc.shape().width()
    }
    #[getter]
    pub fn volume(&self) -> Idx {
        self.ecc.shape().volume()
    }
    #[getter]
    pub fn height(&self) -> Idx {
        self.ecc.shape().height()
    }
    #[getter]
    pub fn channels(&self) -> Idx {
        self.ecc.shape().channels()
    }
    #[getter]
    pub fn area(&self) -> Idx {
        self.ecc.shape().area()
    }
    #[text_signature = "(column_idx)"]
    pub fn column_min(&self, column_idx: usize) -> D {
        self.ecc.column_min(column_idx)
    }
    #[text_signature = "(column_idx)"]
    pub fn column_max(&self, column_idx: usize) -> D {
        self.ecc.column_max(column_idx)
    }
    #[text_signature = "(k,output_sdr)"]
    pub fn topk(&self, k: usize, output: &mut CpuSDR) {
        self.ecc.topk(k, &mut output.sdr)
    }
    #[text_signature = "(k,threshold,output_sdr)"]
    pub fn top1_per_region_thresholded(&self, k: Idx, threshold: D, output: &mut CpuSDR) {
        self.ecc.top1_per_region_thresholded(k, threshold, &mut output.sdr)
    }
    #[text_signature = "(k,output_sdr)"]
    pub fn top1_per_region(&self, k: Idx, output: &mut CpuSDR) {
        self.ecc.top1_per_region(k, &mut output.sdr)
    }
    #[text_signature = "(multiplicative_bias,k,threshold,output_sdr)"]
    pub fn top1_per_region_thresholded_multiplicative(&self, multiplicative_bias: &Tensor, k: Idx, threshold: D, output: &mut CpuSDR) {
        self.ecc.top1_per_region_thresholded_multiplicative(&multiplicative_bias.ecc, k, threshold, &mut output.sdr)
    }
    #[text_signature = "(multiplicative_bias,min_multiplicative_bias,k,threshold,output_sdr)"]
    pub fn top1_per_region_thresholded_multiplicative_with_cached_min(&self, multiplicative_bias: &Tensor, min_multiplicative_bias: &Tensor, k: Idx, threshold: D, output: &mut CpuSDR) {
        self.ecc.top1_per_region_thresholded_multiplicative_with_cached_min(&multiplicative_bias.ecc, &min_multiplicative_bias.ecc, k, threshold, &mut output.sdr)
    }
    #[text_signature = "(multiplicative_bias,k,threshold,output_sdr)"]
    pub fn top1_per_region_thresholded_additive(&self, additive_bias: &Tensor, k: Idx, threshold: D, output: &mut CpuSDR) {
        self.ecc.top1_per_region_thresholded_additive(&additive_bias.ecc, k, threshold, &mut output.sdr)
    }
    #[text_signature = "(lhs_sdr)"]
    pub fn sparse_sum(&self, lhs_sdr: &CpuSDR) -> D {
        self.ecc.sparse_sum(&lhs_sdr.sdr)
    }
    #[text_signature = "(xy_indices,channel,scalar)"]
    pub fn sparse_add_assign_scalar_to_area(&mut self, xy_indices:&CpuSDR, channel:Idx, scalar:D){
        self.ecc.sparse_add_assign_scalar_to_area(&xy_indices.sdr,channel,scalar)
    }
    #[text_signature = "(x,y_indices,scalar)"]
    pub fn mat_sparse_add_assign_scalar_to_column(&mut self, x: Idx, y_indices: &CpuSDR, scalar: D) {
        self.ecc.mat_sparse_add_assign_scalar_to_column(x, &y_indices.sdr, scalar)
    }
    #[text_signature = "(x_indices,y,scalar)"]
    pub fn mat_sparse_add_assign_scalar_to_row(&mut self, x_indices: &CpuSDR, y: Idx, scalar: D) {
        self.ecc.mat_sparse_add_assign_scalar_to_row(&x_indices.sdr, y, scalar)
    }
    #[text_signature = "(x,y_indices,scalar)"]
    pub fn mat_sparse_sub_assign_scalar_to_column(&mut self, x: Idx, y_indices: &CpuSDR, scalar: D) {
        self.ecc.mat_sparse_sub_assign_scalar_to_column(x, &y_indices.sdr, scalar)
    }
    #[text_signature = "(x_indices,y,scalar)"]
    pub fn mat_sparse_sub_assign_scalar_to_row(&mut self, x_indices: &CpuSDR, y: Idx, scalar: D) {
        self.ecc.mat_sparse_sub_assign_scalar_to_row(&x_indices.sdr, y, scalar)
    }
    #[text_signature = "(lhs_sdr)"]
    pub fn mat_sparse_dot_lhs_new_vec(&self, lhs_sdr: &CpuSDR) -> Tensor {
        Tensor { ecc: self.ecc.mat_sparse_dot_lhs_new_vec(&lhs_sdr.sdr) }
    }
    #[text_signature = "(lhs_sdr,output)"]
    pub fn mat_sparse_dot_lhs_vec(&self, lhs_sdr: &CpuSDR, output: &mut Tensor) {
        self.ecc.mat_sparse_dot_lhs_vec(&lhs_sdr.sdr, &mut output.ecc)
    }
    #[text_signature = "(x,scalar)"]
    pub fn mat_div_column(&mut self, x: Idx, scalar: D) {
        self.ecc.mat_div_column(x, scalar)
    }
    #[text_signature = "(y,scalar)"]
    pub fn mat_div_row(&mut self, y: Idx, scalar: D) {
        self.ecc.mat_div_row(y, scalar)
    }
    #[text_signature = "(x,l)"]
    pub fn mat_sum_column(&self, x: Idx, l: Idx) -> D {
        if l == 1 {
            self.ecc.mat_sum_column::<L<1>>(x)
        } else {
            self.ecc.mat_sum_column::<L<2>>(x)
        }
    }
    #[text_signature = "(y,l)"]
    pub fn mat_sum_row(&self, y: Idx, l: Idx) -> D {
        if l == 1 {
            self.ecc.mat_sum_row::<L<1>>(y)
        } else {
            self.ecc.mat_sum_row::<L<2>>(y)
        }
    }
    #[text_signature = "(x,l)"]
    pub fn mat_norm_column(&self, x: Idx, l: Idx) -> D {
        if l == 1 {
            self.ecc.mat_norm_column::<L<1>>(x)
        } else {
            self.ecc.mat_norm_column::<L<2>>(x)
        }
    }
    #[text_signature = "(y,l)"]
    pub fn mat_norm_row(&self, y: Idx, l: Idx) -> D {
        if l == 1 {
            self.ecc.mat_norm_row::<L<1>>(y)
        } else {
            self.ecc.mat_norm_row::<L<2>>(y)
        }
    }
    #[text_signature = "(x,l)"]
    pub fn mat_norm_assign_column(&mut self, x: Idx, l: Idx) {
        if l == 1 {
            self.ecc.mat_norm_assign_column::<L<1>>(x)
        } else {
            self.ecc.mat_norm_assign_column::<L<2>>(x)
        }
    }
    #[text_signature = "(y,l)"]
    pub fn mat_norm_assign_row(&mut self, y: Idx, l: Idx) {
        if l == 1 {
            self.ecc.mat_norm_assign_row::<L<1>>(y)
        } else {
            self.ecc.mat_norm_assign_row::<L<2>>(y)
        }
    }
    #[text_signature = "(l)"]
    pub fn mat_norm_assign_columnwise(&mut self, l: Idx) {
        if l == 1 {
            self.ecc.mat_norm_assign_columnwise::<L<1>>()
        } else {
            self.ecc.mat_norm_assign_columnwise::<L<2>>()
        }
    }
    #[text_signature = "(l)"]
    pub fn mat_norm_assign_rowwise(&mut self, l: Idx) {
        if l == 1 {
            self.ecc.mat_norm_assign_rowwise::<L<1>>()
        } else {
            self.ecc.mat_norm_assign_rowwise::<L<2>>()
        }
    }
    #[text_signature = "(other)"]
    pub fn add_assign(&mut self, other: &Tensor) {
        self.ecc.add_assign(&other.ecc)
    }
    #[text_signature = "(other)"]
    pub fn sub_assign(&mut self, other: &Tensor) {
        self.ecc.sub_assign(&other.ecc)
    }
    #[text_signature = "(other)"]
    pub fn mul_assign(&mut self, other: &Tensor) {
        self.ecc.mul_assign(&other.ecc)
    }
    #[text_signature = "(other)"]
    pub fn div_assign(&mut self, other: &Tensor) {
        self.ecc.div_assign(&other.ecc)
    }
    #[text_signature = "(sparse_mask,scalar)"]
    pub fn sparse_add_assign_scalar(&mut self, sparse_mask: &CpuSDR, scalar: D) {
        self.ecc.sparse_add_assign_scalar(&sparse_mask.sdr, scalar)
    }
    #[text_signature = "(sparse_mask,scalar)"]
    pub fn sparse_sub_assign_scalar(&mut self, sparse_mask: &CpuSDR, scalar: D) {
        self.ecc.sparse_sub_assign_scalar(&sparse_mask.sdr, scalar)
    }
    #[text_signature = "(sparse_mask,scalar)"]
    pub fn sparse_div_assign_scalar(&mut self, sparse_mask: &CpuSDR, scalar: D) {
        self.ecc.sparse_div_assign_scalar(&sparse_mask.sdr, scalar)
    }
    #[text_signature = "(sparse_mask,scalar)"]
    pub fn sparse_mul_assign_scalar(&mut self, sparse_mask: &CpuSDR, scalar: D) {
        self.ecc.sparse_mul_assign_scalar(&sparse_mask.sdr, scalar)
    }
    #[text_signature = "(column_min,sparse_mask,scalar)"]
    pub fn sparse_sub_assign_scalar_and_update_colum_min(&mut self, column_min: &mut Tensor, sparse_mask: &CpuSDR, scalar: D) {
        self.ecc.sparse_sub_assign_scalar_and_update_colum_min(&mut column_min.ecc, &sparse_mask.sdr, scalar)
    }
    #[text_signature = "(conv_tensor)"]
    pub fn kernel_column_sum_assign(&mut self, conv_tensor: &ConvTensor) {
        self.ecc.kernel_column_sum_assign(&conv_tensor.ecc)
    }
    #[text_signature = "(conv_tensor)"]
    pub fn kernel_column_sum_add_assign(&mut self, conv_tensor: &ConvTensor) {
        self.ecc.kernel_column_sum_add_assign(&conv_tensor.ecc)
    }
    #[text_signature = "(tensor)"]
    pub fn kernel_column_sum_sub_assign(&mut self, conv_tensor: &ConvTensor) {
        self.ecc.kernel_column_sum_sub_assign(&conv_tensor.ecc)
    }
    #[text_signature = "()"]
    pub fn rand_assign(&mut self) {
        self.ecc.rand_assign(&mut rand::thread_rng())
    }
    #[text_signature = "()"]
    fn argmax(&self) -> usize {
        self.ecc.argmax()
    }
    #[text_signature = "(x)"]
    fn mat_argmax_in_column(&self, x: Idx) -> usize {
        self.ecc.mat_argmax_in_column(x)
    }
    #[text_signature = "(y)"]
    fn mat_argmax_in_row(&self, y: Idx) -> usize {
        self.ecc.mat_argmax_in_row(y)
    }
    #[new]
    pub fn new(py: Python, shape: PyObject, initial_value: Option<f32>) -> PyResult<Self> {
        let shape = arrX(py, &shape, 1, 1, 1)?;
        Ok(Self {
            ecc: if let Some(initial_value) = initial_value {
                htm::Tensor::new(shape, initial_value)
            } else {
                unsafe { htm::Tensor::empty(shape) }
            }
        })
    }
    #[text_signature = "()"]
    pub fn numpy<'py>(&self, py: Python<'py>) -> &'py PyArray<D, numpy::Ix3> {
        let mut arr = PyArray3::new(py, self.ecc.shape().map(Idx::as_usize), false);
        let s = unsafe { arr.as_slice_mut().unwrap() };
        s.copy_from_slice(self.ecc.as_slice());
        arr
    }
}

#[pyproto]
impl PySequenceProtocol for Tensor {
    fn __len__(&self) -> usize {
        self.ecc.len()
    }
    fn __getitem__(&self, idx: isize) -> f32 {
        assert!(idx >= 0);
        self.ecc.get(idx as usize)
    }

    fn __setitem__(&mut self, idx: isize, value: f32) {
        assert!(idx >= 0);
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
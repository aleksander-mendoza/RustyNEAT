use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol, PyIterProtocol, PySequenceProtocol, PyTypeInfo, PyDowncastError, AsPyPointer, PyNumberProtocol};
use pyo3::PyResult;
use rusty_neat_core::{cppn, neat, gpu};
use std::collections::HashSet;
use rusty_neat_core::activations::{STR_TO_IDX, ALL_ACT_FN};
use pyo3::exceptions::PyValueError;
use rusty_neat_core::cppn::CPPN;
use std::iter::FromIterator;
use pyo3::types::{PyString, PyDateTime, PyDateAccess, PyTimeAccess, PyList, PyInt, PyFloat, PyIterator};
use rusty_neat_core::gpu::{FeedForwardNetOpenCL, FeedForwardNetPicbreeder, FeedForwardNetSubstrate};
use pyo3::basic::CompareOp;
use numpy::{PyReadonlyArrayDyn, PyArrayDyn, IntoPyArray, PyArray,Ix1,Ix3,Ix4, PY_ARRAY_API, npyffi, Element, ToNpyDims, DataType, PyReadonlyArray, PyArray3, PyArray1};
use numpy::npyffi::{NPY_ORDER, npy_intp, NPY_ARRAY_WRITEABLE};
use std::os::raw::c_int;
use crate::ocl_err_to_py_ex;
use crate::py_ndalgebra::{DynMat, try_as_dtype};
use crate::py_ocl::Context;
use htm::{Encoder, EncoderTarget, EncoderRange, Shape, VectorFieldOne, Synapse, SDR, as_usize, Idx, Shape3, ConvShape};
use std::time::SystemTime;
use std::ops::Deref;
use chrono::Utc;
use std::borrow::Borrow;
use std::io::BufWriter;
use std::fs::{File, OpenOptions};
use serde::Serialize;
use crate::util::*;
use std::any::{Any, TypeId};
use pyo3::ffi::PyFloat_Type;
use rand::{thread_rng, Rng};
use crate::py_htm::CpuSDR;
use crate::py_ecc::CpuEccDense;
use crate::py_ecc_population::{ConvWeights, CpuEccPopulation};
use crate::util::{impl_save_load};
#[pyclass]
pub struct CpuSdrDataset {
    pub(crate) sdr: htm::CpuSdrDataset,
}

#[pyclass]
pub struct SubregionIndices {
    pub(crate) sdr: htm::SubregionIndices,
}

#[pymethods]
impl SubregionIndices {
    #[getter]
    fn get_in_shape(&self) -> Vec<Idx>{
        self.sdr.shape().to_vec()
    }
    #[getter]
    fn get_dataset_len(&self) -> usize{
        self.sdr.dataset_len()
    }
    #[text_signature = "(idx)"]
    fn get_sample_idx(&self, idx:usize) -> u32{
        self.sdr[idx].channels()
    }
    #[text_signature = "(idx)"]
    fn get_output_column_pos(&self, idx:usize) -> Vec<Idx>{
        self.sdr[idx].grid().to_vec()
    }
}

#[pyclass]
pub struct LinearClassifier {
    pub(crate) c: htm::LinearClassifier,
}

#[pymethods]
impl LinearClassifier {
    #[text_signature = "(input_idx,label)"]
    pub fn prob(&self,input_idx:Idx,label:usize)->f32{
        self.prob(input_idx,label)
    }
    #[text_signature = "()"]
    pub fn occurrences(&self)->Vec<f32>{
        self.occurrences().to_vec()
    }
    #[text_signature = "()"]
    pub fn square_weights(&mut self){
        self.c.sqrt_weights()
    }
    #[text_signature = "()"]
    pub fn sqrt_weights(&mut self){
        self.c.sqrt_weights()
    }
    #[text_signature = "()"]
    pub fn exp_weights(&mut self){
        self.c.exp_weights()
    }
    #[text_signature = "()"]
    pub fn log_weights(&mut self){
        self.c.log_weights()
    }
    #[text_signature = "(label)"]
    pub fn occurrences_for_label(&self,lbl:usize)->Vec<f32>{
        self.occurrences_for_label(lbl).to_vec()
    }
    #[getter]
    fn num_classes(&self)->usize{
        self.c.num_classes()
    }
    #[getter]
    fn get_shape(&self) -> Vec<Idx>{
        self.c.shape().to_vec()
    }
    #[text_signature = "(sdr)"]
    fn classify(&self, sdr: &CpuSDR) -> usize {
        self.c.classify(&sdr.sdr)
    }
    #[text_signature = "(sdr_dataset)"]
    fn batch_classify<'py>(&self, py:Python<'py>, sdr: &CpuSdrDataset) -> &'py PyArray<u32, Ix1> {
        let v = self.c.batch_classify(&sdr.sdr);
        PyArray::from_vec(py,v)
    }
}
impl CpuSdrDataset{
    fn to_numpy_<'py,T:Element+Copy>(&self,py:Python<'py>,idx:usize,one:T) -> &'py PyArray<T, Ix3>{
        let sdr = &self.sdr[idx];
        let mut arr = PyArray3::zeros(py,self.sdr.shape().map(as_usize),false);
        let s = unsafe{arr.as_slice_mut().unwrap()};
        sdr.iter().for_each(|&i|s[as_usize(i)]=one);
        arr
    }
}
#[pymethods]
impl CpuSdrDataset {
    #[new]
    pub fn new(shape: [Idx; 3], sdr: Option<Vec<PyRef<CpuSDR>>>) -> Self {
        Self {
            sdr: if let Some(s) = sdr {
                let mut d = htm::CpuSdrDataset::with_capacity(s.len(),shape);
                d.extend(s.iter().map(|p| p.sdr.clone()));
                d
            } else {
                htm::CpuSdrDataset::new(shape)
            }
        }
    }
    #[text_signature = "(idx)"]
    fn to_numpy<'py>(&self,py:Python<'py>,idx:usize) -> &'py PyArray<u32, Ix3>{
        self.to_numpy_(py,idx,1)
    }
    #[text_signature = "(idx)"]
    fn to_bool_numpy<'py>(&self,py:Python<'py>,idx:usize) -> &'py PyArray<bool, Ix3>{
        self.to_numpy_(py,idx,true)
    }
    #[text_signature = "(idx)"]
    fn to_f32_numpy<'py>(&self,py:Python<'py>,idx:usize) -> &'py PyArray<f32, Ix3>{
        self.to_numpy_(py,idx,1.)
    }
    #[text_signature = "(idx)"]
    fn to_f64_numpy<'py>(&self,py:Python<'py>,idx:usize) -> &'py PyArray<f64, Ix3>{
        self.to_numpy_(py,idx,1.)
    }
    #[getter]
    fn get_shape(&self) -> Vec<Idx>{
        self.sdr.shape().to_vec()
    }
    #[getter]
    fn get_grid(&self) -> Vec<Idx>{
        self.sdr.shape().grid().to_vec()
    }
    #[getter]
    fn get_volume(&self) -> Idx{
        self.sdr.shape().size()
    }
    #[getter]
    fn get_channels(&self) -> Idx{
        self.sdr.shape().channels()
    }
    #[getter]
    fn get_width(&self) -> Idx{
        self.sdr.shape().width()
    }
    #[getter]
    fn get_height(&self) -> Idx{
        self.sdr.shape().height()
    }
    #[getter]
    fn get_area(&self) -> Idx{
        self.sdr.shape().grid().product()
    }
    #[text_signature = "(min_cardinality)"]
    pub fn filter_by_cardinality_threshold(&mut self, min_cardinality:Idx){
        self.sdr.filter_by_cardinality_threshold(min_cardinality)
    }
    #[text_signature = "(number_of_samples,drift,patches_per_sample,ecc,log)"]
    pub fn train_with_patches(&self, number_of_samples: usize, drift:[Idx;2], patches_per_sample:usize, ecc: &mut CpuEccDense,log:Option<usize>) {
        if let Some(log) = log {
            assert!(log>0,"Logging interval must be greater than 0");
            self.sdr.train_with_patches(number_of_samples,drift,patches_per_sample,&mut ecc.ecc,&mut rand::thread_rng(),|i|if i % log == 0{println!("Processed samples {}", i+1)})
        } else{
            self.sdr.train_with_patches(number_of_samples,drift,patches_per_sample,&mut ecc.ecc,&mut rand::thread_rng(),|i|{})
        }
    }
    #[text_signature = "(number_of_samples,ecc,log)"]
    pub fn train(&self, number_of_samples: usize, ecc: &mut CpuEccDense,log:Option<usize>) {
        if let Some(log) = log {
            self.sdr.train(number_of_samples,&mut ecc.ecc,&mut rand::thread_rng(),|i|if i % log == 0{println!("Processed samples {}", i+1)})
        } else{
            self.sdr.train(number_of_samples,&mut ecc.ecc,&mut rand::thread_rng(),|i|{})
        }
    }
    #[text_signature = "()"]
    fn clear(&mut self) {
        self.sdr.clear()
    }
    #[text_signature = "(sdr)"]
    fn push(&mut self, sdr: &CpuSDR) {
        self.sdr.push(sdr.sdr.clone())
    }
    #[text_signature = "()"]
    fn pop(&mut self) ->Option<CpuSDR>{
        self.sdr.pop().map(|sdr|CpuSDR{sdr})
    }
    #[text_signature = "()"]
    fn rand(&self) ->Option<CpuSDR>{
        self.sdr.rand(&mut rand::thread_rng()).map(|sdr|CpuSDR { sdr:sdr.clone()})
    }
    #[text_signature = "(ecc,indices)"]
    fn conv_subregion_indices_with_ecc(&self, ecc:&CpuEccDense, indices: &SubregionIndices) -> Self{
        Self{sdr:self.sdr.conv_subregion_indices_with_ecc(&ecc.ecc,&indices.sdr)}
    }
    #[text_signature = "(kernel,stride,indices)"]
    fn conv_subregion_indices_with_ker(&self, kernel:[Idx;2],stride:[Idx;2], indices: &SubregionIndices) -> Self{
        Self{sdr:self.sdr.conv_subregion_indices_with_ker(kernel,stride,&indices.sdr)}
    }
    #[text_signature = "(kernel,stride,number_of_samples,original_dataset)"]
    fn extend_from_conv_rand_subregion(&mut self, kernel:[Idx;2],stride:[Idx;2],number_of_samples:usize, original:&CpuSdrDataset) {
        let conv = ConvShape::new_in(*original.sdr.shape(),1,kernel,stride);
        self.sdr.extend_from_conv_rand_subregion(&conv,number_of_samples,&original.sdr,&mut rand::thread_rng())
    }
    #[text_signature = "(kernel,stride,indices,original_dataset)"]
    fn extend_from_conv_subregion_indices(&mut self, kernel:[Idx;2],stride:[Idx;2],indices:&SubregionIndices, original:&CpuSdrDataset) {
        let conv = ConvShape::new_in(*original.sdr.shape(),1,kernel,stride);
        self.sdr.extend_from_conv_subregion_indices(&conv,&indices.sdr,&original.sdr)
    }
    #[text_signature = "(out_shape,number_of_samples)"]
    fn gen_rand_conv_subregion_indices(&self, out_shape:[Idx;2],number_of_samples:usize) -> SubregionIndices{
        SubregionIndices{sdr:self.sdr.gen_rand_conv_subregion_indices(out_shape,number_of_samples,&mut rand::thread_rng())}
    }
    #[text_signature = "(kernel,stride,number_of_samples)"]
    fn gen_rand_conv_subregion_indices_with_ker(&self, kernel:[Idx;2], stride:[Idx;2],number_of_samples:usize) -> SubregionIndices{
        SubregionIndices{sdr:self.sdr.gen_rand_conv_subregion_indices_with_ker(&kernel,&stride,number_of_samples,&mut rand::thread_rng())}
    }
    #[text_signature = "(ecc_dense,number_of_samples)"]
    fn gen_rand_conv_subregion_indices_with_ecc(&self, ecc:&CpuEccDense,number_of_samples:usize) -> SubregionIndices{
        SubregionIndices{sdr:self.sdr.gen_rand_conv_subregion_indices_with_ecc(&ecc.ecc,number_of_samples,&mut rand::thread_rng())}
    }
    #[text_signature = "(labels,number_of_classes)"]
    fn count_per_label(&self, labels:&PyAny,number_of_classes:usize) -> PyResult<Vec<f32>>{
        let array = unsafe {
            if npyffi::PyArray_Check(labels.as_ptr()) == 0 {
                return Err(PyDowncastError::new(labels, "PyArray<T, D>").into());
            }
            &*(labels as *const PyAny as *const PyArrayDyn<u8>)
        };
        if !array.is_c_contiguous(){
            return Err(PyValueError::new_err("Numpy array is not C contiguous"));
        }
        let dtype = array.dtype().get_datatype().ok_or_else(|| PyValueError::new_err("No numpy array has no dtype"))?;
        fn f<T:Element>(sdr:&htm::CpuSdrDataset,labels:&PyAny,number_of_classes:usize,f:impl Fn(&T)->usize) -> PyResult<Vec<f32>>{
            let labels = unsafe { &*(labels as *const PyAny as *const PyArrayDyn<T>) };
            let labels = unsafe{labels.as_slice()?};
            Ok(sdr.count_per_label(labels,number_of_classes,f))
        }
        match dtype{
            u8::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&u8|*f as usize),
            u32::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&u32|*f as usize),
            u64::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&u64|*f as usize),
            i8::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&i8|*f as usize),
            i32::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&i32|*f as usize),
            i64::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&i64|*f as usize),
            usize::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&usize|*f),
            d => Err(PyValueError::new_err(format!("Unexpected dtype {:?} of numpy array ", d)))
        }
    }
    #[text_signature = "()"]
    fn count<'py>(&self, py:Python<'py>) -> PyResult<&'py PyArray<u32, Ix3>> {
        let v = self.sdr.count();
        let a = PyArray::from_vec(py,v);
        a.reshape(self.sdr.shape().map(as_usize))
    }
    #[text_signature = "(outputs)"]
    fn measure_receptive_fields<'py>(&self, py:Python<'py>,outputs:&CpuSdrDataset) ->  PyResult<&'py PyArray<u32, Ix4>> {
        let ov = outputs.sdr.shape().size();
        let v = self.sdr.measure_receptive_fields(&outputs.sdr);
        let a = PyArray::from_vec(py,v);
        let s = self.sdr.shape();
        let s = [s[0],s[1],s[2],ov].map(as_usize);
        a.reshape(s)
    }
    #[text_signature = "(ecc_dense)"]
    fn batch_infer(&self,ecc:&CpuEccDense) -> CpuSdrDataset{
        Self{sdr:self.sdr.batch_infer(&ecc.ecc)}
    }
    #[text_signature = "(start,end)"]
    pub fn subdataset(&self,start:usize,end:Option<usize>) -> Self{
        let end = end.unwrap_or(self.sdr.len());
        Self{sdr:self.sdr.subdataset(start..end)}
    }

    #[text_signature = "(ecc_dense,target)"]
    fn batch_infer_conv_weights(&self,ecc:&ConvWeights,target:&CpuEccPopulation) -> CpuSdrDataset{
        Self{sdr:self.sdr.batch_infer_conv_weights(&ecc.ecc,target.ecc.clone())}
    }
    #[text_signature = "(ecc_dense)"]
    fn batch_infer_and_measure_s_expectation(&self,ecc:&CpuEccDense) -> (CpuSdrDataset,f32,u32){
        let (sdr,s_exp,missed) = self.sdr.batch_infer_and_measure_s_expectation(&ecc.ecc);
        (Self{sdr},s_exp,missed)
    }
    #[text_signature = "(ecc_dense,target)"]
    fn batch_infer_conv_weights_and_measure_s_expectation(&self, ecc:&ConvWeights, target:&CpuEccPopulation) -> (CpuSdrDataset, f32, u32) {
        let (sdr,s_exp,missed) = self.sdr.batch_infer_conv_weights_and_measure_s_expectation(&ecc.ecc,target.ecc.clone());
        (Self{sdr},s_exp,missed)
    }
    // #[text_signature = "(labels)"]
    // fn fit_naive_bayes(&self, labels:&PyAny) -> PyResult<CpuSDR>{
    //     Ok()
    // }
    #[text_signature = "(labels,number_of_classes)"]
    fn fit_linear_regression(&self, labels:&PyAny, number_of_classes:usize) -> PyResult<LinearClassifier>{
        let array = unsafe {
            if npyffi::PyArray_Check(labels.as_ptr()) == 0 {
                return Err(PyDowncastError::new(labels, "PyArray<T, D>").into());
            }
            &*(labels as *const PyAny as *const PyArrayDyn<u8>)
        };
        if !array.is_c_contiguous(){
            return Err(PyValueError::new_err("Numpy array is not C contiguous"));
        }
        let dtype = array.dtype().get_datatype().ok_or_else(|| PyValueError::new_err("No numpy array has no dtype"))?;
        fn f<T:Element>(sdr:&htm::CpuSdrDataset,labels:&PyAny,number_of_classes:usize,f:impl Fn(&T)->usize) -> PyResult<htm::LinearClassifier>{
            let labels = unsafe { &*(labels as *const PyAny as *const PyArrayDyn<T>) };
            let labels = unsafe{labels.as_slice()?};
            Ok(sdr.fit_linear_regression(labels,number_of_classes,f))
        }
        match dtype{
            u8::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&u8|*f as usize),
            u32::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&u32|*f as usize),
            u64::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&u64|*f as usize),
            i8::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&i8|*f as usize),
            i32::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&i32|*f as usize),
            i64::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&i64|*f as usize),
            usize::DATA_TYPE => f(&self.sdr,labels,number_of_classes,|f:&usize|*f),
            d => Err(PyValueError::new_err(format!("Unexpected dtype {:?} of numpy array ", d)))
        }.map(|c|LinearClassifier{c})
    }
}

impl_save_load!(CpuSdrDataset,sdr);
impl_save_load!(SubregionIndices,sdr);
impl_save_load!(LinearClassifier,c);

#[pyproto]
impl PySequenceProtocol for CpuSdrDataset {
    fn __len__(&self) -> usize {
        self.sdr.len()
    }
    fn __getitem__(&self, idx: isize) -> CpuSDR {
        assert!(idx>=0);
        CpuSDR{sdr:self.sdr[idx as usize].clone()}
    }

    fn __setitem__(&mut self, idx: isize, value: PyRef<CpuSDR>) {
        assert!(idx>=0);
        self.sdr[idx as usize] = value.sdr.clone();
    }
}


#[pyproto]
impl PySequenceProtocol for SubregionIndices {
    fn __len__(&self) -> usize {
        self.sdr.len()
    }
    fn __getitem__(&self, idx: isize) -> Vec<u32> {
        assert!(idx>=0);
        self.sdr[idx as usize].to_vec()
    }

    fn __setitem__(&mut self, idx: isize, value: [Idx;3]) {
        assert!(idx>=0);
        self.sdr[idx as usize] = value;
    }
}
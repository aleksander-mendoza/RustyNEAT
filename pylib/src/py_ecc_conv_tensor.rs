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
use htm::{VectorFieldOne, Idx, SDR, w_idx, ConvShape, Shape3, Shape2, HasConvShape, HasShape, D, ConvTensorTrait};
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
/// ConvWeights(output, kernel, stride, in_channels)
///
///
#[pyclass]
pub struct ConvTensor {
    pub ecc: htm::ConvTensor<D>,
}

#[pymethods]
impl ConvTensor {
    #[getter]
    pub fn get_out_grid(&self) -> Vec<Idx> {
        self.ecc.out_grid().to_vec()
    }
    #[getter]
    pub fn get_in_grid(&self) -> Vec<Idx> {
        self.ecc.in_grid().to_vec()
    }
    #[getter]
    pub fn in_shape(&self) -> Vec<Idx> {
        self.ecc.in_shape().to_vec()
    }
    #[getter]
    pub fn in_volume(&self) -> Idx {
        self.ecc.in_volume()
    }
    #[getter]
    pub fn in_channels(&self) -> Idx {
        self.ecc.in_channels()
    }
    #[getter]
    pub fn out_shape(&self) -> Vec<Idx> {
        self.ecc.out_shape().to_vec()
    }
    #[getter]
    pub fn out_volume(&self) -> Idx {
        self.ecc.out_volume()
    }
    #[getter]
    pub fn out_channels(&self) -> Idx {
        self.ecc.out_channels()
    }
    #[getter]
    pub fn get_kernel(&self) -> Vec<Idx> {
        self.ecc.kernel().to_vec()
    }
    #[getter]
    pub fn get_stride(&self) -> Vec<Idx> {
        self.ecc.stride().to_vec()
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

    // #[staticmethod]
    // #[text_signature = "(layers)"]
    // pub fn concat(layers: Vec<PyRef<Self>>) -> Self {
    //     Self { ecc: htm::ConvTensor::concat(&layers, |s| &s.ecc) }
    // }
    // #[new]
    // pub fn new(output: [Idx; 3], kernel: [Idx; 2], stride: [Idx; 2], in_channels: Idx) -> Self{
    //     let mut rng = rand::thread_rng();
    //     Self { ecc: htm::ConvTensorVec::new(htm::ConvShape::new_out(in_channels, output, kernel, stride), &mut rng) }
    // }
    // #[staticmethod]
    // #[text_signature = "(input, kernel, stride, out_channels)"]
    // pub fn new_in(input: [Idx; 3], kernel: [Idx; 2], stride: [Idx; 2], out_channels: Idx) -> Self{
    //     let mut rng = rand::thread_rng();
    //     Self { ecc: htm::ConvTensor::new(htm::ConvShape::new_in(input, out_channels, kernel, stride), &mut rng) }
    // }
    // #[staticmethod]
    // #[text_signature = "(input,output)"]
    // pub fn new_linear(input:Idx,output:Idx) -> Self {
    //     Self { ecc: htm::ConvTensor::new(ConvShape::new_linear(input, output), &mut rand::thread_rng()) }
    // }
    // #[staticmethod]
    // #[text_signature = "(shape)"]
    // pub fn new_identity(py:Python,shape:PyObject) -> PyResult<Self> {
    //     let shape = arrX(py,&shape,1,1,1)?;
    //     Ok(Self { ecc: htm::ConvTensor::new(ConvShape::new_identity(shape), &mut rand::thread_rng()) })
    // }
    // #[text_signature = "()"]
    // pub fn restore_dropped_out_weights(&mut self) {
    //     self.ecc.restore_dropped_out_weights()
    // }
    // #[text_signature = "(number_of_connections_to_drop,per_kernel,normalise)"]
    // pub fn dropout(&mut self, py: Python, number_of_connections_to_drop: PyObject,per_kernel:Option<bool>,normalise:Option<bool>) -> PyResult<()> {
    //     let per_kernel = per_kernel.unwrap_or(false);
    //     let normalise = normalise.unwrap_or(true);
    //     if PyFloat::is_exact_type_of(number_of_connections_to_drop.as_ref(py)) {
    //         if per_kernel{
    //             self.ecc.dropout_per_kernel_f32(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
    //         }else{
    //             self.ecc.dropout_f32(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
    //         }
    //     } else {
    //         if per_kernel{
    //             self.ecc.dropout_per_kernel(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
    //         }else{
    //             self.ecc.dropout(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
    //         }
    //     }
    //     if normalise{
    //         self.ecc.normalize_all()
    //     }
    //     Ok(())
    // }
    //
    // #[text_signature = "(output_idx)"]
    // pub fn normalize(&mut self, output_idx: Option<Idx>) {
    //     if let Some(output_idx) = output_idx {
    //             self.ecc.normalize(output_idx)
    //     }else{
    //             self.ecc.normalize_all()
    //     }
    //
    // }
    //
    // #[text_signature = "(output_neuron)"]
    // pub fn get_dropped_weights_count(&self, output_neuron: Option<Idx>) -> usize {
    //     if let Some(output_neuron_idx) = output_neuron {
    //         self.ecc.get_dropped_weights_of_kernel_column_count(output_neuron_idx)
    //     } else {
    //         self.ecc.get_dropped_weights_count()
    //     }
    // }
    // #[text_signature = "(sums,reset)"]
    // pub fn store_all_incoming_weight_sums(&self, sums:&mut WeightSums, reset:Option<bool>) {
    //     if reset.unwrap_or(false){
    //         self.ecc.store_all_incoming_weight_sums(&mut sums.ecc)
    //     }else{
    //         self.ecc.reset_and_store_all_incoming_weight_sums(&mut sums.ecc)
    //     }
    // }
    // #[text_signature = "(output_neuron)"]
    // pub fn get_weight_sum(&self, output_neuron: Idx) -> f32 {
    //         self.ecc.incoming_weight_sum(output_neuron)
    // }
    // #[text_signature = "(output_idx)"]
    // pub fn get_weights(&mut self, output_neuron_idx: Option<Idx>) -> Vec<f32>{
    //     if let Some(output_neuron_idx) = output_neuron_idx{
    //         self.ecc.incoming_weight_copy(output_neuron_idx).to_vec()
    //     } else {
    //         self.ecc.as_slice().to_vec()
    //     }
    // }
}

impl_save_load!(ConvTensor,ecc);


//
// #[pyproto]
// impl PyObjectProtocol for WeightSums {
//     fn __str__(&self) -> PyResult<String> {
//         Ok(format!("{:?}", self.ecc.target()))
//     }
//     fn __repr__(&self) -> PyResult<String> {
//         self.__str__()
//     }
// }
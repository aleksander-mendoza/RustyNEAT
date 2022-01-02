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
    ecc: Vec<SparseOrDense>,
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
        let layers:PyResult<Vec<SparseOrDense>> = (0..layers).map(|i| {
            let in_channels = channels[i];
            let out_channels = channels[i + 1];
            let k = k[i];
            let kernel = arr2(py, &kernels[i])?;
            let stride = arr2(py, &strides[i])?;
            Ok(if let Some(connections_per_output) = connections_per_output[i] {
                SparseOrDense::Sparse(htm::EccSparse::new(output, kernel, stride, in_channels, out_channels, k, connections_per_output, &mut rng))
            } else {
                SparseOrDense::Dense(htm::EccDense::new(output, kernel, stride, in_channels, out_channels, k, &mut rng))
            })
        }).collect();
        Ok(Self { ecc: layers? })
    }

    // #[getter]
    // pub fn get_in_shape(&self) -> Vec<usize> {
    //     self.ecc.in_shape().to_vec()
    // }
    // #[getter]
    // pub fn get_out_shape(&self) -> Vec<usize> {
    //     self.ecc.out_shape().to_vec()
    // }
    // #[text_signature = "(layer)"]
    // pub fn get_k(&self,layer:usize) -> usize {
    //     self.ecc[layer].k
    // }
    // #[text_signature = "(layer, k)"]
    // pub fn set_k(&mut self, layer:usize,k: usize) {
    //     self.ecc[layer].k = k
    // }
    // #[text_signature = "(layer)"]
    // pub fn get_threshold(&self, layer:usize) -> u16 {
    //     self.ecc[layer].threshold
    // }
    // #[text_signature = "(layer, threshold)"]
    // pub fn set_threshold(&mut self, layer:usize, threshold: PyAny) {
    //     self.ecc[layer].threshold = threshold
    // }
    //
    // #[text_signature = "(input_sdr)"]
    // pub fn run(&mut self, input: &CpuSDR) -> CpuSDR {
    //     CpuSDR { sdr: self.ecc.run(&input.sdr) }
    // }
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

    #[getter]
    pub fn get_rand_seed(&self) -> usize {
        self.ecc.rand_seed
    }
    #[setter]
    pub fn set_rand_seed(&mut self, rand_seed: usize) {
        self.ecc.rand_seed = rand_seed
    }

    #[text_signature = "(input_sdr)"]
    pub fn run(&mut self, input: &CpuSDR) -> CpuSDR {
        CpuSDR { sdr: self.ecc.run(&input.sdr) }
    }
    #[text_signature = "()"]
    pub fn min_activity(&self) -> f32 {
        self.ecc.min_activity()
    }
    #[text_signature = "(output_idx)"]
    pub fn activity(&self, output_idx: usize) -> f32 {
        self.ecc.activity(output_idx)
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
    pub fn get_sums(&self) -> Vec<f32> {
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
    pub fn incoming_weight_sum(&self, output_neuron_idx: usize) -> f32 {
        self.ecc.incoming_weight_sum(output_neuron_idx)
    }
    #[text_signature = "(output_neuron_idx)"]
    pub fn normalise_weights(&mut self, output_neuron_idx: usize) {
        self.ecc.normalise_weights(output_neuron_idx)
    }
    #[text_signature = "()"]
    pub fn normalise_all_weights(&mut self) {
        self.ecc.normalise_all_weights()
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
    pub fn run(&mut self, input: &CpuSDR) -> CpuSDR {
        CpuSDR { sdr: self.ecc.run(&input.sdr) }
    }
    #[text_signature = "(file)"]
    pub fn save(&self, file: String) -> PyResult<()> {
        pickle(&self.ecc, file)
    }
}

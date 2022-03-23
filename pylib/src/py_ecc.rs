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
use htm::{VectorFieldOne, Idx, EccLayer, Rand, SDR, w_idx, ConvShape, Shape2, ConvWeightVec, HasConvShape, ConvShapeTrait, HasConvShapeMut, HasShape};
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
use crate::py_ecc_population::{ConvWeights, CpuEccPopulation, WeightSums};

pub type M = htm::MetricL1;

///
/// CpuEccDense(output: list[int], kernel: list[int], stride: list[int], in_channels: int, out_channels: int, k: int)
///
///
#[pyclass]
pub struct CpuEccDense {
    pub(crate) ecc: htm::CpuEccDense<M>,
}

///
/// CpuEccMachine(output: list[int], kernels: list[list[int]], strides: list[list[int]], channels: list[int], k: list[int])
///
///
#[pyclass]
pub struct CpuEccMachine {
    pub(crate) ecc: htm::CpuEccMachine<M>
}


#[pyproto]
impl PySequenceProtocol for CpuEccMachine {
    fn __len__(&self) -> usize {
        self.ecc.len()
    }
}



#[pyproto]
impl PySequenceProtocol for CpuEccDense {
    fn __len__(&self) -> usize {
        self.ecc.len()
    }
}


#[pymethods]
impl CpuEccMachine {
    #[text_signature = "(up_to_layer)"]
    pub fn composed_kernel_and_stride(&self, idx: Option<usize>) -> ([Idx; 2], [Idx; 2]) {
        let idx = idx.unwrap_or(self.ecc.len());
        self.ecc.composed_kernel_and_stride_up_to(idx)
    }
    #[getter]
    pub fn len(&self) -> usize {
        self.ecc.len()
    }
    #[getter]
    pub fn in_shape(&self) -> Option<Vec<Idx>> {
        self.ecc.in_shape().map(|f| f.to_vec())
    }
    #[getter]
    pub fn in_volume(&self) -> Option<u32> {
        self.ecc.in_volume()
    }
    #[getter]
    pub fn in_channels(&self) -> Option<u32> {
        self.ecc.in_channels()
    }
    #[getter]
    pub fn out_shape(&self) -> Option<Vec<Idx>> {
        self.ecc.out_shape().map(|f| f.to_vec())
    }
    #[getter]
    pub fn out_volume(&self) -> Option<u32> {
        self.ecc.out_volume()
    }
    #[getter]
    pub fn out_channels(&self) -> Option<u32> {
        self.ecc.out_channels()
    }
    #[getter]
    pub fn get_out_grid(&self) -> Option<Vec<Idx>> {
        self.ecc.out_grid().map(|f| f.to_vec())
    }
    #[getter]
    pub fn get_in_grid(&self) -> Option<Vec<Idx>> {
        self.ecc.in_grid().map(|f| f.to_vec())
    }
    #[text_signature = "(layer)"]
    pub fn get_in_shape(&self, layer: usize) -> Vec<Idx> {
        self.ecc[layer].cshape().in_shape().to_vec()
    }
    #[text_signature = "(layer)"]
    pub fn get_out_shape(&self, layer: usize) -> Vec<Idx> {
        self.ecc[layer].cshape().out_shape().to_vec()
    }
    #[text_signature = "(layer)"]
    pub fn get_kernel(&self, layer: usize) -> Vec<Idx> {
        self.ecc[layer].cshape().kernel().to_vec()
    }
    #[text_signature = "(layer)"]
    pub fn get_in_channels(&self, layer: usize) -> Idx {
        self.ecc[layer].cshape().in_channels()
    }
    #[text_signature = "(layer)"]
    pub fn get_out_channels(&self, layer: usize) -> Idx {
        self.ecc[layer].cshape().out_channels()
    }
    #[text_signature = "(layer)"]
    pub fn get_in_volume(&self, layer: usize) -> Idx {
        self.ecc[layer].cshape().in_volume()
    }
    #[text_signature = "(layer)"]
    pub fn get_out_volume(&self, layer: usize) -> Idx {
        self.ecc[layer].cshape().out_volume()
    }
    #[text_signature = "(layer)"]
    pub fn get_stride(&self, layer: usize) -> Vec<Idx> {
        self.ecc[layer].cshape().stride().to_vec()
    }
    #[text_signature = "(layer)"]
    pub fn get_k(&self, layer: usize) -> Idx {
        self.ecc[layer].k()
    }
    #[text_signature = "(layer, k)"]
    pub fn set_k(&mut self, layer: usize, k: Idx) {
        self.ecc[layer].set_k(k)
    }
    #[text_signature = "(input_sdr)"]
    pub fn learnable_parameters(&self) -> usize {
        self.ecc.learnable_parameters()
    }
    #[text_signature = "(input_sdr, up_to_layer, learn, update_activity)"]
    pub fn run(&mut self, input: &CpuSDR, up_to_layer: Option<usize>, learn: Option<bool>, update_activity: Option<bool>) {
        let up_to_layer = up_to_layer.unwrap_or(self.ecc.len());
        self.ecc.infer_up_to_layer(up_to_layer, &input.sdr);
        if update_activity.unwrap_or(true) {
            self.ecc.decrement_activities_up_to_layer(up_to_layer)
        }
        if learn.unwrap_or(false) {
            self.ecc.learn_up_to_layer(up_to_layer)
        }
    }
    #[text_signature = "(layer_idx)"]
    pub fn get_layer(&self, layer: usize) -> CpuEccDense {
        CpuEccDense { ecc: self.ecc[layer].clone() }
    }
    #[text_signature = "(layer_idx,layer_instance)"]
    pub fn set_layer(&mut self, layer: usize, net: &CpuEccDense) {
        assert_eq!(self.ecc[layer].in_shape(), net.ecc.in_shape(), "Input shapes don't match");
        assert_eq!(self.ecc[layer].out_shape(), net.ecc.out_shape(), "Output shapes don't match");
        self.ecc[layer] = net.ecc.clone()
    }
    #[text_signature = "(input_sdr,learn)"]
    pub fn infer(&mut self, input: &CpuSDR, up_to_layer: Option<usize>, learn: Option<bool>) {
        self.run(input, up_to_layer, learn, Some(false));
    }
    #[text_signature = "(input_sdr,layer,learn)"]
    pub fn infer_layer(&mut self, input: &CpuSDR, layer: usize, learn: Option<bool>) -> CpuSDR {
        self.run_layer(input, layer, learn, Some(false))
    }

    #[text_signature = "(input_sdr,layer,learn,update_activity)"]
    pub fn run_layer(&mut self, input: &CpuSDR, layer: usize, learn: Option<bool>, update_activity: Option<bool>) -> CpuSDR {
        let out = self.ecc[layer].infer(&input.sdr);
        if update_activity.unwrap_or(true) {
            self.ecc[layer].decrement_activities(&out)
        }
        if learn.unwrap_or(false) {
            self.ecc[layer].learn(&input.sdr, &out)
        }
        CpuSDR { sdr: out }
    }
    #[text_signature = "(up_to_layer)"]
    pub fn learn(&mut self, up_to_layer: Option<usize>) {
        self.ecc.learn_up_to_layer(up_to_layer.unwrap_or(self.ecc.len()))
    }
    #[text_signature = "()"]
    pub fn last_output_sdr(&self) -> CpuSDR {
        CpuSDR { sdr: self.ecc.last_output_sdr().clone() }
    }
    #[text_signature = "()"]
    pub fn last_output_shape(&self) -> Vec<Idx> {
        self.ecc.last().cshape().out_shape().to_vec()
    }
    #[text_signature = "()"]
    pub fn last_output_channels(&self) -> Idx {
        self.ecc.last().cshape().out_channels()
    }
    #[text_signature = "()"]
    pub fn item(&self) -> Option<Idx> {
        let o = self.ecc.last_output_sdr();
        if o.is_empty() { None } else { Some(o.item()) }
    }
    #[text_signature = "(layer_index)"]
    pub fn output_sdr(&self, layer_index: usize) -> CpuSDR {
        CpuSDR { sdr: self.ecc.output_sdr(layer_index).clone() }
    }
    #[text_signature = "(layer_index)"]
    pub fn input_sdr(&self, layer_index: usize) -> CpuSDR {
        CpuSDR { sdr: self.ecc.input_sdr(layer_index).clone() }
    }
    #[text_signature = "(layer)"]
    pub fn get_threshold(&self, layer: usize) -> f32 {
        self.ecc[layer].get_threshold()
    }
    #[text_signature = "(layer, threshold)"]
    pub fn set_threshold(&mut self, layer: usize, threshold: f32) {
        self.ecc[layer].set_threshold(threshold)
    }
    #[text_signature = "(final_column_grid)"]
    pub fn repeat_column(&self, py: Python, final_column_grid: PyObject) -> PyResult<Self> {
        let final_column_grid = arr2(py, &final_column_grid)?;
        Ok(Self { ecc: htm::CpuEccMachine::from_repeated_column(final_column_grid, &self.ecc) })
    }
    #[text_signature = "(layer)"]
    pub fn get_plasticity(&self, layer: usize) -> f32 {
        self.ecc[layer].get_plasticity()
    }
    #[text_signature = "(layer, plasticity)"]
    pub fn set_plasticity(&mut self, layer: usize, plasticity: f32) {
        self.ecc[layer].set_plasticity(plasticity)
    }
    #[text_signature = "(plasticity)"]
    pub fn set_plasticity_everywhere(&mut self, plasticity: f32) {
        self.ecc.set_plasticity_everywhere(plasticity)
    }
    #[text_signature = "(layer)"]
    pub fn get_max_incoming_synapses(&self, layer: usize) -> Idx {
        self.ecc[layer].get_max_incoming_synapses()
    }
    #[text_signature = "(layer,file)"]
    pub fn save_layer(&self, layer: usize, file: String) -> PyResult<()> {
        pickle(&self.ecc[layer], file)
    }
    #[new]
    pub fn new(output: Option<PyObject>, kernels: Option<Vec<PyObject>>, strides: Option<Vec<PyObject>>, channels: Option<Vec<Idx>>, k: Option<Vec<Idx>>) -> PyResult<Self> {
        let kernels = kernels.unwrap_or(vec![]);
        let strides = strides.unwrap_or(vec![]);
        let channels = channels.unwrap_or(vec![]);
        let k = k.unwrap_or(vec![]);
        let layers = kernels.len();
        if layers != strides.len() {
            return Err(PyValueError::new_err(format!("{}==len(kernels)!=len(strides)=={}", layers, strides.len())));
        }
        if layers != k.len() {
            return Err(PyValueError::new_err(format!("{}==len(kernels)!=len(k)=={}", layers, k.len())));
        }
        let output = if let Some(output) = output {
            output
        } else {
            if layers != 0 {
                return Err(PyValueError::new_err(format!("0==len(kernels) but output is missing")));
            }
            return Ok(Self { ecc: htm::CpuEccMachine::new_empty() });
        };
        if layers + 1 != channels.len() {
            return Err(PyValueError::new_err(format!("{}==len(kernels)+1!=len(channels)=={}", layers + 1, channels.len())));
        }
        let gil = Python::acquire_gil();
        let py = gil.python();
        let output = arr2(py, &output)?;
        let mut rng = rand::thread_rng();
        let kernels: PyResult<Vec<[Idx; 2]>> = kernels.into_iter().map(|k| arr2(py, &k)).collect();
        let strides: PyResult<Vec<[Idx; 2]>> = strides.into_iter().map(|k| arr2(py, &k)).collect();
        Ok(Self { ecc: htm::CpuEccMachine::new_cpu(output, &kernels?, &strides?, &channels, &k, &mut rng) })
    }
    #[text_signature = "(layer)"]
    pub fn restore_dropped_out_weights(&mut self, layer: usize) {
        self.ecc[layer].restore_dropped_out_weights()
    }
    #[text_signature = "(layer,number_of_connections_to_drop,per_kernel,normalise)"]
    pub fn dropout(&mut self, py: Python, layer: usize, number_of_connections_to_drop: PyObject, per_kernel: Option<bool>, normalise: Option<bool>) -> PyResult<()> {
        let per_kernel = per_kernel.unwrap_or(false);
        let normalise = normalise.unwrap_or(true);
        let a = &mut self.ecc[layer];
        if PyFloat::is_exact_type_of(number_of_connections_to_drop.as_ref(py)) {
            if per_kernel {
                a.dropout_per_kernel_f32(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
            } else {
                a.dropout_f32(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
            }
        } else {
            if per_kernel {
                a.dropout_per_kernel(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
            } else {
                a.dropout(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
            }
        }
        if normalise {
            a.normalize_all()
        }
        Ok(())
    }
    #[text_signature = "(layer,output_idx)"]
    pub fn normalize(&mut self, layer: usize, output_idx: Option<Idx>) {
        if let Some(output_idx) = output_idx {
            self.ecc[layer].normalize(output_idx)
        } else {
            self.ecc[layer].normalize_all()
        }
    }
    #[text_signature = "(layer)"]
    pub fn get_sums(&self, py: Python, layer: usize) -> PyObject {
        self.ecc[layer].population().sums.to_object(py)
    }

    #[text_signature = "()"]
    pub fn pop(&mut self) -> Option<CpuEccDense> {
        self.ecc.pop().map(|ecc| CpuEccDense { ecc })
    }

    #[text_signature = "()"]
    pub fn pop_front(&mut self) -> Option<CpuEccDense> {
        self.ecc.pop_front().map(|ecc| CpuEccDense { ecc })
    }

    #[text_signature = "(top_layer)"]
    pub fn push(&mut self, top: &CpuEccDense) {
        self.ecc.push(top.ecc.clone())
    }
    #[text_signature = "(bottom_layer)"]
    pub fn prepend(&mut self, bottom: &CpuEccDense) {
        self.ecc.prepend(bottom.ecc.clone())
    }
    #[text_signature = "(top_layer,column_pos)"]
    pub fn push_repeated_column(&mut self, top: &CpuEccDense, column_pos: Option<[Idx; 2]>) {
        self.ecc.push_repeated_column(&top.ecc, column_pos.unwrap_or([0, 0]))
    }
    #[text_signature = "(bottom_layer,column_pos)"]
    pub fn prepend_repeated_column(&mut self, bottom: &CpuEccDense, column_pos: Option<[Idx; 2]>) {
        self.ecc.prepend_repeated_column(&bottom.ecc, column_pos.unwrap_or([0, 0]))
    }
    #[text_signature = "(layer)"]
    pub fn set_initial_activity(&mut self, layer: usize) {
        self.ecc[layer].population_mut().set_initial_activity()
    }
    #[text_signature = "(layer)"]
    pub fn reset_activity(&mut self, layer: usize) {
        self.ecc[layer].population_mut().reset_activity()
    }
    #[text_signature = "(layer)"]
    pub fn get_activity(&self, layer: usize) -> Vec<f32> {
        self.ecc[layer].population().get_activity().to_vec()
    }

    #[text_signature = "(layer,output_neuron)"]
    pub fn get_dropped_weights_count(&self, layer: usize, output_neuron: Option<Idx>) -> usize {
        let a = &self.ecc[layer];
        if let Some(output_neuron_idx) = output_neuron {
            a.get_dropped_weights_of_kernel_column_count(output_neuron_idx)
        } else {
            a.get_dropped_weights_count()
        }
    }
    #[text_signature = "(layer,output_neuron)"]
    pub fn get_weight_sum(&self, layer: usize, output_neuron: Idx) -> f32 {
        self.ecc[layer].incoming_weight_sum(output_neuron)
    }
    #[text_signature = "(layer,output_neuron_idx)"]
    pub fn get_weights(&self, layer: usize, output_neuron_idx: Option<Idx>) -> Vec<f32> {
        if let Some(output_neuron_idx) = output_neuron_idx {
            self.ecc[layer].weights().incoming_weight_copy(output_neuron_idx).to_vec()
        } else {
            self.ecc[layer].weight_slice().to_vec()
        }
    }
    #[text_signature = "(layer)"]
    pub fn sums_for_output_sdr(&self, layer: usize) -> f32 {
        self.ecc[layer].population().sums_for_sdr(self.ecc.output_sdr(layer))
    }
    #[text_signature = "(layer, sdr)"]
    pub fn sums_for_sdr(&self, layer: usize, sdr: &CpuSDR) -> f32 {
        self.ecc[layer].population().sums_for_sdr(&sdr.sdr)
    }
}

impl_save_load!(CpuEccMachine,ecc);
impl_save_load!(CpuEccDense,ecc);


#[pymethods]
impl CpuEccDense {
    #[text_signature = "(list_of_input_sdrs)"]
    pub fn batch_infer(&self, input: Vec<PyRef<CpuSDR>>) -> Vec<CpuSDR> {
        self.ecc.batch_infer(&input, |s| &s.sdr, |o| CpuSDR { sdr: o })
    }
    #[text_signature = "()"]
    pub fn reset_sums(&mut self) {
        self.ecc.population_mut().reset_sums()
    }
    #[text_signature = "(output)"]
    pub fn decrement_activities(&mut self, output: &mut CpuSDR) {
        self.ecc.decrement_activities(&mut output.sdr)
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
    #[text_signature = "(input_sdr,output_sdr,learn, stored_sums, update_activity)"]
    pub fn run_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR, learn: Option<bool>, stored_sums: Option<&mut WeightSums>, update_activity: Option<bool>) {
        self.ecc.infer_in_place(&input.sdr, &mut output.sdr);
        if update_activity.unwrap_or(true) {
            self.ecc.decrement_activities(&output.sdr)
        }
        if learn.unwrap_or(false) {
            self.learn(input, output, stored_sums)
        }
    }
    #[text_signature = "(input_sdr,output_sdr,learn, stored_sums)"]
    pub fn infer_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR, learn: Option<bool>, stored_sums: Option<&mut WeightSums>) {
        self.run_in_place(input, output, learn, stored_sums, Some(false))
    }

    #[text_signature = "(input_sdr, learn, stored_sums, update_activity)"]
    pub fn run(&mut self, input: &CpuSDR, learn: Option<bool>, stored_sums: Option<&mut WeightSums>, update_activity: Option<bool>) -> CpuSDR {
        let out = self.ecc.infer(&input.sdr);
        if update_activity.unwrap_or(true) {
            self.ecc.decrement_activities(&out)
        }
        let out = CpuSDR { sdr: out };
        if learn.unwrap_or(false) {
            self.learn(input, &out, stored_sums)
        }
        out
    }
    #[text_signature = "(input_sdr, learn, stored_sums)"]
    pub fn infer(&mut self, input: &CpuSDR, learn: Option<bool>, stored_sums: Option<&mut WeightSums>) -> CpuSDR {
        self.run(input, learn, stored_sums, Some(false))
    }

    #[text_signature = "(input_sdr,output_sdr,stored_sums)"]
    pub fn learn(&mut self, input: &CpuSDR, output: &CpuSDR, stored_sums: Option<&mut WeightSums>) {
        if let Some(stored_sums) = stored_sums {
            self.ecc.learn_and_store_sums(&input.sdr, &output.sdr, &mut stored_sums.ecc)
        } else {
            self.ecc.learn(&input.sdr, &output.sdr)
        }
    }
    #[text_signature = "()"]
    pub fn to_machine(&self) -> CpuEccMachine {
        CpuEccMachine { ecc: self.ecc.clone().into_machine() }
    }
    #[text_signature = "()"]
    pub fn restore_dropped_out_weights(&mut self) {
        self.ecc.restore_dropped_out_weights()
    }
    #[text_signature = "(number_of_connections_to_drop,per_kernel,normalise)"]
    pub fn dropout(&mut self, py: Python, number_of_connections_to_drop: PyObject, per_kernel: Option<bool>, normalise: Option<bool>) -> PyResult<()> {
        let per_kernel = per_kernel.unwrap_or(false);
        let normalise = normalise.unwrap_or(true);
        if PyFloat::is_exact_type_of(number_of_connections_to_drop.as_ref(py)) {
            if per_kernel {
                self.ecc.dropout_per_kernel_f32(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
            } else {
                self.ecc.dropout_f32(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
            }
        } else {
            if per_kernel {
                self.ecc.dropout_per_kernel(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
            } else {
                self.ecc.dropout(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
            }
        }
        if normalise {
            self.ecc.normalize_all()
        }
        Ok(())
    }
    #[text_signature = "(output_idx)"]
    pub fn normalize(&mut self, output_idx: Option<Idx>) {
        if let Some(output_idx) = output_idx {
            self.ecc.normalize(output_idx)
        } else {
            self.ecc.normalize_all()
        }
    }
    #[getter]
    pub fn get_region_size(&self) -> Idx {
        self.ecc.population().get_region_size()
    }
    #[new]
    pub fn new(output: PyObject, kernel: PyObject, stride: PyObject, in_channels: Idx, out_channels: Idx, k: Idx) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let output = arr2(py, &output)?;
        let kernel = arr2(py, &kernel)?;
        let stride = arr2(py, &stride)?;
        Ok(CpuEccDense { ecc: htm::CpuEccDense::new(ConvShape::new(output, kernel, stride, in_channels, out_channels), k, &mut rand::thread_rng()) })
    }
    #[staticmethod]
    #[text_signature = "(input, kernel, stride,in_channels, out_channels)"]
    pub fn new_in(input: PyObject, kernel: PyObject, stride: PyObject, in_channels: Idx, out_channels: Idx, k: Idx) -> PyResult<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let input = arr2(py, &input)?;
        let kernel = arr2(py, &kernel)?;
        let stride = arr2(py, &stride)?;
        Ok(CpuEccDense { ecc: htm::CpuEccDense::new(ConvShape::new_in(input.add_channels(in_channels), out_channels, kernel, stride), k, &mut rand::thread_rng()) })
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
        Ok(Self { ecc: htm::CpuEccDense::from_repeated_column(output, &self.ecc, pretrained_column_pos) })
    }
    #[staticmethod]
    #[text_signature = "(weights, population)"]
    pub fn new_from(weights: &ConvWeights, population: &CpuEccPopulation) -> Self {
        assert_eq!(weights.ecc.out_shape(), population.ecc.shape());
        Self { ecc: htm::CpuEccDense::from(weights.ecc.clone(), population.ecc.clone()) }
    }
    #[staticmethod]
    #[text_signature = "(input,output,k)"]
    pub fn new_linear(input: Idx, output: Idx, k: Option<Idx>) -> Self {
        let k = k.unwrap_or(1);
        Self { ecc: htm::CpuEccDense::new(ConvShape::new_linear(input, output), k, &mut rand::thread_rng()) }
    }
    #[staticmethod]
    #[text_signature = "(shape,k)"]
    pub fn new_identity(py: Python, shape: PyObject, k: Option<Idx>) -> PyResult<Self> {
        let k = k.unwrap_or(1);
        let shape = arrX(py, &shape, 1, 1, 1)?;
        Ok(Self { ecc: htm::CpuEccDense::new(ConvShape::new_identity(shape), k, &mut rand::thread_rng()) })
    }
    #[staticmethod]
    #[text_signature = "(layers)"]
    pub fn concat(layers: Vec<PyRef<Self>>) -> Self {
        Self { ecc: htm::CpuEccDense::concat(&layers, |s| &s.ecc) }
    }

    #[text_signature = "()"]
    pub fn min_activity(&self) -> f32 {
        self.ecc.population().min_activity()
    }
    #[text_signature = "()"]
    pub fn max_activity(&self) -> f32 {
        self.ecc.population().max_activity()
    }
    #[text_signature = "(output_idx)"]
    pub fn activity(&self, output_idx: usize) -> f32 {
        self.ecc.population().activity(output_idx)
    }
    #[text_signature = "()"]
    pub fn get_activity(&self) -> Vec<f32> {
        self.ecc.population().get_activity().to_vec()
    }
    #[text_signature = "(input_pos,output_pos)"]
    pub fn w_index(&self, input_pos: PyObject, output_pos: PyObject) -> PyResult<Idx> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let input_pos = arr3(py, &input_pos)?;
        let output_pos = arr3(py, &output_pos)?;
        Ok(self.ecc.w_index(&input_pos, &output_pos))
    }
    #[text_signature = "()"]
    pub fn set_initial_activity(&mut self) {
        self.ecc.population_mut().set_initial_activity()
    }
    #[text_signature = "()"]
    pub fn reset_activity(&mut self) {
        self.ecc.population_mut().reset_activity()
    }
    #[getter]
    pub fn get_sums(&self) -> WeightSums {
        WeightSums { ecc: htm::ShapedArray::from(*self.ecc.out_shape(), self.ecc.population().sums.clone()) }
    }

    #[text_signature = "(output_neuron_idx)"]
    pub fn incoming_weight_sum(&self, output_neuron_idx: Idx) -> f32 {
        self.ecc.incoming_weight_sum(output_neuron_idx)
    }
    #[text_signature = "(sdr)"]
    pub fn sums_for_sdr(&self, sdr: &CpuSDR) -> f32 {
        self.ecc.population().sums_for_sdr(&sdr.sdr)
    }
    #[text_signature = "(output_neuron_idx)"]
    pub fn get_weights(&self, output_neuron_idx: Option<Idx>) -> Vec<f32> {
        if let Some(output_neuron_idx) = output_neuron_idx {
            self.ecc.weights().incoming_weight_copy(output_neuron_idx).to_vec()
        } else {
            self.ecc.weight_slice().to_vec()
        }
    }
}




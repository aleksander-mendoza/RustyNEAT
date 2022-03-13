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
use htm::{VectorFieldOne, Idx, EccLayer, Rand, SDR, w_idx, ConvShape, Shape3, Shape2, ShapedArray, HasConvShape, ConvWeightVec, Metric, MetricL1, MetricL2, HasShape, ForwardTarget};
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
pub struct WeightSums {
    pub ecc: htm::ShapedArray<f32>,
}

///
/// CpuEccPopulation(shape, k: int)
///
///
#[pyclass]
pub struct CpuEccPopulation {
    pub ecc: htm::CpuEccPopulation<MetricL2>,
}

///
/// ConvWeights(output, kernel, stride, in_channels)
///
///
#[pyclass]
pub struct ConvWeights {
    pub ecc: htm::ConvWeights<MetricL2>,
}

#[pymethods]
impl WeightSums {
    #[text_signature = "(index)"]
    pub fn get(&self, i:usize) -> f32 {
        self.ecc[i]
    }
    #[text_signature = "()"]
    pub fn as_list(&self) -> Vec<f32> {
        self.ecc.to_vec()
    }
    #[text_signature = "(index, value)"]
    pub fn set(&mut self, i:usize, v:f32) {
        self.ecc[i] = v
    }
    #[text_signature = "(value)"]
    pub fn fill_all(&mut self, value:f32) {
        self.ecc.fill(value)
    }
    #[text_signature = "(value,sdr,parallel)"]
    pub fn fill(&mut self,value:f32, sdr:&CpuSDR, parallel:Option<bool>) {
        if parallel.unwrap_or(false) {
            sdr.sdr.parallel_fill_into(value,&mut self.ecc)
        }else{
            sdr.sdr.fill_into(value,&mut self.ecc)
        }
    }
    #[new]
    pub fn new(shape: [Idx; 3],initial_value:Option<f32>) -> Self{
        Self { ecc: htm::ShapedArray::new(shape, initial_value.unwrap_or(0.)) }
    }
}

#[pyproto]
impl PySequenceProtocol for WeightSums {
    fn __len__(&self) -> usize {
        self.ecc.len()
    }
    fn __getitem__(&self, idx: isize) -> f32 {
        assert!(idx>=0);
        self.ecc[idx as usize]
    }

    fn __setitem__(&mut self, idx: isize, value: f32) {
        assert!(idx>=0);
        self.ecc[idx as usize] = value;
    }
}


#[pymethods]
impl ConvWeights {
    #[getter]
    pub fn get_out_grid(&self) -> Vec<Idx> {
        self.ecc.out_grid().to_vec()
    }
    #[getter]
    pub fn get_in_grid(&self) -> Vec<Idx> {
        self.ecc.in_grid().to_vec()
    }
    #[getter]
    fn get_plasticity(&self) -> f32 {
        self.ecc.get_plasticity()
    }
    #[setter]
    fn set_plasticity(&mut self, plasticity: f32) {
        self.ecc.set_plasticity(plasticity)
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
    #[text_signature = "(input_sdr,target_population)"]
    pub fn forward(&mut self, input: &CpuSDR, pop:&PyAny) -> PyResult<()>{
        if let Ok(mut pop) = pop.extract::<PyRefMut<CpuEccPopulation>>(){
            assert_eq!(pop.ecc.shape(),self.ecc.out_shape());
            self.ecc.forward(&input.sdr, &mut pop.ecc);
        } else {
            let mut pop = pop.extract::<PyRefMut<WeightSums>>()?;
            assert_eq!(pop.ecc.shape(),self.ecc.out_shape());
                self.ecc.forward(&input.sdr, &mut pop.ecc);
        }
        Ok(())
    }
    #[text_signature = "(input_sdr,target_population)"]
    pub fn inhibit(&mut self, input: &CpuSDR, pop:&PyAny) -> PyResult<()>{
        if let Ok(mut pop) = pop.extract::<PyRefMut<CpuEccPopulation>>(){
            assert_eq!(pop.ecc.shape(),self.ecc.out_shape());
            self.ecc.inhibit(&input.sdr, &mut pop.ecc);
        } else {
            let mut pop = pop.extract::<PyRefMut<WeightSums>>()?;
            assert_eq!(pop.ecc.shape(),self.ecc.out_shape());
            self.ecc.inhibit(&input.sdr, &mut pop.ecc);
        }
        Ok(())
    }
    #[text_signature = "(input_sdr,target_population)"]
    pub fn reset_and_forward(&mut self, input: &CpuSDR, pop:&mut CpuEccPopulation) {
        self.ecc.reset_and_forward(&input.sdr, &mut pop.ecc);
    }
    #[text_signature = "(output_sdr,weight_sums)"]
    pub fn normalize_with_stored_sums(&mut self, output: &CpuSDR, sums:&mut WeightSums) {
        assert_eq!(sums.ecc.shape(),self.ecc.out_shape());
            self.ecc.normalize_with_stored_sums(&output.sdr, &mut sums.ecc);
    }
    #[text_signature = "(input_sdr,output_sdr,target_population,learn, stored_sums, update_activity)"]
    pub fn run_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR, target:&mut CpuEccPopulation, learn: Option<bool>, stored_sums:Option<&mut WeightSums>, update_activity:Option<bool>) {
        assert_eq!(target.ecc.shape(),self.ecc.out_shape());
            self.ecc.infer_in_place(&input.sdr, &mut output.sdr, &mut target.ecc);
        if update_activity.unwrap_or(true) {
            target.decrement_activities(&output)
        }
        if learn.unwrap_or(false) {
            self.learn(input, output,stored_sums)
        }
    }
    #[text_signature = "(input_sdr,output_sdr,learn, stored_sums)"]
    pub fn infer_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR, target:&mut CpuEccPopulation,learn: Option<bool>, stored_sums:Option<&mut WeightSums>) {
        self.run_in_place(input,output,target,learn,stored_sums,Some(false))
    }

    #[text_signature = "(input_sdr, target_population, learn, stored_sums, update_activity)"]
    pub fn run(&mut self, input: &CpuSDR, target:&mut CpuEccPopulation,learn: Option<bool>, stored_sums:Option<&mut WeightSums>, update_activity:Option<bool>) -> CpuSDR {
        let out = self.ecc.infer(&input.sdr, &mut target.ecc);
        if update_activity.unwrap_or(true) {
            target.ecc.decrement_activities(&out)
        }
        let out = CpuSDR { sdr: out };
        if learn.unwrap_or(false) {
            self.learn(input,&out,stored_sums)
        }
        out
    }
    #[text_signature = "(list_of_input_sdrs, target_population)"]
    pub fn batch_infer(&self, input: Vec<PyRef<CpuSDR>>, target:&CpuEccPopulation) -> Vec<CpuSDR> {
        self.ecc.batch_infer(&input,|s|&s.sdr,target.ecc.clone(),|o|CpuSDR{sdr:o})
    }
    #[text_signature = "(input_sdr, target_population, learn, stored_sums)"]
    pub fn infer(&mut self, input: &CpuSDR, target:&mut CpuEccPopulation,learn: Option<bool>, stored_sums:Option<&mut WeightSums>) -> CpuSDR {
        self.run(input,target,learn,stored_sums,Some(false))
    }

    #[text_signature = "(input_sdr,output_sdr,stored_sums)"]
    pub fn learn(&mut self, input: &CpuSDR, output: &CpuSDR,stored_sums:Option<&mut WeightSums>) {
            if let Some(stored_sums) = stored_sums{
                self.ecc.learn_and_store_sums(&input.sdr, &output.sdr, &mut stored_sums.ecc)
            }else{
                self.ecc.learn(&input.sdr, &output.sdr)
            }
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
        Ok(Self { ecc: htm::ConvWeights::from_repeated_column(output, &self.ecc, pretrained_column_pos) })
    }

    #[staticmethod]
    #[text_signature = "(layers)"]
    pub fn concat(layers: Vec<PyRef<Self>>) -> Self {
        Self { ecc: htm::ConvWeights::concat(&layers, |s| &s.ecc) }
    }
    #[new]
    pub fn new(output: [Idx; 3], kernel: [Idx; 2], stride: [Idx; 2], in_channels: Idx) -> Self{
        let mut rng = rand::thread_rng();
        Self { ecc: htm::ConvWeights::new(htm::ConvShape::new_out(in_channels,output, kernel, stride),  &mut rng) }
    }
    #[staticmethod]
    #[text_signature = "(input, kernel, stride, out_channels)"]
    pub fn new_in(input: [Idx; 3], kernel: [Idx; 2], stride: [Idx; 2], out_channels: Idx) -> Self{
        let mut rng = rand::thread_rng();
        Self { ecc: htm::ConvWeights::new(htm::ConvShape::new_in(input, out_channels,kernel, stride),  &mut rng) }
    }
    #[staticmethod]
    #[text_signature = "(input,output)"]
    pub fn new_linear(input:Idx,output:Idx) -> Self {
        Self { ecc: htm::ConvWeights::new(ConvShape::new_linear(input,output),&mut rand::thread_rng()) }
    }
    #[staticmethod]
    #[text_signature = "(shape)"]
    pub fn new_identity(py:Python,shape:PyObject) -> PyResult<Self> {
        let shape = arrX(py,&shape,1,1,1)?;
        Ok(Self { ecc: htm::ConvWeights::new(ConvShape::new_identity(shape),&mut rand::thread_rng()) })
    }
    #[text_signature = "()"]
    pub fn restore_dropped_out_weights(&mut self) {
        self.ecc.restore_dropped_out_weights()
    }
    #[text_signature = "(number_of_connections_to_drop,per_kernel,normalise)"]
    pub fn dropout(&mut self, py: Python, number_of_connections_to_drop: PyObject,per_kernel:Option<bool>,normalise:Option<bool>) -> PyResult<()> {
        let per_kernel = per_kernel.unwrap_or(false);
        let normalise = normalise.unwrap_or(true);
        if PyFloat::is_exact_type_of(number_of_connections_to_drop.as_ref(py)) {
            if per_kernel{
                self.ecc.dropout_per_kernel_f32(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
            }else{
                self.ecc.dropout_f32(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
            }
        } else {
            if per_kernel{
                self.ecc.dropout_per_kernel(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
            }else{
                self.ecc.dropout(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
            }
        }
        if normalise{
            self.ecc.normalize_all()
        }
        Ok(())
    }

    #[text_signature = "(output_idx)"]
    pub fn normalize(&mut self, output_idx: Option<Idx>) {
        if let Some(output_idx) = output_idx {
                self.ecc.normalize(output_idx)
        }else{
                self.ecc.normalize_all()
        }

    }

    #[text_signature = "(output_neuron)"]
    pub fn get_dropped_weights_count(&self, output_neuron: Option<Idx>) -> usize {
        if let Some(output_neuron_idx) = output_neuron {
            self.ecc.get_dropped_weights_of_kernel_column_count(output_neuron_idx)
        } else {
            self.ecc.get_dropped_weights_count()
        }
    }
    #[text_signature = "(sums,reset)"]
    pub fn store_all_incoming_weight_sums(&self, sums:&mut WeightSums, reset:Option<bool>) {
        if reset.unwrap_or(false){
            self.ecc.store_all_incoming_weight_sums(&mut sums.ecc)
        }else{
            self.ecc.reset_and_store_all_incoming_weight_sums(&mut sums.ecc)
        }

    }
    #[text_signature = "(output_neuron)"]
    pub fn get_weight_sum(&self, output_neuron: Idx) -> f32 {
            self.ecc.incoming_weight_sum(output_neuron)
    }
    #[text_signature = "(output_idx)"]
    pub fn get_weights(&mut self, output_neuron_idx: Option<Idx>) -> Vec<f32>{
        if let Some(output_neuron_idx) = output_neuron_idx{
            self.ecc.incoming_weight_copy(output_neuron_idx).to_vec()
        } else {
            self.ecc.weight_slice().to_vec()
        }
    }
}

impl_save_load!(CpuEccPopulation,ecc);
impl_save_load!(ConvWeights,ecc);
impl_save_load!(WeightSums,ecc);

#[pymethods]
impl CpuEccPopulation {
    #[getter]
    pub fn get_shape(&self) -> Vec<Idx> {
        self.ecc.shape().to_vec()
    }
    #[getter]
    pub fn get_volume(&self) -> Idx {
        self.ecc.shape().product()
    }
    #[getter]
    pub fn get_area(&self) -> Idx {
        self.ecc.shape().grid().product()
    }
    #[getter]
    pub fn get_channels(&self) -> Idx {
        self.ecc.shape().channels()
    }
    #[getter]
    pub fn get_grid(&self) -> Vec<Idx> {
        self.ecc.shape().grid().to_vec()
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
    pub fn get_region_size(&self) -> Idx {
        self.ecc.get_region_size()
    }
    #[new]
    pub fn new(shape: [Idx; 3], k: Idx) -> Self {
        CpuEccPopulation { ecc: htm::CpuEccPopulation::new(shape, k) }
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
        Ok(Self { ecc: htm::CpuEccPopulation::from_repeated_column(output, &self.ecc, pretrained_column_pos) })
    }

    #[staticmethod]
    #[text_signature = "(layers)"]
    pub fn concat(layers: Vec<PyRef<Self>>) -> Self {
        Self { ecc: htm::CpuEccPopulation::concat(&layers, |s| &s.ecc) }
    }
    #[text_signature = "()"]
    pub fn reset_sums(&mut self) {
        self.ecc.reset_sums()
    }
    #[text_signature = "(output)"]
    pub fn determine_winners_with_threshold_in_place(&self,threshold:f32,output:&mut CpuSDR) {
        self.ecc.determine_winners_with_threshold(threshold,&mut output.sdr)
    }
    #[text_signature = "(output)"]
    pub fn determine_winners_topk_in_place(&self,output:&mut CpuSDR) {
        self.ecc.determine_winners_topk(&mut output.sdr)
    }
    #[text_signature = "(output)"]
    pub fn determine_winners_top1_per_region_in_place(&self,output:&mut CpuSDR) {
        self.ecc.determine_winners_top1_per_region(&mut output.sdr)
    }
    #[text_signature = "(threshold)"]
    pub fn determine_winners_with_threshold(&self,threshold:f32) ->CpuSDR{
        let mut output = CpuSDR{sdr:htm::CpuSDR::new()};
        self.ecc.determine_winners_with_threshold(threshold,&mut output.sdr);
        output
    }
    #[text_signature = "()"]
    pub fn determine_winners_topk(&self) ->CpuSDR{
        let mut output = CpuSDR{sdr:htm::CpuSDR::new()};
        self.ecc.determine_winners_topk(&mut output.sdr);
        output
    }
    #[text_signature = "()"]
    pub fn determine_winners_top1_per_region(&self)->CpuSDR {
        let mut output = CpuSDR{sdr:htm::CpuSDR::new()};
        self.ecc.determine_winners_top1_per_region(&mut output.sdr);
        output
    }
    #[text_signature = "(sdr)"]
    pub fn sums_for_sdr(&self, output:&CpuSDR)-> f32{
        self.ecc.sums_for_sdr(&output.sdr)
    }
    #[text_signature = "(sdr)"]
    pub fn decrement_activities(&mut self,sdr:&CpuSDR) {
        self.ecc.decrement_activities(&sdr.sdr)
    }
    #[text_signature = "()"]
    pub fn min_activity(&self) -> f32 {
        self.ecc.min_activity()
    }
    #[text_signature = "()"]
    pub fn max_activity(&self) -> f32 {
        self.ecc.max_activity()
    }
    #[text_signature = "(output_idx)"]
    pub fn activity(&self, output_idx: usize) -> f32 {
        self.ecc.activity(output_idx)
    }
    #[text_signature = "()"]
    pub fn get_activity(&self) -> Vec<f32> {
        self.ecc.get_activity().to_vec()
    }
    #[text_signature = "()"]
    pub fn set_initial_activity(&mut self) {
        self.ecc.set_initial_activity()
    }
    #[text_signature = "()"]
    pub fn reset_activity(&mut self) {
        self.ecc.reset_activity()
    }
    #[getter]
    pub fn get_sums(&self) -> WeightSums {
        WeightSums{ecc:ShapedArray::from_pop(&self.ecc)}
    }

}



#[pyproto]
impl PyObjectProtocol for WeightSums {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.ecc.target()))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}
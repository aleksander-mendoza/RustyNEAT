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
use htm::{VectorFieldOne, Idx, SDR, w_idx, ConvShape, Shape2, HasConvShape, ConvShapeTrait, HasConvShapeMut, HasShape, EccLayerTrait, HasEccConfig, D, TensorTrait, ConvTensorTrait, L};
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
use crate::py_ecc_layer::EccLayer;
use crate::py_ecc_net_sdrs::EccNetSDRs;
use crate::py_ecc_config::EccConfig;
use crate::py_ecc_tensor::Tensor;


///
/// EccNet(output: list[int], kernels: list[list[int]], strides: list[list[int]], channels: list[int], k: list[int])
///
///
#[pyclass]
pub struct EccNet {
    pub(crate) ecc: htm::EccNet
}


#[pyproto]
impl PySequenceProtocol for EccNet {
    fn __len__(&self) -> usize {
        self.ecc.len()
    }
}



#[pymethods]
impl EccNet {
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
        self.ecc.layer(layer).in_shape().to_vec()
    }
    #[text_signature = "(layer)"]
    pub fn get_out_shape(&self, layer: usize) -> Vec<Idx> {
        self.ecc.layer(layer).out_shape().to_vec()
    }
    #[text_signature = "(layer)"]
    pub fn get_kernel(&self, layer: usize) -> Vec<Idx> {
        self.ecc.layer(layer).kernel().to_vec()
    }
    #[text_signature = "(layer)"]
    pub fn get_in_channels(&self, layer: usize) -> Idx {
        self.ecc.layer(layer).in_channels()
    }
    #[text_signature = "(layer)"]
    pub fn get_out_channels(&self, layer: usize) -> Idx {
        self.ecc.layer(layer).out_channels()
    }
    #[text_signature = "(layer)"]
    pub fn get_in_volume(&self, layer: usize) -> Idx {
        self.ecc.layer(layer).in_volume()
    }
    #[text_signature = "(layer)"]
    pub fn get_out_volume(&self, layer: usize) -> Idx {
        self.ecc.layer(layer).out_volume()
    }
    #[text_signature = "(layer)"]
    pub fn get_stride(&self, layer: usize) -> Vec<Idx> {
        self.ecc.layer(layer).stride().to_vec()
    }
    #[text_signature = "(layer)"]
    pub fn get_k(&self, layer: usize) -> Idx {
        self.ecc.layer(layer).k()
    }
    #[text_signature = "(layer, k)"]
    pub fn set_k(&mut self, layer: usize, k: Idx) {
        self.ecc.layer_mut(layer).set_k(k)
    }
    #[text_signature = "(input_sdr)"]
    pub fn learnable_parameters(&self) -> usize {
        self.ecc.learnable_parameters()
    }
    #[text_signature = "(layer_idx)"]
    pub fn get_layer(&self, layer: usize) -> EccLayer {
        EccLayer { ecc: self.ecc.layer(layer).clone() }
    }
    #[text_signature = "(layer_idx,layer_instance)"]
    pub fn set_layer(&mut self, layer: usize, net: &EccLayer) {
        assert_eq!(self.ecc.layer(layer).in_shape(), net.ecc.in_shape(), "Input shapes don't match");
        assert_eq!(self.ecc.layer(layer).out_shape(), net.ecc.out_shape(), "Output shapes don't match");
        assert!(self.ecc.layer(layer).cfg_compatible(&net.ecc),"Config not compatible");
        *self.ecc.layer_mut(layer) = net.ecc.clone()
    }
    #[text_signature = "(input_sdr,output_sdrs,learn)"]
    pub fn infer(&mut self, input: &CpuSDR, output:&mut EccNetSDRs, learn: Option<bool>) {
        self.ecc.infer(&input.sdr, &mut output.ecc, learn.unwrap_or(false))
    }
    #[text_signature = "(input_sdr,output_sdrs,learn)"]
    pub fn infer_rotating(&mut self, input: &CpuSDR, output:&mut EccNetSDRs, learn: Option<bool>) -> usize {
        self.ecc.infer_rotating(&input.sdr, &mut output.ecc, learn.unwrap_or(false))
    }
    #[text_signature = "(input_sdr,layer,learn)"]
    pub fn infer_new_sdr(&mut self, input: &CpuSDR, layer: usize, learn: Option<bool>) -> CpuSDR {
        CpuSDR{sdr:self.ecc.layer_mut(layer).infer_new_sdr(&input.sdr,  learn.unwrap_or(false))}
    }
    #[text_signature = "(input_sdr,output_sdr,layer,learn)"]
    pub fn infer_push(&mut self, input: &CpuSDR, output: &mut CpuSDR, layer: usize, learn: Option<bool>){
        self.ecc.layer_mut(layer).infer_push(&input.sdr, &mut output.sdr, learn.unwrap_or(false))
    }
    #[text_signature = "(input_sdr,output_sdr,layer,learn)"]
    pub fn infer_layer(&mut self, input: &CpuSDR, output: &mut CpuSDR, layer: usize, learn: Option<bool>){
        self.ecc.layer_mut(layer).infer(&input.sdr, &mut output.sdr, learn.unwrap_or(false))
    }
    #[text_signature = "(input_sdr,output_sdrs)"]
    pub fn learn(&mut self, input: &CpuSDR, output:&EccNetSDRs) {
        self.ecc.learn(&input.sdr,&output.ecc)
    }
    #[text_signature = "()"]
    pub fn last_output_shape(&self) -> Option<Vec<Idx>> {
        self.ecc.out_shape().map(|l|l.to_vec())
    }
    #[text_signature = "()"]
    pub fn last_output_channels(&self) -> Option<Idx> {
        self.ecc.out_channels()
    }
    #[text_signature = "(layer)"]
    pub fn get_threshold(&self, layer: usize) -> f32 {
        self.ecc.layer(layer).get_threshold()
    }
    #[text_signature = "(layer, threshold)"]
    pub fn set_threshold(&mut self, layer: usize, threshold: f32) {
        self.ecc.layer_mut(layer).set_threshold(threshold)
    }
    #[text_signature = "(final_column_grid)"]
    pub fn repeat_column(&self, py: Python, final_column_grid: PyObject) -> PyResult<Self> {
        let final_column_grid = arr2(py, &final_column_grid)?;
        Ok(Self { ecc: htm::EccNet::from_repeated_column(final_column_grid, &self.ecc) })
    }
    #[text_signature = "(layer)"]
    pub fn get_plasticity(&self, layer: usize) -> f32 {
        self.ecc.layer(layer).get_plasticity()
    }
    #[text_signature = "(layer, plasticity)"]
    pub fn set_plasticity(&mut self, layer: usize, plasticity: f32) {
        self.ecc.layer_mut(layer).set_plasticity(plasticity)
    }
    #[text_signature = "(plasticity)"]
    pub fn set_plasticity_everywhere(&mut self, plasticity: f32) {
        self.ecc.set_plasticity_everywhere(plasticity)
    }
    #[text_signature = "(layer)"]
    pub fn kernel_column_volume(&self, layer: usize) -> Idx {
        self.ecc.layer(layer).kernel_column_volume()
    }
    #[text_signature = "(layer,file)"]
    pub fn save_layer(&self, layer: usize, file: String) -> PyResult<()> {
        pickle(&self.ecc.layer(layer), file)
    }
    #[new]
    pub fn new(cfg:Option<&EccConfig>,output: Option<PyObject>, kernels: Option<Vec<PyObject>>, strides: Option<Vec<PyObject>>, channels: Option<Vec<Idx>>, k: Option<Vec<Idx>>) -> PyResult<Self> {
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
            return Ok(Self { ecc: htm::EccNet::empty() });
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
        if let Some(cfg) = cfg{
            Ok(Self { ecc: htm::EccNet::new(cfg.ecc.clone(),output, &kernels?, &strides?, &channels, &k, &mut rng) })
        }else{
            Err(PyValueError::new_err("EccConfig parameter is missing!"))
        }

    }
    // #[text_signature = "(layer)"]
    // pub fn restore_dropped_out_weights(&mut self, layer: usize) {
    //     self.ecc.layer(layer).restore_dropped_out_weights()
    // }
    // #[text_signature = "(layer,number_of_connections_to_drop,per_kernel,normalise)"]
    // pub fn dropout(&mut self, py: Python, layer: usize, number_of_connections_to_drop: PyObject, per_kernel: Option<bool>, normalise: Option<bool>) -> PyResult<()> {
    //     let per_kernel = per_kernel.unwrap_or(false);
    //     let normalise = normalise.unwrap_or(true);
    //     let a = &mut self.ecc.layer(layer);
    //     if PyFloat::is_exact_type_of(number_of_connections_to_drop.as_ref(py)) {
    //         if per_kernel {
    //             a.dropout_per_kernel_f32(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
    //         } else {
    //             a.dropout_f32(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
    //         }
    //     } else {
    //         if per_kernel {
    //             a.dropout_per_kernel(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
    //         } else {
    //             a.dropout(number_of_connections_to_drop.extract(py)?, &mut rand::thread_rng())
    //         }
    //     }
    //     if normalise {
    //         a.normalize_all()
    //     }
    //     Ok(())
    // }
    #[text_signature = "(layer,sdr)"]
    pub fn sparse_normalise(&mut self, layer: usize, sdr: &CpuSDR) {
        self.ecc.layer_mut(layer).sparse_normalise(&sdr.sdr)
    }
    #[text_signature = "(layer,output_idx)"]
    pub fn kernel_column_normalise(&mut self, layer: usize, output_idx: Idx) {
        self.ecc.layer_mut(layer).kernel_column_normalise(output_idx)
    }
    #[text_signature = "(layer)"]
    pub fn normalise(&mut self, layer: usize) {
        self.ecc.layer_mut(layer).normalise()
    }
    #[text_signature = "(layer)"]
    pub fn get_sums(&self, py: Python, layer: usize) -> Tensor {
        Tensor{ecc:self.ecc.layer(layer).sums.clone()}
    }

    #[text_signature = "()"]
    pub fn pop(&mut self) -> Option<EccLayer> {
        self.ecc.pop().map(|ecc| EccLayer { ecc })
    }

    #[text_signature = "()"]
    pub fn pop_front(&mut self) -> Option<EccLayer> {
        self.ecc.pop_front().map(|ecc| EccLayer { ecc })
    }

    #[text_signature = "(top_layer)"]
    pub fn push(&mut self, top: &EccLayer) {
        self.ecc.push(top.ecc.clone())
    }
    #[text_signature = "(bottom_layer)"]
    pub fn prepend(&mut self, bottom: &EccLayer) {
        self.ecc.prepend(bottom.ecc.clone())
    }
    #[text_signature = "(top_layer,column_pos)"]
    pub fn push_repeated_column(&mut self, top: &EccLayer, column_pos: Option<[Idx; 2]>) {
        self.ecc.push_repeated_column(&top.ecc, column_pos.unwrap_or([0, 0]))
    }
    #[text_signature = "(bottom_layer,column_pos)"]
    pub fn prepend_repeated_column(&mut self, bottom: &EccLayer, column_pos: Option<[Idx; 2]>) {
        self.ecc.prepend_repeated_column(&bottom.ecc, column_pos.unwrap_or([0, 0]))
    }
    #[text_signature = "(layer,value)"]
    pub fn fill_activity(&mut self, layer: usize, value:D) {
        self.ecc.layer_mut(layer).fill_activity(value)
    }
    #[text_signature = "(layer)"]
    pub fn get_activity(&self, layer: usize) -> Vec<f32> {
        self.ecc.layer(layer).activity().as_slice().to_vec()
    }

    // #[text_signature = "(layer,output_neuron)"]
    // pub fn get_dropped_weights_count(&self, layer: usize, output_neuron: Option<Idx>) -> usize {
    //     let a = &self.ecc.layer(layer);
    //     if let Some(output_neuron_idx) = output_neuron {
    //         a.get_dropped_weights_of_kernel_column_count(output_neuron_idx)
    //     } else {
    //         a.get_dropped_weights_count()
    //     }
    // }
    #[text_signature = "(layer,output_neuron)"]
    pub fn get_weight_sum(&self, layer: usize, output_neuron: Idx) -> D {
        self.ecc.layer(layer).weights().kernel_column_sum(output_neuron)
    }
    #[text_signature = "(layer,output_neuron)"]
    pub fn get_weight_square_sum(&self, layer: usize, output_neuron: Idx) -> D {
        self.ecc.layer(layer).weights().kernel_column_pow_sum::<L<2>>(output_neuron)
    }
    #[text_signature = "(layer,output_neuron_idx)"]
    pub fn get_weights(&self, layer: usize, output_neuron_idx: Option<Idx>) -> Vec<D> {
        if let Some(output_neuron_idx) = output_neuron_idx {
            self.ecc.layer(layer).weights().kernel_column_copy(output_neuron_idx)
        } else {
            self.ecc.layer(layer).weights().as_slice().to_vec()
        }
    }
    #[text_signature = "(layer)"]
    pub fn sums_sparse_dot(&self, layer: usize,output_sdr:&CpuSDR) -> f32 {
        self.ecc.layer(layer).sums().sparse_dot(&output_sdr.sdr)
    }
}

impl_save_load!(EccNet,ecc);





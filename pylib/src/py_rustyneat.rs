
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


#[pyfunction]
pub fn random_activation_fn() -> String {
    String::from(rusty_neat_core::activations::random_activation_fn().name())
}


#[pyfunction]
pub fn activation_functions() -> Vec<String> {
    Vec::from_iter(ALL_ACT_FN.iter().map(|s| String::from(s.name())))
}


#[pyclass]
pub struct FeedForwardNet32 {
    net: cppn::FeedForwardNet<f32>,
}

#[pyclass]
pub struct CPPN32 {
    cppn: cppn::CPPN<f32>,
}

#[pyclass]
#[text_signature = "(input_size, output_size, activation_functions, /)"]
pub struct Neat32 {
    neat: neat::Neat,
}

#[pyclass]
pub struct FeedForwardNetOpenCL32 {
    net: gpu::FeedForwardNetOpenCL,
}

#[pyclass]
pub struct FeedForwardNetPicbreeder32 {
    net: gpu::FeedForwardNetPicbreeder,
}

#[pyclass]
pub struct FeedForwardNetSubstrate32 {
    net: gpu::FeedForwardNetSubstrate,
}



#[pymethods]
impl Neat32 {
    #[text_signature = "(/)"]
    pub fn get_activation_functions(&self) -> Vec<String> {
        Vec::from_iter(self.neat.get_activation_functions().iter().map(|s| String::from(s.name())))
    }
    #[text_signature = "(/)"]
    pub fn get_random_activation_function(&self) -> String {
        String::from(self.neat.get_random_activation_function().name())
    }
    #[new]
    pub fn new(input_size: usize, output_size: usize, activations: Option<Vec<String>>) -> PyResult<Self> {
        let ac_fn =
            if let Some(activations) = activations {
                let mut ac_fn = vec![];
                for name in activations {
                    match STR_TO_IDX.get(&name) {
                        None => return Err(PyValueError::new_err(name + " is not a known function name")),
                        Some(&idx) => {
                            ac_fn.push(&ALL_ACT_FN[idx]);
                        }
                    }
                }
                ac_fn
            } else {
                Vec::from_iter(ALL_ACT_FN.iter())
            };
        Ok(Neat32 { neat: neat::Neat::new(ac_fn, input_size, output_size) })
    }
    #[text_signature = "(/)"]
    pub fn new_cppn(&mut self) -> CPPN32 {
        CPPN32 {
            cppn: self.neat.new_cppn(),
        }
    }
    #[text_signature = "(population_size, /)"]
    pub fn new_cppns(&mut self, num: usize) -> Vec<CPPN32> {
        self.neat.new_cppns(num).into_iter().map(|cppn| CPPN32 { cppn }).collect()
    }
    #[getter]
    fn input_size(&self) -> usize {
        self.neat.get_input_size()
    }
    #[getter]
    fn output_size(&self) -> usize {
        self.neat.get_output_size()
    }

    #[getter]
    fn current_innovation_number(&self) -> usize {
        self.neat.get_global_innovation_no()
    }

    #[text_signature = "(cppn, from_node, to_node, /)"]
    fn add_connection(&mut self, cppn: &mut CPPN32, from: usize, to: usize) -> PyResult<bool> {
        if from >= cppn.node_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} nodes but provided source index {}", cppn.node_count(), from)));
        }
        if to >= cppn.node_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} nodes but provided destination index {}", cppn.node_count(), from)));
        }
        Ok(self.neat.add_connection_if_possible(&mut cppn.cppn, from, to))
    }

    #[text_signature = "(cppn, /)"]
    fn add_random_connection(&mut self, cppn: &mut CPPN32) -> bool {
        self.neat.add_random_connection(&mut cppn.cppn)
    }

    #[text_signature = "(cppn, edge_to_split, /)"]
    fn add_node(&mut self, cppn: &mut CPPN32, edge: usize) -> PyResult<()> {
        if edge >= cppn.edge_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} edges but provided index {}", cppn.edge_count(), edge)));
        }
        Ok(self.neat.add_node(&mut cppn.cppn, edge))
    }

    #[text_signature = "(population,node_insertion_prob,edge_insertion_prob,activation_fn_mutation_prob,weight_mutation_prob,enable_edge_prob,disable_edge_prob /)"]
    fn mutate_population(&mut self, mut cppn: Vec<PyRefMut<CPPN32>>, node_insertion_prob: f32,
                         edge_insertion_prob: f32,
                         activation_fn_mutation_prob: f32,
                         weight_mutation_prob: f32,
                         enable_edge_prob: f32,
                         disable_edge_prob: f32) {
        self.neat.mutate_population(cppn.iter_mut().map(|c| &mut c.cppn),
                                    node_insertion_prob,
                                    edge_insertion_prob,
                                    activation_fn_mutation_prob,
                                    weight_mutation_prob,
                                    enable_edge_prob,
                                    disable_edge_prob)
    }

    #[text_signature = "(cppn,node_insertion_prob,edge_insertion_prob,activation_fn_mutation_prob,weight_mutation_prob,enable_edge_prob,disable_edge_prob /)"]
    fn mutate(&mut self, cppn: &mut CPPN32, node_insertion_prob: f32,
              edge_insertion_prob: f32,
              activation_fn_mutation_prob: f32,
              weight_mutation_prob: f32,
              enable_edge_prob: f32,
              disable_edge_prob: f32) {
        self.neat.mutate(&mut cppn.cppn,
                         node_insertion_prob,
                         edge_insertion_prob,
                         activation_fn_mutation_prob,
                         weight_mutation_prob,
                         enable_edge_prob,
                         disable_edge_prob)
    }

    #[text_signature = "(cppn, /)"]
    fn add_random_node(&mut self, cppn: &mut CPPN32) {
        self.neat.add_random_node(&mut cppn.cppn)
    }
    #[text_signature = "( /)"]
    pub fn random_weight(&self) -> f32 {
        f32::random()
    }
    #[text_signature = "(cppn, node_index, /)"]
    pub fn set_random_activation_function(&mut self, cppn: &mut CPPN32, node: usize) -> PyResult<bool> {
        if node >= cppn.node_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} nodes but provided index {}", cppn.node_count(), node)));
        }
        Ok(cppn.cppn.set_activation(node, self.neat.get_random_activation_function()))
    }
}


#[pymethods]
impl CPPN32 {
    #[getter]
    fn get_input_size(&self) -> usize {
        self.cppn.get_input_size()
    }
    #[getter]
    fn get_output_size(&self) -> usize {
        self.cppn.get_output_size()
    }
    #[text_signature = "(less_fit, /)"]
    fn crossover(&self, less_fit: &CPPN32) -> PyResult<CPPN32> {
        let input_size = self.get_input_size();
        let output_size = self.get_output_size();
        if less_fit.get_input_size() != input_size {
            return Err(PyValueError::new_err(format!("Fittter (right) CPPN has input size {} and less fit (left) CPPN has {}", input_size, less_fit.get_input_size())));
        }
        if less_fit.get_output_size() != output_size {
            return Err(PyValueError::new_err(format!("Fittter (right) CPPN has output size {} and less fit (left) CPPN has {}", output_size, less_fit.get_output_size())));
        }
        Ok(CPPN32 { cppn: self.cppn.crossover(&less_fit.cppn) })
    }
    #[text_signature = "(/)"]
    fn build_feed_forward_net(&self) -> FeedForwardNet32 {
        FeedForwardNet32 {
            net: self.cppn.build_feed_forward_net(),
        }
    }
    #[text_signature = "(node_index, function_name, /)"]
    fn set_activation_function(&mut self, node: usize, func: String) -> PyResult<bool> {
        if node >= self.node_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} nodes but provided index {}", self.node_count(), node)));
        }
        if let Some(&func) = STR_TO_IDX.get(&func) {
            Ok(self.cppn.set_activation(node, &ALL_ACT_FN[func]))
        } else {
            Err(PyValueError::new_err(format!("Unknown function {}", func)))
        }
    }

    #[text_signature = "(node_index, function_name, /)"]
    fn get_activation_function(&mut self, node: usize) -> PyResult<Option<String>> {
        if node >= self.node_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} nodes but provided index {}", self.node_count(), node)));
        }
        Ok(self.cppn.get_activation(node).map(|x| String::from(x.name())))
    }

    #[text_signature = "(/)"]
    fn node_count(&self) -> usize {
        self.cppn.node_count()
    }

    #[text_signature = "(/)"]
    fn get_random_node(&self) -> usize {
        self.cppn.get_random_node()
    }

    #[text_signature = "(/)"]
    fn get_random_edge(&self) -> usize {
        self.cppn.get_random_edge()
    }

    #[text_signature = "(/)"]
    fn edge_count(&self) -> usize {
        self.cppn.edge_count()
    }

    #[text_signature = "(edge_index, weight, /)"]
    fn set_weight(&mut self, edge: usize, weight: f32) -> PyResult<()> {
        if edge >= self.edge_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} edges but provided index {}", self.edge_count(), edge)));
        }
        Ok(self.cppn.set_weight(edge, weight))
    }

    #[text_signature = "(edge_index, /)"]
    fn get_weight(&mut self, edge: usize) -> PyResult<f32> {
        if edge >= self.edge_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} edges but provided index {}", self.edge_count(), edge)));
        }
        Ok(self.cppn.get_weight(edge))
    }
    #[text_signature = "(source_node, destination_node, /)"]
    fn search_connection_by_endpoints(&mut self, source_node: usize, destination_node: usize) -> PyResult<Option<usize>> {
        if source_node >= self.node_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} nodes but provided source node {}", self.node_count(), source_node)));
        }
        if destination_node >= self.node_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} nodes but provided destination node {}", self.node_count(), destination_node)));
        }
        Ok(self.cppn.search_connection_by_endpoints(source_node, destination_node))
    }

    #[text_signature = "(edge_index, /)"]
    fn flip_enabled(&mut self, edge: usize) -> PyResult<bool> {
        if edge >= self.edge_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} edges but provided index {}", self.edge_count(), edge)));
        }
        Ok(self.cppn.flip_enabled(edge))
    }

    #[text_signature = "(edge_index, enabled, /)"]
    fn set_enabled(&mut self, edge: usize, enabled: bool) -> PyResult<()> {
        if edge >= self.edge_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} edges but provided index {}", self.edge_count(), edge)));
        }
        Ok(self.cppn.set_enabled(edge, enabled))
    }


    #[text_signature = "(edge_index, /)"]
    fn is_enabled(&self, edge: usize) -> PyResult<bool> {
        if edge >= self.edge_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} edges but provided index {}", self.edge_count(), edge)));
        }
        Ok(self.cppn.is_enabled(edge))
    }

    #[text_signature = "(edge_index, /)"]
    fn edge_dest_node(&mut self, edge: usize) -> PyResult<usize> {
        if edge >= self.edge_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} edges but provided index {}", self.edge_count(), edge)));
        }
        Ok(self.cppn.edge_dest(edge))
    }

    #[text_signature = "(edge_index, /)"]
    fn edge_src_node(&mut self, edge: usize) -> PyResult<usize> {
        if edge >= self.edge_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} edges but provided index {}", self.edge_count(), edge)));
        }
        Ok(self.cppn.edge_src(edge))
    }

    #[text_signature = "(edge_index, /)"]
    fn edge_innovation_number(&mut self, edge: usize) -> PyResult<usize> {
        if edge >= self.edge_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} edges but provided index {}", self.edge_count(), edge)));
        }
        Ok(self.cppn.edge_innovation_no(edge))
    }
}


#[pymethods]
impl FeedForwardNet32 {
    #[call]
    fn __call__(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        if input.len() != self.get_input_size() {
            Err(PyValueError::new_err(format!("Expected input of size {} but got {}", self.get_input_size(), input.len())))
        } else {
            let mut out = vec![0f32; self.get_output_size()];
            self.net.run(input.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }

    // #[text_signature = "(input_tensor, /)"]
    // fn numpy<'py>(&self, py: Python<'py>, input: PyReadonlyArrayDyn<'_, f32>) -> PyResult<&'py PyArrayDyn<f32>> {
    //     let s = input.shape();
    //     if s.len() != 1 {
    //         return Err(PyValueError::new_err(format!("Expected input to be a 1 dimensional matrix but got {} dimensions {:?}", s.len(), s)));
    //     }
    //     if s[0] != self.get_input_size() {
    //         return Err(PyValueError::new_err(format!("Expected input to have length {} but got {}", self.get_input_size(), s[0])));
    //     }
    //     let mut out = vec![0f32; self.get_output_size()];
    //     self.net.run(input.as_slice()?, out.as_mut_slice());
    //     new_ndarray(py, [self.get_output_size()], out).map(|v| v.to_dyn())
    // }

    #[text_signature = "(neat_context, /)"]
    fn to(&self, context: &mut Context) -> PyResult<FeedForwardNetOpenCL32> {
        let lin_alg = context.compile_lin_alg_program()?;
        let n = self.net.to(lin_alg).map_err(ocl_err_to_py_ex)?;
        Ok(FeedForwardNetOpenCL32 { net: n })
    }
    #[text_signature = "(platform, device, /)"]
    fn to_picbreeder(&self, center: Option<Vec<f32>>, bias: Option<bool>, context: &mut Context) -> PyResult<FeedForwardNetPicbreeder32> {
        let lin_alg = context.compile_lin_alg_program()?;
        Ok(FeedForwardNetPicbreeder32 { net: self.net.to_picbreeder( center.as_ref(), bias.unwrap_or(false), lin_alg).map_err(ocl_err_to_py_ex)? })
    }
    #[text_signature = "(input_dimensions, output_dimensions, platform, device, /)"]
    fn to_substrate(&self, input_dimensions: usize, output_dimensions: Option<usize>, context: &mut Context) -> PyResult<FeedForwardNetSubstrate32> {
        let lin_alg = context.compile_lin_alg_program()?;
        Ok(FeedForwardNetSubstrate32 { net: self.net.to_substrate(input_dimensions, output_dimensions, lin_alg).map_err(ocl_err_to_py_ex)? })
    }
    #[getter]
    fn get_input_size(&self) -> usize {
        self.net.get_input_size()
    }
    #[getter]
    fn get_output_size(&self) -> usize {
        self.net.get_output_size()
    }
    #[text_signature = "( /)"]
    fn opencl_view(&self) -> String {
        format!("{}", self.net.opencl_view())
    }
    #[text_signature = "( /)"]
    fn picbreeder_view(&self, center: Option<Vec<f32>>, bias: bool) -> PyResult<String> {
        let p = self.net.picbreeder_view(center.as_ref().map(|v| v.as_slice()), bias).map_err(|e| PyValueError::new_err(e))?;
        Ok(format!("{}", p))
    }
    #[text_signature = "( /)"]
    fn substrate_view(&self, input_dimensions: usize, output_dimensions: Option<usize>) -> PyResult<String> {
        let expected_out_dims = self.net.get_input_size().checked_sub(input_dimensions).ok_or_else(|| PyValueError::new_err(format!("Substrate input dimensions {} is larger than CPPN's {}", input_dimensions, self.net.get_input_size())))?;
        let output_dimensions = output_dimensions.unwrap_or(expected_out_dims);
        let p = self.net.substrate_view(input_dimensions, output_dimensions).map_err(|e| PyValueError::new_err(e))?;
        Ok(format!("{}", p))
    }
}

#[pymethods]
impl FeedForwardNetPicbreeder32 {
    #[getter]
    fn get_input_size(&self) -> usize {
        self.net.get_input_size()
    }
    #[getter]
    fn get_output_size(&self) -> usize {
        self.net.get_output_size()
    }
    #[getter]
    fn get_device(&self) -> String {
        self.net.lin_alg().device().to_string()
    }
    #[call]
    // #[text_signature = "(pixel_count_per_dimension, pixel_size_per_dimension, location_offset_per_dimension,/)"]
    fn __call__(&self, pixel_count_per_dimension: Vec<usize>, pixel_size_per_dimension: Option<Vec<f32>>, location_offset_per_dimension: Option<Vec<f32>>) -> PyResult<DynMat> {
        let pixel_size_per_dimension = pixel_size_per_dimension.unwrap_or_else(|| vec![1f32; self.get_input_size()]);
        let location_offset_per_dimension = location_offset_per_dimension.unwrap_or_else(|| vec![0f32; self.get_input_size()]);
        if pixel_count_per_dimension.len() != self.get_input_size() {
            Err(PyValueError::new_err(format!("Expected pixel_count_per_dimension of size {} but got {}", self.get_input_size(), pixel_count_per_dimension.len())))
        } else if pixel_size_per_dimension.len() != self.get_input_size() {
            Err(PyValueError::new_err(format!("Expected pixel_size_per_dimension of size {} but got {}", self.get_input_size(), pixel_size_per_dimension.len())))
        } else if location_offset_per_dimension.len() != self.get_input_size() {
            Err(PyValueError::new_err(format!("Expected location_offset_per_dimension of size {} but got {}", self.get_input_size(), location_offset_per_dimension.len())))
        } else {
            self.net.run(pixel_count_per_dimension.as_slice(), pixel_size_per_dimension.as_slice(), location_offset_per_dimension.as_slice()).map(DynMat::from).map_err(ocl_err_to_py_ex)
        }
    }
}

#[pymethods]
impl FeedForwardNetOpenCL32 {
    #[getter]
    fn get_input_size(&self) -> usize {
        self.net.get_input_size()
    }
    #[getter]
    fn get_device(&self) -> String {
        self.net.lin_alg().device().to_string()
    }
    #[getter]
    fn get_output_size(&self) -> usize {
        self.net.get_output_size()
    }
    #[call]
    fn __call__(&self, input: &DynMat) -> PyResult<DynMat> {
        self.net.run(input.try_as_dtype::<f32>()?).map(DynMat::from).map_err(ocl_err_to_py_ex)
    }
}


#[pymethods]
impl FeedForwardNetSubstrate32 {
    #[getter]
    fn get_input_size(&self) -> usize {
        self.net.get_input_size()
    }
    #[getter]
    fn get_device(&self) -> String {
        self.net.lin_alg().device().to_string()
    }
    #[getter]
    fn get_output_size(&self) -> usize {
        self.net.get_output_size()
    }
    #[getter]
    fn get_weight_size(&self) -> usize {
        self.net.get_weight_size()
    }
    #[call]
    fn __call__(&self, input_neurons: &DynMat, output_neurons: &DynMat) -> PyResult<DynMat> {
        self.net.run(input_neurons.try_as_dtype::<f32>()?, output_neurons.try_as_dtype::<f32>()?).map(DynMat::from).map_err(ocl_err_to_py_ex)
    }
}


#[pyproto]
impl PyObjectProtocol for CPPN32 {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.cppn))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}



#[pyproto]
impl PyObjectProtocol for FeedForwardNet32 {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.net))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

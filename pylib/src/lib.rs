mod slice_box;

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
use ndarray::{ArrayViewD, IntoDimension, Dimension};
use numpy::{PyReadonlyArrayDyn, PyArrayDyn, IntoPyArray, PyArray, PY_ARRAY_API, npyffi, Element, ToNpyDims};
use numpy::npyffi::{NPY_ORDER, npy_intp, NPY_ARRAY_WRITEABLE};
use std::os::raw::c_int;
use rusty_neat_core::envs::evol::{AGENT_ATTRIBUTES, LIDAR_ATTRIBUTES};

#[pyfunction]
pub fn random_activation_fn() -> String {
    String::from(rusty_neat_core::activations::random_activation_fn().name())
}


#[pyfunction]
pub fn activation_functions() -> Vec<String> {
    Vec::from_iter(ALL_ACT_FN.iter().map(|s| String::from(s.name())))
}

#[pyclass]
#[derive(Clone, Copy)]
pub struct Platform {
    p: rusty_neat_core::Platform,
}

#[pyfunction]
pub fn platforms() -> Vec<Platform> {
    rusty_neat_core::opencl_platforms().into_iter().map(|p| Platform { p }).collect()
}

#[pyclass]
#[text_signature = "(platform, /)"]
#[derive(Clone, Copy)]
pub struct Device {
    d: rusty_neat_core::Device,
}

#[pyfunction]
#[text_signature = "(platform,/)"]
pub fn devices(p: Option<&Platform>) -> Vec<Device> {
    let d: Vec<rusty_neat_core::Device> = if let Some(p) = p {
        rusty_neat_core::device_list(&p.p)
    } else if let Some(p) = rusty_neat_core::opencl_platforms().into_iter().next() {
        rusty_neat_core::device_list(&p)
    } else {
        vec![]
    };
    d.into_iter().map(|d| Device { d }).collect()
}

// #[pyclass]
// pub struct Output32 {
//     out: Vec<f32>,
//     #[pyo3(get)]
//     input_size: usize,
//     #[pyo3(get)]
//     output_size: usize,
// }

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
impl Platform {
    #[text_signature = "( /)"]
    fn info(&self) -> String {
        self.p.to_string()
    }
    #[new]
    pub fn new() -> Self {
        Platform { p: rusty_neat_core::opencl_default_platform() }
    }
}

#[pymethods]
impl Device {
    #[text_signature = "( /)"]
    fn info(&self) -> String {
        self.d.to_string()
    }
    #[new]
    pub fn new(p: Option<Platform>) -> PyResult<Self> {
        rusty_neat_core::default_device(&p.map(|p| p.p).unwrap_or_else(|| rusty_neat_core::opencl_default_platform())).map(|d| Device { d }).ok_or_else(|| PyValueError::new_err("No device for default platform"))
    }
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
    fn is_enabled(&mut self, edge: usize) -> PyResult<bool> {
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

fn ocl_err_to_py_ex(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
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

    #[text_signature = "(input_tensor, /)"]
    fn numpy<'py>(&self, py: Python<'py>, input: PyReadonlyArrayDyn<'_, f32>) -> PyResult<&'py PyArrayDyn<f32>> {
        let s = input.shape();
        if s.len() != 1 {
            return Err(PyValueError::new_err(format!("Expected input to be a 1 dimensional matrix but got {} dimensions {:?}", s.len(), s)));
        }
        if s[0] != self.get_input_size() {
            return Err(PyValueError::new_err(format!("Expected input to have length {} but got {}", self.get_input_size(), s[0])));
        }
        let mut out = vec![0f32; self.get_output_size()];
        self.net.run(input.as_slice()?, out.as_mut_slice());
        new_ndarray(py, [self.get_output_size()], out).map(|v| v.to_dyn())
    }

    #[text_signature = "(platform, device, /)"]
    fn to(&self, platform: Option<Platform>, device: Option<Device>) -> PyResult<FeedForwardNetOpenCL32> {
        let p = platform.map(|p| p.p).unwrap_or_else(|| rusty_neat_core::opencl_default_platform());
        let d = device.map(|d| d.d).or_else(|| rusty_neat_core::default_device(&p)).ok_or_else(|| PyValueError::new_err(format!("No device for {}", &p)))?;
        let n = self.net.to(p, d).map_err(ocl_err_to_py_ex)?;
        Ok(FeedForwardNetOpenCL32 { net: n })
    }
    #[text_signature = "(platform, device, /)"]
    fn to_picbreeder(&self, center: Option<Vec<f32>>, bias: Option<bool>, platform: Option<Platform>, device: Option<Device>) -> PyResult<FeedForwardNetPicbreeder32> {
        let p = platform.map(|p| p.p).unwrap_or_else(|| rusty_neat_core::opencl_default_platform());
        let d = device.map(|d| d.d).or_else(|| rusty_neat_core::default_device(&p)).ok_or_else(|| PyValueError::new_err(format!("No device for {}", &p)))?;
        Ok(FeedForwardNetPicbreeder32 { net: self.net.to_picbreeder(center.as_ref(), bias.unwrap_or(false), p, d).map_err(ocl_err_to_py_ex)? })
    }
    #[text_signature = "(input_dimensions, output_dimensions, platform, device, /)"]
    fn to_substrate(&self, input_dimensions: usize, output_dimensions: Option<usize>, platform: Option<Platform>, device: Option<Device>) -> PyResult<FeedForwardNetSubstrate32> {
        let p = platform.map(|p| p.p).unwrap_or_else(|| rusty_neat_core::opencl_default_platform());
        let d = device.map(|d| d.d).or_else(|| rusty_neat_core::default_device(&p)).ok_or_else(|| PyValueError::new_err(format!("No device for {}", &p)))?;
        Ok(FeedForwardNetSubstrate32 { net: self.net.to_substrate(input_dimensions, output_dimensions, p, d).map_err(ocl_err_to_py_ex)? })
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
    fn get_device(&self) -> Device {
        Device { d: self.net.get_device() }
    }
    #[getter]
    fn get_output_size(&self) -> usize {
        self.net.get_output_size()
    }
    #[call]
    // #[text_signature = "(pixel_count_per_dimension, pixel_size_per_dimension, location_offset_per_dimension,/)"]
    fn __call__<'py>(&self, py: Python<'py>, pixel_count_per_dimension: Vec<usize>, pixel_size_per_dimension: Option<Vec<f32>>, location_offset_per_dimension: Option<Vec<f32>>) -> PyResult<&'py PyArrayDyn<f32>> {
        let pixel_size_per_dimension = pixel_size_per_dimension.unwrap_or_else(|| vec![1f32; self.get_input_size()]);
        let location_offset_per_dimension = location_offset_per_dimension.unwrap_or_else(|| vec![0f32; self.get_input_size()]);
        if pixel_count_per_dimension.len() != self.get_input_size() {
            Err(PyValueError::new_err(format!("Expected pixel_count_per_dimension of size {} but got {}", self.get_input_size(), pixel_count_per_dimension.len())))
        } else if pixel_size_per_dimension.len() != self.get_input_size() {
            Err(PyValueError::new_err(format!("Expected pixel_size_per_dimension of size {} but got {}", self.get_input_size(), pixel_size_per_dimension.len())))
        } else if location_offset_per_dimension.len() != self.get_input_size() {
            Err(PyValueError::new_err(format!("Expected location_offset_per_dimension of size {} but got {}", self.get_input_size(), location_offset_per_dimension.len())))
        } else {
            let out = self.net.run(pixel_count_per_dimension.as_slice(), pixel_size_per_dimension.as_slice(), location_offset_per_dimension.as_slice()).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let mut shape = Vec::with_capacity(self.get_input_size() + /*channels*/1);
            shape.extend_from_slice(pixel_count_per_dimension.as_slice());
            shape.push(self.get_output_size());
            new_ndarray(py, shape.as_slice(), out).map(|v| v.to_dyn())
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
    fn get_device(&self) -> Device {
        Device { d: self.net.get_device() }
    }
    #[getter]
    fn get_output_size(&self) -> usize {
        self.net.get_output_size()
    }
    #[call]
    fn __call__(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        if input.len() != self.get_input_size() {
            Err(PyValueError::new_err(format!("Expected input of size {} but got {}", self.get_input_size(), input.len())))
        } else {
            self.net.run(input.as_slice(), true).map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }
    #[text_signature = "(input_tensor, /)"]
    fn numpy<'py>(&self, py: Python<'py>, input: PyReadonlyArrayDyn<'_, f32>) -> PyResult<&'py PyArrayDyn<f32>> {
        let s = input.shape();
        if s.len() != 2 {
            return Err(PyValueError::new_err(format!("Expected input to be a 2 dimensional matrix but got {} dimensions", s.len())))
        }
        if s[1] != self.get_input_size() {
            return Err(PyValueError::new_err(format!("Expected input to have {} columns but got {}", self.get_input_size(), s[1])))
        }
        let data= input.as_slice()?;
        let strides = input.strides();
        if strides[0] < 0 || strides[1] < 0 {
            return Err(PyValueError::new_err(format!("Negative strides are not supported")));
        }
        let row = strides[0] as usize / std::mem::size_of::<f32>();
        let col = strides[1] as usize / std::mem::size_of::<f32>();
        let out = self.net.run_with_strides(data, col, row, 1, self.get_output_size())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        new_ndarray(py, [s[0], self.get_output_size()], out).map(|v| v.to_dyn())
    }
}

unsafe fn from_boxed_slice<T: Element, D: Dimension, ID>(
    py: Python,
    dims: ID,
    flags: c_int,
    strides: *const npy_intp,
    slice: Box<[T]>,
) -> &PyArray<T, D>
    where
        ID: IntoDimension<Dim=D>,
{
    let dims = dims.into_dimension();
    let container = slice_box::SliceBox::new(slice);
    let data_ptr = container.data;
    let cell = pyo3::PyClassInitializer::from(container)
        .create_cell(py)
        .expect("Object creation failed.");
    let ptr = PY_ARRAY_API.PyArray_New(
        PY_ARRAY_API.get_type_object(npyffi::NpyTypes::PyArray_Type),
        dims.ndim_cint(),
        dims.as_dims_ptr(),
        T::npy_type() as i32,
        strides as *mut _,          // strides
        data_ptr as _,              // data
        std::mem::size_of::<T>() as i32, // itemsize
        flags,                          // flag
        std::ptr::null_mut(),            //obj
    );
    PY_ARRAY_API.PyArray_SetBaseObject(ptr as *mut npyffi::PyArrayObject, cell as _);
    PyArray::from_owned_ptr(py, ptr)
}

pub fn new_ndarray<T: Element, D: Dimension, ID>(py: Python, dims: ID, vec: Vec<T>) -> PyResult<&PyArray<T, D>>
    where ID: IntoDimension<Dim=D> {
    let vec = vec.into_boxed_slice();
    let len = vec.len();
    let strides = [std::mem::size_of::<T>() as npy_intp];
    let vec = unsafe { from_boxed_slice(py, [len], NPY_ARRAY_WRITEABLE, strides.as_ptr(), vec) };
    vec.reshape(dims)
}


#[pymethods]
impl FeedForwardNetSubstrate32 {
    #[getter]
    fn get_input_size(&self) -> usize {
        self.net.get_input_size()
    }
    #[getter]
    fn get_device(&self) -> Device {
        Device { d: self.net.get_device() }
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
    fn __call__<'py>(&self, py: Python<'py>, input_neurons: PyReadonlyArrayDyn<'_, f32>, output_neurons: PyReadonlyArrayDyn<'_, f32>) -> PyResult<&'py PyArrayDyn<f32>> {
        let s = input_neurons.shape();
        if s.len() != 2 {
            return Err(PyValueError::new_err(format!("Expected input to be a 2 dimensional matrix but got {} dimensions", s.len())));
        }
        let in_neurons_count = s[0];
        if s[1] != self.get_input_size() {
            return Err(PyValueError::new_err(format!("Expected input to have {} columns but got {}", self.get_input_size(), s[1])));
        }
        let s = output_neurons.shape();
        if s.len() != 2 {
            return Err(PyValueError::new_err(format!("Expected output to be a 2 dimensional matrix but got {} dimensions", s.len())));
        }
        let out_neurons_count = s[0];
        if s[1] != self.get_input_size() {
            return Err(PyValueError::new_err(format!("Expected output to have {} columns but got {}", self.get_output_size(), s[1])));
        }
        let in_data = input_neurons.as_slice().map_err(|e| PyValueError::new_err(format!("Provided input neurons are not contiguous")))?;
        let out_data = output_neurons.as_slice().map_err(|e| PyValueError::new_err(format!("Provided output neurons are not contiguous")))?;
        let in_strides = input_neurons.strides();
        let out_strides = output_neurons.strides();
        if in_strides[0] < 0 || in_strides[1] < 0 || out_strides[0] < 0 || out_strides[1] < 0 {
            return Err(PyValueError::new_err(format!("Negative strides are not supported")));
        }
        let in_row_stride = in_strides[0] as usize / std::mem::size_of::<f32>();
        let in_col_stride = in_strides[1] as usize / std::mem::size_of::<f32>();
        let out_row_stride = out_strides[0] as usize / std::mem::size_of::<f32>();
        let out_col_stride = out_strides[1] as usize / std::mem::size_of::<f32>();
        let weights_depth_stride = 1;
        let weights_col_stride = self.get_weight_size() * weights_depth_stride;
        let weights_row_stride = weights_col_stride * in_neurons_count;
        let mut out = self.net.run(in_data, out_data,
                                   weights_row_stride, weights_col_stride, weights_depth_stride,
                                   in_col_stride, out_col_stride,
                                   in_row_stride, out_row_stride)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let shape = [out_neurons_count, in_neurons_count, self.get_weight_size()];
        new_ndarray(py, shape, out).map(|v| v.to_dyn())
    }
}
//
// #[pymethods]
// impl Output32 {
//     #[text_signature = "(/)"]
//     fn get_output(&self) -> Vec<f32> {
//         Vec::from(&self.out[self.input_size..self.input_size + self.output_size])
//     }
//
//     #[text_signature = "(/)"]
//     fn get_input(&self) -> Vec<f32> {
//         Vec::from(&self.out[..self.input_size])
//     }
//
//     #[text_signature = "(array, /)"]
//     fn set_input(&mut self, array: Vec<f32>) -> PyResult<()> {
//         if array.len() != self.input_size {
//             return Err(PyValueError::new_err(format!("Expected input of length {} but got {}", self.input_size, array.len())));
//         }
//         self.out[..self.input_size].clone_from_slice(array.as_slice());
//         Ok(())
//     }
//
//
// }
//
// #[pyproto]
// impl PyObjectProtocol for Output32 {
//     fn __str__(&self) -> PyResult<String>{
//         Ok(format!("Len={}, Input={:?}, Output={:?}", self.out.len(), self.get_input(), self.get_output()))
//     }
//     fn __repr__(&self) -> PyResult<String> {
//         self.__str__()
//     }
// }

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
impl PyObjectProtocol for Platform {
    fn __richcmp__(&self, other: PyRef<Platform>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.p.as_core() == other.p.as_core()),
            CompareOp::Ne => Ok(self.p.as_core() != other.p.as_core()),
            op => Err(PyValueError::new_err("Cannot compare platforms"))
        }
    }
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("Platform('{}')", self.p.name().unwrap_or_else(|e| e.to_string())))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyproto]
impl PyObjectProtocol for Device {
    fn __richcmp__(&self, other: PyRef<Device>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.d == other.d),
            CompareOp::Ne => Ok(self.d != other.d),
            op => Err(PyValueError::new_err("Cannot compare platforms"))
        }
    }
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("Device('{}')", self.d.name().unwrap_or_else(|e| e.to_string())))
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

#[pymodule]
pub fn envs(py: Python, m: &PyModule) -> PyResult<()> {
    use rusty_neat_core::envs::*;
    #[pyclass]
    #[text_signature = "(borders, width, height, hunger_change_per_step, lidars, lidar_steps, step_len, platform, device/)"]
    pub struct Evol {
        e: evol::Evol,
    }

    #[pymethods]
    impl Evol {
        #[getter]
        fn get_height(&self)->usize{
            self.e.get_height()
        }
        #[getter]
        fn get_width(&self)->usize{
            self.e.get_width()
        }
        #[getter]
        fn get_borders<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArrayDyn<u8>> {
            let v = self.e.get_borders().map_err(ocl_err_to_py_ex)?;
            new_ndarray(py, [self.get_height(), self.get_width()], v).map(|v| v.to_dyn())
        }
        #[new]
        fn new(borders: &PyArrayDyn<u8>, width: usize, height: usize, hunger_change_per_step: f32, lidars: Vec<f32>, lidar_steps: usize, step_len: f32, platform: Option<Platform>, device: Option<Device>) -> PyResult<Self> {
            let p = platform.map(|p| p.p).unwrap_or_else(|| rusty_neat_core::opencl_default_platform());
            let d = device.map(|d| d.d).or_else(|| rusty_neat_core::default_device(&p)).ok_or_else(|| PyValueError::new_err(format!("No device for {}", &p)))?;
            Ok(Self { e: evol::Evol::new(borders.to_vec()?, width, height, hunger_change_per_step, lidars.as_slice(), lidar_steps, step_len, p, d).map_err(ocl_err_to_py_ex)? })
        }
        #[call]
        fn __call__(&self, agents: &PyArrayDyn<f32>, lidars: &PyArrayDyn<f32>) -> PyResult<()> {
            let s = agents.shape();
            if s.len() != 2 {
                return Err(PyValueError::new_err(format!("agents matrix must have 2 dimensions")));
            }
            if s[1] != AGENT_ATTRIBUTES {
                return Err(PyValueError::new_err(format!("agents matrix must have {} columns (x,y,angle,hunger)", AGENT_ATTRIBUTES)));
            }
            let strides = agents.strides();
            if strides[0] < 0 || strides[1] < 0 {
                return Err(PyValueError::new_err(format!("agents matrix has negative strides")));
            }
            let strides = [strides[0] as usize / std::mem::size_of::<f32>(), strides[1] as usize / std::mem::size_of::<f32>()];
            let exp_strides = [AGENT_ATTRIBUTES, 1];
            if strides != exp_strides {
                return Err(PyValueError::new_err(format!("agents matrix has strides {:?} but expected {:?}", strides, exp_strides)));
            }
            let lidar_s = lidars.shape();
            if lidar_s.len() != 3 {
                return Err(PyValueError::new_err(format!("lidar matrix must have 3 dimensions (agent, lidar, attribute)")));
            }
            if lidar_s[0] != s[0] {
                return Err(PyValueError::new_err(format!("agents matrix has {} rows and lidars have {} but should be equal", s[0], lidar_s[0])));
            }
            if lidar_s[1] != self.e.get_lidar_count() {
                return Err(PyValueError::new_err(format!("lidar matrix has {} columns {} was expected", lidar_s[1], self.e.get_lidar_count())));
            }
            if lidar_s[2] != LIDAR_ATTRIBUTES {
                return Err(PyValueError::new_err(format!("lidar matrix has depth {} but {} was expected", lidar_s[2], LIDAR_ATTRIBUTES)));
            }
            let lidar_strides = lidars.strides();
            if lidar_strides[0] < 0 || lidar_strides[1] < 0 || lidar_strides[2] < 0 {
                return Err(PyValueError::new_err(format!("lidar matrix has negative strides")));
            }
            let lidar_strides = [lidar_strides[0] as usize / std::mem::size_of::<f32>(),
                lidar_strides[1] as usize / std::mem::size_of::<f32>(),
                lidar_strides[2] as usize / std::mem::size_of::<f32>()];
            let exp_s = [LIDAR_ATTRIBUTES * self.e.get_lidar_count(), LIDAR_ATTRIBUTES, 1];
            if lidar_strides != exp_s {
                return Err(PyValueError::new_err(format!("lidar matrix has strides {:?} but expected {:?}", lidar_strides, exp_s)));
            }
            unsafe { self.e.run(agents.as_slice_mut()?, lidars.as_slice_mut()?).map_err(ocl_err_to_py_ex) }
        }
    }
    m.add_class::<Evol>()?;
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn rusty_neat(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(random_activation_fn, m)?)?;
    m.add_function(wrap_pyfunction!(activation_functions, m)?)?;
    m.add_function(wrap_pyfunction!(platforms, m)?)?;
    m.add_function(wrap_pyfunction!(devices, m)?)?;
    m.add_wrapped(wrap_pymodule!(envs))?;
    m.add_class::<Device>()?;
    m.add_class::<Platform>()?;
    m.add_class::<CPPN32>()?;
    m.add_class::<Neat32>()?;
    m.add_class::<FeedForwardNet32>()?;
    m.add_class::<FeedForwardNetSubstrate32>()?;
    m.add_class::<FeedForwardNetOpenCL32>()?;
    m.add_class::<FeedForwardNetPicbreeder32>()?;
    Ok(())
}


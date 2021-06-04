

use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, PyObjectProtocol};
use pyo3::PyResult;
use rusty_neat_core::{cppn, neat};
use std::collections::HashSet;
use rusty_neat_core::activations::{ STR_TO_IDX, ALL_ACT_FN};
use pyo3::exceptions::PyValueError;
use rusty_neat_core::cppn::CPPN;
use std::iter::FromIterator;
use pyo3::types::PyString;
use rusty_neat_core::num::Num;

#[pyfunction]
pub fn random_activation_fn() -> String {
    String::from(rusty_neat_core::activations::random_activation_fn().name())
}


#[pyfunction]
pub fn activation_functions() -> Vec<String> {
    Vec::from_iter(ALL_ACT_FN.iter().map(|s| String::from(s.name())))
}

#[pyclass]
pub struct Platform {
    p:rusty_neat_core::Platform
}

#[pyfunction]
pub fn platforms() -> Vec<Platform> {
    rusty_neat_core::opencl_platforms().into_iter().map(|p|Platform{p}).collect()
}

#[pyclass]
pub struct Device {
    d:rusty_neat_core::Device
}

#[pyfunction]
#[text_signature = "(platform,/)"]
pub fn devices(p:Option<&Platform>) -> Vec<Device> {
    let d:Vec<rusty_neat_core::Device> = if let Some(p) = p{
        rusty_neat_core::device_list(&p.p)
    }else if let Some(p) = rusty_neat_core::opencl_platforms().into_iter().next(){
        rusty_neat_core::device_list(&p)
    }else{
        vec![]
    };
    d.into_iter().map(|d|Device{d}).collect()
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
pub struct Neat32 { neat: neat::Neat }


#[pymethods]
impl Neat32 {

    #[text_signature = "(/)"]
    pub fn get_activation_functions(&self)->Vec<String>{
        Vec::from_iter(self.neat.get_activation_functions().iter().map(|s|String::from(s.name())))
    }
    #[text_signature = "(/)"]
    pub fn get_random_activation_function(&self)->String{
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
        let input_size = self.input_size();
        let output_size = self.output_size();
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
        self.neat.attempt_to_add_random_connection(&mut cppn.cppn)
    }

    #[text_signature = "(cppn, edge_to_split, /)"]
    fn add_node(&mut self, cppn: &mut CPPN32, edge: usize) -> PyResult<()> {
        if edge >= cppn.edge_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} edges but provided index {}", cppn.edge_count(), edge)));
        }
        Ok(self.neat.add_node(&mut cppn.cppn, edge))
    }

    #[text_signature = "(population,node_insertion_prob,edge_insertion_prob,activation_fn_mutation_prob,weight_mutation_prob /)"]
    fn mutate_population(&mut self, mut cppn: Vec<PyRefMut<CPPN32>>, node_insertion_prob: f32,
                         edge_insertion_prob: f32,
                         activation_fn_mutation_prob: f32,
                         weight_mutation_prob: f32){
        self.neat.mutate_population(cppn.iter_mut().map(|c| &mut c.cppn),
                                    node_insertion_prob,
                                    edge_insertion_prob,
                                    activation_fn_mutation_prob,
                                    weight_mutation_prob)
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
    pub fn set_random_activation_function(&mut self, cppn: &mut CPPN32, node:usize) -> PyResult<bool>{
        if node >= cppn.node_count() {
            return Err(PyValueError::new_err(format!("CPPN has {} nodes but provided index {}", cppn.node_count(), node)));
        }
        Ok(cppn.cppn.set_activation(node, self.neat.get_random_activation_function()))
    }



}


#[pymethods]
impl CPPN32 {
    #[getter]
    fn get_input_size(&self)->usize{
        self.cppn.get_input_size()
    }
    #[getter]
    fn get_output_size(&self)->usize{
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
        Ok(self.cppn.get_activation(node).map(|x|String::from(x.name())))
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

#[pymethods]
impl FeedForwardNet32 {
    #[call]
    fn __call__(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        if input.len() != self.get_input_size(){
            Err(PyValueError::new_err(format!("Expected input of size {} but got {}", self.get_input_size(), input.len())))
        }else {
            let mut out = vec![0f32; self.get_output_size()];
            self.net.run(input.as_slice(), out.as_mut_slice());
            Ok(out)
        }
    }
    #[getter]
    fn get_input_size(&self)->usize{
        self.net.get_input_size()
    }
    #[getter]
    fn get_output_size(&self)->usize{
        self.net.get_output_size()
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
    fn __str__(&self) -> PyResult<String>{
        Ok(format!("{}",self.cppn))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyproto]
impl PyObjectProtocol for Platform {
    fn __str__(&self) -> PyResult<String>{
        Ok(format!("Platform('{}')",self.p.name().unwrap_or_else(|e|e.to_string())))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyproto]
impl PyObjectProtocol for Device {
    fn __str__(&self) -> PyResult<String>{
        Ok(format!("Device('{}')",self.d.name().unwrap_or_else(|e|e.to_string())))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}
#[pyproto]
impl PyObjectProtocol for FeedForwardNet32 {
    fn __str__(&self) -> PyResult<String>{
        Ok(format!("{}",self.net))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rusty_neat(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(random_activation_fn, m)?)?;
    m.add_function(wrap_pyfunction!(activation_functions, m)?)?;
    m.add_function(wrap_pyfunction!(platforms, m)?)?;
    m.add_function(wrap_pyfunction!(devices, m)?)?;
    m.add_class::<CPPN32>()?;
    m.add_class::<Neat32>()?;
    m.add_class::<FeedForwardNet32>()?;
    Ok(())
}


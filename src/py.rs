use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::PyResult;
use crate::{cppn, neat};
use std::collections::HashSet;
use crate::activations::{identity, ALL_STR, ALL_F64, STR_TO_IDX};
use pyo3::exceptions::PyValueError;
use crate::cppn::CPPN;

/// Formats the sum of two numbers as string.
#[pyfunction]
pub fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyclass]
pub struct Output64 {
    out: Vec<f64>,
    #[pyo3(get)]
    input_size: usize,
    #[pyo3(get)]
    output_size: usize,
}

#[pyclass]
pub struct FeedForwardNet64 {
    net: cppn::FeedForwardNet<f64>,
    #[pyo3(get)]
    input_size: usize,
    #[pyo3(get)]
    output_size: usize,
}

#[pyclass]
pub struct CPPN64 {
    cppn: cppn::CPPN<f64>,
    #[pyo3(get)]
    input_size: usize,
    #[pyo3(get)]
    output_size: usize,
}


#[pyclass]
pub struct CPPN32 {
    cppn: cppn::CPPN<f32>
}

#[pyclass]
#[text_signature = "(input_size, output_size, activation_functions, /)"]
pub struct Neat64 { neat: neat::Neat<f64> }

#[pyclass]
pub struct Neat32 { neat: neat::Neat<f32> }


#[pymethods]
impl Neat64 {
    #[new]
    pub fn new(input_size: usize, output_size: usize, activations: Option<Vec<String>>) -> PyResult<Self> {
        let ac_fn =
            if let Some(activations) = activations {
                let mut ac_fn = vec![];
                for name in activations {
                    match STR_TO_IDX.get(&name) {
                        None => return Err(PyValueError::new_err(name + " is not a known function name")),
                        Some(&idx) => {
                            ac_fn.push(ALL_F64[idx]);
                        }
                    }
                }
                ac_fn
            } else {
                Vec::from(ALL_F64)
            };
        Ok(Neat64 { neat: neat::Neat::new(ac_fn, input_size, output_size) })
    }
    #[text_signature = "(/)"]
    pub fn new_cppn(&mut self) -> CPPN64 {
        CPPN64 {
            cppn: self.neat.new_cppn(),
            input_size: self.input_size(),
            output_size: self.output_size(),
        }
    }
    #[getter]
    fn input_size(&self) -> usize {
        self.neat.get_input_size()
    }
    #[getter]
    fn output_size(&self) -> usize {
        self.neat.get_input_size()
    }

    #[text_signature = "(fitter, less_fit, /)"]
    fn crossover(&self, fitter: &CPPN64, less_fit: &CPPN64) -> PyResult<CPPN64> {
        let input_size = self.input_size();
        let output_size = self.output_size();
        if fitter.input_size != input_size {
            return Err(PyValueError::new_err(format!("Left CPPN has input size {} but NEAT instance expected {}", fitter.input_size, input_size)));
        }
        if fitter.output_size != output_size {
            return Err(PyValueError::new_err(format!("Left CPPN has output size {} but NEAT instance expected {}", fitter.output_size, output_size)));
        }
        if less_fit.input_size != input_size {
            return Err(PyValueError::new_err(format!("Right CPPN has input size {} but NEAT instance expected {}", less_fit.input_size, input_size)));
        }
        if less_fit.output_size != output_size {
            return Err(PyValueError::new_err(format!("Right CPPN has output size {} but NEAT instance expected {}", less_fit.output_size, output_size)));
        }
        Ok(CPPN64 { cppn: fitter.cppn.crossover(&less_fit.cppn, &mut ||self.neat.random_weight_generator()), input_size, output_size })
    }

    
}


#[pymethods]
impl CPPN64 {
    #[text_signature = "(/)"]
    fn build_feed_forward_net(&self) -> FeedForwardNet64 {
        FeedForwardNet64 {
            net: self.cppn.build_feed_forward_net(),
            input_size: self.input_size,
            output_size: self.output_size,
        }
    }
}

#[pymethods]
impl FeedForwardNet64 {
    #[text_signature = "(output_buffer, /)"]
    fn run(&self, output: &mut Output64) {
        self.net.run(output.out.as_mut_slice())
    }

    #[text_signature = "(/)"]
    fn make_output_buffer(&self) -> Output64 {
        Output64 {
            out: self.net.new_input_buffer(),
            input_size: self.input_size,
            output_size: self.output_size,
        }
    }
}


#[pymethods]
impl Output64 {
    #[text_signature = "(/)"]
    fn get_output(&self) -> Vec<f64> {
        Vec::from(&self.out[self.input_size..self.input_size + self.output_size])
    }

    #[text_signature = "(/)"]
    fn get_input(&self) -> Vec<f64> {
        Vec::from(&self.out[..self.input_size])
    }

    #[text_signature = "(array, /)"]
    fn set_input(&mut self, array: Vec<f64>) -> PyResult<()> {
        if array.len() != self.input_size {
            return Err(PyValueError::new_err(format!("Expected input of length {} but got {}", self.input_size, array.len())));
        }
        self.out[..self.input_size].clone_from_slice(array.as_slice());
        Ok(())
    }
}
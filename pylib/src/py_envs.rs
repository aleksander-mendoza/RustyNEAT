use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol};
use pyo3::PyResult;
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyString;
use numpy::{PyReadonlyArrayDyn, PyArrayDyn, IntoPyArray, PyArray, PY_ARRAY_API, npyffi, Element, ToNpyDims};
use numpy::npyffi::{NPY_ORDER, npy_intp, NPY_ARRAY_WRITEABLE};
use rusty_neat_core::envs::evol::{AGENT_ATTRIBUTES, LIDAR_ATTRIBUTES};

use rusty_neat_core::envs::*;
use crate::py_ndalgebra::DynMat;
use std::task::Context;
use crate::py_ocl::NeatContext;
use crate::ocl_err_to_py_ex;

#[pyclass]
#[text_signature = "(width, height, hunger_change_per_step, lidars, lidar_steps, step_len, context/)"]
pub struct Evol {
    e: evol::Evol,
}

#[pymethods]
impl Evol {
    #[getter]
    fn get_height(&self) -> usize {
        self.e.get_height()
    }
    #[getter]
    fn get_width(&self) -> usize {
        self.e.get_width()
    }

    #[new]
    fn new(width: usize, height: usize, hunger_change_per_step: f32, lidars: Vec<f32>, lidar_steps: usize, step_len: f32, context: &NeatContext) -> PyResult<Self> {
        Ok(Self { e: evol::Evol::new(width, height, hunger_change_per_step, lidars.as_slice(), lidar_steps, step_len, &context.c).map_err(ocl_err_to_py_ex)? })
    }
    #[call]
    fn __call__(&self, borders: &mut DynMat, agents: &mut DynMat, lidars: &mut DynMat) -> PyResult<()> {
        let borders = borders.try_as_dtype_mut::<u8>()?;
        let agents = agents.try_as_dtype_mut::<f32>()?;
        let lidars = lidars.try_as_dtype_mut::<f32>()?;
        self.e.run(borders,agents,lidars).map_err(ocl_err_to_py_ex)
    }
}

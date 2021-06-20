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

#[pyclass]
#[text_signature = "(borders, width, height, hunger_change_per_step, lidars, lidar_steps, step_len, platform, device/)"]
pub struct Evol {
    e: evol::Evol,
}
//
// #[pymethods]
// impl Evol {
//     #[getter]
//     fn get_height(&self) -> usize {
//         self.e.get_height()
//     }
//     #[getter]
//     fn get_width(&self) -> usize {
//         self.e.get_width()
//     }
//     #[getter]
//     fn get_borders<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArrayDyn<u8>> {
//         let v = self.e.get_borders().map_err(ocl_err_to_py_ex)?;
//         new_ndarray(py, [self.get_height(), self.get_width()], v).map(|v| v.to_dyn())
//     }
//     #[new]
//     fn new(borders: &PyArrayDyn<u8>, width: usize, height: usize, hunger_change_per_step: f32, lidars: Vec<f32>, lidar_steps: usize, step_len: f32, platform: Option<Platform>, device: Option<Device>) -> PyResult<Self> {
//         let p = platform.map(|p| p.p).unwrap_or_else(|| rusty_neat_core::opencl_default_platform());
//         let d = device.map(|d| d.d).or_else(|| rusty_neat_core::default_device(&p)).ok_or_else(|| PyValueError::new_err(format!("No device for {}", &p)))?;
//         Ok(Self { e: evol::Evol::new(borders.to_vec()?, width, height, hunger_change_per_step, lidars.as_slice(), lidar_steps, step_len, p, d).map_err(ocl_err_to_py_ex)? })
//     }
//     #[call]
//     fn __call__(&self, agents: &PyArrayDyn<f32>, lidars: &PyArrayDyn<f32>) -> PyResult<()> {
//         let s = agents.shape();
//         if s.len() != 2 {
//             return Err(PyValueError::new_err(format!("agents matrix must have 2 dimensions")));
//         }
//         if s[1] != AGENT_ATTRIBUTES {
//             return Err(PyValueError::new_err(format!("agents matrix must have {} columns (x,y,angle,hunger)", AGENT_ATTRIBUTES)));
//         }
//         let strides = agents.strides();
//         if strides[0] < 0 || strides[1] < 0 {
//             return Err(PyValueError::new_err(format!("agents matrix has negative strides")));
//         }
//         let strides = [strides[0] as usize / std::mem::size_of::<f32>(), strides[1] as usize / std::mem::size_of::<f32>()];
//         let exp_strides = [AGENT_ATTRIBUTES, 1];
//         if strides != exp_strides {
//             return Err(PyValueError::new_err(format!("agents matrix has strides {:?} but expected {:?}", strides, exp_strides)));
//         }
//         let lidar_s = lidars.shape();
//         if lidar_s.len() != 3 {
//             return Err(PyValueError::new_err(format!("lidar matrix must have 3 dimensions (agent, lidar, attribute)")));
//         }
//         if lidar_s[0] != s[0] {
//             return Err(PyValueError::new_err(format!("agents matrix has {} rows and lidars have {} but should be equal", s[0], lidar_s[0])));
//         }
//         if lidar_s[1] != self.e.get_lidar_count() {
//             return Err(PyValueError::new_err(format!("lidar matrix has {} columns {} was expected", lidar_s[1], self.e.get_lidar_count())));
//         }
//         if lidar_s[2] != LIDAR_ATTRIBUTES {
//             return Err(PyValueError::new_err(format!("lidar matrix has depth {} but {} was expected", lidar_s[2], LIDAR_ATTRIBUTES)));
//         }
//         let lidar_strides = lidars.strides();
//         if lidar_strides[0] < 0 || lidar_strides[1] < 0 || lidar_strides[2] < 0 {
//             return Err(PyValueError::new_err(format!("lidar matrix has negative strides")));
//         }
//         let lidar_strides = [lidar_strides[0] as usize / std::mem::size_of::<f32>(),
//             lidar_strides[1] as usize / std::mem::size_of::<f32>(),
//             lidar_strides[2] as usize / std::mem::size_of::<f32>()];
//         let exp_s = [LIDAR_ATTRIBUTES * self.e.get_lidar_count(), LIDAR_ATTRIBUTES, 1];
//         if lidar_strides != exp_s {
//             return Err(PyValueError::new_err(format!("lidar matrix has strides {:?} but expected {:?}", lidar_strides, exp_s)));
//         }
//         unsafe { self.e.run(agents.as_slice_mut()?, lidars.as_slice_mut()?).map_err(ocl_err_to_py_ex) }
//     }
// }

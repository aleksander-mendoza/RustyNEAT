use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol};
use pyo3::PyResult;
use pyo3::basic::CompareOp;
use pyo3::types::PyString;
use rusty_neat_core::context::NeatContext as NC;
use crate::ocl_err_to_py_ex;
use std::any::Any;
use rusty_neat_core::{Platform, Device};

#[pyclass]
pub struct NeatContext {
    pub(crate) c: rusty_neat_core::context::NeatContext,
}

fn platform_to_str(p:Platform)->String{
    p.version().unwrap_or_else(|e|e.to_string())
}

fn device_to_str(p:Device)->String{
    p.name().unwrap_or_else(|e|e.to_string())
}

#[pyfunction]
pub fn devices() -> Vec<(String,String)> {
    rusty_neat_core::context::NeatContext::opencl_platforms().into_iter().flat_map(|p| rusty_neat_core::context::NeatContext::device_list(p).into_iter().map(move |d|(platform_to_str(p),device_to_str(d)))).collect()
}

#[pyfunction]
#[text_signature = "(platform, device, /)"]
pub fn make_new_context(platform:Option<String>, device:Option<String>) -> PyResult<NeatContext> {
    let platform = platform.and_then(|platform| NC::opencl_platforms().into_iter().find(|p|p.version().contains(&platform))).unwrap_or_else(||NC::opencl_default_platform());
    let device = device.and_then(|device| NC::device_list(platform).into_iter().find(|d|d.name().contains(&device)));
    let device = if let Some(device) = device{
        device
    }else{
        NC::opencl_default_device(platform).map_err(ocl_err_to_py_ex)?
    };
    NC::new(platform,device).map(|c|NeatContext { c }).map_err(ocl_err_to_py_ex)
}
#[pyfunction]
#[text_signature = "( /)"]
pub fn make_gpu_context() -> PyResult<NeatContext> {
    NC::gpu().map(|c|NeatContext { c }).map_err(ocl_err_to_py_ex)
}
#[pyfunction]
#[text_signature = "( /)"]
pub fn make_cpu_context() -> PyResult<NeatContext> {
    NC::gpu().map(|c|NeatContext { c }).map_err(ocl_err_to_py_ex)
}

#[pymethods]
impl NeatContext {
    #[text_signature = "( /)"]
    fn device_info(&self) -> String {
        self.c.device().to_string()
    }
    #[text_signature = "( /)"]
    fn device(&self) -> PyResult<String> {
        self.c.device().name().map_err(ocl_err_to_py_ex)
    }
    #[text_signature = "( /)"]
    fn platform(&self) -> PyResult<String> {
        self.c.platform().version().map_err(ocl_err_to_py_ex)
    }
    #[text_signature = "( /)"]
    fn platform_info(&self) -> String {
        self.c.platform().to_string()
    }

}


#[pyproto]
impl PyObjectProtocol for NeatContext {
    fn __richcmp__(&self, other: PyRef<NeatContext>, op: CompareOp) -> PyResult<bool> {
        let eq = self.c.device() == other.c.device() && self.c.platform().as_core()==other.c.platform().as_core();
        match op {
            CompareOp::Eq => Ok(eq),
            CompareOp::Ne => Ok(!eq),
            op => Err(ocl_err_to_py_ex("Cannot compare platforms"))
        }
    }
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("NeatContext(platform='{}', device='{}')", self.c.platform(), self.c.device()))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}
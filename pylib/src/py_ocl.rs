use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol};
use pyo3::PyResult;
use pyo3::basic::CompareOp;
use pyo3::types::PyString;
use crate::ocl_err_to_py_ex;
use std::any::Any;
use ndalgebra::{Platform, Device};
use ndalgebra::context::Context as C;
use htm::EccProgram;
use ndalgebra::lin_alg_program::LinAlgProgram;

#[pyclass]
pub struct Context {
    pub(crate) c: C,
    pub(crate) htm: Option<htm::EccProgram>,
    pub(crate) lin_alg: Option<ndalgebra::lin_alg_program::LinAlgProgram>,
}

impl Context{
    pub fn new(c:C)->Self{
        Self{
            c,
            htm: None,
            lin_alg: None
        }
    }
    pub fn compile_htm_program(&mut self) -> PyResult<&EccProgram> {
        if self.htm.is_none(){
            let htm = htm::EccProgram::new(self.c.clone()).map_err(ocl_err_to_py_ex)?;
            self.htm.insert(htm);
        }
        Ok(self.htm.as_ref().unwrap())
    }
    pub fn compile_lin_alg_program(&mut self) -> PyResult<&LinAlgProgram> {
        if self.lin_alg.is_none(){
            let alg = ndalgebra::lin_alg_program::LinAlgProgram::new(self.c.clone()).map_err(ocl_err_to_py_ex)?;
            self.lin_alg.insert(alg);
        }
        Ok(self.lin_alg.as_ref().unwrap())
    }
}

fn platform_to_str(p: Platform) -> String {
    p.version().unwrap_or_else(|e| e.to_string())
}

fn device_to_str(p: Device) -> String {
    p.name().unwrap_or_else(|e| e.to_string())
}

#[pyfunction]
pub fn devices() -> Vec<(String, String)> {
    C::opencl_platforms().into_iter().flat_map(|p| C::device_list(&p).into_iter().map(move |d| (platform_to_str(p), device_to_str(d)))).collect()
}

#[pyfunction]
#[text_signature = "(platform, device, /)"]
pub fn make_new_context(platform: Option<String>, device: Option<String>) -> PyResult<Context> {
    let platform = platform.and_then(|platform| C::opencl_platforms().into_iter().find(|p| p.version().contains(&platform))).unwrap_or_else(|| C::opencl_default_platform());
    let device = device.and_then(|device| C::device_list(&platform).into_iter().find(|d| d.name().contains(&device)));
    let device = if let Some(device) = device {
        device
    } else {
        C::opencl_default_device(platform).map_err(ocl_err_to_py_ex)?
    };
    C::new(platform, device).map( Context::new).map_err(ocl_err_to_py_ex)
}

#[pyfunction]
#[text_signature = "( /)"]
pub fn make_gpu_context() -> PyResult<Context> {
    C::gpu().map(Context::new).map_err(ocl_err_to_py_ex)
}

#[pyfunction]
#[text_signature = "( /)"]
pub fn make_cpu_context() -> PyResult<Context> {
    C::gpu().map(Context::new).map_err(ocl_err_to_py_ex)
}


#[pymethods]
impl Context {

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
        self.c.platform_version().map(|v| v.to_string()).map_err(ocl_err_to_py_ex)
    }
    #[text_signature = "( /)"]
    fn platform_info(&self) -> PyResult<Option<String>> {
        self.c.platform().map(|v| v.map(|v| v.to_string())).map_err(ocl_err_to_py_ex)
    }

}


#[pyproto]
impl PyObjectProtocol for Context {
    // fn __richcmp__(&self, other: PyRef<Context>, op: CompareOp) -> PyResult<bool> {
    //     let eq = self.c.device() == other.c.device() && self.c.platform().as_core() == other.c.platform().as_core();
    //     match op {
    //         CompareOp::Eq => Ok(eq),
    //         CompareOp::Ne => Ok(!eq),
    //         op => Err(ocl_err_to_py_ex("Cannot compare platforms"))
    //     }
    // }
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("Context(platform='{}', device='{}')", self.c.platform().map_err(ocl_err_to_py_ex)?.map(|v|v.to_string()).unwrap_or(String::from("None")), self.c.device()))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}


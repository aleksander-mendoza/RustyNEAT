#[macro_use] extern crate maplit;
#[macro_use] extern crate lazy_static;


mod neat;
mod activations;
mod num;
mod util;
mod cppn;
mod py;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use num_traits;
use std::collections::HashMap;

use num::Num;
use py::*;




/// A Python module implemented in Rust.
#[pymodule]
fn rusty_neat(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<CPPN32>()?;
    m.add_class::<CPPN64>()?;
    m.add_class::<Neat32>()?;
    m.add_class::<Neat64>()?;
    Ok(())
}


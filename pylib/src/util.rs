
use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol, PyIterProtocol, PySequenceProtocol, PyTypeInfo, PyDowncastError, AsPyPointer, PyNumberProtocol};
use pyo3::PyResult;
use rusty_neat_core::{cppn, neat, gpu};
use std::collections::HashSet;
use rusty_neat_core::activations::{STR_TO_IDX, ALL_ACT_FN};
use pyo3::exceptions::PyValueError;
use rusty_neat_core::cppn::CPPN;
use std::iter::FromIterator;
use pyo3::types::{PyString, PyDateTime, PyDateAccess, PyTimeAccess, PyList, PyInt};
use rusty_neat_core::num::Num;
use rusty_neat_core::gpu::{FeedForwardNetOpenCL, FeedForwardNetPicbreeder, FeedForwardNetSubstrate};
use pyo3::basic::CompareOp;
use numpy::{PyReadonlyArrayDyn, PyArrayDyn, IntoPyArray, PyArray, PY_ARRAY_API, npyffi, Element, ToNpyDims, DataType};
use numpy::npyffi::{NPY_ORDER, npy_intp, NPY_ARRAY_WRITEABLE};
use std::os::raw::c_int;
use crate::ocl_err_to_py_ex;
use crate::py_ndalgebra::{DynMat, try_as_dtype};
use crate::py_ocl::Context;
use htm::{Encoder, EncoderTarget, EncoderRange, Shape, VectorFieldOne, Synapse};
use std::time::SystemTime;
use std::ops::Deref;
use chrono::Utc;
use std::borrow::Borrow;
use std::io::BufWriter;
use std::fs::{File, OpenOptions};
use serde_pickle::SerOptions;
use serde::Serialize;

pub fn arr3<'py, T:Element+Copy+FromPyObject<'py>>(py: Python<'py>, t: &'py PyObject) -> PyResult<[T; 3]> {
    Ok(if let Ok(t) = t.extract::<(T, T, T)>(py) {
        [t.0, t.1, t.2]
    } else if let Ok(t) = t.extract::<Vec<T>>(py) {
        [t[0], t[1], t[2]]
    } else {
        let array = py_any_as_numpy(t.as_ref(py))?;
        let t = unsafe { array.as_slice()? };
        [t[0], t[1], t[2]]
    })
}

pub fn arrX<'py,T:Element+Copy+FromPyObject<'py>>(py: Python<'py>, t: &'py PyObject, default0: T, default1: T, default2: T) -> PyResult<[T; 3]> {
    Ok(if t.is_none(py) {
        [default0, default1, default2]
    } else if let Ok(t) = t.extract::<T>(py) {
        [default0, default1, t]
    } else if let Ok(t) = t.extract::<(T, T)>(py) {
        [default0, t.0, t.1]
    } else if let Ok(t) = t.extract::<(T, T, T)>(py) {
        [t.0, t.1, t.2]
    } else {
        let d = [default0, default1, default2];
        fn to3<T:Element+Copy>(arr: &[T], mut d: [T; 3]) -> [T; 3] {
            for i in 0..arr.len().min(3) {
                d[3 - arr.len() + i] = arr[i];
            }
            d
        }
        if let Ok(t) = t.extract::<Vec<T>>(py) {
            to3(&t, d)
        } else {
            let array = py_any_as_numpy(t.as_ref(py))?;
            let t = unsafe { array.as_slice()? };
            to3(&t, d)
        }
    })
}

pub fn arr2<'py,T:Element+Copy+FromPyObject<'py>>(py: Python<'py>, t: &'py PyObject) -> PyResult<[T; 2]> {
    Ok(if let Ok(t) = t.extract::<(T, T)>(py) {
        [t.0, t.1]
    } else if let Ok(t) = t.extract::<Vec<T>>(py) {
        [t[0], t[1]]
    } else {
        let array = py_any_as_numpy(t.as_ref(py))?;
        let t = unsafe { array.as_slice()? };
        [t[0], t[1]]
    })
}

pub fn pickle<T: Serialize>(val: &T, file: String) -> PyResult<()> {
    let o = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(file)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_pickle::to_writer(&mut BufWriter::new(o), val, SerOptions::new());
    Ok(())
}

pub fn py_any_as_numpy<T:Element>(input: &PyAny) -> Result<&PyArrayDyn<T>, PyErr> {
    let array = unsafe {
        if npyffi::PyArray_Check(input.as_ptr()) == 0 {
            return Err(PyDowncastError::new(input, "PyArray<T, D>").into());
        }
        &*(input as *const PyAny as *const PyArrayDyn<u8>)
    };
    if !array.is_c_contiguous(){
        return Err(PyValueError::new_err("Numpy array is not C contiguous"));
    }
    let actual_dtype = array.dtype().get_datatype().ok_or_else(|| PyValueError::new_err("No numpy array has no dtype"))?;
    if T::DATA_TYPE != actual_dtype {
        return Err(PyValueError::new_err(format!("Expected numpy array of dtype {:?} but got {:?}", T::DATA_TYPE, actual_dtype)));
    }
    let array = unsafe { &*(input as *const PyAny as *const PyArrayDyn<T>) };
    Ok(array)
}

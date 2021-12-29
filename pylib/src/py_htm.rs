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
use htm_vis::visualise_cpu_htm2;

#[pyclass]
pub struct CpuSDR {
    sdr: htm::CpuSDR,
}

#[pyclass]
pub struct CpuBitset {
    bits: htm::CpuBitset,
}

///
/// CpuInput(size)
///
/// Optimised structure for holding both bitset and SDR. As a result most operations become O(1).
///
#[pyclass]
pub struct CpuInput {
    inp: htm::CpuInput,
}


// ///
// /// CpuBigHTM(input_size:int, minicolumns:int, n: int, rand_seed:int)
// ///
// ///
// #[pyclass]
// pub struct CpuBigHTM {
//     htm: htm::CpuBigHTM,
// }

///
/// CpuHTM2(input_size:int, n:int, population:Optional[Population], rand_seed:Optional[int])
///
///
/// The standard implementation of spacial pooler.
///  Randomly generate a new Spacial Pooler. You can provide a random seed manually.
///  Otherwise the millisecond part of system time is used as a seed. Seed is a 32-bit number.
///
///
#[pyclass]
pub struct CpuHTM {
    htm: htm::CpuHTM,
}

#[pyclass]
pub struct OclSDR {
    sdr: htm::OclSDR,
}

#[pyclass]
pub struct OclBitset {
    bits: htm::OclBitset,
}

#[pyclass]
pub struct OclInput {
    inp: htm::OclInput,
}

#[pyclass]
pub struct OclHTM {
    htm: htm::OclHTM,
}


#[pyclass]
pub struct EncoderBuilder {
    enc: htm::EncoderBuilder,
}

///
/// Population(population_size:int,num_of_segments:int)
///
///
#[pyclass]
pub struct Population {
    pop:htm::Population,
}
///
/// Neuron(num_of_segments:int)
///
///
#[pyclass]
pub struct Neuron {
    n:htm::Neuron,
}
///
/// Segment()
///
///
#[pyclass]
pub struct Segment {
    seg:htm::Segment,
}

#[pyclass]
pub struct CategoricalEncoder {
    enc: htm::CategoricalEncoder,
}

#[pyclass]
pub struct BitsEncoder {
    enc: htm::BitsEncoder,
}

#[pyclass]
pub struct IntegerEncoder {
    enc: htm::IntegerEncoder,
}

#[pyclass]
pub struct CircularIntegerEncoder {
    enc: htm::CircularIntegerEncoder,
}

#[pyclass]
pub struct FloatEncoder {
    enc: htm::FloatEncoder,
}

#[pyclass]
pub struct DayOfWeekEncoder {
    enc: htm::DayOfWeekEncoder,
}

#[pyclass]
pub struct DayOfMonthEncoder {
    enc: htm::DayOfMonthEncoder,
}

#[pyclass]
pub struct DayOfYearEncoder {
    enc: htm::DayOfYearEncoder,
}

#[pyclass]
pub struct IsWeekendEncoder {
    enc: htm::IsWeekendEncoder,
}

#[pyclass]
pub struct TimeOfDayEncoder {
    enc: htm::TimeOfDayEncoder,
}

#[pyclass]
pub struct BoolEncoder {
    enc: htm::BoolEncoder,
}

#[pymethods]
impl EncoderBuilder {
    #[new]
    pub fn new() -> Self {
        EncoderBuilder { enc: htm::EncoderBuilder::new() }
    }
    #[getter]
    pub fn input_size(&mut self) -> u32 {
        self.enc.input_size()
    }
    #[text_signature = "(number_of_idle_neurons)"]
    pub fn pad(&mut self, number_of_idle_neurons: u32) {
        self.enc.pad(number_of_idle_neurons)
    }
    #[text_signature = "(from_inclusive,to_exclusive,size,cardinality)"]
    pub fn add_circularinteger(&mut self, from: u32, to: u32, size: u32, cardinality: u32) -> CircularIntegerEncoder {
        CircularIntegerEncoder { enc: self.enc.add_circular_integer(from..to, size, cardinality) }
    }
    #[text_signature = "(number_of_categories, cardinality)"]
    pub fn add_categorical(&mut self, number_of_categories: u32, cardinality: u32) -> CategoricalEncoder {
        CategoricalEncoder { enc: self.enc.add_categorical(number_of_categories, cardinality) }
    }
    #[text_signature = "(size)"]
    pub fn add_bits(&mut self, size: u32) -> BitsEncoder {
        BitsEncoder { enc: self.enc.add_bits(size) }
    }
    #[text_signature = "(from_inclusive,to_exclusive,size,cardinality)"]
    pub fn add_integer(&mut self, from: u32, to: u32, size: u32, cardinality: u32) -> IntegerEncoder {
        IntegerEncoder { enc: self.enc.add_integer(from..to, size, cardinality) }
    }
    #[text_signature = "(from_inclusive,to_exclusive,size,cardinality)"]
    pub fn add_float(&mut self, from: f32, to: f32, size: u32, cardinality: u32) -> FloatEncoder {
        FloatEncoder { enc: self.enc.add_float(from..to, size, cardinality) }
    }
    #[text_signature = "(size,cardinality)"]
    pub fn add_bool(&mut self, size: u32, cardinality: u32) -> BoolEncoder {
        BoolEncoder { enc: self.enc.add_bool(size, cardinality) }
    }
    #[text_signature = "(size,cardinality)"]
    pub fn add_day_of_week(&mut self, size: u32, cardinality: u32) -> DayOfWeekEncoder {
        DayOfWeekEncoder { enc: self.enc.add_day_of_week(size, cardinality) }
    }
    #[text_signature = "(size,cardinality)"]
    pub fn add_day_of_year(&mut self, size: u32, cardinality: u32) -> DayOfYearEncoder {
        DayOfYearEncoder { enc: self.enc.add_day_of_year(size, cardinality) }
    }
    #[text_signature = "(size,cardinality)"]
    pub fn add_day_of_month(&mut self, size: u32, cardinality: u32) -> DayOfMonthEncoder {
        DayOfMonthEncoder { enc: self.enc.add_day_of_month(size, cardinality) }
    }
    #[text_signature = "(size,cardinality)"]
    pub fn add_is_weekend(&mut self, size: u32, cardinality: u32) -> IsWeekendEncoder {
        IsWeekendEncoder { enc: self.enc.add_is_weekend(size, cardinality) }
    }
    #[text_signature = "(size,cardinality)"]
    pub fn add_time_of_day(&mut self, size: u32, cardinality: u32) -> TimeOfDayEncoder {
        TimeOfDayEncoder { enc: self.enc.add_time_of_day(size, cardinality) }
    }
}

fn encode<T, U>(sdr: PyObject, scalar: T, f1: impl FnOnce(&mut htm::CpuSDR, T) -> U, f2: impl FnOnce(&mut htm::CpuBitset, T) -> U, f3: impl FnOnce(&mut htm::CpuInput, T) -> U) -> PyResult<U> {
    encode_err(sdr, scalar, |a, b| Ok(f1(a, b)), |a, b| Ok(f2(a, b)), |a, b| Ok(f3(a, b)))
}

fn encode_err<T, U>(sdr: PyObject, scalar: T, f1: impl FnOnce(&mut htm::CpuSDR, T) -> PyResult<U>, f2: impl FnOnce(&mut htm::CpuBitset, T) -> PyResult<U>, f3: impl FnOnce(&mut htm::CpuInput, T) -> PyResult<U>) -> PyResult<U> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let o = if let Ok(mut sdr) = sdr.extract::<PyRefMut<CpuBitset>>(py) {
        f2(&mut sdr.bits, scalar)
    } else if let Ok(mut sdr) = sdr.extract::<PyRefMut<CpuInput>>(py) {
        f3(&mut sdr.inp, scalar)
    } else {
        let mut sdr = sdr.extract::<PyRefMut<CpuSDR>>(py)?;
        f1(&mut sdr.sdr, scalar)
    };
    o
}

#[pymethods]
impl BitsEncoder {
    pub fn encode(&self, sdr: PyObject, val: &PyAny) -> PyResult<()> {
        if let Ok(bools) = val.extract::<Vec<bool>>() {
            self.encode_from_bools(sdr, bools)
        } else if let Ok(indices) = val.extract::<Vec<u32>>() {
            self.encode_from_indices(sdr, indices)
        } else if let Ok(indices) = val.extract::<PyRef<CpuSDR>>() {
            self.encode_from_sdr(sdr, indices)
        } else {
            self.encode_from_numpy(sdr, val)
        }
    }
    pub fn encode_from_sdr(&self, sdr: PyObject, indices: PyRef<CpuSDR>) -> PyResult<()> {
        encode(sdr, indices.sdr.as_slice(),
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y))
    }
    pub fn encode_from_indices(&self, sdr: PyObject, indices: Vec<u32>) -> PyResult<()> {
        encode(sdr, indices.as_slice(),
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y))
    }
    pub fn encode_from_bools(&self, sdr: PyObject, bools: Vec<bool>) -> PyResult<()> {
        encode(sdr, bools.as_slice(),
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y))
    }
    pub fn encode_from_numpy(&self, sdr: PyObject, numpy: &PyAny) -> PyResult<()> {
        let array = py_any_as_numpy::<bool>(numpy)?;
        let array = unsafe { array.as_slice()? };
        if array.len() != self.enc.neuron_range_len() as usize {
            return Err(PyValueError::new_err(format!("Expected numpy array of size {} but got {}", self.enc.neuron_range_len(), array.len())));
        }
        encode(sdr, array,
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y))
    }
}

#[pymethods]
impl IntegerEncoder {
    pub fn encode(&self, sdr: PyObject, scalar: u32) -> PyResult<()> {
        encode(sdr, scalar,
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y))
    }
}
macro_rules! impl_encoder {
    ($name:ident) => {
#[pymethods]
impl $name {
    pub fn clear(&self, sdr: PyObject) -> PyResult<()>{
        encode(sdr,(),
           |x,_|self.enc.clear(x),
           |x,_|self.enc.clear(x),
           |x,_|self.enc.clear(x))
    }
    #[getter]
    pub fn len(&self)->u32{
        self.enc.neuron_range_len()
    }
    #[getter]
    pub fn begin(&self)->u32{
        self.enc.neuron_range_begin()
    }
    #[getter]
    pub fn end(&self)->u32{
        self.enc.neuron_range_end()
    }
}
    };
}
impl_encoder!(CategoricalEncoder);
impl_encoder!(CircularIntegerEncoder);
impl_encoder!(FloatEncoder);
impl_encoder!(IntegerEncoder);
impl_encoder!(BitsEncoder);
impl_encoder!(BoolEncoder);
impl_encoder!(DayOfMonthEncoder);
impl_encoder!(DayOfYearEncoder);
impl_encoder!(IsWeekendEncoder);
impl_encoder!(TimeOfDayEncoder);
impl_encoder!(DayOfWeekEncoder);
#[pymethods]
impl CategoricalEncoder {
    pub fn encode(&self, sdr: PyObject, scalar: u32) -> PyResult<()> {
        encode(sdr, scalar,
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y))
    }
    #[getter]
    pub fn num_of_categories(&self) -> u32 {
        self.enc.num_of_categories()
    }
    #[getter]
    pub fn sdr_cardinality(&self) -> u32 {
        self.enc.sdr_cardinality()
    }
    #[text_signature = "(sdr, /)"]
    pub fn find_category_with_highest_overlap(&self, sdr: PyObject) -> PyResult<u32> {
        encode(sdr, (),
               |sdr, _| self.enc.find_category_with_highest_overlap(sdr),
               |sdr, _| self.enc.find_category_with_highest_overlap_bitset(sdr),
               |sdr, _| self.enc.find_category_with_highest_overlap(sdr.get_sparse()))
    }
    #[text_signature = "(sdr, /)"]
    pub fn calculate_overlap(&self, sdr: PyObject) -> PyResult<Vec<u32>> {
        encode(sdr, (),
               |sdr, _| self.enc.calculate_overlap(sdr),
               |sdr, _| self.enc.calculate_overlap_bitset(sdr),
               |sdr, _| self.enc.calculate_overlap(sdr.get_sparse()))
    }
}

#[pymethods]
impl CircularIntegerEncoder {
    pub fn encode(&self, sdr: PyObject, scalar: u32) -> PyResult<()> {
        encode(sdr, scalar,
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y))
    }
}

#[pymethods]
impl FloatEncoder {
    pub fn encode(&self, sdr: PyObject, scalar: f32) -> PyResult<()> {
        encode(sdr, scalar,
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y))
    }
}

#[pymethods]
impl DayOfWeekEncoder {
    pub fn encode(&self, sdr: PyObject, scalar: &PyDateTime) -> PyResult<()> {
        let weekday = scalar.call_method("weekday", (), None).unwrap().extract().unwrap();
        encode(sdr, weekday,
               |x, y| self.enc.encode_day_of_week(x, y),
               |x, y| self.enc.encode_day_of_week(x, y),
               |x, y| self.enc.encode_day_of_week(x, y))
    }
}

#[pymethods]
impl DayOfMonthEncoder {
    pub fn encode(&self, sdr: PyObject, scalar: &PyDateTime) -> PyResult<()> {
        let day = scalar.get_day() as u32 - 1;
        encode(sdr, day,
               |x, y| self.enc.encode_day_of_month(x, y),
               |x, y| self.enc.encode_day_of_month(x, y),
               |x, y| self.enc.encode_day_of_month(x, y))
    }
}

#[pymethods]
impl DayOfYearEncoder {
    pub fn encode(&self, sdr: PyObject, scalar: &PyDateTime) -> PyResult<()> {
        let days_up_to_month = [0, 31, 31 + 28, 31 + 28 + 31, 31 + 28 + 31 + 30, 31 + 28 + 31 + 30 + 31, 31 + 28 + 31 + 30 + 31 + 30, 31 + 28 + 31 + 30 + 31 + 30 + 31, 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31, 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30, 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31, 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30];
        let mut day = scalar.get_day() as u32 - 1;
        let month = scalar.get_month() as usize;
        if month >= 2 {
            let year = scalar.get_year();
            if year % 400 == 0 || (year % 4 == 0 && year % 100 != 0) {
                day += 1; // leap year
            }
        }
        day += days_up_to_month[month - 1];
        encode(sdr, day,
               |x, y| self.enc.encode_day_of_year(x, y),
               |x, y| self.enc.encode_day_of_year(x, y),
               |x, y| self.enc.encode_day_of_year(x, y))
    }
}

#[pymethods]
impl IsWeekendEncoder {
    pub fn encode(&self, sdr: PyObject, scalar: &PyDateTime) -> PyResult<()> {
        let day = scalar.call_method("weekday", (), None).unwrap().extract::<u32>().unwrap();
        let is_weekend = day >= 5;

        encode(sdr, is_weekend,
               |x, y| self.enc.encode_is_weekend(x, y),
               |x, y| self.enc.encode_is_weekend(x, y),
               |x, y| self.enc.encode_is_weekend(x, y))
    }
}

#[pymethods]
impl TimeOfDayEncoder {
    pub fn encode(&self, sdr: PyObject, scalar: &PyDateTime) -> PyResult<()> {
        let sec = (60 * scalar.get_hour() as u32 + scalar.get_minute() as u32) * 60 + scalar.get_second() as u32;
        encode(sdr, sec,
               |x, y| self.enc.encode_time_of_day(x, y),
               |x, y| self.enc.encode_time_of_day(x, y),
               |x, y| self.enc.encode_time_of_day(x, y))
    }
}

#[pymethods]
impl BoolEncoder {
    pub fn encode(&self, sdr: PyObject, scalar: bool) -> PyResult<()> {
        encode(sdr, scalar,
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y),
               |x, y| self.enc.encode(x, y))
    }
}


#[pymethods]
impl Population {
    #[new]
    pub fn new(population_size:Option<usize>,num_of_segments:Option<usize>) -> Self {
        Self{pop:htm::Population::new(population_size.unwrap_or(0),num_of_segments.unwrap_or(1))}
    }

    #[text_signature = "(input_range,total_region,subregion_start,subregion_end,synapse_count)"]
    pub fn add_uniform_rand_inputs_from_area(&mut self, input_range:(usize,usize), total_region:PyObject, subregion_start:PyObject,subregion_end:PyObject,synapse_count:usize) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let input_range = input_range.0..input_range.1;
        let total_region = arrX(py,&total_region,1,1,input_range.len())?;
        let subregion_start = arrX(py,&subregion_start,0,0,0)?;
        let subregion_end = arrX(py,&subregion_end,total_region[0],total_region[1],total_region[2])?;
        Ok(self.pop.add_uniform_rand_inputs_from_area(input_range,total_region,subregion_start..subregion_end,synapse_count,&mut rand::thread_rng()))
    }

    #[text_signature = "(input_range,synapse_count)"]
    pub fn add_uniform_rand_inputs_from_range(&mut self, input_range:(usize,usize),synapse_count:usize) -> PyResult<()> {
        Ok(self.pop.add_uniform_rand_inputs_from_range(input_range.0..input_range.1,synapse_count,&mut rand::thread_rng()))
    }

    #[text_signature = "(input_range,total_region,subregion_start,subregion_end)"]
    pub fn add_all_inputs_from_area(&mut self, input_range:(usize,usize), total_region:PyObject, subregion_start:PyObject,subregion_end:PyObject) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let input_range = input_range.0..input_range.1;
        let total_region = arrX(py,&total_region,1,1,input_range.len())?;
        let subregion_start = arrX(py,&subregion_start,0,0,0)?;
        let subregion_end = arrX(py,&subregion_end,total_region[0],total_region[1],total_region[2])?;
        Ok(self.pop.add_all_inputs_from_area(input_range,total_region,subregion_start..subregion_end,&mut rand::thread_rng()))
    }

    #[text_signature = "(input_range)"]
    pub fn add_all_inputs_from_range(&mut self, input_range:(usize,usize)) -> PyResult<()> {
        Ok(self.pop.add_all_inputs_from_range(input_range.0..input_range.1,&mut rand::thread_rng()))
    }

    #[text_signature = "(input_range,neurons_per_column,synapses_per_segment,stride,kernel,input_size)"]
    pub fn add_2d_column_grid_with_3d_input(&mut self, input_range:(usize,usize),neurons_per_column: usize,synapses_per_segment: usize, stride:PyObject, kernel:PyObject,input_size:PyObject) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let stride = arr2(py,&stride)?;
        let kernel = arr2(py,&kernel)?;
        let input_size = arr3(py,&input_size)?;
        Ok(self.pop.add_2d_column_grid_with_3d_input(input_range.0..input_range.1,neurons_per_column,synapses_per_segment,stride,kernel,input_size,&mut rand::thread_rng()))
    }
    #[text_signature = "(input_range,neurons_per_column,segments_per_neuron,synapses_per_segment,stride,kernel,input_size)"]
    pub fn push_add_2d_column_grid_with_3d_input(&mut self, input_range:(usize,usize),neurons_per_column: usize,segments_per_neuron: usize,synapses_per_segment: usize, stride:PyObject, kernel:PyObject,input_size:PyObject) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let stride = arr2(py,&stride)?;
        let kernel = arr2(py,&kernel)?;
        let input_size = arr3(py,&input_size)?;
        let output = [input_size[1],input_size[2]].conv_out_size(&stride,&kernel);
        self.pop.push_neurons((output.product()*neurons_per_column) as usize,segments_per_neuron);
        Ok(self.pop.add_2d_column_grid_with_3d_input(input_range.0..input_range.1,neurons_per_column,synapses_per_segment,stride,kernel,input_size,&mut rand::thread_rng()))
    }
    #[text_signature = "(neurons_per_column,segments_per_neuron,stride,kernel,input_size)"]
    pub fn push_2d_column_grid_with_3d_input(&mut self, neurons_per_column: usize,segments_per_neuron: usize, stride:PyObject, kernel:PyObject,input_size:PyObject) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let stride = arr2::<usize>(py,&stride)?;
        let kernel = arr2(py,&kernel)?;
        let input_size = arr3(py,&input_size)?;
        let output = [input_size[1],input_size[2]].conv_out_size(&stride,&kernel);
        self.pop.push_neurons(output.product()*neurons_per_column,segments_per_neuron);
        Ok(())
    }
    #[text_signature = "()"]
    pub fn set_weights_random(&mut self){
        self.pop.set_weights_random(&mut rand::thread_rng())
    }
    #[text_signature = "()"]
    pub fn set_weights_uniform(&mut self){
        self.pop.set_weights_uniform()
    }
    #[text_signature = "(weight)"]
    pub fn set_weights_const(&mut self, weight:f32){
        self.pop.set_weights_const(weight)
    }
    #[text_signature = "(scale)"]
    pub fn set_weights_scaled(&mut self, scale:f32){
        self.pop.set_weights_scaled(scale)
    }
    #[text_signature = "()"]
    pub fn get_weights_sum(&mut self)->f32{
        self.pop.get_weights_sum()
    }
    #[text_signature = "()"]
    pub fn set_weights_normalized(&mut self){
        self.pop.set_weights_normalized()
    }
    #[text_signature = "()"]
    pub fn total_synapses(&mut self)->usize{
        self.pop.total_synapses()
    }
    #[text_signature = "()"]
    pub fn get_weights_mean(&mut self)->f32{
        self.pop.get_weights_mean()
    }

    #[text_signature = "(other)"]
    pub fn zip_join(&mut self,other:&Population) {
        self.pop.zip_join(&other.pop)
    }
    #[text_signature = "(other)"]
    pub fn append(&mut self,other:&Population) {
        self.pop.append(&mut other.pop.clone())
    }
    #[text_signature = "(other)"]
    pub fn zip_append(&mut self,other:&Population) {
        self.pop.zip_append(&mut other.pop.clone())
    }
    #[text_signature = "()"]
    pub fn clone(&self) -> Self{
        Self{pop:self.pop.clone()}
    }
    #[text_signature = "(neuron)"]
    pub fn push(&mut self, neuron:&Neuron) {
        self.pop.push(neuron.n.clone())
    }
    #[text_signature = "(population_size,segments_per_neuron)"]
    pub fn push_neurons(&mut self, population_size: usize, segments_per_neuron: Option<usize>){
        self.pop.push_neurons(population_size,segments_per_neuron.unwrap_or(1))
    }
    #[text_signature = "(index)"]
    pub fn remove(&mut self, index:usize) -> Neuron {
        Neuron{n:self.pop.remove(index)}
    }
    #[getter]
    fn get_len(&self) -> usize {
        self.pop.len()
    }
}

#[pymethods]
impl Neuron {
    #[new]
    pub fn new(num_of_segments:usize) -> Self {
        Self{n:htm::Neuron::new(num_of_segments)}
    }
    #[text_signature = "(input_range,total_region,subregion_start,subregion_end,synapse_count)"]
    pub fn add_uniform_rand_inputs_from_area(&mut self, input_range:(usize,usize), total_region:PyObject, subregion_start:PyObject,subregion_end:PyObject,synapse_count:usize) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let input_range = input_range.0..input_range.1;
        let total_region = arrX(py,&total_region,1,1,input_range.len())?;
        let subregion_start = arrX(py,&subregion_start,0,0,0)?;
        let subregion_end = arrX(py,&subregion_end,total_region[0],total_region[1],total_region[2])?;
        Ok(self.n.add_uniform_rand_inputs_from_area(input_range,total_region,subregion_start..subregion_end,synapse_count,&mut rand::thread_rng()))
    }

    #[text_signature = "(input_range,synapse_count)"]
    pub fn add_uniform_rand_inputs_from_range(&mut self, input_range:(usize,usize),synapse_count:usize) -> PyResult<()> {
        Ok(self.n.add_uniform_rand_inputs_from_range(input_range.0..input_range.1,synapse_count,&mut rand::thread_rng()))
    }
    #[text_signature = "()"]
    pub fn set_weights_random(&mut self){
        self.n.set_weights_random(&mut rand::thread_rng())
    }
    #[text_signature = "()"]
    pub fn set_weights_uniform(&mut self){
        self.n.set_weights_uniform()
    }
    #[text_signature = "(weight)"]
    pub fn set_weights_const(&mut self, weight:f32){
        self.n.set_weights_const(weight)
    }
    #[text_signature = "(scale)"]
    pub fn set_weights_scaled(&mut self, scale:f32){
        self.n.set_weights_scaled(scale)
    }
    #[text_signature = "()"]
    pub fn get_weights_sum(&mut self)->f32{
        self.n.get_weights_sum()
    }
    #[text_signature = "()"]
    pub fn set_weights_normalized(&mut self){
        self.n.set_weights_normalized()
    }
    #[text_signature = "()"]
    pub fn total_synapses(&mut self)->usize{
        self.n.total_synapses()
    }
    #[text_signature = "()"]
    pub fn get_weights_mean(&mut self)->f32{
        self.n.get_weights_mean()
    }
    #[text_signature = "(input_range,total_region,subregion_start,subregion_end)"]
    pub fn add_all_inputs_from_area(&mut self, input_range:(usize,usize), total_region:PyObject, subregion_start:PyObject,subregion_end:PyObject) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let input_range = input_range.0..input_range.1;
        let total_region = arrX(py,&total_region,1,1,input_range.len())?;
        let subregion_start = arrX(py,&subregion_start,0,0,0)?;
        let subregion_end = arrX(py,&subregion_end,total_region[0],total_region[1],total_region[2])?;
        Ok(self.n.add_all_inputs_from_area(input_range,total_region,subregion_start..subregion_end,&mut rand::thread_rng()))
    }

    #[text_signature = "(input_range)"]
    pub fn add_all_inputs_from_range(&mut self, input_range:(usize,usize)) -> PyResult<()> {
        Ok(self.n.add_all_inputs_from_range(input_range.0..input_range.1,&mut rand::thread_rng()))
    }
    #[text_signature = "(other)"]
    pub fn zip_join(&mut self, other:&Self) {
        self.n.zip_join(&other.n)
    }
    #[text_signature = "()"]
    pub fn dedup_all(&mut self) {
        self.n.dedup_all()
    }

    #[text_signature = "()"]
    pub fn collapse(&mut self) {
        self.n.collapse()
    }
    #[text_signature = "(other)"]
    pub fn append(&mut self,other:&Neuron) {
        self.n.append(&mut other.n.clone())
    }
    #[getter]
    pub fn is_empty(&self) -> bool {
        self.n.is_empty()
    }
    #[text_signature = "()"]
    pub fn clone(&self) -> Self{
        Self{n:self.n.clone()}
    }
    #[text_signature = "(segment)"]
    pub fn push(&mut self, segment:&Segment) {
        self.n.push(segment.seg.clone())
    }
    #[text_signature = "(index)"]
    pub fn remove(&mut self, index:usize) -> Segment {
        Segment{seg:self.n.remove(index)}
    }
    #[getter]
    fn get_len(&self) -> usize {
        self.n.len()
    }

}

#[pymethods]
impl Segment {
    #[new]
    pub fn new() -> Self {
        Self{seg:htm::Segment::new()}
    }
    #[text_signature = "(input_range,total_region,subregion_start,subregion_end,synapse_count)"]
    pub fn add_uniform_rand_inputs_from_area(&mut self, input_range:(usize,usize), total_region:PyObject, subregion_start:PyObject,subregion_end:PyObject,synapse_count:usize) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let input_range = input_range.0..input_range.1;
        let total_region = arrX(py,&total_region,1,1,input_range.len())?;
        let subregion_start = arrX(py,&subregion_start,0,0,0)?;
        let subregion_end = arrX(py,&subregion_end,total_region[0],total_region[1],total_region[2])?;
        Ok(self.seg.add_uniform_rand_inputs_from_area(input_range,total_region,subregion_start..subregion_end,synapse_count,&mut rand::thread_rng()))
    }

    #[text_signature = "(input_range,synapse_count)"]
    pub fn add_uniform_rand_inputs_from_range(&mut self, input_range:(usize,usize),synapse_count:usize) -> PyResult<()> {
        Ok(self.seg.add_uniform_rand_inputs_from_range(input_range.0..input_range.1,synapse_count,&mut rand::thread_rng()))
    }

    #[text_signature = "(other)"]
    pub fn join(&mut self, other:&Self) {
        self.seg.join(&other.seg)
    }
    #[text_signature = "()"]
    pub fn dedup(&mut self) {
        self.seg.dedup()
    }
    #[text_signature = "()"]
    pub fn clone(&self) -> Self{
        Self{seg:self.seg.clone()}
    }
    #[text_signature = "(neuron_idx,weight)"]
    pub fn push(&mut self, neuron_idx:usize, weight:f32) {
        self.seg.synapses.push(Synapse::new(neuron_idx,weight))
    }
    #[text_signature = "(index)"]
    pub fn remove(&mut self, index:usize) -> usize {
        self.seg.synapses.swap_remove(index).input_idx
    }
    #[getter]
    fn get_len(&self) -> usize {
        self.seg.synapses.len()
    }
    #[text_signature = "()"]
    pub fn set_weights_random(&mut self){
        self.seg.set_weights_random(&mut rand::thread_rng())
    }
    #[text_signature = "()"]
    pub fn set_weights_uniform(&mut self){
        self.seg.set_weights_uniform()
    }
    #[text_signature = "(weight)"]
    pub fn set_weights_const(&mut self, weight:f32){
        self.seg.set_weights_const(weight)
    }
    #[text_signature = "(scale)"]
    pub fn set_weights_scaled(&mut self, scale:f32){
        self.seg.set_weights_scaled(scale)
    }
    #[text_signature = "()"]
    pub fn get_weights_sum(&mut self)->f32{
        self.seg.get_weights_sum()
    }
    #[text_signature = "()"]
    pub fn set_weights_normalized(&mut self){
        self.seg.set_weights_normalized()
    }
    #[text_signature = "()"]
    pub fn get_weights_mean(&mut self)->f32{
        self.seg.get_weights_mean()
    }
    #[text_signature = "(input_range,total_region,subregion_start,subregion_end)"]
    pub fn add_all_inputs_from_area(&mut self, input_range:(usize,usize), total_region:PyObject, subregion_start:PyObject,subregion_end:PyObject) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let input_range = input_range.0..input_range.1;
        let total_region = arrX(py,&total_region,1,1,input_range.len())?;
        let subregion_start = arrX(py,&subregion_start,0,0,0)?;
        let subregion_end = arrX(py,&subregion_end,total_region[0],total_region[1],total_region[2])?;
        Ok(self.seg.add_all_inputs_from_area(input_range,total_region,subregion_start..subregion_end,&mut rand::thread_rng()))
    }

    #[text_signature = "(input_range)"]
    pub fn add_all_inputs_from_range(&mut self, input_range:(usize,usize)) -> PyResult<()> {
        Ok(self.seg.add_all_inputs_from_range(input_range.0..input_range.1,&mut rand::thread_rng()))
    }
}


fn arr3<'py, T:Element+Copy+FromPyObject<'py>>(py: Python<'py>, t: &'py PyObject) -> PyResult<[T; 3]> {
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

fn arrX<'py,T:Element+Copy+FromPyObject<'py>>(py: Python<'py>, t: &'py PyObject, default0: T, default1: T, default2: T) -> PyResult<[T; 3]> {
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

fn arr2<'py,T:Element+Copy+FromPyObject<'py>>(py: Python<'py>, t: &'py PyObject) -> PyResult<[T; 2]> {
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

#[pymethods]
impl CpuHTM {
    #[new]
    pub fn new(input_size: u32, n: u32, population: Option<&Population>) -> PyResult<Self> {
        let mut htm = htm::CpuHTM::new(input_size, n);
        if let Some(population) = population {
            htm.add_population(&population.pop);
        }
        Ok(CpuHTM { htm })
    }
    #[text_signature = "(input_shapes,output_shapes,input,output,input_cell_margin, output_cell_margin)"]
    pub fn visualise(&mut self, input_shapes:Vec<PyObject>, output_shapes:Vec<PyObject>,input:Option<&CpuSDR>,output:Option<&CpuSDR>,input_cell_margin: Option<f32>, output_cell_margin: Option<f32>) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let input_cell_margin = input_cell_margin.unwrap_or(0.2);
        let output_cell_margin = output_cell_margin.unwrap_or(0.2);
        let input_shapes:PyResult<Vec<[u32;3]>> = input_shapes.into_iter().map(|o|arrX(py,&o,1,1,self.htm.input_size())).collect();
        let output_shapes:PyResult<Vec<[u32;3]>> = output_shapes.into_iter().map(|o|arrX(py,&o,1,1,self.htm.minicolumns_as_slice().len() as u32)).collect();
        let input = input.map(|s|s.sdr.as_slice()).unwrap_or(&[]);
        let output = output.map(|s|s.sdr.as_slice()).unwrap_or(&[]);
        Ok(visualise_cpu_htm2(&self.htm, &input_shapes?,&output_shapes?,input,output,input_cell_margin,output_cell_margin))
    }

    #[text_signature = "(file)"]
    pub fn pickle(&mut self, file: String) -> PyResult<()> {
        pickle(&self.htm, file)
    }
    #[text_signature = "(permenence)"]
    pub fn set_all_permanences(&mut self, val: f32) {
        self.htm.set_all_permanences(val)
    }
    #[text_signature = "(permenence)"]
    pub fn multiply_all_permanences(&mut self, val: f32) {
        self.htm.multiply_all_permanences(val)
    }
    #[text_signature = "(population)"]
    pub fn add_population(&mut self, population:&Population) {
        self.htm.add_population(&population.pop)
    }
    #[getter]
    fn get_synapse_count(&self) -> u32 {
        self.htm.feedforward_connections_as_slice().len() as u32
    }
    #[text_signature = "(column_idx)"]
    fn get_synapses(&self, column_idx: Option<u32>) -> Vec<(u32, f32)> {
        if let Some(column_idx) = column_idx {
            let s = self.htm.minicolumns_as_slice()[column_idx as usize];
            self.htm.feedforward_connections_as_slice()[s.connection_offset as usize..(s.connection_offset + s.connection_len) as usize].iter().map(|c| (c.input_id, c.permanence)).collect()
        } else {
            self.htm.feedforward_connections_as_slice().iter().map(|c| (c.input_id, c.permanence)).collect()
        }
    }
    #[text_signature = "( /)"]
    fn clone(&self) -> CpuHTM {
        CpuHTM { htm: self.htm.clone() }
    }
    #[getter]
    fn get_input_size(&self) -> u32 {
        self.htm.input_size()
    }
    #[getter]
    fn get_minicolumn_count(&self) -> u32 {
        self.htm.minicolumns_as_slice().len() as u32
    }

    #[getter]
    fn get_n(&self) -> u32 {
        self.htm.n
    }

    #[setter]
    fn set_n(&mut self, n: u32) {
        self.htm.n = n
    }

    #[getter]
    fn get_permanence_decrement(&self) -> f32 {
        self.htm.permanence_decrement_increment[0]
    }

    #[setter]
    fn set_permanence_decrement(&mut self, permanence_decrement: f32) {
        self.htm.permanence_decrement_increment[0] = permanence_decrement
    }

    #[getter]
    fn get_permanence_increment(&self) -> f32 {
        self.htm.permanence_decrement_increment[1]
    }

    #[setter]
    fn set_permanence_increment(&mut self, permanence_increment: f32) {
        self.htm.permanence_decrement_increment[1] = permanence_increment
    }

    #[getter]
    fn get_permanence_threshold(&self) -> f32 {
        self.htm.permanence_threshold
    }

    #[setter]
    fn set_permanence_threshold(&mut self, permanence_threshold: f32) {
        self.htm.permanence_threshold = permanence_threshold
    }

    #[getter]
    fn get_max_overlap(&self) -> u32 {
        self.htm.max_overlap
    }

    #[setter]
    fn set_max_overlap(&mut self, max_overlap: u32) {
        self.htm.max_overlap = max_overlap
    }

    #[call]
    fn __call__(&mut self, bitset_input: &CpuBitset, learn: Option<bool>) -> CpuSDR {
        self.infer(bitset_input, learn)
    }
    #[text_signature = "(bitset_input, learn)"]
    fn infer(&mut self, bitset_input: &CpuBitset, learn: Option<bool>) -> CpuSDR {
        CpuSDR { sdr: self.htm.infer(&bitset_input.bits, learn.unwrap_or(false)) }
    }
    #[text_signature = "(minicolumns_per_column,minicolumn_stride,bitset_input, learn)"]
    fn infer_and_group_into_columns(&mut self, minicolumns_per_column: usize, minicolumn_stride:usize, bitset_input: &CpuBitset, learn: Option<bool>) -> CpuSDR {
        CpuSDR { sdr: self.htm.infer_and_group_into_columns(minicolumns_per_column, minicolumn_stride,&bitset_input.bits, learn.unwrap_or(false)) }
    }
    #[text_signature = "(active_columns,bitset_input)"]
    fn update_permanence(&mut self, active_columns: &CpuSDR, bitset_input: &CpuBitset) {
        self.htm.update_permanence(&active_columns.sdr, &bitset_input.bits)
    }
    #[text_signature = "(top_n_minicolumns,active_columns,bitset_input)"]
    fn update_permanence_ltd(&mut self, top_n_minicolumns: &CpuSDR, active_columns: &CpuBitset, bitset_input: &CpuBitset) {
        self.htm.update_permanence_ltd(&top_n_minicolumns.sdr, &active_columns.bits, &bitset_input.bits)
    }
    #[text_signature = "(top_n_minicolumns,active_columns,bitset_input,penalty_multiplier)"]
    fn update_permanence_and_penalize(&mut self, active_columns: &CpuBitset, bitset_input: &CpuBitset, penalty_multiplier: Option<f32>) {
        self.htm.update_permanence_and_penalize(&active_columns.bits, &bitset_input.bits, penalty_multiplier.unwrap_or(-1.))
    }
    #[text_signature = "(top_n_minicolumns,active_columns,bitset_input,activity_threshold,penalty_multiplier)"]
    fn update_permanence_and_penalize_thresholded(&mut self, active_columns: &CpuBitset, bitset_input: &CpuBitset, activity_threshold: u32, penalty_multiplier: Option<f32>) {
        self.htm.update_permanence_and_penalize_thresholded(&active_columns.bits, &bitset_input.bits, activity_threshold, penalty_multiplier.unwrap_or(-1.))
    }
    #[text_signature = "(bitset_input)"]
    fn compute(&mut self, bitset_input: &CpuBitset) -> CpuSDR {
        CpuSDR { sdr: self.htm.compute(&bitset_input.bits) }
    }

    #[text_signature = "(minicolumns_per_column,bitset_input)"]
    fn compute_and_group_into_columns(&mut self, minicolumns_per_column: usize, minicolumn_stride:usize, bitset_input: &CpuBitset) -> CpuSDR {
        CpuSDR { sdr: self.htm.compute_and_group_into_columns(minicolumns_per_column, minicolumn_stride,&bitset_input.bits) }
    }

    #[text_signature = "(minicolumn_id)"]
    fn get_overlap(&self, minicolumn_id: u32) -> i32 {
        self.htm.minicolumns_as_slice()[minicolumn_id as usize].overlap
    }

    #[text_signature = "(minicolumn_id)"]
    fn get_synapses_range(&self, minicolumn_id: u32) -> (u32, u32) {
        let range = &self.htm.minicolumns_as_slice()[minicolumn_id as usize];
        (range.connection_offset, range.connection_len)
    }

    #[text_signature = "(synapse_id)"]
    fn get_synapse_input_and_permanence(&self, synapse_id: u32) -> (u32, f32) {
        let s = &self.htm.feedforward_connections_as_slice()[synapse_id as usize];
        (s.input_id, s.permanence)
    }

    #[text_signature = "(synapse_id, input_id)"]
    fn set_synapse_input(&mut self, synapse_id: u32, input_id: u32) {
        self.htm.feedforward_connections_as_mut_slice()[synapse_id as usize].input_id = input_id;
    }
    #[text_signature = "(synapse_id, permanence)"]
    fn set_synapse_permanence(&mut self, synapse_id: u32, permanence: f32) {
        self.htm.feedforward_connections_as_mut_slice()[synapse_id as usize].permanence = permanence;
    }
    #[text_signature = "(context)"]
    fn to_ocl(&self, context: &mut Context) -> PyResult<OclHTM> {
        OclHTM::new(context, self)
    }
}


#[pymethods]
impl OclSDR {
    #[new]
    pub fn new(context: &mut Context, max_active_neurons: u32) -> PyResult<Self> {
        context.compile_htm_program().and_then(|prog| htm::OclSDR::new(prog.clone(), max_active_neurons).map_err(ocl_err_to_py_ex)).map(|sdr| OclSDR { sdr })
    }

    #[getter]
    fn get_max_active_neurons(&self) -> usize {
        self.sdr.max_active_neurons()
    }

    #[setter]
    fn set_active_neurons(&mut self, neuron_indices: Vec<u32>) -> PyResult<()> {
        self.sdr.set(neuron_indices.as_slice()).map_err(ocl_err_to_py_ex)
    }

    #[getter]
    fn get_active_neurons(&self) -> PyResult<Vec<u32>> {
        self.sdr.get().map_err(ocl_err_to_py_ex)
    }
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

#[pymethods]
impl CpuSDR {
    #[new]
    pub fn new(sdr: Option<Vec<u32>>) -> Self {
        CpuSDR { sdr: sdr.map(|sdr| htm::CpuSDR::from(sdr)).unwrap_or_else(|| htm::CpuSDR::new()) }
    }
    #[text_signature = "()"]
    fn clear(&mut self) {
        self.sdr.clear()
    }
    #[text_signature = "(neuron_index)"]
    fn push(&mut self, neuron_index: u32) {
        self.sdr.push(neuron_index)
    }
    #[text_signature = "(number_of_bits_to_retain)"]
    fn shrink(&mut self, number_of_bits_to_retain: usize) {
        self.sdr.shrink(number_of_bits_to_retain)
    }
    #[text_signature = "(number_of_bits_to_retain)"]
    fn subsample(&mut self, number_of_bits_to_retain: usize) {
        self.sdr.shrink(number_of_bits_to_retain)
    }
    #[text_signature = "(other_sdr)"]
    pub fn extend(&mut self, other: &CpuSDR) {
        self.sdr.extend(&other.sdr)
    }
    #[text_signature = "(shift)"]
    pub fn shift(&mut self, shift: i32) {
        self.sdr.shift(shift)
    }
    #[text_signature = "(other_sdr)"]
    pub fn subtract(&self, other: &CpuSDR) -> Self {
        let mut c = self.clone();
        c.sdr.subtract(&other.sdr);
        c
    }
    #[text_signature = "(file)"]
    pub fn pickle(&mut self, file: String) -> PyResult<()> {
        pickle(&self.sdr, file)
    }
    #[text_signature = "(n,range_begin_inclusive,range_end_exclusive)"]
    pub fn add_unique_random(&mut self, n: u32, range_begin: u32, range_end: u32) {
        self.sdr.add_unique_random(n, range_begin..range_end)
    }
    #[text_signature = "(other_sdr,n)"]
    pub fn randomly_extend_from<'py>(mut slf: PyRefMut<'py, Self>, other: &CpuSDR, n: Option<u32>) -> PyRefMut<'py, Self> {
        let c = slf.cardinality();
        slf.sdr.randomly_extend_from(&other.sdr, n.unwrap_or(c) as usize);
        slf
    }
    #[text_signature = "(other_sdr)"]
    pub fn subtract_<'py>(mut slf: PyRefMut<'py, Self>, other: &CpuSDR) -> PyRefMut<'py, Self> {
        slf.sdr.subtract(&other.sdr);
        slf
    }
    #[text_signature = "(other_sdr)"]
    pub fn union(&self, other: &CpuSDR) -> CpuSDR {
        CpuSDR { sdr: self.sdr.union(&other.sdr) }
    }
    #[text_signature = "(other_sdr)"]
    pub fn union_<'py>(mut slf: PyRefMut<'py, Self>, other: &CpuSDR) -> PyRefMut<'py, Self> {
        slf.sdr = slf.sdr.union(&other.sdr);
        slf
    }
    #[text_signature = "(other_sdr)"]
    pub fn intersection(&self, other: &CpuSDR) -> CpuSDR {
        CpuSDR { sdr: self.sdr.intersection(&other.sdr) }
    }
    #[text_signature = "(other_sdr)"]
    pub fn intersection_<'py>(mut slf: PyRefMut<'py, Self>, other: &CpuSDR) -> PyRefMut<'py, Self> {
        slf.sdr = slf.sdr.intersection(&other.sdr);
        slf
    }
    #[text_signature = "()"]
    pub fn normalize(&mut self) {
        self.sdr.normalize()
    }
    #[text_signature = "()"]
    pub fn is_normalized(&self) -> bool {
        self.sdr.is_normalized()
    }
    #[text_signature = "(other_sdr)"]
    pub fn overlap(&self, other: PyObject) -> PyResult<u32> {
        encode_err(other, ()
                   , |sdr, _| Ok(self.sdr.overlap(sdr))
                   , |sdr, _| Err(PyValueError::new_err("Cannot compare SDR with bitset"))
                   , |sdr, _| Ok(self.sdr.overlap(sdr.get_sparse())))
    }
    #[setter]
    fn set_active_neurons(&mut self, neuron_indices: Vec<u32>) {
        self.sdr.set(neuron_indices.as_slice())
    }

    #[getter]
    fn get_active_neurons(&self) -> Vec<u32> {
        self.sdr.clone().to_vec()
    }
    #[getter]
    fn cardinality(&self) -> u32 {
        self.sdr.cardinality()
    }
    #[text_signature = "(input_size)"]
    pub fn to_bitset(&self, input_size: u32) -> CpuBitset {
        CpuBitset { bits: htm::CpuBitset::from_sdr(&self.sdr, input_size) }
    }
    #[text_signature = "(input_size)"]
    pub fn to_input(&self, input_size: u32) -> CpuInput {
        CpuInput { inp: htm::CpuInput::from_sparse(self.sdr.clone(), input_size) }
    }
    #[text_signature = "()"]
    pub fn clone(&self) -> Self {
        CpuSDR { sdr: self.sdr.clone() }
    }
    #[text_signature = "(context, max_cardinality)"]
    fn to_ocl(&self, context: &mut Context, max_cardinality: u32) -> PyResult<OclSDR> {
        let ctx = context.compile_htm_program()?;
        let sdr = htm::OclSDR::from_cpu(ctx.clone(), &self.sdr, max_cardinality).map_err(ocl_err_to_py_ex)?;
        Ok(OclSDR { sdr })
    }
}

#[pyfunction]
#[text_signature = "(output_sdrs,stride,kernel_size,grid_size)"]
pub fn vote_conv2d_transpose(output_sdrs: Vec<Vec<PyRef<CpuSDR>>>, stride: PyObject, kernel_size: PyObject, grid_size: PyObject) -> PyResult<Vec<Vec<CpuSDR>>> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let o = htm::CpuSDR::vote_conv2d_transpose_arr(arr2(py, &stride)?, arr2(py, &kernel_size)?, arr2(py, &grid_size)?, &|i0, i1| &output_sdrs[i0 as usize][i1 as usize].sdr);
    Ok(o.into_iter().map(|a| a.into_iter().map(|sdr| CpuSDR { sdr }).collect()).collect())
}

#[pyfunction]
#[text_signature = "(sdrs,n,activity_threshold,stride,kernel_size)"]
pub fn vote_conv2d(sdrs: Vec<Vec<PyRef<CpuSDR>>>, n: usize, threshold: u32, stride: PyObject, kernel_size: PyObject) -> PyResult<Vec<Vec<CpuSDR>>> {
    let grid_size = [sdrs.len() as u32, sdrs[0].len() as u32];
    let gil = Python::acquire_gil();
    let py = gil.python();
    Ok(htm::CpuSDR::vote_conv2d_arr_with(n, threshold, arr2(py, &stride)?, arr2(py, &kernel_size)?, grid_size, &|i0, i1| &sdrs[i0 as usize][i1 as usize].sdr, |sdr| CpuSDR { sdr }))
}

#[pyfunction]
#[text_signature = "(sdrs,n,activity_threshold)"]
pub fn vote(sdrs: Vec<PyRef<CpuSDR>>, n: usize, threshold: u32) -> CpuSDR {
    CpuSDR { sdr: htm::CpuSDR::vote_over_iter(sdrs.as_slice().iter().map(|s| &s.sdr), n, threshold) }
}

#[pyfunction]
#[text_signature = "(input_size,stride,kernel)"]
pub fn conv_out_size(input_size: PyObject, stride: PyObject, kernel: PyObject) -> PyResult<Vec<u32>> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let input_size = arrX(py,&input_size,1,1,1)?;
    let stride = arrX(py,&stride,1,1,1)?;
    let kernel = arrX(py,&kernel,1,1,1)?;
    let out_size = input_size.conv_out_size(&stride,&kernel);
    Ok(out_size.to_vec())
}

#[pyfunction]
#[text_signature = "(output_size,stride,kernel)"]
pub fn conv_in_size(output_size: PyObject, stride: PyObject, kernel: PyObject) -> PyResult<Vec<u32>> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let output_size = arrX(py,&output_size,1,1,1)?;
    let stride = arrX(py,&stride,1,1,1)?;
    let kernel = arrX(py,&kernel,1,1,1)?;
    let in_size = output_size.conv_in_size(&stride,&kernel);
    Ok(in_size.to_vec())
}
#[pyfunction]
#[text_signature = "(input_size,output_size,kernel)"]
pub fn conv_stride(input_size: PyObject, output_size: PyObject, kernel: PyObject) -> PyResult<Vec<u32>> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let input_size = arrX(py,&input_size,1,1,1)?;
    let output_size = arrX(py,&output_size,1,1,1)?;
    let kernel = arrX(py,&kernel,1,1,1)?;
    let stride = input_size.conv_stride(&output_size,&kernel);
    Ok(stride.to_vec())
}

#[pyfunction]
#[text_signature = "(stride1,kernel1,stride2,kernel2)"]
pub fn conv_compose(stride1: PyObject, kernel1: PyObject,stride2: PyObject, kernel2: PyObject) -> PyResult<(Vec<u32>,Vec<u32>)> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let stride1 = arrX(py,&stride1,1,1,1)?;
    let kernel1 = arrX(py,&kernel1,1,1,1)?;
    let stride2 = arrX(py,&stride2,1,1,1)?;
    let kernel2 = arrX(py,&kernel2,1,1,1)?;
    let (stride,kernel) = stride1.conv_compose(&kernel1,&stride2,&kernel2);
    Ok((stride.to_vec(),kernel.to_vec()))
}

fn py_any_as_numpy<T:Element>(input: &PyAny) -> Result<&PyArrayDyn<T>, PyErr> {
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


#[pyfunction]
#[text_signature = "(numpy_array/)"]
pub fn bitset_from_numpy(input: &PyAny) -> Result<CpuBitset, PyErr> {
    let array = py_any_as_numpy::<bool>(input)?;
    let shape = array.shape();
    if shape.len() > 3 {
        return Err(PyValueError::new_err(format!("Maximum possible dimensionality is 3 but numpy array has shape {:?}", shape)));
    }
    let array = unsafe { array.as_slice()? };
    let mut bitset = htm::CpuBitset::from_bools(array);
    if shape.len() == 2 {
        bitset.reshape2d(shape[0] as u32, shape[1] as u32)
    } else if shape.len() == 3 {
        bitset.reshape3d(shape[0] as u32, shape[1] as u32, shape[2] as u32)
    }

    Ok(CpuBitset { bits: bitset })
}

#[pyfunction]
#[text_signature = "(bits)"]
pub fn bitset_from_bools(bits: Vec<bool>) -> CpuBitset {
    CpuBitset { bits: htm::CpuBitset::from_bools(&bits) }
}

#[pyfunction]
#[text_signature = "(bit_indices)"]
pub fn bitset_from_indices(bit_indices: Vec<u32>, input_size: u32) -> CpuBitset {
    CpuBitset { bits: htm::CpuBitset::from_sdr(&bit_indices, input_size) }
}


#[pymethods]
impl CpuBitset {
    #[getter]
    pub fn width(&self) -> u32 {
        self.bits.width()
    }
    #[getter]
    pub fn depth(&self) -> u32 {
        self.bits.depth()
    }
    #[getter]
    pub fn height(&self) -> u32 {
        self.bits.height()
    }
    #[getter]
    pub fn shape(&self) -> Vec<u32> {
        self.bits.shape().to_vec()
    }
    #[new]
    pub fn new(bit_count: u32) -> Self {
        CpuBitset { bits: htm::CpuBitset::new(bit_count) }
    }
    #[getter]
    pub fn size(&self) -> u32 {
        self.bits.size()
    }
    #[text_signature = "(shape)"]
    pub fn resize(&mut self, shape: PyObject) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let [z, y, x] = arrX(py, &shape, 1, 1, self.bits.size())?;
        Ok(self.bits.resize3d(z, y, x))
    }
    #[text_signature = "(shape)"]
    pub fn reshape(&mut self, shape: PyObject) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let [z, y, x] = arrX(py, &shape, 1, 1, self.bits.size())?;
        Ok(self.bits.reshape3d(z, y, x))
    }
    #[text_signature = "(file)"]
    pub fn pickle(&mut self, file: String) -> PyResult<()> {
        pickle(&self.bits, file)
    }
    #[text_signature = "(bitset)"]
    pub fn overlap(&self, bitset: PyObject) -> PyResult<u32> {
        encode_err(bitset, ()
                   , |sdr, _| Err(PyValueError::new_err("Cannot compare SDR with bitset"))
                   , |sdr, _| Ok(self.bits.overlap(sdr))
                   , |sdr, _| Ok(self.bits.overlap(sdr.get_dense())))
    }
    #[text_signature = "(offset1,offset2,size)"]
    pub fn swap_u32(&mut self, offset1: u32, offset2: u32, size: u32) {
        self.bits.swap_u32(offset1, offset2, size)
    }
    #[text_signature = "(from,to,bit_count)"]
    pub fn set_bits_on_rand(&mut self, from: u32, to: u32, bit_count: u32) {
        self.bits.set_bits_on_rand(from, to, bit_count, &mut rand::thread_rng())
    }
    #[text_signature = "(bit_index)"]
    pub fn is_bit_on(&self, bit_index: u32) -> bool {
        self.bits.is_bit_on(bit_index)
    }
    #[text_signature = "(y,x)"]
    pub fn is_bit_on2d(&self, y: u32, x: u32) -> bool {
        self.bits.is_bit_on2d(y, x)
    }
    #[text_signature = "(z,y,x)"]
    pub fn is_bit_on3d(&self, z: u32, y: u32, x: u32) -> bool {
        self.bits.is_bit_on3d(z, y, x)
    }
    #[text_signature = "(sdr)"]
    pub fn append_to_sdr(&self, sdr: &mut CpuSDR) {
        self.bits.append_to_sdr(&mut sdr.sdr)
    }
    #[getter]
    pub fn cardinality(&self) -> u32 {
        self.bits.cardinality()
    }
    #[text_signature = "(sdr)"]
    pub fn clear(&mut self, sdr: Option<&CpuSDR>) {
        if let Some(sdr) = sdr {
            self.bits.clear(&sdr.sdr)
        } else {
            self.bits.clear_all()
        }
    }
    #[text_signature = "(bit_idx)"]
    pub fn set_bit_on(&mut self, bit_idx: u32) {
        self.bits.set_bit_on(bit_idx)
    }
    #[text_signature = "(y,x)"]
    pub fn set_bit_on2d(&mut self, y: u32, x: u32) {
        self.bits.set_bit_on2d(y, x)
    }
    #[text_signature = "(z,y,x)"]
    pub fn set_bit_on3d(&mut self, z: u32, y: u32, x: u32) {
        self.bits.set_bit_on3d(z, y, x)
    }
    #[text_signature = "(sdr)"]
    pub fn set_bit_off(&mut self, bit_idx: u32) {
        self.bits.set_bit_off(bit_idx)
    }
    #[text_signature = "(sdr,input_range)"]
    pub fn set_bits_on(&mut self, sdr: &CpuSDR,input_range:Option<(u32,u32)>) {
        if let Some(input_range) = input_range{
            self.bits.set_bits_on_in_range(input_range.0..input_range.1,&sdr.sdr)
        }else{
            self.bits.set_bits_on(&sdr.sdr)
        }
    }
    #[text_signature = "(sdr,offset,size)"]
    pub fn set_bits_on3d(&mut self, sdr: &CpuSDR, offset: PyObject, size: PyObject) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(self.bits.set_bits_on3d(arr3(py, &offset)?, arr3(py, &size)?, &sdr.sdr))
    }
    #[text_signature = "(sdr,offset,size)"]
    pub fn set_bits_on2d(&mut self, sdr: &CpuSDR, offset: PyObject, size: PyObject) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(self.bits.set_bits_on2d(arr2(py, &offset)?, arr2(py, &size)?, &sdr.sdr))
    }
    #[text_signature = "(sdr,input_range)"]
    pub fn set_bits_off(&mut self, sdr: &CpuSDR,input_range:Option<(u32,u32)>) {
        if let Some(input_range) = input_range{
            self.bits.set_bits_off_in_range(input_range.0..input_range.1,&sdr.sdr)
        }else{
            self.bits.set_bits_off(&sdr.sdr)
        }
    }
    #[text_signature = "(sdr)"]
    pub fn to_sdr(&self) -> CpuSDR {
        CpuSDR { sdr: htm::CpuSDR::from(&self.bits) }
    }
    #[text_signature = "()"]
    pub fn to_input(&self) -> CpuInput {
        CpuInput { inp: htm::CpuInput::from_dense(self.bits.clone()) }
    }
    #[text_signature = "(context)"]
    fn to_ocl(&self, context: &mut Context) -> PyResult<OclBitset> {
        let ctx = context.compile_htm_program()?;
        let bits = htm::OclBitset::from_cpu(&self.bits, ctx.clone()).map_err(ocl_err_to_py_ex)?;
        Ok(OclBitset { bits })
    }
}


#[pymethods]
impl OclBitset {
    #[new]
    pub fn new(bit_count: u32, context: &mut Context) -> PyResult<Self> {
        let ctx = context.compile_htm_program()?;
        let bits = htm::OclBitset::new(bit_count, ctx.clone()).map_err(ocl_err_to_py_ex)?;
        Ok(OclBitset { bits })
    }
    #[getter]
    pub fn size(&self) -> usize {
        self.bits.size()
    }
    #[text_signature = "(sdr)"]
    pub fn clear(&mut self, sdr: Option<&OclSDR>) -> PyResult<()> {
        if let Some(sdr) = sdr {
            self.bits.clear(&sdr.sdr)
        } else {
            self.bits.clear_all()
        }.map_err(ocl_err_to_py_ex)
    }
    #[text_signature = "(sdr)"]
    pub fn set_bits_on(&mut self, sdr: &OclSDR) -> PyResult<()> {
        self.bits.set_bits_on(&sdr.sdr).map_err(ocl_err_to_py_ex)
    }
    #[text_signature = "(sdr)"]
    pub fn copy_from(&mut self, bits: &CpuBitset) -> PyResult<()> {
        self.bits.copy_from(&bits.bits).map_err(ocl_err_to_py_ex)
    }
    #[text_signature = "(sdr)"]
    pub fn to_cpu(&self) -> PyResult<CpuBitset> {
        self.bits.to_cpu().map(|bits| CpuBitset { bits }).map_err(ocl_err_to_py_ex)
    }
}

#[pymethods]
impl CpuInput {
    #[new]
    pub fn new(size: u32) -> Self {
        CpuInput { inp: htm::CpuInput::new(size) }
    }
    #[getter]
    pub fn size(&self) -> u32 {
        self.inp.size()
    }
    #[text_signature = "(file)"]
    pub fn pickle(&mut self, file: String) -> PyResult<()> {
        pickle(&self.inp, file)
    }
    #[getter]
    pub fn cardinality(&self) -> u32 {
        self.inp.cardinality()
    }
    #[text_signature = "(shape)"]
    pub fn reshape(&mut self, size: PyObject) -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let [z, y, x] = arrX(py, &size, 1, 1, self.inp.size())?;
        Ok(self.inp.reshape3d(z, y, x))
    }
    #[text_signature = "(input)"]
    pub fn overlap(&self, input: PyObject) -> PyResult<u32> {
        encode_err(input, ()
                   , |sdr, _| Ok(self.inp.get_sparse().overlap(sdr))
                   , |sdr, _| Ok(self.inp.get_dense().overlap(sdr))
                   , |sdr, _| Ok(self.inp.get_sparse().overlap(sdr.get_sparse())))
    }
    #[text_signature = "(sdr)"]
    pub fn set_sparse(&mut self, sdr: &CpuSDR) {
        self.inp.set_sparse_from_slice(sdr.sdr.as_slice())
    }
    #[text_signature = "(bitset)"]
    pub fn set_dense(&mut self, bitset: &CpuBitset) {
        self.inp.set_dense(bitset.bits.clone());
    }
    #[text_signature = "()"]
    pub fn get_dense(&mut self) -> CpuBitset {
        CpuBitset { bits: self.inp.get_dense().clone() }
    }
    #[text_signature = "()"]
    pub fn get_sparse(&mut self) -> CpuSDR {
        CpuSDR { sdr: self.inp.get_sparse().clone() }
    }
    #[text_signature = "(neuron_index)"]
    pub fn contains(&mut self, neuron_index: u32) -> bool {
        self.inp.contains(neuron_index)
    }
    #[text_signature = "(neuron_index)"]
    pub fn push(&mut self, neuron_index: u32) {
        self.inp.push(neuron_index)
    }
    #[text_signature = "()"]
    pub fn clear(&mut self) {
        self.inp.clear()
    }
    #[text_signature = "(from,to)"]
    pub fn clear_range(&mut self, from: u32, to: u32) {
        self.inp.clear_range(from, to)
    }
}



#[pymethods]
impl OclHTM {
    #[new]
    pub fn new(context: &mut Context, htm: &CpuHTM) -> PyResult<Self> {
        htm::OclHTM::new(&htm.htm, context.compile_htm_program()?.clone()).map(|htm| OclHTM { htm }).map_err(ocl_err_to_py_ex)
    }
    #[getter]
    fn get_input_size(&self) -> u32 {
        self.htm.input_size()
    }

    #[getter]
    fn get_n(&self) -> u32 {
        self.htm.n
    }

    #[setter]
    fn set_n(&mut self, n: u32) {
        self.htm.n = n
    }

    #[getter]
    fn get_permanence_decrement(&self) -> f32 {
        self.htm.permanence_decrement_increment[0]
    }

    #[setter]
    fn set_permanence_decrement(&mut self, permanence_decrement: f32) {
        self.htm.permanence_decrement_increment[0] = permanence_decrement
    }

    #[getter]
    fn get_permanence_increment(&self) -> f32 {
        self.htm.permanence_decrement_increment[1]
    }

    #[setter]
    fn set_permanence_increment(&mut self, permanence_increment: f32) {
        self.htm.permanence_decrement_increment[1] = permanence_increment
    }

    #[getter]
    fn get_permanence_threshold(&self) -> f32 {
        self.htm.permanence_threshold
    }

    #[setter]
    fn set_permanence_threshold(&mut self, permanence_threshold: f32) {
        self.htm.permanence_threshold = permanence_threshold
    }

    #[getter]
    fn get_max_overlap(&self) -> u32 {
        self.htm.max_overlap
    }

    #[setter]
    fn set_max_overlap(&mut self, max_overlap: u32) {
        self.htm.max_overlap = max_overlap
    }

    #[call]
    fn __call__(&mut self, bitset_input: &OclBitset, learn: Option<bool>) -> PyResult<OclSDR> {
        self.infer(bitset_input, learn)
    }
    #[text_signature = "(bitset_input, learn)"]
    fn infer(&mut self, bitset_input: &OclBitset, learn: Option<bool>) -> Result<OclSDR, PyErr> {
        self.htm.infer(&bitset_input.bits, learn.unwrap_or(false)).map(|sdr| OclSDR { sdr }).map_err(ocl_err_to_py_ex)
    }
    #[text_signature = "(minicolumns_per_column, minicolumn_stride, bitset_input, learn)"]
    fn infer_and_group_into_columns(&mut self, minicolumns_per_column: usize,minicolumn_stride:usize, bitset_input: &OclBitset, learn: Option<bool>) -> PyResult<OclSDR> {
        self.htm.infer_and_group_into_columns(minicolumns_per_column,minicolumn_stride, &bitset_input.bits, learn.unwrap_or(false)).map(|sdr| OclSDR { sdr }).map_err(ocl_err_to_py_ex)
    }

    #[text_signature = "(bitset_input)"]
    fn compute(&mut self, bitset_input: &OclBitset) -> Result<OclSDR, PyErr> {
        self.htm.compute(&bitset_input.bits).map(|sdr| OclSDR { sdr }).map_err(ocl_err_to_py_ex)
    }
    #[text_signature = "(minicolumns_per_column,minicolumn_stride, bitset_input)"]
    fn compute_and_group_into_columns(&mut self, minicolumns_per_column: usize, minicolumn_stride:usize, bitset_input: &OclBitset) -> PyResult<OclSDR> {
        self.htm.compute_and_group_into_columns(minicolumns_per_column, minicolumn_stride,&bitset_input.bits).map(|sdr| OclSDR { sdr }).map_err(ocl_err_to_py_ex)
    }
    #[text_signature = "(active_minicolumns, bitset_input)"]
    fn update_permanence(&mut self, active_minicolumns: &OclSDR, bitset_input: &OclBitset) -> PyResult<()> {
        self.htm.update_permanence(&active_minicolumns.sdr, &bitset_input.bits).map_err(ocl_err_to_py_ex)
    }
}

#[pyproto]
impl PySequenceProtocol for CpuSDR {
    fn __len__(&self) -> usize {
        self.sdr.len()
    }
}

#[pyproto]
impl PyObjectProtocol for CpuSDR {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.sdr))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
    fn __richcmp__(&self, other: &PyAny, op: CompareOp) -> PyResult<bool> {
        let equal = if let Ok(sdr) = other.cast_as::<PyCell<CpuSDR>>() {
            let sdr_ref = sdr.try_borrow()?;
            &self.sdr == &sdr_ref.sdr
        } else if let Ok(vec) = other.extract::<Vec<u32>>() {
            &self.sdr == &vec
        } else {
            return Err(ocl_err_to_py_ex("CpuSDR can only be compared with [int] or another CpuSDR"));
        };

        match op {
            CompareOp::Eq => Ok(equal),
            CompareOp::Ne => Ok(!equal),
            op => Err(ocl_err_to_py_ex("Cannot compare platforms"))
        }
    }
}

#[pyclass]
pub struct CpuSDRIter {
    inner: Py<CpuSDR>,
    idx: usize,
}

#[pyproto]
impl PyIterProtocol for CpuSDRIter {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<u32>> {
        let i = slf.idx;
        slf.idx += 1;
        let r = Py::try_borrow(&slf.inner, slf.py())?;
        Ok(if i >= r.sdr.len() {
            None
        } else {
            Some(r.sdr[i])
        })
    }
}

#[pyproto]
impl PyIterProtocol for CpuSDR {
    fn __iter__(slf: Py<Self>) -> CpuSDRIter {
        CpuSDRIter {
            inner: slf,
            idx: 0,
        }
    }
}

#[pyproto]
impl PyObjectProtocol for OclSDR {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.sdr.get().map_err(ocl_err_to_py_ex)?))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyproto]
impl PyObjectProtocol for CpuBitset {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.bits))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyproto]
impl PyObjectProtocol for CpuInput {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inp.get_sparse()))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}


#[pyproto]
impl PyObjectProtocol for Population {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.pop))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}


#[pyproto]
impl PyObjectProtocol for Neuron {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.n))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyproto]
impl PyObjectProtocol for Segment {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.seg))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyproto]
impl PySequenceProtocol for Population {
    fn __len__(&self) -> usize {
        self.pop.len()
    }
}

#[pyproto]
impl PySequenceProtocol for Neuron {
    fn __len__(&self) -> usize {
        self.n.len()
    }
}

#[pyproto]
impl PySequenceProtocol for Segment {
    fn __len__(&self) -> usize {
        self.seg.len()
    }
}



#[pyproto]
impl PyNumberProtocol for Neuron {
    fn __add__(lhs: PyRef<Self>, rhs: PyRef<Self>) -> Self {
        Self{n:lhs.n.clone()+rhs.n.clone()}
    }
    fn __mul__(lhs: PyRef<Self>, rhs: PyRef<Self>) -> Self {
        Self{n:lhs.n.clone()*rhs.n.clone()}
    }
}
#[pyproto]
impl PyNumberProtocol for Population {
    fn __add__(lhs: PyRef<Self>, rhs: PyRef<Self>) -> Self {
        Self{pop:lhs.pop.clone()+rhs.pop.clone()}
    }
    fn __mul__(lhs: PyRef<Self>, rhs: PyRef<Self>) -> Self {
        Self{pop:lhs.pop.clone()*rhs.pop.clone()}
    }
}
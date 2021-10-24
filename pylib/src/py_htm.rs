use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol, PyIterProtocol, PySequenceProtocol, PyTypeInfo};
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
use numpy::{PyReadonlyArrayDyn, PyArrayDyn, IntoPyArray, PyArray, PY_ARRAY_API, npyffi, Element, ToNpyDims};
use numpy::npyffi::{NPY_ORDER, npy_intp, NPY_ARRAY_WRITEABLE};
use std::os::raw::c_int;
use crate::ocl_err_to_py_ex;
use crate::py_ndalgebra::{DynMat, try_as_dtype};
use crate::py_ocl::Context;
use htm::{Encoder, HomSegment, auto_gen_seed, EncoderTarget, EncoderRange};
use std::time::SystemTime;
use std::ops::Deref;
use chrono::Utc;

#[pyclass]
pub struct CpuSDR {
    sdr: htm::CpuSDR,
}

#[pyclass]
pub struct CpuBitset{
    bits: htm::CpuBitset
}

#[pyclass]
pub struct CpuInput{
    inp: htm::CpuInput
}

#[pyclass]
pub struct CpuHTM {
    htm: htm::CpuHTM,
}


#[pyclass]
pub struct CpuHOM {
    hom: htm::CpuHOM,
}


#[pyclass]
pub struct CpuHTM2 {
    htm: htm::CpuHTM2,
}

#[pyclass]
pub struct CpuHTM4 {
    htm: htm::CpuHTM4,
}

#[pyclass]
pub struct OclSDR {
    sdr: htm::OclSDR,
}

#[pyclass]
pub struct OclBitset{
    bits: htm::OclBitset
}

#[pyclass]
pub struct OclInput{
    inp: htm::OclInput
}

#[pyclass]
pub struct OclHTM {
    htm: htm::OclHTM,
}

#[pyclass]
pub struct OclHTM2 {
    htm: htm::OclHTM2,
}

#[pyclass]
pub struct EncoderBuilder {
    enc: htm::EncoderBuilder,
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
    #[text_signature = "(from_inclusive,to_exclusive,size,cardinality)"]
    pub fn add_circularinteger(&mut self, from: u32, to: u32, size: u32, cardinality: u32) -> CircularIntegerEncoder {
        CircularIntegerEncoder { enc: self.enc.add_circular_integer(from..to, size, cardinality) }
    }
    #[text_signature = "(number_of_categories, cardinality)"]
    pub fn add_categorical(&mut self, number_of_categories: u32, cardinality: u32) -> CategoricalEncoder {
        CategoricalEncoder { enc: self.enc.add_categorical( number_of_categories,cardinality) }
    }
    #[text_signature = "(size)"]
    pub fn add_bits(&mut self, size: u32) -> BitsEncoder {
        BitsEncoder { enc: self.enc.add_bits( size) }
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

fn encode<T>(sdr:PyObject,scalar:T,f1:impl FnOnce(&mut htm::CpuSDR, T),f2:impl FnOnce(&mut htm::CpuBitset, T),f3:impl FnOnce(&mut htm::CpuInput, T))->PyResult<()>{
    let gil = Python::acquire_gil();
    let py = gil.python();
    if let Ok(mut sdr) = sdr.extract::<PyRefMut<CpuBitset>>(py) {
        f2(&mut sdr.bits, scalar)
    }else if let Ok(mut sdr) = sdr.extract::<PyRefMut<CpuInput>>(py) {
        f3(&mut sdr.inp, scalar)
    }else{
        let mut sdr = sdr.extract::<PyRefMut<CpuSDR>>(py)?;
        f1(&mut sdr.sdr, scalar)
    }
    Ok(())
}
#[pymethods]
impl BitsEncoder {
    pub fn clear(&self, sdr: PyObject) -> PyResult<()>{
        encode(sdr,(),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x))
    }
    pub fn encode(&self, sdr: PyObject, val: &PyAny) -> PyResult<()>{
        if let Ok(bools) = val.extract::<Vec<bool>>(){
            self.encode_from_bools(sdr, bools)
        }else{
            let indices = val.extract::<Vec<u32>>()?;
            self.encode_from_indices(sdr, indices)
        }
    }
    pub fn encode_from_indices(&self, sdr: PyObject, indices: Vec<u32>) -> PyResult<()>{
        encode(sdr,indices.as_slice(),
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y))
    }
    pub fn encode_from_bools(&self, sdr: PyObject, bools: Vec<bool>) -> PyResult<()> {
        encode(sdr,bools.as_slice(),
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y))
    }
}
#[pymethods]
impl IntegerEncoder {
    pub fn clear(&self, sdr: PyObject) -> PyResult<()>{
        encode(sdr,(),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x))
    }
    pub fn encode(&self, sdr: PyObject, scalar: u32) -> PyResult<()> {
        encode(sdr,scalar,
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y))
    }
}

#[pymethods]
impl CategoricalEncoder {
    pub fn clear(&self, sdr: PyObject) -> PyResult<()>{
        encode(sdr,(),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x))
    }
    pub fn encode(&self, sdr: PyObject, scalar: u32) -> PyResult<()> {
        encode(sdr,scalar,
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y))
    }
}
#[pymethods]
impl CircularIntegerEncoder {
    pub fn clear(&self, sdr: PyObject) -> PyResult<()>{
        encode(sdr,(),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x))
    }
    pub fn encode(&self, sdr: PyObject, scalar: u32) -> PyResult<()> {
        encode(sdr,scalar,
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y))
    }
}

#[pymethods]
impl FloatEncoder {
    pub fn clear(&self, sdr: PyObject) -> PyResult<()>{
        encode(sdr,(),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x))
    }
    pub fn encode(&self, sdr: PyObject, scalar: f32) -> PyResult<()>{
        encode(sdr,scalar,
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y))
    }
}

#[pymethods]
impl DayOfWeekEncoder {
    pub fn clear(&self, sdr: PyObject) -> PyResult<()>{
        encode(sdr,(),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x))
    }
    pub fn encode(&self, sdr: PyObject, scalar: &PyDateTime) -> PyResult<()>{
        let weekday = scalar.call_method("weekday", (), None).unwrap().extract().unwrap();
        encode(sdr,weekday,
               |x,y|self.enc.encode_day_of_week(x,y),
               |x,y|self.enc.encode_day_of_week(x,y),
               |x,y|self.enc.encode_day_of_week(x,y))
    }
}

#[pymethods]
impl DayOfMonthEncoder {
    pub fn clear(&self, sdr: PyObject) -> PyResult<()>{
        encode(sdr,(),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x))
    }
    pub fn encode(&self, sdr: PyObject, scalar: &PyDateTime) -> PyResult<()>{
        let day = scalar.get_day() as u32 - 1;
        encode(sdr,day,
               |x,y|self.enc.encode_day_of_month(x,y),
               |x,y|self.enc.encode_day_of_month(x,y),
               |x,y|self.enc.encode_day_of_month(x,y))
    }
}

#[pymethods]
impl DayOfYearEncoder {
    pub fn clear(&self, sdr: PyObject) -> PyResult<()>{
        encode(sdr,(),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x))
    }
    pub fn encode(&self, sdr: PyObject, scalar: &PyDateTime) -> PyResult<()>{
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
        encode(sdr,day,
               |x,y|self.enc.encode_day_of_year(x,y),
               |x,y|self.enc.encode_day_of_year(x,y),
               |x,y|self.enc.encode_day_of_year(x,y))
    }
}

#[pymethods]
impl IsWeekendEncoder {
    pub fn clear(&self, sdr: PyObject) -> PyResult<()>{
        encode(sdr,(),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x))
    }
    pub fn encode(&self, sdr: PyObject, scalar: &PyDateTime) -> PyResult<()>{
        let day = scalar.call_method("weekday", (), None).unwrap().extract::<u32>().unwrap();
        let is_weekend = day >= 5;

        encode(sdr,is_weekend,
               |x,y|self.enc.encode_is_weekend(x,y),
               |x,y|self.enc.encode_is_weekend(x,y),
               |x,y|self.enc.encode_is_weekend(x,y))
    }
}

#[pymethods]
impl TimeOfDayEncoder {
    pub fn clear(&self, sdr: PyObject) -> PyResult<()>{
        encode(sdr,(),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x))
    }
    pub fn encode(&self, sdr: PyObject, scalar: &PyDateTime) -> PyResult<()>{
        let sec = (60 * scalar.get_hour() as u32 + scalar.get_minute() as u32) * 60 + scalar.get_second() as u32;
        encode(sdr,sec,
               |x,y|self.enc.encode_time_of_day(x,y),
               |x,y|self.enc.encode_time_of_day(x,y),
               |x,y|self.enc.encode_time_of_day(x,y))
    }
}

#[pymethods]
impl BoolEncoder {
    pub fn clear(&self, sdr: PyObject) -> PyResult<()>{
        encode(sdr,(),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x),
               |x,_|self.enc.clear(x))
    }
    pub fn encode(&self, sdr: PyObject, scalar: bool) -> PyResult<()>{
        encode(sdr,scalar,
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y),
               |x,y|self.enc.encode(x,y))
    }
}


#[pymethods]
impl CpuHOM {
    #[new]
    pub fn new(cells_per_minicolumn:u32, minicolumn_count:u32) -> Self {
        CpuHOM { hom: htm::CpuHOM::new(cells_per_minicolumn,minicolumn_count) }
    }

    
    #[getter]
    fn get_max_synapses_per_segment(&self) -> usize{
        self.hom.hyp.max_synapses_per_segment
    }
    #[setter]
    fn set_max_synapses_per_segment(&mut self, param: usize){
        self.hom.hyp.max_synapses_per_segment = param
    }
    #[getter]
    fn get_max_segments_per_cell(&self) ->  usize{
        self.hom.hyp.max_segments_per_cell
    }
    #[setter]
    fn set_max_segments_per_cell(&mut self, param:  usize){
        self.hom.hyp.max_segments_per_cell = param
    }
    #[getter]
    fn get_max_new_synapse_count(&self) -> usize{
        self.hom.hyp.max_new_synapse_count
    }
    #[setter]
    fn set_max_new_synapse_count(&mut self, param: usize){
        self.hom.hyp.max_new_synapse_count = param
    }
    #[getter]
    fn get_initial_permanence(&self) ->  f32{
        self.hom.hyp.initial_permanence
    }
    #[setter]
    fn set_initial_permanence(&mut self, param:  f32){
        self.hom.hyp.initial_permanence = param
    }
    #[getter]
    fn get_min_permanence_to_keep(&self) ->  f32{
        self.hom.hyp.min_permanence_to_keep
    }
    #[setter]
    fn set_min_permanence_to_keep(&mut self, param:  f32){
        self.hom.hyp.min_permanence_to_keep = param
    }
    #[getter]
    fn get_activation_threshold(&self) ->  u32{
        self.hom.hyp.activation_threshold
    }
    #[setter]
    fn set_activation_threshold(&mut self, param:  u32){
        self.hom.hyp.activation_threshold = param
    }
    #[getter]
    fn get_learning_threshold(&self) ->  u32{
        self.hom.hyp.learning_threshold
    }
    #[setter]
    fn set_learning_threshold(&mut self, param:  u32){
        self.hom.hyp.learning_threshold = param
    }
    #[getter]
    fn get_permanence_threshold(&self) ->  f32{
        self.hom.hyp.permanence_threshold
    }
    #[setter]
    fn set_permanence_threshold(&mut self, param:  f32){
        self.hom.hyp.permanence_threshold = param
    }
    #[getter]
    fn get_predicted_decrement(&self) -> f32{
        self.hom.hyp.predicted_decrement
    }
    #[setter]
    fn set_predicted_decrement(&mut self, param: f32){
        self.hom.hyp.predicted_decrement = param
    }
    #[getter]
    fn get_permanence_decrement_increment(&self) ->  [f32; 2]{
        self.hom.hyp.permanence_decrement_increment
    }
    #[setter]
    fn set_permanence_decrement_increment(&mut self, param:  [f32; 2]){
        self.hom.hyp.permanence_decrement_increment = param
    }

    #[call]
    fn __call__(&mut self, active_minicolumns: &CpuSDR, learn: Option<bool>)->CpuSDR{
        CpuSDR{sdr:self.hom.infer(&active_minicolumns.sdr,learn.unwrap_or(false))}
    }

    #[text_signature = "( /)"]
    fn clone(&self) -> CpuHOM {
        CpuHOM { hom: self.hom.clone() }
    }

}

#[pymethods]
impl CpuHTM {
    #[new]
    pub fn new(input_size: u32, minicolumns: u32, n: u32, inputs_per_minicolumn: u32, rand_seed:Option<u32>) -> PyResult<Self> {
        if inputs_per_minicolumn > minicolumns{
            return Err(PyValueError::new_err(format!("There are {} inputs per minicolumn but only {} minicolumns",inputs_per_minicolumn, minicolumns)))
        }
        Ok(CpuHTM { htm: htm::CpuHTM::new_globally_uniform_prob(input_size, minicolumns, n, inputs_per_minicolumn,rand_seed.unwrap_or_else(auto_gen_seed)) })
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
    fn __call__(&mut self, input: &CpuInput, learn: Option<bool>) -> CpuSDR {
        CpuSDR{sdr:self.htm.infer(&input.inp, learn.unwrap_or(false))}
    }

    #[text_signature = "( /)"]
    fn to_htm2(&self) -> CpuHTM2 {
        CpuHTM2 { htm: htm::CpuHTM2::from(&self.htm) }
    }

    #[text_signature = "( /)"]
    fn clone(&self) -> CpuHTM {
        CpuHTM { htm: self.htm.clone() }
    }
}


#[pymethods]
impl CpuHTM2 {
    /// new(input_size, minicolumns, inputs_per_minicolumn, n, rand_seed)
    /// --
    ///
    /// Randomly generate a new Spacial Pooler. You can provide a random seed manually.
    /// Otherwise the millisecond part of system time is used as a seed. Seed is a 32-bit number.
    ///
    #[new]
    pub fn new(input_size: u32, minicolumns: u32,n: u32, inputs_per_minicolumn: u32, rand_seed:Option<u32>) -> PyResult<Self> {
        if inputs_per_minicolumn > minicolumns{
            return Err(PyValueError::new_err(format!("There are {} inputs per minicolumn but only {} minicolumns",inputs_per_minicolumn, minicolumns)))
        }
        Ok(CpuHTM2 { htm: htm::CpuHTM2::new_globally_uniform_prob(input_size, minicolumns, n, inputs_per_minicolumn, rand_seed.unwrap_or_else(auto_gen_seed)) })
    }
    #[text_signature = "( /)"]
    fn clone(&self) -> CpuHTM2 {
        CpuHTM2 { htm: self.htm.clone() }
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
        CpuSDR{sdr:self.htm.infer2(&bitset_input.bits, learn.unwrap_or(false))}
    }
}

#[pymethods]
impl CpuHTM4 {
    #[text_signature = "( /)"]
    fn clone(&self) -> CpuHTM4 {
        CpuHTM4 { htm: self.htm.clone() }
    }
    #[new]
    pub fn new(input_size: u32, minicolumns: u32, n: u32, inputs_per_minicolumn: u32,  excitatory_connection_probability:f32, rand_seed:Option<u32>) -> PyResult<Self> {
        if inputs_per_minicolumn > minicolumns{
            return Err(PyValueError::new_err(format!("There are {} inputs per minicolumn but only {} minicolumns",inputs_per_minicolumn, minicolumns)))
        }
        Ok(CpuHTM4 { htm: htm::CpuHTM4::new_globally_uniform_prob(input_size, minicolumns, n, inputs_per_minicolumn,excitatory_connection_probability,rand_seed.unwrap_or_else(auto_gen_seed)) })
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
        CpuSDR{sdr:self.htm.infer4(&bitset_input.bits, learn.unwrap_or(false))}
    }
}


#[pymethods]
impl OclSDR {
    #[new]
    pub fn new(context: &mut Context, max_active_neurons: u32) -> PyResult<Self> {
        context.compile_htm_program().and_then(|prog|htm::OclSDR::new(prog.clone(), max_active_neurons).map_err(ocl_err_to_py_ex)).map(|sdr| OclSDR { sdr })
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
    fn push_active_neuron(&mut self, neuron_index: u32) {
        self.sdr.push(neuron_index)
    }
    #[text_signature = "(number_of_bits_to_retain)"]
    fn shrink(&mut self, number_of_bits_to_retain:usize){
        self.sdr.shrink(number_of_bits_to_retain)
    }
    #[text_signature = "(number_of_bits_to_retain)"]
    fn subsample(&mut self, number_of_bits_to_retain:usize){
        self.sdr.shrink(number_of_bits_to_retain)
    }
    #[text_signature = "(other_sdr)"]
    pub fn extend(&mut self, other:&CpuSDR){
        self.sdr.extend(&other.sdr)
    }
    #[text_signature = "(other_sdr)"]
    pub fn union(&self, other:&CpuSDR)->CpuSDR{
        CpuSDR{sdr:self.sdr.union(&other.sdr)}
    }
    #[text_signature = "()"]
    pub fn normalize(&mut self){
        self.sdr.normalize()
    }
    #[text_signature = "(other_sdr)"]
    pub fn overlap(&self, other:&CpuSDR) -> u32 {
        self.sdr.overlap(&other.sdr)
    }
    #[setter]
    fn set_active_neurons(&mut self, neuron_indices: Vec<u32>) {
        self.sdr.set(neuron_indices.as_slice())
    }

    #[getter]
    fn get_active_neurons(&self) -> Vec<u32> {
        self.sdr.clone().to_vec()
    }
    #[text_signature = "(sdr)"]
    pub fn to_bitset(&self, input_size:u32)->CpuBitset{
        CpuBitset{bits:htm::CpuBitset::from_sdr(&self.sdr, input_size)}
    }
}
#[pyfunction]
#[text_signature = "(bits)"]
pub fn bitset_from_bools(bits: Vec<bool>) -> CpuBitset{
    CpuBitset{bits:htm::CpuBitset::from_bools(&bits)}
}
#[pyfunction]
#[text_signature = "(bit_indices)"]
pub fn bitset_from_indices(bit_indices: Vec<u32>, input_size:u32) -> CpuBitset{
    CpuBitset{bits:htm::CpuBitset::from_sdr(&bit_indices,input_size)}
}
#[pymethods]
impl CpuBitset{
    #[new]
    pub fn new(bit_count:u32)->Self{
        CpuBitset{bits:htm::CpuBitset::new(bit_count)}
    }
    #[getter]
    pub fn size(&self)->u32{
        self.bits.size()
    }
    #[text_signature = "(bit_index)"]
    pub fn is_bit_on(&self, bit_index:u32)->bool{
        self.bits.is_bit_on(bit_index)
    }
    #[text_signature = "(sdr)"]
    pub fn append_to_sdr(&self, sdr:&mut CpuSDR){
        self.bits.append_to_sdr(&mut sdr.sdr)
    }
    #[getter]
    pub fn cardinality(&self)->u32{
        self.bits.cardinality()
    }
    #[text_signature = "(sdr)"]
    pub fn clear(&mut self, sdr:Option<&CpuSDR>){
        if let Some(sdr) = sdr{
            self.bits.clear(&sdr.sdr)
        }else{
            self.bits.clear_all()
        }
    }
    #[text_signature = "(min_size)"]
    pub fn ensure_size(&mut self, min_size:u32){
        self.bits.ensure_size(min_size)
    }
    #[text_signature = "(size)"]
    pub fn set_size(&mut self, size:u32){
        self.bits.set_size(size)
    }
    #[text_signature = "(bit_idx)"]
    pub fn set_bit_on(&mut self, bit_idx:u32){
        self.bits.set_bit_on(bit_idx)
    }
    #[text_signature = "(sdr)"]
    pub fn set_bit_off(&mut self, bit_idx:u32){
        self.bits.set_bit_off(bit_idx)
    }
    #[text_signature = "(sdr)"]
    pub fn set_bits_on(&mut self, sdr:&CpuSDR){
        self.bits.set_bits_on(&sdr.sdr)
    }
    #[text_signature = "(sdr)"]
    pub fn set_bits_off(&mut self, sdr:&CpuSDR){
        self.bits.set_bits_off(&sdr.sdr)
    }
    #[text_signature = "(sdr)"]
    pub fn to_sdr(&self)->CpuSDR{
        CpuSDR{sdr:htm::CpuSDR::from(&self.bits)}
    }
}

#[pymethods]
impl CpuInput{
    #[new]
    pub fn new(size:u32)->Self{
        CpuInput{inp:htm::CpuInput::new(size)}
    }
    #[getter]
    pub fn size(&self)->u32{
        self.inp.size()
    }
    #[getter]
    pub fn cardinality(&self)->u32{
        self.inp.cardinality()
    }
    #[text_signature = "(size)"]
    pub fn set_size(&mut self, size:u32){
        self.inp.set_size(size)
    }

}


#[pymethods]
impl OclHTM {
    #[new]
    pub fn new(context: &mut Context, htm: &CpuHTM) -> PyResult<Self> {
        htm::OclHTM::new(&htm.htm, context.compile_htm_program()?.clone()).map(|htm| OclHTM { htm }).map_err(ocl_err_to_py_ex)
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
    fn __call__(&mut self, input: &OclInput, learn: Option<bool>) -> PyResult<OclSDR> {
        self.htm.infer(&input.inp, learn.unwrap_or(false)).map(|sdr| OclSDR { sdr }).map_err(ocl_err_to_py_ex)
    }
}

#[pymethods]
impl OclHTM2 {
    #[new]
    pub fn new(context: &mut Context, htm: &CpuHTM2) -> PyResult<Self> {
        htm::OclHTM2::new(&htm.htm, context.compile_htm_program()?.clone()).map(|htm| OclHTM2 { htm }).map_err(ocl_err_to_py_ex)
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
        self.htm.infer2(&bitset_input.bits, learn.unwrap_or(false)).map(|sdr| OclSDR { sdr }).map_err(ocl_err_to_py_ex)
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
        let equal = if let Ok(sdr) = other.cast_as::<PyCell<CpuSDR>>(){
            let sdr_ref = sdr.try_borrow()?;
            &self.sdr == &sdr_ref.sdr
        } else if let Ok(vec) = other.extract::<Vec<u32>>(){
            &self.sdr == &vec
        } else {
            return Err(ocl_err_to_py_ex("CpuSDR can only be compared with [int] or another CpuSDR"));
        };

        match op{
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

#[pyclass]
pub struct CpuHOMIter {
    inner: Py<CpuHOM>,
    idx: usize,
    minicolumn_count:usize,
}

#[pyproto]
impl PyIterProtocol for CpuHOMIter {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<CpuHOMMinicolumn>> {
        let i = slf.idx;
        slf.idx += 1;
        Ok(if i >= slf.minicolumn_count {
            None
        } else {
            let inner : Py<CpuHOM> = slf.inner.clone();
            Some(CpuHOMMinicolumn{ inner, minicolumn_idx: i })
        })
    }
}

#[pyclass]
pub struct CpuHOMMinicolumn {
    inner: Py<CpuHOM>,
    minicolumn_idx: usize,
}

#[pyclass]
pub struct CpuHOMMinicolumnIter {
    inner: Py<CpuHOM>,
    minicolumn_idx: usize,
    segment_idx: usize,
}

#[pyproto]
impl PyIterProtocol for CpuHOMMinicolumn {
    fn __iter__(slf: PyRef<Self>) -> CpuHOMMinicolumnIter {
        let minicolumn_idx = slf.minicolumn_idx;
        CpuHOMMinicolumnIter {
            inner: slf.inner.clone(),
            minicolumn_idx,
            segment_idx: 0
        }
    }
}



#[pyproto]
impl PyIterProtocol for CpuHOM {
    fn __iter__(slf: Py<Self>) -> PyResult<CpuHOMIter> {
        let minicolumn_count = {
            let gil = Python::acquire_gil();
            let py = gil.python();
            let r:PyRef<CpuHOM> = Py::try_borrow(&slf, py)?;
            r.hom.minicolumns.len()
        };
        Ok(CpuHOMIter {
            inner: slf,
            idx: 0,
            minicolumn_count
        })
    }
}

#[pyclass]
pub struct CpuHOMSegment {
    inner: Py<CpuHOM>,
    minicolumn_idx: usize,
    segment_idx: usize,
}
impl CpuHOMSegment {
    fn segment<X>(&self, f:impl FnOnce(&HomSegment)->X)->PyResult<X>{
        let gil = Python::acquire_gil();
        let py = gil.python();
        let r:PyRef<CpuHOM> = Py::try_borrow(&self.inner, py)?;
        if self.minicolumn_idx < r.hom.minicolumns.len(){
            let minicolumn = &r.hom.minicolumns[self.minicolumn_idx];
            if self.segment_idx < minicolumn.segments.len(){
                Ok(f(&minicolumn.segments[self.segment_idx]))
            }else{
                Err(PyValueError::new_err("Segment has been removed!"))
            }
        }else{
            Err(PyValueError::new_err("Minicolumn has been removed!"))
        }
    }
}
#[pymethods]
impl CpuHOMSegment {
    #[getter]
    fn get_synapses(&self)->PyResult<Vec<(u32,f32)>>{
        self.segment(|segment|segment.synapses.iter().map(|s|(s.cell_id,s.permanence)).collect())
    }
    #[getter]
    fn get_postsynaptic_cell_idx(&self)->PyResult<u8>{
        self.segment(|s|s.cell_idx)
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

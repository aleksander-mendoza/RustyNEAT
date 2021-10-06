use ocl::{ProQue,Error, SpatialDims, flags, Platform, Device, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use ndalgebra::buffer::Buffer;
use crate::htm_program::HtmProgram;

#[derive(Clone)]
pub struct CpuSDR(Vec<u32>);

impl From<Vec<u32>> for CpuSDR{
    fn from(v: Vec<u32>) -> Self {
        Self(v)
    }
}
impl CpuSDR {
    pub fn to_vec(self)->Vec<u32>{
        let Self(v) = self;
        v
    }
    pub fn set(&mut self, active_neurons:&[u32]){
        unsafe{self.0.set_len(0)}
        self.0.extend_from_slice(active_neurons)
    }
    pub fn as_slice(&self)->&[u32]{
        self.0.as_slice()
    }
    pub fn number_of_active_neurons(&self)->usize{
        self.0.len()
    }
    /**SDR ust be sparse. Hence we might as well put a cap on the maximum number of active neurons.
    If your system is designed correctly, then you should never have to worry about exceeding this limit.*/
    pub fn max_active_neurons(&self)->usize{
        self.0.capacity()
    }
    pub fn new(max_active_neurons:usize) -> Self{
        Self(Vec::with_capacity(max_active_neurons))
    }
    pub fn as_mut_slice(&mut self) -> &mut [u32]{
        self.0.as_mut_slice()
    }
    pub fn iter(&self)->std::slice::Iter<u32>{
        self.0.iter()
    }
}


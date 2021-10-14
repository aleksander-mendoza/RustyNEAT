use ocl::{ProQue,Error, SpatialDims, flags, Platform, Device, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use ndalgebra::buffer::Buffer;
use crate::htm_program::HtmProgram;
use ndalgebra::context::Context;
use crate::CpuSDR;

#[derive(Clone)]
pub struct OclSDR {
    context:Context,
    buffer:Buffer<u32>,
    number_of_active_neurons:usize,
}

impl OclSDR {
    pub fn buffer(&self)->&Buffer<u32>{
        &self.buffer
    }
    pub fn queue(&self)->&Queue{
        &self.context.queue()
    }
    pub fn read(&self, offset:usize, dst:&mut [u32]) -> Result<(), Error> {
        self.buffer.read(self.queue(), offset, dst).map_err(Error::from)
    }
    pub fn get(&self)->Result<Vec<u32>,Error>{
        let mut v = Vec::with_capacity(self.number_of_active_neurons);
        self.buffer.read(self.queue(), 0, v.as_mut_slice())?;
        Ok(v)
    }
    pub fn to_cpu(&self)->Result<CpuSDR,Error>{
        self.get().map(CpuSDR::from)
    }
    pub fn number_of_active_neurons(&self)->usize{
        self.number_of_active_neurons
    }
    /**SDR ust be sparse. Hence we might as well put a cap on the maximum number of active neurons.
    If your system is designed correctly, then you should never have to worry about exceeding this limit.*/
    pub fn max_active_neurons(&self)->usize{
        self.buffer.len()
    }
    pub fn from_sdr(context:Context,sdr:&CpuSDR, max_active_neurons:usize) -> Result<Self,Error>{
        Self::from_slice(context,sdr,max_active_neurons)
    }
    pub fn from_slice(context:Context,sdr:&[u32],max_active_neurons:usize) -> Result<Self,Error>{
        let mut ocl_sdr = Self::new(context,max_active_neurons)?;
        ocl_sdr.set(sdr)?;
        Ok(ocl_sdr)
    }
    pub fn from_buff(context:Context,buffer:Buffer<u32>, number_of_active_neurons:usize) -> Self{
        Self{
            context,
            buffer,
            number_of_active_neurons
        }
    }
    pub fn new(context:Context,max_active_neurons:usize) -> Result<Self,Error>{
        let buffer = unsafe{Buffer::empty(context.context(),flags::MEM_READ_WRITE,max_active_neurons)}?;
        Ok(Self{buffer,context,number_of_active_neurons:0})
    }
    pub fn set(&mut self, neuron_indices:&[u32]) -> Result<(), Error> {
        self.number_of_active_neurons = neuron_indices.len();
        self.buffer.write(self.queue(), 0, neuron_indices)
    }
    pub fn iter(&self)->Result<std::vec::IntoIter<u32>,Error>{
        let mut val = Vec::with_capacity(self.number_of_active_neurons);
        unsafe{val.set_len(self.number_of_active_neurons)}
        self.buffer.read(self.queue(), 0, &mut val)?;
        Ok(val.into_iter())
    }
}


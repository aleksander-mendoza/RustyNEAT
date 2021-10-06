use ocl::{ProQue,Error, SpatialDims, flags, Platform, Device, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use ndalgebra::buffer::Buffer;
use crate::htm_program::HtmProgram;

#[derive(Clone)]
pub struct OclSDR {
    program:HtmProgram,
    buffer:Buffer<u32>,
    number_of_active_neurons:usize,
}

impl OclSDR {
    pub fn buffer(&self)->&Buffer<u32>{
        &self.buffer
    }
    pub fn read(&self, offset:usize, dst:&mut [u32]) -> Result<(), Error> {
        self.buffer.read(self.program.queue(), offset, dst).map_err(Error::from)
    }
    pub fn get(&self)->Result<Vec<u32>,Error>{
        let mut v = Vec::with_capacity(self.number_of_active_neurons);
        self.buffer.read(self.program.queue(), 0, v.as_mut_slice())?;
        Ok(v)
    }
    pub fn number_of_active_neurons(&self)->usize{
        self.number_of_active_neurons
    }
    /**SDR ust be sparse. Hence we might as well put a cap on the maximum number of active neurons.
    If your system is designed correctly, then you should never have to worry about exceeding this limit.*/
    pub fn max_active_neurons(&self)->usize{
        self.buffer.len()
    }
    pub fn from_buff(program:HtmProgram,buffer:Buffer<u32>, number_of_active_neurons:usize) -> Self{
        Self{
            program,
            buffer,
            number_of_active_neurons
        }
    }
    pub fn new(program:HtmProgram,max_active_neurons:usize) -> Result<Self,Error>{
        let buffer = unsafe{program.buffer_empty(flags::MEM_READ_WRITE,max_active_neurons)}?;
        Ok(Self{buffer,program,number_of_active_neurons:0})
    }
    pub fn set(&mut self, neuron_indices:&[u32]) -> Result<(), Error> {
        self.number_of_active_neurons = neuron_indices.len();
        self.buffer.write(self.program.queue(), 0, neuron_indices)
    }
    pub fn iter(&self)->Result<std::vec::IntoIter<u32>,Error>{
        let mut val = Vec::with_capacity(self.number_of_active_neurons);
        unsafe{val.set_len(self.number_of_active_neurons)}
        self.buffer.read(self.program.queue(), 0, &mut val)?;
        Ok(val.into_iter())
    }
}


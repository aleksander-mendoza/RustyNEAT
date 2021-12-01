use ocl::{ProQue,Error, SpatialDims, flags, Platform, Device, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use ndalgebra::buffer::Buffer;
use crate::htm_program::HtmProgram;
use ndalgebra::context::Context;
use crate::{CpuSDR, OclBitset};

#[derive(Clone)]
pub struct OclSDR {
    prog:HtmProgram,
    buffer:Buffer<u32>,
    cardinality:u32,
}

impl OclSDR {
    pub fn prog(&self)->&HtmProgram{
        &self.prog
    }
    pub fn buffer(&self)->&Buffer<u32>{
        &self.buffer
    }
    pub fn queue(&self)->&Queue{
        &self.prog.queue()
    }
    pub fn read(&self, offset:usize, dst:&mut [u32]) -> Result<(), Error> {
        self.buffer.read(self.queue(), offset, dst).map_err(Error::from)
    }
    pub fn get(&self)->Result<Vec<u32>,Error>{
        let mut v = Vec::with_capacity(self.cardinality as usize);
        unsafe{v.set_len(self.cardinality as usize)}
        self.buffer.read(self.queue(), 0, v.as_mut_slice())?;
        Ok(v)
    }
    pub fn to_cpu(&self)->Result<CpuSDR,Error>{
        self.get().map(CpuSDR::from)
    }
    /**number of active neurons*/
    pub fn cardinality(&self) ->u32{
        self.cardinality
    }
    /**SDR ust be sparse. Hence we might as well put a cap on the maximum number of active neurons.
    If your system is designed correctly, then you should never have to worry about exceeding this limit.*/
    pub fn max_active_neurons(&self)->usize{
        self.buffer.len()
    }
    pub fn from_cpu(prog:HtmProgram, sdr:&CpuSDR, max_cardinality:u32) -> Result<Self,Error>{
        Self::from_slice(prog, &sdr[0..sdr.cardinality() as usize], max_cardinality)
    }
    pub fn in_place_from_bitset(&mut self, bits:&OclBitset) -> Result<(),Error>{
        let Self{ prog, buffer, cardinality } = self;
        let cardinality_buffer = prog.buffer_filled(MemFlags::WRITE_ONLY, 1, 0u32)?;
        prog.kernel_builder("bitset_to_sdr")?.
            add_buff(&cardinality_buffer)?. //__global uint * sdr_cardinality
            add_buff(buffer)?. // __global uint * sdr_input
            add_buff(bits.buffer())?. //__global uint * bitset_input
            enq(prog.queue(),&[bits.size(),1,1]).
            map_err(Error::from)?;
        cardinality_buffer.read(prog.queue(),0,std::slice::from_mut(cardinality))?;
        Ok(())
    }
    pub fn from_slice(prog:HtmProgram,sdr:&[u32],max_cardinality:u32) -> Result<Self,Error>{
        let mut ocl_sdr = Self::new(prog,max_cardinality)?;
        ocl_sdr.set(sdr)?;
        Ok(ocl_sdr)
    }
    pub fn from_buff(prog:HtmProgram,buffer:Buffer<u32>, cardinality:u32) -> Self{
        Self{
            prog,
            buffer,
            cardinality
        }
    }
    pub fn new(prog:HtmProgram,max_cardinality:u32) -> Result<Self,Error>{
        let buffer = unsafe{Buffer::empty(prog.context(),flags::MEM_READ_WRITE,max_cardinality as usize)}?;
        Ok(Self{buffer,prog, cardinality:0})
    }
    pub fn set(&mut self, neuron_indices:&[u32]) -> Result<(), Error> {
        self.cardinality = neuron_indices.len() as u32;
        self.buffer.write(self.queue(), 0, neuron_indices)
    }
    pub fn iter(&self)->Result<std::vec::IntoIter<u32>,Error>{
        let mut val = Vec::with_capacity(self.cardinality as usize);
        unsafe{val.set_len(self.cardinality as usize)}
        self.buffer.read(self.queue(), 0, &mut val)?;
        Ok(val.into_iter())
    }
    pub fn clear(&mut self){
        self.cardinality = 0;
    }
}


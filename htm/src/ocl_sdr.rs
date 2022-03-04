use ocl::{ProQue,Error, SpatialDims, flags, Platform, Device, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use ndalgebra::buffer::Buffer;
use crate::ecc_program::EccProgram;
use ndalgebra::context::Context;
use crate::{CpuSDR, OclBitset, as_usize};
use crate::sdr::SDR;
use crate::as_usize::AsUsize;

#[derive(Clone)]
pub struct OclSDR {
    prog: EccProgram,
    buffer:Buffer<u32>,
    cardinality:u32,
}

impl SDR for OclSDR{
    fn clear(&mut self) {
        self.cardinality = 0;
    }

    fn item(&self) -> u32 {
        assert_eq!(self.cardinality,1,"SDR is not a singleton");
        self.read_at(0).unwrap()
    }

    /**number of active neurons*/
    fn cardinality(&self) ->u32{
        self.cardinality
    }

    fn set_from_slice(&mut self, other: &[u32]) {
        OclSDR::set(self,other).unwrap()
    }

    fn set_from_sdr(&mut self, other: &Self) {
        assert!(other.cardinality as usize<=self.max_active_neurons());
        self.cardinality = other.cardinality;
        self.buffer.copy_with_offset_from(other.prog.queue(),other.buffer(),0,0,other.cardinality.as_usize());
    }

    fn to_vec(&self) -> Vec<u32> {
        self.read_all().unwrap()
    }

    fn into_vec(self) -> Vec<u32> {
        self.to_vec()
    }
}

impl OclSDR {
    pub fn prog(&self)->&EccProgram {
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
    pub fn read_at(&self, offset:usize) -> Result<u32, Error> {
        let mut o=0;
        self.buffer.read(self.queue(), offset, std::slice::from_mut(&mut o)).map_err(Error::from)?;
        Ok(o)
    }
    pub fn read_all(&self) ->Result<Vec<u32>,Error>{
        let mut v = Vec::with_capacity(self.cardinality as usize);
        unsafe{v.set_len(self.cardinality as usize)}
        self.buffer.read(self.queue(), 0, v.as_mut_slice())?;
        Ok(v)
    }
    pub fn to_cpu(&self)->Result<CpuSDR,Error>{
        self.read_all().map(CpuSDR::from)
    }
    /**number of active neurons*/
    pub unsafe fn set_cardinality(&mut self,cardinality:u32){
        assert!(cardinality as usize<=self.buffer().len(),"{}<={}",cardinality,self.buffer().len());
        self.cardinality=cardinality
    }

    /**SDR ust be sparse. Hence we might as well put a cap on the maximum number of active neurons.
    If your system is designed correctly, then you should never have to worry about exceeding this limit.*/
    pub fn max_active_neurons(&self)->usize{
        self.buffer.len()
    }
    pub fn from_cpu(prog: EccProgram, sdr:&CpuSDR, max_cardinality:u32) -> Result<Self,Error>{
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
    pub fn from_slice(prog: EccProgram, sdr:&[u32], max_cardinality:u32) -> Result<Self,Error>{
        let mut ocl_sdr = Self::new(prog,max_cardinality)?;
        ocl_sdr.set(sdr)?;
        Ok(ocl_sdr)
    }
    pub fn from_buff(prog: EccProgram, buffer:Buffer<u32>, cardinality:u32) -> Self{
        Self{
            prog,
            buffer,
            cardinality
        }
    }
    pub fn new(prog: EccProgram, max_cardinality:u32) -> Result<Self,Error>{
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
}


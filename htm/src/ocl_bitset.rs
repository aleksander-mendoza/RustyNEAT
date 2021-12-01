use crate::{CpuSDR, EncoderTarget, HtmProgram, OclSDR};
use crate::cpu_bitset::{CpuBitset, bit_count_to_vec_size};
use ndalgebra::buffer::Buffer;
use ocl::{MemFlags, Error, Queue};

pub struct OclBitset {
    prog: HtmProgram,
    bits: Buffer<u32>,
}

impl OclBitset {
    pub fn prog(&self)->&HtmProgram{
        &self.prog
    }
    pub fn buffer(&self)->&Buffer<u32>{
        &self.bits
    }
    pub fn queue(&self)->&Queue{
        &self.prog.queue()
    }
    pub fn size(&self) ->usize{
        self.bits.len()*32
    }
    pub fn to_cpu(&self)->Result<CpuBitset,Error>{
        let mut v = Vec::with_capacity(self.bits.len());
        unsafe{v.set_len(self.bits.len())}
        self.bits.read(self.queue(), 0, v.as_mut_slice())?;
        Ok(CpuBitset::from(v))
    }
    pub fn from_sdr(sdr: &OclSDR, input_size:u32) -> Result<Self,Error> {
        let mut bitset = Self::new(input_size, sdr.prog().clone())?;
        bitset.set_bits_on(sdr)?;
        Ok(bitset)
    }
    pub fn from_cpu(bitset: &CpuBitset, prog: HtmProgram) -> Result<OclBitset, Error> {
        prog.buffer_from_slice(MemFlags::READ_WRITE,bitset.as_slice())
            .map(|bits|Self{ prog, bits })
    }
    pub fn new(input_size: u32, prog: HtmProgram) -> Result<OclBitset, Error> {
        prog.buffer_filled(MemFlags::READ_WRITE,bit_count_to_vec_size(input_size), 0).map(|bits|Self { prog, bits })
    }

    pub fn set_bits_on(&mut self, sdr: &OclSDR) -> Result<(), Error> {
        let Self{ prog, bits } = self;
        prog.kernel_builder("bitset_set_active_inputs")?.
            add_buff(sdr.buffer())?.//__global uint * sdr_input,
            add_buff(bits)?.// __global uint * bitset_input
            enq(self.prog.queue(),&[sdr.cardinality() as usize,1,1]).
            map_err(Error::from)
    }

    pub fn clear(&mut self, sdr: &OclSDR) -> Result<(), Error> {
        let Self{ prog, bits } = self;
        prog.kernel_builder("bitset_clean_up_active_inputs")?.
            add_buff(sdr.buffer())?.//__global uint * sdr_input,
            add_buff(bits)?.// __global uint * bitset_input
            enq(self.prog.queue(),&[sdr.cardinality() as usize,1,1]).
            map_err(Error::from)
    }
    pub fn clear_all(&mut self) -> Result<(), Error> {
        let Self{ prog, bits } = self;
        bits.fill(prog.queue(),0)
    }
    pub fn copy_from(&mut self, bitset:&CpuBitset) -> Result<(), Error> {
        assert_eq!(bitset.size(),self.size() as u32);
        let Self{ prog, bits } = self;
        bits.write(prog.queue(),0,bitset.as_slice())
    }

}
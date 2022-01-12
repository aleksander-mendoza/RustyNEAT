use crate::{CpuSDR, CpuBitset, EncoderTarget, OclSDR, OclBitset, EccProgram, CpuInput};
use ocl::Error;
use crate::sdr::SDR;

pub struct OclInput{
    sdr:OclSDR,
    bitset:OclBitset,
}

impl OclInput{
    pub fn from_cpu(input:&CpuInput, prog: EccProgram, max_cardinality:u32) -> Result<Self,Error>{
        let bitset = OclBitset::from_cpu(input.get_dense(),prog.clone())?;
        let sdr = OclSDR::from_cpu(prog,input.get_sparse(),max_cardinality)?;
        Ok(Self{bitset,sdr})
    }
    pub fn from_sparse_cpu(sdr:&CpuSDR, prog: EccProgram, max_cardinality:u32, input_size:u32) -> Result<Self,Error>{
        OclSDR::from_cpu(prog, sdr, max_cardinality).and_then(|sdr| Self::from_sparse(sdr, input_size))
    }
    pub fn from_sparse(sdr:OclSDR, input_size:u32) -> Result<Self,Error>{
        let bitset = OclBitset::from_sdr(&sdr,input_size)?;
        Ok(Self{sdr,bitset})
    }
    pub fn from_dense_cpu(bitset:&CpuBitset, prog: EccProgram, max_cardinality:u32) -> Result<Self,Error>{
        OclBitset::from_cpu(bitset,prog).and_then(|bitset| Self::from_dense(bitset,max_cardinality))
    }
    pub fn from_dense(bitset:OclBitset, max_cardinality:u32) -> Result<Self,Error>{
        let mut sdr = OclSDR::new(bitset.prog().clone(),max_cardinality)?;
        sdr.in_place_from_bitset(&bitset)?;
        Ok(Self{sdr,bitset})
    }
    pub fn new(prog: EccProgram, input_size:u32, max_cardinality:u32) ->Result<Self,Error>{
        Ok(Self{ sdr: OclSDR::new(prog.clone(),max_cardinality)?, bitset: OclBitset::new(input_size, prog)? })
    }
    pub fn get_sparse(&self)->&OclSDR{
        &self.sdr
    }
    pub fn get_dense(&self)->&OclBitset{
        &self.bitset
    }

    /**returns previous SDR*/
    pub fn set_sparse(&mut self, sdr:OclSDR)->OclSDR{
        self.bitset.clear(&self.sdr);
        self.bitset.set_bits_on(&sdr);
        let mut sdr = sdr;
        std::mem::swap(&mut self.sdr,&mut sdr);
        sdr
    }
    pub fn set_sparse_from_slice(&mut self, neuron_indices:&[u32])->Result<(),Error>{
        let Self{ sdr, bitset } = self;
        bitset.clear(&sdr)?;
        sdr.set(neuron_indices)?;
        bitset.set_bits_on(sdr)
    }
    pub fn set_dense(&mut self, bitset:OclBitset)->OclBitset{
        self.sdr.in_place_from_bitset(&bitset);
        let mut bitset = bitset;
        std::mem::swap(&mut self.bitset,&mut bitset);
        bitset
    }
    pub fn to_cpu(&self) -> Result<CpuInput, Error> {
        let Self{ sdr, bitset } = self;
        let sdr = sdr.to_cpu()?;
        let bitset = bitset.to_cpu()?;
        Ok(unsafe{CpuInput::new_unchecked(sdr,bitset)})
    }
    pub fn cardinality(&self)->u32{
        self.sdr.cardinality()
    }
    pub fn size(&mut self)->usize{
        self.bitset.size() * 32
    }
    pub fn clear(&mut self){
        let Self{ sdr, bitset } = self;
        bitset.clear(sdr);
        sdr.clear();
    }
}
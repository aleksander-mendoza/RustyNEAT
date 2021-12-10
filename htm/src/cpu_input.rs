use crate::{CpuSDR, CpuBitset, EncoderTarget};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct CpuInput{
    sdr:CpuSDR,
    bitset:CpuBitset,
}
impl From<CpuBitset> for CpuInput{
    fn from(bitset:CpuBitset) -> Self{
        let sdr = CpuSDR::from(&bitset);
        Self{sdr,bitset}
    }
}
impl EncoderTarget for CpuInput{
    fn push(&mut self, neuron_index: u32) {
        self.sdr.push(neuron_index);
        self.bitset.push(neuron_index);
    }

    fn clear_range(&mut self, from: u32, to: u32) {
        self.sdr.clear_range(from,to);
        self.bitset.clear_range(from,to);
    }

    fn contains(&self, neuron_index: u32) -> bool {
        self.bitset.contains(neuron_index)
    }
}
impl CpuInput{
    pub fn new( size:u32)->Self{
        Self::new2d(1,size)
    }
    pub fn new2d( height:u32,width:u32)->Self{
        Self::new3d(1,height,width)
    }
    pub fn new3d(depth:u32, height:u32,width:u32)->Self{
        Self{sdr:CpuSDR::new(),bitset:CpuBitset::new3d(depth,height,width)}
    }
    pub unsafe fn new_unchecked(sdr:CpuSDR,bitset:CpuBitset)->Self{
        Self{sdr,bitset}
    }
    pub fn from_dense_bools(bools:&[bool]) -> Self{
        Self::from_dense(CpuBitset::from_bools(bools))
    }
    pub fn from_dense(bitset:CpuBitset) -> Self{
        let mut sdr = CpuSDR::new();
        bitset.append_to_sdr(&mut sdr);
        Self{sdr,bitset}
    }
    pub fn from_sparse_slice(sdr:&[u32],input_size:u32) -> Self{
        Self::from_sparse(CpuSDR::from(sdr),input_size)
    }
    pub fn from_sparse(sdr:CpuSDR, input_size:u32) -> Self{
        let bitset = CpuBitset::from_sdr(&sdr,input_size);
        Self{sdr,bitset}
    }
    pub fn get_sparse(&self)->&CpuSDR{
        &self.sdr
    }
    pub fn get_dense(&self)->&CpuBitset{
        &self.bitset
    }
    /**returns previous SDR*/
    pub fn set_sparse(&mut self, sdr:CpuSDR)->CpuSDR{
        self.bitset.clear(&self.sdr);
        self.bitset.set_bits_on(&sdr);
        let mut sdr = sdr;
        std::mem::swap(&mut self.sdr,&mut sdr);
        sdr
    }
    /**returns previous Bitset*/
    pub fn set_dense(&mut self, bitset:CpuBitset)->CpuBitset{
        self.sdr.clear();
        bitset.append_to_sdr(&mut self.sdr);
        let mut bitset = bitset;
        std::mem::swap(&mut self.bitset,&mut bitset);
        bitset
    }
    pub fn set_sparse_from_slice(&mut self, sdr:&[u32]){
        self.bitset.clear(&self.sdr);
        self.bitset.set_bits_on(sdr);
        self.sdr.clear();
        self.sdr.extend_from_slice(sdr);
    }
    pub fn normalize(&mut self){
        self.sdr.normalize()
    }
    pub fn cardinality(&self)->u32{
        self.sdr.cardinality()
    }
    pub fn size(&self)->u32{
        self.bitset.size()
    }
    pub fn shape(&self) -> &[u32; 3] {
        self.bitset.shape()
    }
    pub fn reshape2d(&mut self,height:u32,width:u32) {
        self.bitset.reshape2d(height,width)
    }
    pub fn reshape(&mut self) {
        self.bitset.reshape()
    }
    pub fn reshape3d(&mut self,depth:u32, height:u32,width:u32) {
        self.bitset.reshape3d(depth,height,width)
    }
    pub fn clear(&mut self){
        let Self{ sdr, bitset } = self;
        bitset.clear(sdr);
        sdr.clear();
    }
}
use crate::{CpuSDR, EncoderTarget};
use std::fmt::{Debug, Formatter};

#[derive( Eq, PartialEq)]
pub struct CpuBitset {
    bits: Vec<u32>,
}
impl Debug for CpuBitset{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f,"[")?;
        let mut i = self.bits.iter();
        if let Some(&first) = i.next() {
            write!(f,"{:032b}", first)?;
            for &u in i {
                write!(f," {:032b}", u)?;
            }
        }
        write!(f,"]")
    }
}
impl From<Vec<u32>> for CpuBitset {
    fn from(bits: Vec<u32>) -> Self {
        Self{bits}
    }
}
pub fn bit_count_to_vec_size(bit_count:u32)->usize{
    (bit_count as usize + 31) / 32
}
impl EncoderTarget for CpuBitset{
    fn push(&mut self, neuron_index: u32) {
        self.set_bit_on(neuron_index)
    }

    fn clear_range(&mut self, from: u32, to: u32) {
        let from_u32 = (from/32) as usize;
        let to_u32 = (to/32) as usize;
        let all_ones = u32::MAX;
        if from_u32==to_u32{
            self.bits[from_u32] &= !(all_ones >> (from&31)) | (all_ones >> (to&31));
        }else{
            self.bits[from_u32+1..to_u32].fill(0);
            self.bits[from_u32] &= !(all_ones >> (from&31));
            if to_u32<self.bits.len(){
                self.bits[to_u32] &= (all_ones >> (to&31));
            }
        }

    }
}
impl CpuBitset {
    pub fn from_bools(bools:&[bool])->Self{
        let mut bitset = Self::new(bools.len() as u32);
        bools.iter().cloned().enumerate().filter(|(i,b)|*b).map(|(i,_)|i as u32).for_each(|i|bitset.set_bit_on(i));
        bitset
    }
    pub fn from_sdr(sdr: &[u32], input_size:u32) -> Self {
        let mut bitset = Self::new(input_size);
        bitset.set_bits_on(sdr);
        bitset
    }
    pub fn size(&self)->u32{
        self.bits.len() as u32* 32
    }
    pub fn as_slice(&self)->&[u32]{
        self.bits.as_slice()
    }
    pub fn empty() -> Self {
        Self { bits: vec![] }
    }

    pub fn set_size(&mut self, bit_count:u32){
        self.bits.resize(bit_count_to_vec_size(bit_count),0)
    }
    pub fn ensure_size(&mut self, min_size:u32) {
        let new_size = bit_count_to_vec_size(min_size);
        if new_size > self.bits.len(){
            self.bits.resize(new_size,0)
        }
    }
    pub fn new(bit_count: u32) -> Self {
        Self { bits: vec![0; bit_count_to_vec_size(bit_count)]}
    }


    pub fn set_bits_on(&mut self, sdr: &[u32]) {
        for &index in sdr.iter() {
            self.set_bit_on(index);
        }
    }

    pub fn set_bits_off(&mut self, sdr: &[u32]) {
        for &index in sdr.iter() {
            self.set_bit_off(index);
        }
    }
    pub fn clear_all(&mut self) {
        self.bits.fill(0)
    }
    pub fn clear(&mut self, sdr: &CpuSDR) {
        for &index in sdr.iter() {
            self.clear_u32_containing_bit(index);
        }
    }

    pub fn set_bit_on(&mut self, index: u32) {
        // u32 has 32 bits
        // self.inputs stores one bit per input in form of u32 integers
        // index/32 gives index of the u32 integer that contains index-th bit
        // index/32 == index>>5
        // index%32 gives us index of the index-th bit but relative to the u32 that contains it
        // index%32 == index&31
        // we might either do  1<<(index&31) or 2147483648>>(index&31) . Both are equivalent. It only changes the order in which we store bits within each u32
        let i = (index >> 5) as usize;
        assert!(i<self.bits.len(),"Index {} out of bounds for bitset of {} bits", index, self.size());
        self.bits[i] |= 2147483648>>(index&31);
    }

    pub fn clear_u32_containing_bit(&mut self, index: u32) {
        self.bits[(index >> 5) as usize] = 0;
    }

    pub fn set_bit_off(&mut self, index: u32) {
        self.bits[(index >> 5) as usize] &= !(2147483648>>(index&31));
    }

    pub fn is_bit_on(&self, index: u32) -> bool {
        (self.bits[(index >> 5) as usize] & (2147483648>>(index&31))) != 0
    }


    pub fn cardinality(&self) -> u32 {
        self.bits.iter().map(|&s| s.count_ones()).sum()
    }

    pub fn append_to_sdr(&self, sdr:&mut CpuSDR) {
        for (u32_index, &u32_val) in self.bits.iter().enumerate(){
            let u32_index = u32_index as u32;
            let mut u32_val = u32_val;
            let mut bit_index = 0;
            while u32_val != 0 {
                if (u32_val & 1) != 0{
                    sdr.push(u32_index*32+bit_index)
                }
                u32_val = u32_val>>1;
                bit_index += 1;
            }
        }

    }
}
impl From<&CpuBitset> for CpuSDR{
    fn from(bitset: &CpuBitset) -> Self {
        let mut sdr = CpuSDR::new();
        bitset.append_to_sdr(&mut sdr);
        sdr
    }
}
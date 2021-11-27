use crate::{CpuSDR, EncoderTarget, CpuBitset, Shape};
use std::fmt::{Debug, Formatter};
use crate::rand::xorshift32;
use std::ops::{Deref, DerefMut};

use serde::{Serialize, Deserialize};

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CpuBitset2d {
    bits: CpuBitset,
    shape: Shape
}

impl Debug for CpuBitset2d{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut all_bits = String::new();
        let mut col = 0;
        let mut row = 0;
        'outer: for &bits in self.bits.as_slice(){
            for bit in format!("{:032b}", bits).bytes(){
                all_bits.push(bit as char);
                col+=1;
                if col == self.width(){
                    col = 0;
                    row +=1;
                    if row == self.height(){
                        break 'outer;
                    }
                    all_bits.push('\n');
                }
            }
        }
        write!(f,"{}", all_bits)
    }
}

impl Deref for CpuBitset2d{
    type Target = CpuBitset;

    fn deref(&self) -> &Self::Target {
        &self.bits
    }
}
impl DerefMut for CpuBitset2d{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.bits
    }
}
impl CpuBitset2d {
    pub fn width(&self)->u32{
        self.shape.width()
    }
    pub fn height(&self)->u32{
        self.shape.height()
    }
    pub fn as_bitset(&self)->&CpuBitset{
        &self.bits
    }
    pub fn as_bitset_mut(&mut self)->&mut CpuBitset{
        &mut self.bits
    }
    pub fn new(bits: CpuBitset, height:u32,width:u32)->Self{
        Self{bits,shape:Shape::new(height,width)}
    }
    pub fn shape(&self)->&Shape{
        &self.shape
    }
    pub fn is_bit_at(&self, y:u32,x:u32)->bool{
        self.bits.is_bit_on(self.shape.index(y,x))
    }
    pub fn set_bit_at(&mut self, y:u32,x:u32){
        self.bits.set_bit_on(self.shape.index(y,x))
    }
    pub fn set_bits_at(&mut self,y:u32,x:u32,height:u32,width:u32,sdr:&CpuSDR){
        assert!(x+width<=self.shape.width(), "The subregion x+width={}+{} is out of bounds {}",x,width,self.shape.width());
        assert!(y+height<=self.shape.height(), "The subregion y+height={}+{} is out of bounds {}",y,height,self.shape.height());
        for &neuron in sdr.as_slice(){
            let ny = neuron/width;
            let nx = neuron%width;
            assert!(ny<y+height,"SDR contained neuron {} which was resolved to coordinates y={} x={} lying outside of bounds {}x{}",neuron,ny,nx,height,width);
            self.set_bit_at(y+ny,x+nx);
        }
    }

}
impl EncoderTarget for CpuBitset2d{
    fn push(&mut self, neuron_index: u32) {
        self.as_bitset_mut().push(neuron_index)
    }

    fn clear_range(&mut self, from: u32, to: u32) {
        self.as_bitset_mut().clear_range(from, to)
    }

    fn contains(&self, neuron_index: u32) ->bool{
        self.is_bit_on(neuron_index)
    }
}
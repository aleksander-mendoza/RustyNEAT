use crate::{CpuSDR, EncoderTarget, CpuBitset, Shape2, Shape3};
use std::fmt::{Debug, Formatter};
use crate::rnd::xorshift32;
use std::ops::{Deref, DerefMut};

use serde::{Serialize, Deserialize};

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CpuBitset3d {
    bits: CpuBitset,
    shape: [u32;3]
}

impl Debug for CpuBitset3d{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut all_bits = String::new();
        let mut col = 0;
        let mut row = 0;
        let mut slice = 0;
        'outer: for &bits in self.bits.as_slice(){
            for bit in format!("{:032b}", bits).bytes(){
                all_bits.push(bit as char);
                col+=1;
                if col == self.width(){
                    col = 0;
                    row +=1;
                    if row == self.height(){
                        row = 0;
                        slice += 1;
                        if slice == self.depth() {
                            break 'outer;
                        }
                        all_bits.push('\n');
                    }
                    all_bits.push('\n');
                }
            }
        }
        write!(f,"{}", all_bits)
    }
}

impl Deref for CpuBitset3d{
    type Target = CpuBitset;

    fn deref(&self) -> &Self::Target {
        &self.bits
    }
}
impl DerefMut for CpuBitset3d{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.bits
    }
}
impl CpuBitset3d {
    pub fn width(&self)->u32{
        self.shape.width()
    }
    pub fn height(&self)->u32{
        self.shape.height()
    }
    pub fn depth(&self)->u32{
        self.shape.depth()
    }
    pub fn as_bitset(&self)->&CpuBitset{
        &self.bits
    }
    pub fn as_bitset_mut(&mut self)->&mut CpuBitset{
        &mut self.bits
    }
    pub fn new(bits: CpuBitset, depth:u32, height:u32,width:u32)->Self{
        Self{bits,shape: [depth, height, width]}
    }
    pub fn shape(&self)->&[u32;3] {
        &self.shape
    }
    pub fn is_bit_at(&self,z:u32, y:u32,x:u32)->bool{
        self.bits.is_bit_on(self.shape.index(z,y,x))
    }
    pub fn set_bit_at(&mut self, z:u32,y:u32,x:u32){
        self.bits.set_bit_on(self.shape.index(z,y,x))
    }
    pub fn set_bits_at(&mut self,z:u32,y:u32,x:u32,depth:u32, height:u32,width:u32,sdr:&CpuSDR){
        assert!(x+width<=self.shape.width(), "The subregion x+width={}+{} is out of bounds {}",x,width,self.shape.width());
        assert!(y+height<=self.shape.height(), "The subregion y+height={}+{} is out of bounds {}",y,height,self.shape.height());
        assert!(z+depth<=self.shape.depth(), "The subregion z+depth={}+{} is out of bounds {}",z,depth,self.shape.depth());
        for &neuron in sdr.as_slice(){
            let nx = neuron%width;
            let nzy = neuron/width;
            let ny = neuron%height;
            let nz = neuron/height;
            assert!(nz<depth, "SDR contained neuron {} which was resolved to coordinates z={} y={} x={} lying outside of bounds {}x{}x{}",neuron,nz,ny,nx,depth,height,width);
            self.set_bit_at(z+nz,y+ny,x+nx);
        }
    }

}
impl EncoderTarget for CpuBitset3d{
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
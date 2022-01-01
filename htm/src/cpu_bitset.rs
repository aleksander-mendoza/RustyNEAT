use crate::{CpuSDR, EncoderTarget, Shape, resolve_range, Shape3};
use std::fmt::{Debug, Formatter};
use serde::{Serialize, Deserialize};
use crate::vector_field::{VectorFieldOne, VectorFieldAdd, VectorFieldPartialOrd};
use std::ops::RangeBounds;
use std::collections::Bound;
use rand::Rng;

#[derive(Serialize, Deserialize, Clone, Eq, PartialEq)]
pub struct CpuBitset {
    bits: Vec<u32>,
    shape: [u32;3]
}

impl Debug for CpuBitset{
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
                        if slice == self.channels() {
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
pub fn bit_count_to_vec_size(bit_count:u32)->usize{
    (bit_count as usize + 31) / 32
}

impl EncoderTarget for CpuBitset{
    fn push(&mut self, neuron_index: u32) {
        self.set_bit_on(neuron_index)
    }

    fn clear_range(&mut self, from: u32, to: u32) {
        assert!(from<=to,"Range's left bound {} is greater than right bound {}", from,to);
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
    fn contains(&self, neuron_index: u32) ->bool{
        self.is_bit_on(neuron_index)
    }
}
impl CpuBitset {
    pub fn from_raw(bits: Vec<u32>, shape:[u32;3]) -> Self {
        assert!(bits.len() as u32 * 32 >= shape.product(),"Shape {:?} has size {} but raw bitset holds only {} bits",shape,shape.product(),bits.len()*32);
        Self{bits,shape}
    }
    pub fn width(&self)->u32{
        self.shape.width()
    }
    pub fn height(&self)->u32{
        self.shape.height()
    }
    pub fn channels(&self)->u32{
        self.shape.channels()
    }
    pub fn cardinality_in_range(&self, from: u32, to: u32) -> u32 {
        assert!(from<=to,"Range's left bound {} is greater than right bound {}", from,to);
        let from_u32 = (from/32) as usize;
        let to_u32 = (to/32) as usize;
        let all_ones = u32::MAX;

        if from_u32==to_u32{
            (self.bits[from_u32] & (all_ones >> (from&31)) & !(all_ones >> (to&31))).count_ones()
        }else{
            let mut sum = self.bits[from_u32+1..to_u32].iter().map(|&s| s.count_ones()).sum();
            sum += (self.bits[from_u32] & (all_ones >> (from&31))).count_ones();
            if to_u32<self.bits.len(){
                sum += (self.bits[to_u32] & !(all_ones >> (to&31))).count_ones();
            }
            sum
        }
    }
    pub fn from_bools(bools:&[bool])->Self{
        let mut bitset = Self::new(bools.len() as u32);
        bools.iter().cloned().enumerate().filter(|(i,b)|*b).map(|(i,_)|i as u32).for_each(|i|bitset.set_bit_on(i));
        bitset
    }
    pub fn from_sdr(sdr: &[u32], input_size:u32) -> Self {
        Self::from_sdr2d(sdr,1,input_size)
    }
    pub fn from_sdr2d(sdr: &[u32], height:u32,width:u32) -> Self {
        Self::from_sdr3d(sdr,1,height,width)
    }
    pub fn from_sdr3d(sdr: &[u32], depth:u32,height:u32,width:u32) -> Self {
        let mut bitset = Self::new3d(depth,height,width);
        bitset.set_bits_on(sdr);
        bitset
    }
    pub fn size(&self)->u32{
        self.shape.product()
    }
    pub fn as_slice(&self)->&[u32]{
        self.bits.as_slice()
    }


    pub fn new(bit_count: u32) -> Self {
        Self::new2d(1,bit_count)
    }
    pub fn new2d(height:u32,width:u32)->Self{
        Self::new3d(1,height, width)
    }
    pub fn new3d(depth:u32, height:u32,width:u32)->Self{
        let shape = [depth, height, width];
        Self{ bits: vec![0; bit_count_to_vec_size(shape.product())], shape}
    }
    pub fn rand(width:u32, rand_seed:&mut impl rand::Rng) -> Self {
        Self::rand2d(1,width,rand_seed)
    }
    pub fn rand2d(height:u32,width:u32, rand_seed:&mut impl rand::Rng) -> Self {
        Self::rand3d(1,height,width,rand_seed)
    }
    pub fn rand3d(depth:u32, height:u32,width:u32, rand_seed:&mut impl rand::Rng) -> Self {
        let shape = [depth, height, width];
        let bit_count = shape.product();
        let u32_count = bit_count_to_vec_size(bit_count);
        Self { bits: (0..u32_count).map(|_|{
            rand_seed.gen()
        }).collect(), shape}
    }
    pub fn empty(width:u32) -> Self {
        Self::empty2d(1,width)
    }
    pub fn empty2d( height:u32,width:u32) -> Self {
        Self::empty3d(1,height,width)
    }
    pub fn empty3d(depth:u32, height:u32,width:u32) -> Self {
        Self { bits: vec![] ,shape:[depth,height,width]}
    }
    pub fn rand_of_cardinality(bit_count: u32, cardinality:u32,rand_seed:&mut impl rand::Rng) -> Self {
        let mut slf = Self::new(bit_count);
        slf.set_bits_on_rand(0,bit_count,cardinality,rand_seed);
        slf
    }

    pub fn set_bits_on_rand(&mut self,from:u32,to:u32,bit_count: u32, rand_seed:&mut impl rand::Rng) {
        let len = to-from;
        for _ in 0..bit_count{
            let r:u32 = rand_seed.gen();
            self.set_bit_on(from + r%len);
        }
    }



    pub fn set_bits_off(&mut self, sdr: &[u32]) {
        for &index in sdr.iter() {
            self.set_bit_off(index);
        }
    }
    pub fn overlap(&self, other:&CpuBitset) -> u32 {
        self.bits.iter().cloned().zip(other.bits.iter().cloned()).map(| (bits,other_bits)|(bits&other_bits).count_ones()).sum()
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
        assert!(index<self.size(),"Index {} out of bounds for bitset of {:?}={} bits", index, self.shape(),self.size());
        // u32 has 32 bits
        // self.inputs stores one bit per input in form of u32 integers
        // index/32 gives index of the u32 integer that contains index-th bit
        // index/32 == index>>5
        // index%32 gives us index of the index-th bit but relative to the u32 that contains it
        // index%32 == index&31
        // we might either do  1<<(index&31) or 2147483648>>(index&31) . Both are equivalent. It only changes the order in which we store bits within each u32
        let i = (index >> 5) as usize;
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
                if (u32_val & 2147483648) != 0{
                    sdr.push(u32_index*32+bit_index)
                }
                u32_val = u32_val<<1;
                bit_index += 1;
            }
        }

    }

    pub fn swap_u32(&mut self, offset1:u32, offset2:u32, size:u32){
        assert!(!(offset1 <= offset2 && offset2 < offset1+size), "the two regions overlap");
        for i in 0..size{
            self.bits.swap((offset1+i) as usize,(offset2+i) as usize)
        }
    }

    pub fn shape(&self)->&[u32;3] {
        &self.shape
    }
    pub fn reshape(&mut self){
        self.reshape3d(1,1,self.shape.product())
    }
    pub fn reshape2d(&mut self,height:u32,width:u32){
        self.reshape3d(1,height,width)
    }
    pub fn reshape3d(&mut self,depth:u32, height:u32,width:u32){
        let new_shape = [depth,height,width];
        assert_eq!(new_shape.product(),self.shape.product(),"New shape {:?} is incompatible with old {:?}",new_shape,self.shape());
        self.shape = new_shape
    }
    pub fn resize(&mut self,width:u32){
        self.resize2d(1,width);
    }
    pub fn resize2d(&mut self, height:u32,width:u32){
        self.resize3d(1,height,width);
    }
    pub fn resize3d(&mut self,depth:u32, height:u32,width:u32){
        let new_shape = [depth,height,width].product();
        self.bits.resize(bit_count_to_vec_size(new_shape),0);
    }
    pub fn is_bit_on2d(&self, y:u32,x:u32)->bool{
        self.is_bit_on3d(0,y,x)
    }
    pub fn is_bit_on3d(&self,z:u32, y:u32,x:u32)->bool{
        self.is_bit_on(self.shape.idx([z,y,x]))
    }
    pub fn set_bit_on2d(&mut self, y:u32,x:u32){
        self.set_bit_on3d(0,y,x)
    }
    pub fn set_bit_on3d(&mut self, z:u32,y:u32,x:u32){
        self.set_bit_on(self.shape.idx([z,y,x]))
    }
    pub fn set_bits_on(&mut self, sdr: &[u32]) {
        for &index in sdr.iter() {
            self.set_bit_on(index);
        }
    }
    pub fn set_bits_on_in_range(&mut self, input_range:impl RangeBounds<u32>,sdr: &[u32]) {
        let range = resolve_range(self.size(),input_range);
        for &index in sdr.iter() {
            let i = range.start + index;
            assert!(i<range.end,"Index {} is out of range {:?} size",index,range);
            self.set_bit_on(i);
        }
    }
    pub fn set_bits_off_in_range(&mut self, input_range:impl RangeBounds<u32>,sdr: &[u32]) {
        let range = resolve_range(self.size(),input_range);
        for &index in sdr.iter() {
            let i = range.start + index;
            assert!(i<range.end,"Index {} is out of range {:?} size",index,range);
            self.set_bit_off(i);
        }
    }
    pub fn set_bits_on3d(&mut self,offset:[u32;3],size:[u32;3],sdr:&[u32]){
        assert!(offset.add(&size).all_le(&self.shape), "The subregion {:?}..{:?} is out of bounds {:?}",offset,size,self.shape);
        for &neuron in sdr{
            self.set_bit_on(self.shape.idx(offset.add(&size.pos(neuron))));
        }
    }
    pub fn set_bits_on2d(&mut self,offset:[u32;2],size:[u32;2],sdr:&[u32]){
        assert_eq!(self.channels(),1, "This bitset has depth dimension! Use set_bits_on3d instead");
        assert!(offset.add(&size).all_le(self.shape().grid()), "The subregion {:?}..{:?} is out of bounds {:?}",offset,size,self.shape);
        for &neuron in sdr{
            let o = self.shape().grid().idx(offset.add(&size.pos(neuron)));
            self.set_bit_on(o);
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
use ocl::{ProQue,Error, SpatialDims, flags, Platform, Device, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, Deref, DerefMut};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use ndalgebra::buffer::Buffer;
use crate::htm_program::HtmProgram;

#[derive(Clone, Eq, PartialEq)]
pub struct CpuSDR(Vec<u32>);

impl PartialEq<Vec<u32>> for CpuSDR{
    fn eq(&self, other: &Vec<u32>) -> bool {
        self.0.eq(other)
    }
}
impl From<Vec<u32>> for CpuSDR{
    fn from(v: Vec<u32>) -> Self {
        Self(v)
    }
}

impl Deref for CpuSDR{
    type Target = [u32];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}
impl DerefMut for CpuSDR{

    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut_slice()
    }
}
impl Debug for CpuSDR{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl CpuSDR {
    pub fn as_slice(&self)->&[u32]{
        self.0.as_slice()
    }
    pub fn as_mut_slice(&mut self)->&mut [u32]{
        self.0.as_mut_slice()
    }
    pub fn push(&mut self, neuron_index:u32){
        self.0.push(neuron_index)
    }
    pub fn to_vec(self)->Vec<u32>{
        let Self(v) = self;
        v
    }
    pub fn clear(&mut self){
        self.0.clear()
    }
    pub fn set(&mut self, active_neurons:&[u32]){
        unsafe{self.0.set_len(0)}
        self.0.extend_from_slice(active_neurons)
    }
    pub fn number_of_active_neurons(&self)->usize{
        self.0.len()
    }
    /**SDR ust be sparse. Hence we might as well put a cap on the maximum number of active neurons.
    If your system is designed correctly, then you should never have to worry about exceeding this limit.*/
    pub fn max_active_neurons(&self)->usize{
        self.0.capacity()
    }
    pub fn new() -> Self{
        Self(Vec::new())
    }
    pub fn with_capacity(capacity:usize) -> Self{
        Self(Vec::with_capacity(capacity))
    }
    /**Sorts and removes duplicates*/
    pub fn normalize(&mut self){
        self.0.sort();
        self.0.dedup();
    }
    /**This method requires that both SDRs are first normalized*/
    pub fn overlap(&self, other:&CpuSDR)->u32{
        if self.is_empty() || other.is_empty(){return 0}
        let mut i1=0;
        let mut i2=0;
        let mut overlap = 0;
        let (s1,s2) = if self.0[0] < other.0[0]{(self,other)}else{(other,self)};
        loop {
            while s1.0[i1] < s2.0[i2] {
                i1 += 1;
                if i1 >= s1.len() { return overlap }
            }
            if s1.0[i1] == s2.0[i2] {
                overlap += 1;
                i1 += 1;
                if i1 >= s1.len() { return overlap }
            }
            while s1.0[i1] > s2.0[i2] {
                i2 += 1;
                if i2 >= s2.len() { return overlap }
            }
            if s1.0[i1] == s2.0[i2] {
                overlap += 1;
                i2 += 1;
                if i2 >= s2.len() { return overlap }
            }
        }
    }
    pub fn binary_search(&self, neuron_index:u32)->bool{
        self.0.binary_search(&neuron_index).is_ok()
    }
    pub fn is_normalized(&self)->bool{
        if self.0.is_empty(){return true}
        let mut prev = self.0[0];
        for &i in &self.0[1..]{
            if i <= prev{return false}
            prev = i;
        }
        true
    }
    pub fn extend(&mut self, other:&CpuSDR){
        self.0.extend_from_slice(other)
    }
    /**Requires that both SDRs are normalized. The resulting SDR is already in normalized form*/
    pub fn intersection(&self, other:&CpuSDR)->CpuSDR{
        let mut intersection = CpuSDR::with_capacity(self.len()+other.len());
        let mut i=0;
        if other.is_empty(){return intersection}
        for &neuron_index in &self.0{
            while other[i] < neuron_index{
                i+=1;
                if i >= other.len() {return intersection}
            }
            if other[i] == neuron_index{
                intersection.push(neuron_index);
            }
        }
        intersection
    }
    pub fn union(&self, other:&CpuSDR)->CpuSDR{
        let mut union = CpuSDR::with_capacity(self.len()+other.len());
        if self.is_empty() || other.is_empty(){return union}
        let mut i1=0;
        let mut i2=0;
        let (s1,s2) = if self.0[0] < other.0[0]{(self,other)}else{(other,self)};
        loop {
            while s1.0[i1] < s2.0[i2] {
                union.0.push(s1.0[i1]);
                i1 += 1;
                if i1 >= s1.len() { return union }
            }
            if s1.0[i1] == s2.0[i2] {
                union.0.push(s1.0[i1]);
                i1 += 1;
                if i1 >= s1.len() { return union }
            }
            while s1.0[i1] > s2.0[i2] {
                union.0.push(s2.0[i2]);
                i2 += 1;
                if i2 >= s2.len() { return union }
            }
            if s1.0[i1] == s2.0[i2] {
                union.0.push(s2.0[i2]);
                i2 += 1;
                if i2 >= s2.len() { return union }
            }
        }
    }
    pub fn shrink(&mut self, number_of_bits_to_retain:usize){
        if number_of_bits_to_retain < self.len() {
            unsafe{self.0.set_len(number_of_bits_to_retain)}
        }
    }
    /**Similar to shrink, but removes random bits*/
    pub fn subsample(&mut self, number_of_bits_to_retain:usize){
        while self.len() > number_of_bits_to_retain{
            self.0.swap_remove(rand::random::<usize>() % self.len());
        }
    }
}


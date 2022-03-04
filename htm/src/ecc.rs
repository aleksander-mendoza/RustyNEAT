use crate::{VectorFieldOne, Shape2, Shape3, VectorFieldPartialOrd, EccProgram, CpuSDR, CpuEccDense, DenseWeight, top_small_k_indices, EncoderTarget, Shape, range_contains, from_xyz, w_idx, ConvShape, ConvShapeTrait, HasShape, HasShapeMut, Metric};
use crate::xorshift::xorshift32;
use std::ops::{Deref, DerefMut, IndexMut, Index};
use itertools::Itertools;
use rand::Rng;
use serde::{Serialize,Deserialize};
use crate::sdr::SDR;
use ocl::Error;
use std::fmt::Debug;

pub type Idx = u32;
pub type Rand = u32;
pub fn xorshift_rand(rand:Rand)->Rand{
    xorshift32(rand)
}

pub fn as_idx(i:usize)->Idx{
    i as u32
}



pub trait EccLayer:HasShapeMut {
    type A:SDR;
    type D:DenseWeight;
    fn get_threshold(&self) -> Self::D;
    fn set_threshold(&mut self, threshold: Self::D);
    fn get_plasticity(&self) -> Self::D;
    fn set_plasticity(&mut self, threshold: Self::D);
    fn k(&self) -> Idx;
    fn set_k(&mut self, k: Idx);

    fn learnable_parameters(&self) -> usize;

    fn get_max_incoming_synapses(&self) -> Idx;


    fn new_empty_sdr(&self,capacity:Idx)->Self::A;
    fn new_empty_output_sdr(&self)->Self::A {
        let k = self.k();
        let a = self.shape().out_area();
        self.new_empty_sdr(a*k)
    }
    fn run(&mut self, input: &Self::A) -> Self::A {
        let output = self.infer(input);
        self.decrement_activities(&output);
        output
    }
    fn infer(&mut self, input: &Self::A) -> Self::A {
        let mut output = self.new_empty_output_sdr();
        self.infer_in_place(input, &mut output);
        output
    }
    fn run_in_place(&mut self, input: &Self::A, output: &mut Self::A){
        self.infer_in_place(input,output);
        self.decrement_activities(output)
    }
    fn infer_in_place(&mut self, input: &Self::A, output: &mut Self::A);
    fn decrement_activities(&mut self,output: &Self::A);
    fn learn(&mut self, input: &Self::A, output: &Self::A);

    // fn top_large_k_by_channel<T>(&self, sums: &[T], candidates_per_value: &mut [Idx], f: fn(&T) -> Idx, threshold: impl Fn(Idx) -> bool) -> CpuSDR {
    //     let a = as_usize(self.out_area());
    //     let c = as_usize(self.out_channels());
    //     let k = as_usize(self.k());
    //     let mut top_k = CpuSDR::with_capacity(k * a);
    //     for column_idx in 0..a {
    //         let offset = c * column_idx;
    //         let range = offset..offset + c;
    //         top_large_k_indices(k, &sums[range], candidates_per_value, f, |t| if threshold(t) { top_k.push(offset + t) });
    //     }
    //     top_k
    // }

}
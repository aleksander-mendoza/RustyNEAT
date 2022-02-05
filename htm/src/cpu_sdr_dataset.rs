use ocl::{ProQue, Error, SpatialDims, flags, Platform, Device, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, Deref, DerefMut};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use ndalgebra::buffer::Buffer;
use crate::ecc_program::EccProgram;
use crate::{CpuBitset, EncoderTarget, Shape, Idx, as_idx, as_usize, OclSDR, range_contains, VectorFieldSub, VectorFieldPartialOrd, VectorFieldRem, VectorFieldAdd, CpuSDR, ConvShape, CpuEccDense, DenseWeight, ConvWeights, CpuEccPopulation, VectorFieldRng, Shape2, Shape3, VectorFieldOne, EccLayer};
use std::collections::{HashMap, HashSet};
use std::borrow::Borrow;
use serde::{Serialize, Deserialize};
use crate::vector_field::{VectorField, VectorFieldMul};
use crate::sdr::SDR;
use rand::Rng;
use num_traits::Num;
use itertools::Itertools;
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct CpuSdrDataset {
    data: Vec<CpuSDR>,
    shape: [Idx; 3],
}

impl Deref for CpuSdrDataset {
    type Target = Vec<CpuSDR>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for CpuSdrDataset {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl Debug for CpuSdrDataset {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dataset{{shape:{:?},len:{:}}}", self.shape, self.data.len())
    }
}


impl CpuSdrDataset {
    pub fn from_slice(s: &[CpuSDR], shape: [Idx; 3]) -> Self {
        Self {
            data: Vec::from(s),
            shape,
        }
    }
    pub fn shape(&self) -> &[Idx; 3] {
        &self.shape
    }
    pub fn new(shape: [Idx; 3]) -> Self {
        Self {
            data: vec![],
            shape,
        }
    }
    pub fn with_capacity(capacity: usize, shape: [Idx; 3]) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            shape,
        }
    }
    pub fn rand(&self, rng: &mut impl Rng) -> Option<&CpuSDR> {
        if self.is_empty() {
            None
        } else {
            Some(&self[rng.gen_range(0..self.len())])
        }
    }
    pub fn conv_rand_subregion(&self, conv: &ConvShape, number_of_samples: usize, rng: &mut impl Rng) -> Self {
        let mut s = Self::new(conv.kernel_column());
        s.extend_from_conv_rand_subregion(conv, number_of_samples, self, rng);
        s
    }

    pub fn extend_from_conv_rand_subregion(&mut self, conv: &ConvShape, number_of_samples: usize, original: &CpuSdrDataset, rng: &mut impl Rng) {
        assert!(!original.is_empty() || number_of_samples == 0);
        assert_eq!(original.shape(), conv.in_shape(), "Original shape must be equal convolution input shape");
        assert_eq!(self.shape(), &conv.kernel_column(), "Patch shape must be equal kernel column");
        for _ in 0..number_of_samples {
            let sdr = original.rand(rng).unwrap();
            let sdr = sdr.conv_rand_subregion(conv, rng);
            self.push(sdr);
        }
    }
    pub fn extend_from_conv_subregion_indices(&mut self, conv:&ConvShape, indices: &SubregionIndices, original: &CpuSdrDataset) {
        assert!(!original.is_empty() || indices.is_empty());
        assert_eq!(original.shape(), conv.in_shape(), "Original shape must be equal convolution input shape");
        assert_eq!(self.shape(), &conv.kernel_column(), "Patch shape must be equal kernel column");
        assert_eq!(indices.shape(), conv.out_shape(), "Indices refer to different shape");
        assert_eq!(original.len(),indices.dataset_len, "Dataset lengths don't match");
        for i in &indices.indices {
            let sdr = &original[as_usize(i.channels())];
            let sdr = sdr.conv_subregion(conv, i.grid());
            self.push(sdr);
        }
    }
    pub fn train<D:DenseWeight>(&self, number_of_samples: usize, ecc: &mut CpuEccDense<D>, rng: &mut impl Rng,progress_callback:impl Fn(usize)) {
        assert_eq!(self.shape(),ecc.in_shape(),"Shapes don't match");
        assert_ne!(self.len(),0,"Dataset is empty");
        for i in 0..number_of_samples{
            let sample_idx = rng.gen_range(0..self.len());
            let sample = &self[sample_idx];
            let out = ecc.run(sample);
            ecc.learn(sample,&out);
            progress_callback(i);
        }
    }
    /**Setting drift==[1,1] will disable drift. Setting it to [n,n] will mean that patch can randomly jiggle within an area of n-by-n output columns*/
    pub fn train_with_patches<D:DenseWeight>(&self, number_of_samples: usize, drift:[Idx;2], patches_per_sample:usize, ecc: &mut CpuEccDense<D>, rng: &mut impl Rng,progress_callback:impl Fn(usize)) {
        assert_eq!(ecc.out_grid(),&[1,1],"The ecc network should consist of only a single column");
        assert_eq!(self.shape().channels(),ecc.in_channels(),"Channels don't match");
        assert_ne!(self.len(),0,"Dataset is empty");
        assert!(drift.all_gt_scalar(0),"Drift can't be 0");
        let conv = ConvShape::new_in(*self.shape(),ecc.out_channels(),*ecc.kernel(),*ecc.stride());
        assert!(conv.out_grid().all_ge(&drift),"Drift {:?} is larger than output grid {:?}",drift,conv.out_grid());
        for i in 0..number_of_samples{
            let i = rng.gen_range(0..self.len());
            let sample = &self[i];
            let pos = conv.out_grid().sub(&drift.sub_scalar(1)).rand_vec(rng);
            let mut out=CpuSDR::new();
            for _ in 0..patches_per_sample {
                let drifted_pos = pos.add(&drift.rand_vec(rng));
                let patch = sample.conv_subregion(&conv, &drifted_pos);
                if out.is_empty() {
                    ecc.infer_in_place(&patch, &mut out);
                }
                ecc.learn(&patch, &out);
            }
            ecc.decrement_activities(&out);
            progress_callback(i);
        }
    }
    pub fn count(&self) -> Vec<u32>{
        let mut counts = vec![0;as_usize(self.shape().product())];
        for i in &self.data{
            for &i in i.iter(){
                counts[as_usize(i)]+=1
            }
        }
        counts
    }
    pub fn measure_receptive_fields(&self, outputs: &Self) -> Vec<u32>{
        assert_eq!(self.len(),outputs.len());
        let ov =outputs.shape().size();
        let iv = self.shape().size();
        let mut receptive_fields = vec![0;as_usize(ov*iv)];
        for (o_sdr,i_sdr) in outputs.iter().zip(self.iter()) {
            for &o in o_sdr.iter() {
                for&i in i_sdr.iter(){
                    receptive_fields[as_usize(i + o * iv)]+=1;
                }
            }
        }
        receptive_fields
    }
    pub fn batch_infer<T: DenseWeight + Sync + Send>(&self, ecc: &CpuEccDense<T>) -> Self {
        assert_eq!(self.shape(), ecc.in_shape());
        let data = ecc.batch_infer(&self.data, |d| d, |o| o);
        Self {
            data,
            shape: ecc.output_shape(),
        }
    }
    pub fn batch_infer_and_measure_s_expectation<T: DenseWeight + Sync + Send>(&self, ecc: &CpuEccDense<T>) -> (Self,T,u32) {
        assert_eq!(self.shape(), ecc.in_shape());
        let (data,s_exp,missed) = ecc.batch_infer_and_measure_s_expectation(&self.data, |d| d, |o| o);
        (Self {
            data,
            shape: ecc.output_shape(),
        },s_exp,missed)
    }
    pub fn batch_infer_conv_weights<T: DenseWeight + Sync + Send>(&self, ecc: &ConvWeights<T>, target: CpuEccPopulation<T>) -> Self {
        assert_eq!(self.shape(), ecc.in_shape());
        let data = ecc.batch_infer(&self.data, |d| d, target, |o| o);
        Self {
            data,
            shape: ecc.output_shape(),
        }
    }
    pub fn batch_infer_conv_weights_and_measure_s_expectation<T: DenseWeight + Sync + Send>(&self, ecc: &ConvWeights<T>, target: CpuEccPopulation<T>) -> (Self,T,u32) {
        assert_eq!(self.shape(), ecc.in_shape());
        let (data,s_exp,missed) = ecc.batch_infer_and_measure_s_expectation(&self.data, |d| d, target, |o| o);
        (Self {
            data,
            shape: ecc.output_shape(),
        },s_exp,missed)
    }

    /**set num_of_classes==0 in order to automatically detect it using the maximum value found in labels array*/
    pub fn count_per_label<T>(&self, labels: &[T], num_of_classes: usize,f:impl Fn(&T)->usize) -> Vec<f32> {
        assert_eq!(labels.len(), self.len());
        let volume = as_usize(self.shape().size());
        let mut occurrences = vec![0f32; num_of_classes * volume];
        for (sdr, label) in self.iter().zip(labels.iter().map(f)) {
            if label < num_of_classes {
                let offset = label * volume;
                let occurrences_for_label = &mut occurrences[offset..offset + volume];
                sdr.iter().cloned().for_each(|i| occurrences_for_label[as_usize(i)] += 1.);
            }
        }
        occurrences
    }
    pub fn gen_rand_conv_subregion_indices(&self, out_shape: [Idx;3], number_of_samples: usize, rng: &mut impl Rng) -> SubregionIndices {
        SubregionIndices::gen_rand_conv_subregion_indices(self.len(),out_shape,number_of_samples,rng)
    }
    // pub fn fit_naive_bayes<T: Into<usize>>(&self, labels: &[T]) -> NaiveBayesClassifier {
    //
    // }
    pub fn fit_linear_regression<T>(&self, labels: &[T],num_of_classes: usize,f:impl Fn(&T)->usize) -> LinearClassifier {
        let mut occurrences = self.count_per_label(labels,num_of_classes,f);
        let volume = as_usize(self.shape().size());
        for i in 0..volume{
            let s:f32 = (0..num_of_classes).map(|lbl|occurrences[lbl*volume+i]).sum();
            (0..num_of_classes).for_each(|lbl|occurrences[lbl*volume+i]/=s)
        }
        LinearClassifier{
            occurrences,
            num_classes:num_of_classes,
            volume,
            in_shape: self.shape
        }
    }
}

pub struct SubregionIndices {
    dataset_len:usize,
    indices: Vec<[Idx; 3]>,
    shape: [Idx;3],
}
impl Deref for SubregionIndices{
    type Target = Vec<[Idx; 3]>;

    fn deref(&self) -> &Self::Target {
        &self.indices
    }
}
impl DerefMut for SubregionIndices{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.indices
    }
}
impl SubregionIndices {
    pub fn shape(&self)->&[Idx;3]{
        &self.shape
    }
    pub fn gen_rand_conv_subregion_indices(dataset_len: usize, out_shape: [Idx;3], number_of_samples: usize, rng: &mut impl Rng) -> SubregionIndices {
        Self {
            dataset_len,
            indices: (0..number_of_samples).map(|_| {
                let sdr_idx = as_idx(rng.gen_range(0..dataset_len));
                let pos = out_shape.grid().rand_vec(rng);
                pos.add_channels(sdr_idx)
            }).collect(),
            shape: out_shape,
        }
    }
}

pub struct LinearClassifier {
    occurrences: Vec<f32>,
    num_classes: usize,
    volume: usize,
    //cached volume of input shape
    in_shape: [Idx; 3],
}
impl LinearClassifier{
    pub fn prob(&self,input_idx:Idx,label:usize)->f32{
        self.occurrences[label*self.volume+as_usize(input_idx)]
    }
    pub fn occurrences(&self)->&[f32]{
        &self.occurrences
    }
    pub fn occurrences_for_label(&self,lbl:usize)->&[f32]{
        let offset = self.volume*lbl;
        &self.occurrences[offset..offset+self.volume]
    }
    pub fn num_classes(&self)->usize{
        self.num_classes
    }
    pub fn shape(&self)->&[Idx;3]{
        &self.in_shape
    }
    pub fn classify(&self,sdr:&CpuSDR)->usize{
        (0..self.num_classes).map(|lbl|sdr.iter().map(|&i|self.prob(i,lbl)).sum::<f32>()).position_max_by(|&a,&b|if a>b{Ordering::Greater}else{Ordering::Less}).unwrap()
    }
    pub fn batch_classify(&self,sdr:&CpuSdrDataset)->Vec<u32>{
        assert_eq!(sdr.shape(),self.shape());
        sdr.data.par_iter().map(|sdr|self.classify(sdr) as u32).collect()
    }
}

pub struct NaiveBayesClassifier {
    occurrences: Vec<f32>,
    num_classes: usize,
    volume: usize,
    //cached volume of input shape
    in_shape: [Idx; 3],
}

//
// impl NaiveBayesClassifier{
//     pub fn occurrences_of(&self,input_idx:Idx,label:usize)->u32{
//         self.occurrences[label*self.volume+input_idx]
//     }
//     pub fn classify(&self, sdr:&CpuSDR)->usize{
//         for &i in sdr.iter(){
//             for label in 0..self.num_classes {
//                 self.occurrences_of(i, )
//             }
//         }
//     }
// }

#[cfg(test)]
mod tests {}
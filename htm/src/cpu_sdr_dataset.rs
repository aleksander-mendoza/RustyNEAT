use ocl::{ProQue, Error, SpatialDims, flags, Platform, Device, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, Deref, DerefMut};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use ndalgebra::buffer::Buffer;
use crate::ecc_program::EccProgram;
use crate::{CpuBitset, EncoderTarget, Shape, Idx, as_idx, as_usize, OclSDR, range_contains, VectorFieldSub, VectorFieldPartialOrd, VectorFieldRem, VectorFieldAdd, CpuSDR, ConvShape, CpuEccDense, DenseWeight, ConvWeights, CpuEccPopulation, VectorFieldRng, Shape2, Shape3, VectorFieldOne, EccLayer, from_xyz, from_xy, ShapedArray, CpuEccMachine, ConvShapeTrait, ConvWeightVec, HasConvShape, Metric, D};
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
use crate::as_usize::AsUsize;

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
    pub fn filter_by_cardinality_threshold(&mut self, min_cardinality: Idx) {
        self.data.retain(|sdr| sdr.cardinality() >= min_cardinality)
    }
    pub fn shape(&self) -> &[Idx; 3] {
        &self.shape
    }
    pub fn subdataset(&self, range: Range<usize>) -> Self {
        let sub = self.data[range].to_vec();
        Self { data: sub, shape: self.shape }
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
    pub fn conv_subregion_indices_with_ecc<M:Metric<D>>(&self, ecc: &CpuEccDense<M>, indices: &SubregionIndices) -> Self {
        assert_eq!(ecc.cshape().out_grid(), &[1, 1], "EccDense should have only a single output column");
        self.conv_subregion_indices_with_ker(*ecc.cshape().kernel(), *ecc.cshape().stride(), indices)
    }
    pub fn conv_subregion_indices_with_machine<M:Metric<D>>(&self, ecc: &CpuEccMachine<M>, indices: &SubregionIndices) -> Self {
        assert_eq!(ecc.out_grid(), Some(&[1, 1]), "EccMachine should have only a single output column");
        let (k, s) = ecc.composed_kernel_and_stride();
        self.conv_subregion_indices_with_ker(k, s, indices)
    }
    pub fn conv_subregion_indices_with_ker(&self, kernel: [Idx; 2], stride: [Idx; 2], indices: &SubregionIndices) -> Self {
        let conv = ConvShape::new_in(self.shape, 1, kernel, stride);
        self.conv_subregion_indices(&conv, indices)
    }
    pub fn conv_subregion_indices(&self, conv: &ConvShape, indices: &SubregionIndices) -> Self {
        let mut s = Self::new(conv.kernel_column());
        s.extend_from_conv_subregion_indices(conv, indices, self);
        s
    }
    pub fn extend_from_conv_subregion_indices(&mut self, conv: &ConvShape, indices: &SubregionIndices, original: &CpuSdrDataset) {
        assert!(!original.is_empty() || indices.is_empty());
        assert_eq!(original.shape(), conv.in_shape(), "Original shape must be equal convolution input shape");
        assert_eq!(self.shape(), &conv.kernel_column(), "Patch shape must be equal kernel column");
        assert_eq!(indices.shape(), conv.out_grid(), "Indices refer to different shape");
        assert_eq!(original.len(), indices.dataset_len, "Dataset lengths don't match");
        for i in &indices.indices {
            let sdr = &original[i.channels().as_usize()];
            let sdr = sdr.conv_subregion(conv, i.grid());
            self.push(sdr);
        }
    }
    pub fn extend_from_rand_subregions(&mut self, number_of_samples:usize, original: &CpuSdrDataset,rng: &mut impl Rng) {
        assert!(!original.is_empty());
        assert_eq!(self.shape.channels(),original.shape.channels(),"Number of channels doesn't match");
        assert!(self.shape.grid().all_le(original.shape.grid()),"Patch size is larger than original size");
        for i in 0..number_of_samples{
            let sdr = original.rand(rng).unwrap();
            let sdr = sdr.rand_subregion(original.shape(),self.shape(),rng);
            self.push(sdr);
        }
    }
    pub fn train<M:Metric<D>>(&self, decrement_activities:bool, number_of_samples: usize, ecc: &mut CpuEccDense<M>, rng: &mut impl Rng, progress_callback: impl Fn(usize)) {
        assert_eq!(self.shape(), ecc.cshape().in_shape(), "Shapes don't match");
        assert_ne!(self.len(), 0, "Dataset is empty");
        for i in 0..number_of_samples {
            let sample_idx = rng.gen_range(0..self.len());
            let sample = &self[sample_idx];
            let out = ecc.infer(sample);
            if decrement_activities {
                ecc.decrement_activities(&out)
            }
            ecc.learn(sample, &out);
            progress_callback(i);
        }
    }
    /**Setting drift==[1,1] will disable drift. Setting it to [n,n] will mean that patch can randomly jiggle within an area of n-by-n output columns*/
    pub fn train_with_patches<M:Metric<D>>(&self, decrement_activities:bool,number_of_samples: usize, drift: [Idx; 2], patches_per_sample: usize, ecc: &mut CpuEccDense<M>, rng: &mut impl Rng, progress_callback: impl Fn(usize)) {
        assert_eq!(ecc.cshape().out_grid(), &[1, 1], "The ecc network should consist of only a single column");
        assert_eq!(self.shape().channels(), ecc.cshape().in_channels(), "Channels don't match");
        assert_ne!(self.len(), 0, "Dataset is empty");
        assert!(drift.all_gt_scalar(0), "Drift can't be 0");
        let conv = ConvShape::new_in(*self.shape(), ecc.cshape().out_channels(), *ecc.cshape().kernel(), *ecc.cshape().stride());
        assert!(conv.out_grid().all_ge(&drift), "Drift {:?} is larger than output grid {:?}", drift, conv.out_grid());
        for i in 0..number_of_samples {
            let idx = rng.gen_range(0..self.len());
            let sample = &self[idx];
            let pos = conv.out_grid().sub(&drift.sub_scalar(1)).rand_vec(rng);
            let mut out = CpuSDR::new();
            for _ in 0..patches_per_sample {
                let drifted_pos = pos.add(&drift.rand_vec(rng));
                let patch = sample.conv_subregion(&conv, &drifted_pos);
                if out.is_empty() {
                    ecc.infer_in_place(&patch, &mut out);
                }
                ecc.learn(&patch, &out);
            }
            if decrement_activities {
                ecc.decrement_activities(&out);
            }
            progress_callback(i);
        }
    }
    pub fn train_machine_with_patches<M:Metric<D>>(&self, decrement_activities:bool, number_of_samples: usize, ecc: &mut CpuEccMachine<M>, rng: &mut impl Rng, progress_callback: impl Fn(usize)) {
        assert_eq!(ecc.out_grid(), Some(&[1, 1]), "The ecc network should consist of only a single column");
        assert_eq!(Some(self.shape().channels()), ecc.in_channels(), "Channels don't match");
        assert_ne!(self.len(), 0, "Dataset is empty");
        for i in 0..number_of_samples {
            let idx = rng.gen_range(0..self.len());
            let sample = &self[idx];
            let patch = sample.rand_subregion(self.shape(), ecc.in_shape().unwrap(), rng);
            ecc.infer(&patch);
            if decrement_activities {
                ecc.decrement_activities()
            }
            ecc.learn();
            progress_callback(i);
        }
    }
    pub fn count(&self) -> Vec<u32> {
        let mut counts = vec![0; self.shape().product().as_usize()];
        for i in &self.data {
            for &i in i.iter() {
                counts[i.as_usize()] += 1
            }
        }
        counts
    }
    /**The shape of receptive fields is [h,w,c,o] where [h,w,c] is the shape of input SDRs (this dataset)
    and o is the volume of output shape (outputs dataset)*/
    pub fn measure_receptive_fields(&self, outputs: &Self) -> Vec<u32> {
        assert_eq!(self.len(), outputs.len());
        let ov = outputs.shape().size();
        let iv = self.shape().size();
        let receptive_fields_shape = [iv, ov];
        let mut receptive_fields = vec![0; (ov * iv).as_usize()];
        for (o_sdr, i_sdr) in outputs.iter().zip(self.iter()) {
            for &o in o_sdr.iter() {
                for &i in i_sdr.iter() {
                    assert!(i<iv,"Input idx is {} but input shape {:?} has volume {}",i,self.shape(),iv);
                    assert!(o<ov,"Output idx is {} but output shape {:?} has volume {}",o,outputs.shape(),ov);
                    let idx = receptive_fields_shape.idx([i, o]);
                    receptive_fields[idx.as_usize()] += 1;
                }
            }
        }
        receptive_fields
    }
    pub fn batch_infer<M:Metric<D>>(&self, ecc: &CpuEccDense<M>) -> Self {
        assert_eq!(self.shape(), ecc.cshape().in_shape());
        let data = ecc.batch_infer(&self.data, |d| d, |o| o);
        Self {
            data,
            shape: ecc.cshape().output_shape(),
        }
    }
    pub fn machine_infer<M:Metric<D>>(&self, ecc: &mut CpuEccMachine<M>) -> Self {
        assert_eq!(Some(self.shape()), ecc.in_shape());
        let mut data = self.data.iter().map(|s|ecc.infer(s).clone()).collect();
        Self {
            data,
            shape: *ecc.out_shape().unwrap(),
        }
    }
    pub fn batch_infer_and_measure_s_expectation<M:Metric<D>>(&self, ecc: &CpuEccDense<M>) -> (Self, D, u32) {
        assert_eq!(self.shape(), ecc.cshape().in_shape());
        let (data, s_exp, missed) = ecc.batch_infer_and_measure_s_expectation(&self.data, |d| d, |o| o);
        (Self {
            data,
            shape: ecc.cshape().output_shape(),
        }, s_exp, missed)
    }
    pub fn batch_infer_conv_weights<M:Metric<D>>(&self, ecc: &ConvWeights<M>, target: CpuEccPopulation<M>) -> Self {
        assert_eq!(self.shape(), ecc.cshape().in_shape());
        let data = ecc.batch_infer(&self.data, |d| d, target, |o| o);
        Self {
            data,
            shape: ecc.cshape().output_shape(),
        }
    }
    pub fn batch_infer_conv_weights_and_measure_s_expectation<M:Metric<D>>(&self, ecc: &ConvWeights<M>, target: CpuEccPopulation<M>) -> (Self, D, u32) {
        assert_eq!(self.shape(), ecc.cshape().in_shape());
        let (data, s_exp, missed) = ecc.batch_infer_and_measure_s_expectation(&self.data, |d| d, target, |o| o);
        (Self {
            data,
            shape: ecc.cshape().output_shape(),
        }, s_exp, missed)
    }

    /**set num_of_classes==0 in order to automatically detect it using the maximum value found in labels array*/
    pub fn count_per_label<T>(&self, labels: &[T], num_of_classes: usize, f: impl Fn(&T) -> usize) -> Occurrences {
        assert_eq!(labels.len(), self.len());
        let volume = self.shape().size().as_usize();
        let mut occurrences = vec![0f32; num_of_classes * volume];
        for (sdr, label) in self.iter().zip(labels.iter().map(f)) {
            if label < num_of_classes {
                let offset = label * volume;
                let occurrences_for_label = &mut occurrences[offset..offset + volume];
                sdr.iter().cloned().for_each(|i| occurrences_for_label[i.as_usize()] += 1.);
            }
        }
        Occurrences {
            occurrences,
            class_prob: vec![0.; num_of_classes],
            num_classes: num_of_classes,
            volume,
            in_shape: self.shape,
        }
    }
    pub fn gen_rand_2d_patches(&self,patch_size:[Idx;2],number_of_samples:usize,rng:&mut impl Rng)->Self{
        self.gen_rand_subregions(patch_size.add_channels(self.shape.channels()),number_of_samples,rng)
    }
    pub fn gen_rand_subregions(&self,subregion:[Idx;3],number_of_samples:usize,rng:&mut impl Rng)->Self{
        assert!(subregion.all_le(self.shape()),"Patch size is larger than original");
        let mut d = Self::with_capacity(number_of_samples, subregion);
        d.extend_from_rand_subregions(number_of_samples,self,rng);
        d
    }
    pub fn gen_rand_conv_subregion_indices(&self, out_shape: [Idx; 2], number_of_samples: usize, rng: &mut impl Rng) -> SubregionIndices {
        SubregionIndices::gen_rand_conv_subregion_indices(self.len(), out_shape, number_of_samples, rng)
    }
    pub fn gen_rand_conv_subregion_indices_with_ker(&self, kernel: &[Idx; 2], stride: &[Idx; 2], number_of_samples: usize, rng: &mut impl Rng) -> SubregionIndices {
        self.gen_rand_conv_subregion_indices(self.shape().grid().conv_out_size(stride, kernel), number_of_samples, rng)
    }
    pub fn gen_rand_conv_subregion_indices_with_ecc<M:Metric<D>>(&self, ecc: &CpuEccDense<M>, number_of_samples: usize, rng: &mut impl Rng) -> SubregionIndices {
        assert_eq!(ecc.out_grid(), &[1, 1], "EccDense should have only single output column");
        self.gen_rand_conv_subregion_indices_with_ker(ecc.kernel(), ecc.stride(), number_of_samples, rng)
    }
    pub fn gen_rand_conv_subregion_indices_with_machine<M:Metric<D>>(&self, ecc: &CpuEccMachine<M>, number_of_samples: usize, rng: &mut impl Rng) -> SubregionIndices {
        assert_eq!(ecc.out_grid(), Some(&[1, 1]), "EccMachine should have only single output column");
        let (k, s) = ecc.composed_kernel_and_stride();
        self.gen_rand_conv_subregion_indices_with_ker(&k, &s, number_of_samples, rng)
    }
    // pub fn fit_naive_bayes<T: Into<usize>>(&self, labels: &[T]) -> NaiveBayesClassifier {
    //
    // }
    pub fn fit_naive_bayes<T>(&self, labels: &[T], num_of_classes: usize, invariant_to_column: bool, f: impl Fn(&T) -> usize) -> Occurrences {
        let mut occurrences = self.count_per_label(labels, num_of_classes, f);
        occurrences.compute_class_probs();
        if invariant_to_column {
            occurrences.aggregate_invariant_to_column();
            #[cfg(debug_assertions)] {
                let probs = occurrences.class_prob.clone();
                occurrences.compute_class_probs();
                debug_assert_eq!(probs, occurrences.class_prob);
            }
        }
        occurrences.normalise_wrt_labels();
        occurrences.log();
        occurrences
    }
}

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct SubregionIndices {
    dataset_len: usize,
    indices: Vec<[Idx; 3]>,
    shape: [Idx; 2],
}

impl Deref for SubregionIndices {
    type Target = Vec<[Idx; 3]>;

    fn deref(&self) -> &Self::Target {
        &self.indices
    }
}

impl DerefMut for SubregionIndices {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.indices
    }
}

impl SubregionIndices {
    pub fn shape(&self) -> &[Idx; 2] {
        &self.shape
    }
    pub fn dataset_len(&self) -> usize {
        self.dataset_len
    }
    pub fn gen_rand_conv_subregion_indices(dataset_len: usize, out_shape: [Idx; 2], number_of_samples: usize, rng: &mut impl Rng) -> SubregionIndices {
        Self {
            dataset_len,
            indices: (0..number_of_samples).map(|_| {
                let sdr_idx = as_idx(rng.gen_range(0..dataset_len));
                let pos = out_shape.rand_vec(rng);
                pos.add_channels(sdr_idx)
            }).collect(),
            shape: out_shape,
        }
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Occurrences {
    occurrences: Vec<f32>,
    class_prob: Vec<f32>,
    num_classes: usize,
    /**cached volume of input shape*/
    volume: usize,
    in_shape: [Idx; 3],
}

impl Occurrences {
    pub fn compute_class_sums(&mut self) {
        let Self { occurrences, class_prob, volume, .. } = self;
        let v = *volume;
        for (lbl, prob) in class_prob.iter_mut().enumerate() {
            *prob = (0..v).map(|i| occurrences[Self::prob_idx_(i, v, lbl)]).sum();
        }
    }
    pub fn class_prob(&self) -> &[f32] {
        &self.class_prob
    }
    pub fn normalise_class_probs(&mut self) {
        let sum: f32 = self.class_prob.iter().sum();
        self.class_prob.iter_mut().for_each(|p| *p /= sum);
    }
    pub fn compute_class_probs(&mut self) {
        self.compute_class_sums();
        self.normalise_class_probs();
    }
    pub fn sum_over_labels(&self, input_idx: usize) -> f32 {
        (0..self.num_classes).map(|lbl| self.prob(input_idx, lbl)).sum()
    }
    pub fn normalise_wrt_labels(&mut self) {
        for i in 0..self.volume {
            let s: f32 = self.sum_over_labels(i);
            (0..self.num_classes).for_each(|lbl| *self.prob_mut(i, lbl) /= s)
        }
    }

    /** This can be used to make all columns vote on the class, but each column does so
    independently from others. If you don't use this method then by default the naive bayes
    takes all columns into account and may try to look for specific features at specific locations.
    When you use this method, then every column looks only at features in its own specific location
    and tries to guess the class.*/
    pub fn aggregate_invariant_to_column(&mut self) {
        for channel in 0..self.shape().channels() {
            for lbl in 0..self.num_classes {
                let mut sum = 0.;
                for x in 0..self.shape().width() {
                    for y in 0..self.shape().height() {
                        let i = self.shape().idx(from_xyz(x, y, channel)).as_usize();
                        sum += self.prob(i, lbl);
                    }
                }
                for x in 0..self.shape().width() {
                    for y in 0..self.shape().height() {
                        let i = self.shape().idx(from_xyz(x, y, channel)).as_usize();
                        *self.prob_mut(i, lbl) = sum;
                    }
                }
            }
        }
    }
    pub fn prob_idx_(input_idx: usize, volume: usize, label: usize) -> usize {
        label * volume + input_idx
    }
    pub fn prob_idx(&self, input_idx: usize, label: usize) -> usize {
        debug_assert_eq!(self.occurrences.len(), self.num_classes * self.volume);
        debug_assert_eq!(self.shape().product().as_usize(), self.volume);
        debug_assert!(input_idx < self.volume);
        debug_assert!(label < self.num_classes);
        Self::prob_idx_(input_idx, self.volume, label)
    }
    pub fn clear_class_prob(&mut self) {
        self.class_prob.fill(0.)
    }
    pub fn prob(&self, input_idx: usize, label: usize) -> f32 {
        self.occurrences[self.prob_idx(input_idx, label)]
    }
    pub fn prob_mut(&mut self, input_idx: usize, label: usize) -> &mut f32 {
        let i = self.prob_idx(input_idx, label);
        &mut self.occurrences[i]
    }
    pub fn occurrences(&self) -> &[f32] {
        &self.occurrences
    }
    pub fn occurrences_for_label(&self, lbl: usize) -> &[f32] {
        let offset = self.volume * lbl;
        &self.occurrences[offset..offset + self.volume]
    }
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
    pub fn shape(&self) -> &[Idx; 3] {
        &self.in_shape
    }
    pub fn square_weights(&mut self) {
        self.occurrences.iter_mut().for_each(|w| *w = *w * *w)
    }
    pub fn sqrt_weights(&mut self) {
        self.occurrences.iter_mut().for_each(|w| *w = w.sqrt())
    }
    pub fn exp_weights(&mut self) {
        self.occurrences.iter_mut().for_each(|w| *w = w.exp2())
    }
    pub fn log_weights(&mut self) {
        self.occurrences.iter_mut().for_each(|w| *w = w.log2())
    }
    pub fn square_class_probs(&mut self) {
        self.class_prob.iter_mut().for_each(|w| *w = *w * *w)
    }
    pub fn sqrt_class_probs(&mut self) {
        self.class_prob.iter_mut().for_each(|w| *w = w.sqrt())
    }
    pub fn exp_class_probs(&mut self) {
        self.class_prob.iter_mut().for_each(|w| *w = w.exp2())
    }
    pub fn log_class_probs(&mut self) {
        self.class_prob.iter_mut().for_each(|w| *w = w.log2())
    }
    pub fn log(&mut self) {
        self.log_weights();
        self.log_class_probs();
    }
    pub fn collect_votes_per_column_and_lbl(&self, sdr: &CpuSDR) -> ShapedArray<f32> {
        let area = self.shape().grid().product().as_usize();
        let mut votes = vec![0.; self.num_classes * area];
        let votes_shape = self.shape().grid().add_channels(as_idx(self.num_classes));
        for i in 0..area {
            let offset = i * self.num_classes;
            votes[offset..offset + self.num_classes].clone_from_slice(&self.class_prob);
        }
        #[cfg(debug_assertions)] {
            for x in 0..self.shape().width() {
                for y in 0..self.shape().height() {
                    for lbl in 0..as_idx(self.num_classes) {
                        let i = votes_shape.idx(from_xyz(x, y, lbl)).as_usize();
                        debug_assert_eq!(votes[i], self.class_prob[lbl.as_usize()]);
                    }
                }
            }
        }
        for lbl in 0..self.num_classes {
            for &i in sdr.iter() {
                let prob = self.prob(i.as_usize(), lbl);
                let mut pos = self.shape().pos(i);
                *pos.channels_mut() = as_idx(lbl);
                votes[votes_shape.idx(pos).as_usize()] += prob;
            }
        }
        ShapedArray::from(votes_shape, votes)
    }
    pub fn classify_per_column(&self, sdr: &CpuSDR, min_deviation_from_mean: f32) -> ShapedArray<isize> {
        let area = self.shape().area().as_usize();
        let votes = self.collect_votes_per_column_and_lbl(sdr);
        let votes = (0..area).map(|i| {
            self.get_vote(i, &votes, min_deviation_from_mean).map(|a| a as isize).unwrap_or(-1)
        }).collect();
        ShapedArray::from(self.shape().grid().add_channels(1), votes)
    }
    fn get_vote(&self, i: usize, votes: &ShapedArray<f32>, min_deviation_from_mean: f32) -> Option<usize> {
        let offset = i * self.num_classes;
        let slice = &votes[offset..offset + self.num_classes];
        let (max_idx, max_score) = slice.iter().cloned().enumerate().max_by(|(_, a), (_, b)| a.cmp_naive(b)).unwrap();
        if min_deviation_from_mean > 0. {
            let mean = slice.iter().sum::<f32>() / self.num_classes as f32;
            let deviation = (mean - max_score).abs();
            if deviation < min_deviation_from_mean {
                return None;
            }
        }
        Some(max_idx)
    }
    /**Sometimes te column might be undecided (like in the case when no neuron has been activated and all classes get score 0).
    Then we might want to ignore the vote of such column (if all scores are 0 then the max is 0 and the vote is always cast on the first class by default).
    Use min_deviation_from_mean to set a minimum threshold by which scores must deviate from the mean in order to count as a vote.
    (so if all scores are 0 then mean is 0 and 0-0=0, so any min_deviation_from_mean>0 will filter out such scenario)*/
    pub fn classify_and_count_votes_per_lbl(&self, sdr: &CpuSDR, min_deviation_from_mean: f32) -> Vec<u32> {
        let area = self.shape().area().as_usize();
        let votes = self.collect_votes_per_column_and_lbl(sdr);
        let mut votes_per_lbl = vec![0; self.num_classes];
        for i in 0..area {
            if let Some(max_idx) = self.get_vote(i, &votes, min_deviation_from_mean) {
                votes_per_lbl[max_idx] += 1;
            }
        }
        votes_per_lbl
    }
    pub fn classify_with_most_votes(&self, sdr: &CpuSDR, min_deviation_from_mean: f32) -> usize {
        self.classify_and_count_votes_per_lbl(sdr, min_deviation_from_mean).iter().position_max().unwrap()
    }
    pub fn classify(&self, sdr: &CpuSDR) -> usize {
        (0..self.num_classes).map(|lbl| self.class_prob[lbl] + sdr.iter().map(|&i| self.prob(i.as_usize(), lbl)).sum::<f32>()).position_max_by(D::cmp_naive).unwrap()
    }
    pub fn batch_classify(&self, sdr: &CpuSdrDataset) -> Vec<u32> {
        assert_eq!(sdr.shape(), self.shape());
        sdr.data.par_iter().map(|sdr| self.classify(sdr) as u32).collect()
    }
    pub fn batch_classify_invariant_to_column(&self, sdr: &CpuSdrDataset, min_deviation_from_mean: f32) -> Vec<u32> {
        assert_eq!(sdr.shape(), self.shape());
        sdr.data.par_iter().map(|sdr| as_idx(self.classify_with_most_votes(sdr, min_deviation_from_mean))).collect()
    }
}


#[cfg(test)]
mod tests {
    use crate::{CpuSdrDataset, CpuSDR, as_usize, VectorFieldOne, Idx, AsUsize};
    use rand::random;

    #[test]
    fn test1() {
        let i = test(32, 2, [4, 3, 2], 2, 16);
        assert!(i <= 4, "{}", i);
    }

    fn test(samples: usize, card: Idx, shape: [Idx; 3], classes: Idx, tests: usize) -> u32 {
        let mut d = CpuSdrDataset::new(shape);
        let size = d.shape().product();
        for _ in 0..samples {
            d.push(CpuSDR::rand(card, size));
        }
        let lbls: Vec<u32> = (0..d.len()).map(|_| random::<u32>() % classes).collect();
        let mut n = d.fit_naive_bayes(&lbls, classes.as_usize(), true, |&a| a.as_usize());
        n.clear_class_prob();
        let mut incorrent = 0;
        for j in 0..tests {
            let i = random::<usize>() % d.len();
            let sdr = &d[i];
            let m = 0.01;
            let lbl = n.classify_with_most_votes(sdr, m);
            println!("{}:{:?}, {} == {} -> {:?} \n {:?} \n {:?}", i, sdr, lbl, lbls[i],
                     n.classify_and_count_votes_per_lbl(sdr, m),
                     n.collect_votes_per_column_and_lbl(sdr),
                     n.classify_per_column(sdr, m));
            if lbl != lbls[i].as_usize() {
                incorrent += 1
            }
        }
        incorrent
    }
}
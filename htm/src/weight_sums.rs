use crate::{DenseWeight, Idx, CpuSDR, as_idx, as_usize, VectorFieldOne};
use rayon::prelude::*;
use std::ops::{Index, IndexMut, Deref, DerefMut};
use serde::{Serialize, Deserialize};
use std::slice::{Iter, IterMut};

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct WeightSums<D: DenseWeight> {
    sums: Vec<D>,
    shape: [Idx; 3],
}

impl<D: DenseWeight> WeightSums<D> {
    pub fn shape(&self) -> &[Idx; 3] {
        &self.shape
    }
    pub fn as_slice(&self) -> &[D] {
        &self.sums
    }
    pub fn new(shape: [Idx; 3], initial_value: D) -> Self {
        Self {
            sums: vec![initial_value; as_usize(shape.product())],
            shape,
        }
    }
    pub fn fill(&mut self, value: D, sdr: &CpuSDR) {
        for &i in sdr.as_slice() {
            self[i] = value
        }
    }
    pub fn fill_all(&mut self, value: D) {
        self.sums.iter_mut().for_each(|a| *a = value)
    }
    pub fn len(&self) -> usize {
        self.sums.len()
    }
    pub fn iter(&self) -> Iter<D> {
        self.sums.iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<D> {
        self.sums.iter_mut()
    }
    pub fn as_mut_slice(&mut self) -> &mut [D] {
        self.sums.as_mut_slice()
    }
    pub fn as_mut_ptr(&mut self) -> *mut D {
        self.sums.as_mut_ptr()
    }
    pub fn as_ptr(&self) -> *const D {
        self.sums.as_ptr()
    }
}

impl<'a, D: DenseWeight + Send + Sync> WeightSums<D> {
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<D> {
        self.sums.par_iter_mut()
    }
    pub fn par_iter(&self) -> rayon::slice::Iter<D> {
        self.sums.par_iter()
    }
    pub fn parallel_fill(&mut self, value: D, sdr: &CpuSDR) {
        let sums_len = self.sums.len();
        let sums_ptr = self.sums.as_mut_ptr() as usize;
        sdr.par_iter().for_each(|&output_idx| {
            let sums_slice = unsafe { std::slice::from_raw_parts_mut(sums_ptr as *mut D, sums_len) };
            sums_slice[as_usize(output_idx)] = value
        })
    }
    pub fn parallel_fill_all(&mut self, value: D) {
        self.sums.par_iter_mut().for_each(|a| *a = value)
    }
}

impl<D: DenseWeight> Deref for WeightSums<D> {
    type Target = [Idx; 3];

    fn deref(&self) -> &Self::Target {
        &self.shape
    }
}

impl<D: DenseWeight> Index<usize> for WeightSums<D> {
    type Output = D;

    fn index(&self, index: usize) -> &Self::Output {
        &self.sums[index]
    }
}

impl<D: DenseWeight> Index<Idx> for WeightSums<D> {
    type Output = D;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.sums[as_usize(index)]
    }
}

impl<D: DenseWeight> IndexMut<usize> for WeightSums<D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.sums[index]
    }
}

impl<D: DenseWeight> IndexMut<Idx> for WeightSums<D> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.sums[as_usize(index)]
    }
}
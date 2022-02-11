use ocl::{ProQue, Error, SpatialDims, flags, Platform, Device, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, Deref, DerefMut};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use ndalgebra::buffer::Buffer;
use crate::ecc_program::EccProgram;
use crate::{CpuBitset, EncoderTarget, Shape, Idx, as_idx, as_usize, OclSDR, range_contains, VectorFieldSub, VectorFieldPartialOrd, VectorFieldRem, VectorFieldAdd, ConvShape, Shape3, Shape2, VectorFieldRng};
use std::collections::{HashMap, HashSet};
use std::borrow::Borrow;
use serde::{Serialize, Deserialize};
use crate::vector_field::{VectorField, VectorFieldMul};
use crate::sdr::SDR;
use rand::Rng;
use rayon::iter::{IntoParallelRefMutIterator, IntoParallelRefIterator, ParallelIterator};

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct CpuSDR(Vec<Idx>);

impl PartialEq<Vec<Idx>> for CpuSDR {
    fn eq(&self, other: &Vec<Idx>) -> bool {
        self.0.eq(other)
    }
}

impl From<&[Idx]> for CpuSDR {
    fn from(v: &[Idx]) -> Self {
        Self(Vec::from(v))
    }
}

impl From<Vec<u32>> for CpuSDR {
    fn from(v: Vec<Idx>) -> Self {
        Self(v)
    }
}

impl Deref for CpuSDR {
    type Target = [Idx];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl DerefMut for CpuSDR {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut_slice()
    }
}

impl Debug for CpuSDR {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl EncoderTarget for CpuSDR {
    fn push(&mut self, neuron_index: Idx) {
        self.0.push(neuron_index)
    }

    fn clear_range(&mut self, from: Idx, to: Idx) {
        self.0.retain(|&i| i >= to || i < from)
    }

    fn contains(&self, neuron_index: Idx) -> bool {
        debug_assert!(self.is_normalized());
        self.binary_search(neuron_index)
    }
}

impl SDR for CpuSDR {
    fn clear(&mut self) {
        self.0.clear()
    }
    fn item(&self) -> Idx {
        assert_eq!(self.len(), 1, "The SDR is not a singleton");
        self.0[0]
    }
    fn cardinality(&self) -> Idx {
        as_idx(self.0.len())
    }
    fn set_from_slice(&mut self, active_neurons: &[Idx]) {
        self.clear();
        self.0.extend_from_slice(active_neurons)
    }

    fn set_from_sdr(&mut self, other: &Self) {
        self.set_from_slice(other.as_slice())
    }

    fn to_vec(&self) -> Vec<Idx> {
        self.0.clone()
    }
    fn into_vec(self) -> Vec<Idx> {
        let Self(v) = self;
        v
    }
}

impl CpuSDR {
    pub fn from_slice(s: &[Idx]) -> Self {
        Self::from(s)
    }
    pub fn as_slice(&self) -> &[Idx] {
        self.0.as_slice()
    }
    pub fn to_ocl(&self, prog:EccProgram, max_cardinality:Idx) -> Result<OclSDR, Error> {
        OclSDR::from_cpu(prog,self,max_cardinality)
    }
    pub fn shift(&mut self, shift: i32) {
        for i in &mut self.0 {
            let new_i = *i as i32 + shift;
            if new_i < 0 { panic!("Shifting neuron {} by {} produces negative index {}", *i, shift, new_i) }
            *i = new_i as u32;
        }
    }
    pub fn swap_remove(&mut self, idx: usize) -> Idx {
        self.0.swap_remove(idx)
    }
    pub fn remove(&mut self, idx: usize) -> Idx {
        self.0.remove(idx)
    }
    pub fn as_mut_slice(&mut self) -> &mut [Idx] {
        self.0.as_mut_slice()
    }


    pub fn retain(&mut self, mut predicate: impl FnMut(Idx) -> bool) {
        self.0.retain(|&x| predicate(x))
    }

    /**SDR ust be sparse. Hence we might as well put a cap on the maximum number of active neurons.
    If your system is designed correctly, then you should never have to worry about exceeding this limit.*/
    pub fn max_active_neurons(&self) -> usize {
        self.0.capacity()
    }
    pub fn new() -> Self {
        Self(Vec::new())
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }
    /**Sorts and removes duplicates*/
    pub fn normalize(&mut self) {
        self.0.sort();
        self.0.dedup();
    }
    /**This method requires that both SDRs are first normalized*/
    pub fn overlap(&self, other: &CpuSDR) -> Idx {
        if self.is_empty() || other.is_empty() { return 0; }
        let mut i1 = 0;
        let mut i2 = 0;
        let mut overlap = 0;
        let (s1, s2) = if self.0[0] < other.0[0] { (self, other) } else { (other, self) };
        loop {
            while s1.0[i1] < s2.0[i2] {
                i1 += 1;
                if i1 >= s1.len() { return overlap; }
            }
            if s1.0[i1] == s2.0[i2] {
                overlap += 1;
                i1 += 1;
                if i1 >= s1.len() { return overlap; }
            }
            while s1.0[i1] > s2.0[i2] {
                i2 += 1;
                if i2 >= s2.len() { return overlap; }
            }
            if s1.0[i1] == s2.0[i2] {
                overlap += 1;
                i2 += 1;
                if i2 >= s2.len() { return overlap; }
            }
        }
    }
    pub fn binary_search(&self, neuron_index: u32) -> bool {
        self.0.binary_search(&neuron_index).is_ok()
    }
    pub fn sorted(mut self) -> Self {
        self.sort();
        self
    }
    pub fn is_normalized(&self) -> bool {
        if self.0.is_empty() { return true; }
        let mut prev = self.0[0];
        for &i in &self.0[1..] {
            if i <= prev { return false; }
            prev = i;
        }
        true
    }
    pub fn extend(&mut self, other: &CpuSDR) {
        self.0.extend_from_slice(other)
    }
    pub fn extend_from_iter(&mut self, other: impl IntoIterator<Item=Idx>) {
        self.0.extend(other)
    }
    pub fn extend_from_slice(&mut self, other: &[Idx]) {
        self.0.extend_from_slice(other)
    }
    /**Requires that both SDRs are normalized. The resulting SDR is already in normalized form*/
    pub fn intersection(&self, other: &CpuSDR) -> CpuSDR {
        let mut intersection = CpuSDR::with_capacity(self.len() + other.len());
        let mut i = 0;
        if other.is_empty() { return intersection; }
        for &neuron_index in &self.0 {
            while other[i] < neuron_index {
                i += 1;
                if i >= other.len() { return intersection; }
            }
            if other[i] == neuron_index {
                intersection.push(neuron_index);
            }
        }
        intersection
    }
    pub fn union(&self, other: &CpuSDR) -> CpuSDR {
        let mut union = CpuSDR::with_capacity(self.len() + other.len());
        let mut i1 = 0;
        let mut i2 = 0;
        if self.0.len() > 0 && other.0.len() > 0 {
            'outer: loop {
                while self.0[i1] < other.0[i2] {
                    union.0.push(self.0[i1]);
                    i1 += 1;
                    if i1 >= self.len() { break 'outer; }
                }
                if self.0[i1] == other.0[i2] {
                    union.0.push(self.0[i1]);
                    i1 += 1;
                    i2 += 1;
                    if i1 >= self.len() || i2 >= other.len() { break 'outer; }
                }
                while self.0[i1] > other.0[i2] {
                    union.0.push(other.0[i2]);
                    i2 += 1;
                    if i2 >= other.len() { break 'outer; }
                }
                if self.0[i1] == other.0[i2] {
                    union.0.push(other.0[i2]);
                    i1 += 1;
                    i2 += 1;
                    if i1 >= self.len() || i2 >= other.len() { break 'outer; }
                }
            }
        }
        if i1 < self.0.len() {
            union.0.extend_from_slice(&self.0[i1..])
        } else {
            union.0.extend_from_slice(&other.0[i2..])
        }
        union
    }
    pub fn subtract(&mut self, other: &CpuSDR) {
        let mut i1 = 0;
        let mut i2 = 0;
        let mut j = 0;
        if self.0.len() > 0 && other.0.len() > 0 {
            'outer: loop {
                while self.0[i1] < other.0[i2] {
                    self.0[j] = self.0[i1];
                    j += 1;
                    i1 += 1;
                    if i1 >= self.len() { break 'outer; }
                }
                if self.0[i1] == other.0[i2] {
                    i1 += 1;
                    i2 += 1;
                    if i1 >= self.len() || i2 >= other.len() { break 'outer; }
                }
                while self.0[i1] > other.0[i2] {
                    i2 += 1;
                    if i2 >= other.len() { break 'outer; }
                }
                if self.0[i1] == other.0[i2] {
                    i1 += 1;
                    i2 += 1;
                    if i1 >= self.len() || i2 >= other.len() { break 'outer; }
                }
            }
        }
        while i1 < self.0.len() {
            self.0[j] = self.0[i1];
            j += 1;
            i1 += 1;
        }
        self.0.truncate(j);
    }
    pub fn shrink(&mut self, number_of_bits_to_retain: usize) {
        if number_of_bits_to_retain < self.len() {
            unsafe { self.0.set_len(number_of_bits_to_retain) }
        }
    }
    /**Similar to shrink, but removes random bits*/
    pub fn subsample(&mut self, number_of_bits_to_retain: usize) {
        while self.len() > number_of_bits_to_retain {
            self.0.swap_remove(rand::random::<usize>() % self.len());
        }
    }
    pub fn add_unique_random(&mut self, n: u32, range: Range<u32>) {
        let len = range.end - range.start;
        assert!(len >= n, "The range of values {}..{} has {} elements. Can't get unique {} elements out of it!", range.start, range.end, len, n);
        let mut set = HashSet::new();
        for _ in 0..n {
            let mut r = range.start + rand::random::<u32>() % len;
            while !set.insert(r) {
                r += 1;
                if r >= range.end {
                    r = range.start;
                }
            }
            self.0.push(r);
        }
    }
    /**Iterates over all lower-level columns. For each one looks up all the connected higher-level columns and takes the union of their activities.
    This union could then be used for training the lower-level columns. The function returns a 2d array of such unions.*/
    pub fn vote_conv2d_transpose_arr<'a>(stride: [u32; 2], kernel_size: [u32; 2], grid_size: [u32; 2], output_sdr_grid: &'a impl Fn(u32, u32) -> &'a CpuSDR) -> Vec<Vec<CpuSDR>> {
        let out_grid_size = grid_size.conv_out_size(&stride, &kernel_size);
        let mut apical_feedback_input_grid: Vec<Vec<CpuSDR>> = (0..grid_size[0]).map(|_| (0..grid_size[1]).map(|_| CpuSDR::new()).collect()).collect();
        for out0 in 0..out_grid_size[0] {
            for out1 in 0..out_grid_size[1] {
                let out_sdr: &CpuSDR = output_sdr_grid(out0, out1);
                let in_begin = (out0 * stride[0], out1 * stride[1]);
                for in0 in in_begin.0..in_begin.0 + kernel_size[0] {
                    for in1 in in_begin.1..in_begin.1 + kernel_size[1] {
                        let union_sdr = &mut apical_feedback_input_grid[in0 as usize][in1 as usize];
                        let mut new_union = union_sdr.union(out_sdr);
                        std::mem::swap(&mut new_union, union_sdr);
                    }
                }
            }
        }
        apical_feedback_input_grid
    }

    pub fn vote_conv2d_arr<T: Borrow<CpuSDR>>(n: usize, threshold: u32, stride: [u32; 2], kernel_size: [u32; 2], grid_size: [u32; 2], input_sdr_grid: &[impl AsRef<[T]>]) -> Vec<Vec<CpuSDR>> {
        Self::vote_conv2d_arr_with(n, threshold, stride, kernel_size, grid_size, &|c0, c1| input_sdr_grid[c0 as usize].as_ref()[c1 as usize].borrow(), |a| a)
    }
    pub fn vote_conv2d_arr_with<'a, O>(n: usize, threshold: u32, stride: [u32; 2], kernel_size: [u32; 2], grid_size: [u32; 2], input_sdr_grid: &'a impl Fn(u32, u32) -> &'a CpuSDR, out_sdr: impl Fn(CpuSDR) -> O) -> Vec<Vec<O>> {
        let out_grid_size = grid_size.conv_out_size(&stride, &kernel_size);
        let mut out_grid: Vec<Vec<O>> = (0..out_grid_size[0]).map(|_| {
            let mut v = Vec::with_capacity(out_grid_size[1] as usize);
            unsafe { v.set_len(out_grid_size[1] as usize) }
            v
        }).collect();
        Self::vote_conv2d(n, threshold, stride, kernel_size, grid_size, input_sdr_grid, |c0, c1, sdr| {
            let o = &mut out_grid[c0 as usize][c1 as usize];
            let o = o as *mut O;
            unsafe { std::ptr::write(o, out_sdr(sdr)) }
        });
        out_grid
    }
    pub fn vote_conv2d<'a>(n: usize, threshold: u32, stride: [u32; 2], kernel_size: [u32; 2], grid_size: [u32; 2], input_sdr_grid: &'a impl Fn(u32, u32) -> &'a CpuSDR, mut output_sdr_grid: impl FnMut(u32, u32, CpuSDR)) {
        let out_grid_size = grid_size.conv_out_size(&stride, &kernel_size);
        for out0 in 0..out_grid_size[0] {
            for out1 in 0..out_grid_size[1] {
                let in_begin = stride.mul(&[out0, out1]);
                let sdr = Self::vote_over_iter((in_begin[0]..in_begin[0] + kernel_size[0]).flat_map(|in0| (in_begin[1]..in_begin[1] + kernel_size[1]).map(move |in1| input_sdr_grid(in0, in1))), n, threshold);
                output_sdr_grid(out0, out1, sdr);
            }
        }
    }
    pub fn vote<T: Borrow<Self>>(sdrs: &[T], n: usize, threshold: u32) -> Self {
        Self::vote_over_iter(sdrs.iter().map(|c| c.borrow()), n, threshold)
    }
    pub fn vote_over_iter<T: Borrow<Self>>(mut sdrs: impl Iterator<Item=T>, n: usize, threshold: u32) -> Self {
        let mut map = HashMap::<u32, u32>::new();
        for sdr in sdrs {
            let sdr = sdr.borrow();
            for &active_neuron in sdr.as_slice() {
                map.entry(active_neuron).and_modify(|x| *x += 1).or_insert(1);
            }
        }
        let mut n_best = Vec::<(u32, u32)>::with_capacity(n);
        for (&active_neuron, &count) in map.iter() {
            if count >= threshold {
                n_best.push((active_neuron, count));
            }
        }
        n_best.sort_by_key(|&(_, c)| c);
        let mut highest_voted_neurons: Vec<u32> = n_best.iter().rev().map(|&(an, _)| an).take(n as usize).collect();
        highest_voted_neurons.sort();
        Self::from(highest_voted_neurons)
    }
    pub fn rand(cardinality:Idx, size: Idx) -> Self{
        assert!(cardinality<=size);
        let mut s= Self::with_capacity(as_usize(cardinality));
        s.add_unique_random(cardinality,0..size);
        s
    }
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<Idx> {
        self.0.par_iter_mut()
    }
    pub fn par_iter(&self) -> rayon::slice::Iter<Idx> {
        self.0.par_iter()
    }
    pub fn fill_into<D:Copy>(&self, value: D, array: &mut [D]) {
        for &i in self.as_slice() {
            array[as_usize(i)] = value
        }
    }
    pub fn parallel_fill_into<D:Copy+Send+Sync>(&self, value: D, array: &mut[D]) {
        let sums_len = array.len();
        let sums_ptr = array.as_mut_ptr() as usize;
        self.par_iter().for_each(|&output_idx| {
            let sums_slice = unsafe { std::slice::from_raw_parts_mut(sums_ptr as *mut D, sums_len) };
            sums_slice[as_usize(output_idx)] = value
        })
    }
    /**Randomly picks some neurons that a present in other SDR but not in self SDR.
    Requires that both SDRs are already normalized.
    It will only add so many elements so that self.len() <= n*/
    pub fn randomly_extend_from(&mut self, other: &Self, n: usize) {
        debug_assert!(self.is_normalized());
        debug_assert!(other.is_normalized());
        assert!(other.len() <= n, "The limit {} is less than the size of SDR {}", n, other.len());
        self.subtract(other);
        while self.len() + other.len() > n {
            let idx = rand::random::<usize>() % self.0.len();
            self.0.swap_remove(idx);
        }
        self.0.extend_from_slice(other.as_slice());
        self.0.sort()
    }

    pub fn subregion(&self, total_shape:&[Idx;3], subregion_range:&Range<[Idx;3]>)->CpuSDR{
        CpuSDR(self.iter().cloned().filter(|&i|range_contains(subregion_range,&total_shape.pos(i))).collect())
    }
    pub fn subregion2d(&self, total_shape:&[Idx;3], subregion_range:&Range<[Idx;2]>)->CpuSDR{
        CpuSDR(self.iter().cloned().filter(|&i|range_contains(subregion_range,total_shape.pos(i).grid())).collect())
    }
    pub fn conv_rand_subregion(&self, shape:&ConvShape, rng:&mut impl Rng)->CpuSDR{
        self.conv_subregion(shape,&shape.out_grid().rand_vec(rng))
    }
    pub fn conv_subregion(&self, shape:&ConvShape, output_column_position:&[Idx;2])->CpuSDR{
        let mut s = CpuSDR::new();
        let r = shape.in_range(output_column_position);
        let kc = shape.kernel_column();
        for &i in self.iter(){
            let pos = shape.in_shape().pos(i);
            if range_contains(&r,pos.grid()){
                let pos_within_subregion = pos.grid().sub(&r.start).add_channels(pos.channels());
                s.push(kc.idx(pos_within_subregion))
            }
        }
        s
    }
    pub fn rand_subregion(&self, total_shape:&[Idx;3], subregion_size:&[Idx;3], rng:&mut impl Rng)->CpuSDR{
        assert!(subregion_size.all_le(total_shape),"subregion exceeds total region");
        let offset:[Idx;3] = rng.gen();
        let offset = offset.rem(&total_shape.sub(subregion_size).add_scalar(1));
        self.subregion(total_shape,&(offset..offset.add(subregion_size)))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ocl::{SpatialDims, Platform, Device, Program, Queue, Buffer, flags, Kernel, ProQue};
    use ocl::core::{BufferRegion, Error};
    use crate::ecc_program::EccProgram;
    use crate::ocl_sdr::OclSDR;
    use crate::cpu_sdr::CpuSDR;
    use ndalgebra::context::Context;
    use crate::encoder::{EncoderBuilder, Encoder};
    use crate::population::Population;
    use rand::SeedableRng;
    use crate::{CpuInput, OclInput};


    #[test]
    fn test2() -> Result<(), String> {
        let mut encoder = EncoderBuilder::new();
        let scalar = encoder.add_integer(50..100, 100, 5);

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 50);
        assert_eq!(vec![0, 1, 2, 3, 4], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 51);
        assert_eq!(vec![1, 2, 3, 4, 5], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 52);
        assert_eq!(vec![3, 4, 5, 6, 7], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 100);
        assert_eq!(vec![95, 96, 97, 98, 99], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 99);
        assert_eq!(vec![95, 96, 97, 98, 99], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 98);
        assert_eq!(vec![93, 94, 95, 96, 97], sdr.to_vec());
        Ok(())
    }

    #[test]
    fn test3() -> Result<(), String> {
        let mut encoder = EncoderBuilder::new();
        let scalar = encoder.add_float(50f32..100f32, 100, 5);

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 50.);
        assert_eq!(vec![0, 1, 2, 3, 4], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 51.);
        assert_eq!(vec![1, 2, 3, 4, 5], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 52.);
        assert_eq!(vec![3, 4, 5, 6, 7], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 100.);
        assert_eq!(vec![95, 96, 97, 98, 99], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 99.);
        assert_eq!(vec![93, 94, 95, 96, 97], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 98.);
        assert_eq!(vec![91, 92, 93, 94, 95], sdr.to_vec());
        Ok(())
    }

    #[test]
    fn test4() -> Result<(), String> {
        let mut encoder = EncoderBuilder::new();
        let scalar = encoder.add_circular_integer(50..100, 100, 5);

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 50);
        assert_eq!(vec![0, 1, 2, 3, 4], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 51);
        assert_eq!(vec![2, 3, 4, 5, 6], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 52);
        assert_eq!(vec![4, 5, 6, 7, 8], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 101);
        assert_eq!(vec![2, 3, 4, 5, 6], sdr.to_vec());


        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 100);
        assert_eq!(vec![0, 1, 2, 3, 4], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 99);
        assert_eq!(vec![98, 99, 0, 1, 2], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 98);
        assert_eq!(vec![96, 97, 98, 99, 0], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 97);
        assert_eq!(vec![94, 95, 96, 97, 98], sdr.to_vec());
        Ok(())
    }

    #[test]
    fn test5() -> Result<(), String> {
        let mut encoder = EncoderBuilder::new();
        let scalar1 = encoder.add_circular_integer(50..100, 100, 5);
        let scalar2 = encoder.add_float(50f32..100f32, 100, 5);

        let mut sdr = CpuSDR::new();
        scalar1.encode(&mut sdr, 99);
        scalar2.encode(&mut sdr, 50.);
        assert_eq!(vec![98, 99, 0, 1, 2, 100, 101, 102, 103, 104], sdr.to_vec());
        Ok(())
    }

    #[test]
    fn test6() -> Result<(), String> {
        fn overlap(a: &[u32], b: &[u32]) -> u32 {
            let mut sdr1 = CpuSDR::new();
            sdr1.set_from_slice(a);
            let mut sdr2 = CpuSDR::new();
            sdr2.set_from_slice(b);
            sdr1.normalize();
            sdr2.normalize();
            sdr1.overlap(&sdr2)
        }
        assert_eq!(overlap(&[1, 5, 6, 76], &[1]), 1);
        assert_eq!(overlap(&[1, 5, 6, 76], &[]), 0);
        assert_eq!(overlap(&[], &[]), 0);
        assert_eq!(overlap(&[], &[1]), 0);
        assert_eq!(overlap(&[1, 5, 6, 76], &[1, 5, 6, 76]), 4);
        assert_eq!(overlap(&[1, 5, 6, 76], &[5, 76, 6, 1]), 4);
        assert_eq!(overlap(&[1, 5, 6, 76], &[53, 746, 6, 1]), 2);
        assert_eq!(overlap(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]), 3);
        Ok(())
    }

    #[test]
    fn test7() -> Result<(), String> {
        fn intersect(a: &[u32], b: &[u32]) -> CpuSDR {
            let mut sdr1 = CpuSDR::new();
            sdr1.set_from_slice(a);
            let mut sdr2 = CpuSDR::new();
            sdr2.set_from_slice(b);
            sdr1.normalize();
            sdr2.normalize();
            sdr1.intersection(&sdr2)
        }
        assert_eq!(intersect(&[1, 5, 6, 76], &[1]).as_slice(), &[1]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[]).as_slice(), &[]);
        assert_eq!(intersect(&[], &[]).as_slice(), &[]);
        assert_eq!(intersect(&[], &[1]).as_slice(), &[]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[1, 5, 6, 76]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[5, 76, 6, 1]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[53, 746, 6, 1]).as_slice(), &[1, 6]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]).as_slice(), &[1, 5, 6]);
        Ok(())
    }

    #[test]
    fn test7_union() -> Result<(), String> {
        fn union(a: &[u32], b: &[u32]) -> CpuSDR {
            let mut sdr1 = CpuSDR::new();
            sdr1.set_from_slice(a);
            let mut sdr2 = CpuSDR::new();
            sdr2.set_from_slice(b);
            sdr1.normalize();
            sdr2.normalize();
            sdr1.union(&sdr2)
        }
        assert_eq!(union(&[1, 5, 6, 76], &[1]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[1, 5, 6, 76], &[]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[], &[]).as_slice(), &[]);
        assert_eq!(union(&[1], &[]).as_slice(), &[1]);
        assert_eq!(union(&[], &[1]).as_slice(), &[1]);
        assert_eq!(union(&[1, 5, 6, 76], &[1, 5, 6, 76]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[1, 5, 6, 76], &[5, 76, 6, 1]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[1, 5, 6, 76], &[53, 746, 6, 1]).as_slice(), &[1, 5, 6, 53, 76, 746]);
        assert_eq!(union(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]).as_slice(), &[1, 3, 5, 6, 7, 53, 76, 78, 746]);
        Ok(())
    }

    #[test]
    fn test7_subtract() -> Result<(), String> {
        fn subtract(a: &[u32], b: &[u32]) -> CpuSDR {
            let mut sdr1 = CpuSDR::new();
            sdr1.set_from_slice(a);
            let mut sdr2 = CpuSDR::new();
            sdr2.set_from_slice(b);
            sdr1.normalize();
            sdr2.normalize();
            sdr1.subtract(&sdr2);
            sdr1
        }
        assert_eq!(subtract(&[1, 5, 6, 76], &[1]).as_slice(), &[5, 6, 76]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(subtract(&[], &[]).as_slice(), &[]);
        assert_eq!(subtract(&[1], &[]).as_slice(), &[1]);
        assert_eq!(subtract(&[], &[1]).as_slice(), &[]);
        assert_eq!(subtract(&[1], &[1]).as_slice(), &[]);
        assert_eq!(subtract(&[1], &[2]).as_slice(), &[1]);
        assert_eq!(subtract(&[1, 2], &[2]).as_slice(), &[1]);
        assert_eq!(subtract(&[2, 3], &[2]).as_slice(), &[3]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[1, 5, 6, 76]).as_slice(), &[]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[5, 76, 6, 1]).as_slice(), &[]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[53, 746, 6, 1]).as_slice(), &[5, 76]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]).as_slice(), &[76]);
        Ok(())
    }

    #[test]
    fn test9() -> Result<(), String> {
        let p = EccProgram::default()?;
        let input = CpuInput::from_sparse_slice(&[1, 2, 4, 7, 15], 16);
        let ocl_input = OclInput::from_cpu(&input, p.clone(), 16)?;
        assert_eq!(input.cardinality(), ocl_input.cardinality(), "cardinality");
        let input2 = ocl_input.to_cpu()?;
        assert_eq!(input.get_sparse(), input2.get_sparse(), "sparse");
        assert_eq!(input.get_dense(), input2.get_dense(), "dense");
        Ok(())
    }

    #[test]
    fn test10() -> Result<(), String> {
        let p = EccProgram::default()?;
        let input = CpuInput::from_dense_bools(&[true, false, false, true, true, false, false, true]);
        let ocl_input = OclInput::from_cpu(&input, p.clone(), 16)?;
        assert_eq!(input.cardinality(), ocl_input.cardinality(), "cardinality");
        let input2 = ocl_input.to_cpu()?;
        assert_eq!(input.get_sparse(), input2.get_sparse(), "sparse");
        assert_eq!(input.get_dense(), input2.get_dense(), "dense");
        Ok(())
    }

    #[test]
    fn test11() -> Result<(), String> {
        let p = EccProgram::default()?;
        let input = CpuInput::from_sparse_slice(&[1, 2, 4, 7, 15], 16);
        let mut ocl_input = OclInput::from_cpu(&input, p.clone(), 16)?;
        ocl_input.set_sparse_from_slice(&[1, 5, 13])?;
        let input = CpuInput::from_sparse_slice(&[1, 5, 13], 16);
        let input2 = ocl_input.to_cpu()?;
        assert_eq!(input.get_sparse(), input2.get_sparse(), "sparse");
        assert_eq!(input.get_dense(), input2.get_dense(), "dense");
        Ok(())
    }


    #[test]
    fn test16() {
        fn test(from: u32, to: u32) {
            let mut bits = CpuBitset::from_bools(&[true; 64]);
            bits.clear_range(from, to);
            for i in from..to {
                assert!(!bits.is_bit_on(i), "{},{}->{}", from, to, i);
            }
            for i in 0..from {
                assert!(bits.is_bit_on(i), "{},{}->{}", from, to, i);
            }
            for i in to..64 {
                assert!(bits.is_bit_on(i), "{},{}->{}", from, to, i);
            }
        }
        test(0, 3);
        test(1, 3);
        test(0, 32);
        test(0, 33);
        test(32, 33);
        test(0, 64);
        test(32, 64);
        test(50, 64);
        test(50, 55);
    }

    #[test]
    fn test19() {
        let mut bits = CpuBitset::new(64);
        bits.set_bits_on(&[0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(bits.cardinality_in_range(0, 9), 9);
        assert_eq!(bits.cardinality_in_range(0, 8), 8);
        assert_eq!(bits.cardinality_in_range(1, 9), 8);
        assert_eq!(bits.cardinality_in_range(2, 2), 0);
        assert_eq!(bits.cardinality_in_range(1, 8), 7);

        bits.set_bits_on(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 32, 33, 34]);
        assert_eq!(bits.cardinality_in_range(0, 35), 12);
        assert_eq!(bits.cardinality_in_range(0, 34), 11);
        assert_eq!(bits.cardinality_in_range(0, 32), 9);
        assert_eq!(bits.cardinality_in_range(1, 32), 8);
        assert_eq!(bits.cardinality_in_range(9, 32), 0);
        assert_eq!(bits.cardinality_in_range(9, 35), 3);
        assert_eq!(bits.cardinality_in_range(32, 35), 3);
    }

    #[test]
    fn test20() {
        let enc = EncoderBuilder::new().add_categorical(5, 10);
        let mut i = CpuInput::new(64);
        i.set_sparse_from_slice(&[1, 4, 10, 21, 33, 32, 34, 40]);
        assert_eq!(enc.find_category_with_highest_overlap_bitset(i.get_dense()), 3);
        assert_eq!(enc.find_category_with_highest_overlap(i.get_sparse()), 3);
        i.set_sparse_from_slice(&[1, 4, 10, 21, 34, 40]);
        assert_eq!(enc.find_category_with_highest_overlap_bitset(i.get_dense()), 0);
        assert_eq!(enc.find_category_with_highest_overlap(i.get_sparse()), 0);
        i.set_sparse_from_slice(&[1, 4, 5, 6, 10, 21, 33, 32, 34, 40]);
        assert_eq!(enc.find_category_with_highest_overlap_bitset(i.get_dense()), 0);
        assert_eq!(enc.find_category_with_highest_overlap(i.get_sparse()), 0);
        i.set_sparse_from_slice(&[1]);
        assert_eq!(enc.find_category_with_highest_overlap_bitset(i.get_dense()), 0);
        assert_eq!(enc.find_category_with_highest_overlap(i.get_sparse()), 0);
        i.set_sparse_from_slice(&[34]);
        assert_eq!(enc.find_category_with_highest_overlap_bitset(i.get_dense()), 3);
        assert_eq!(enc.find_category_with_highest_overlap(i.get_sparse()), 3);
    }

    #[test]
    fn test22() {
        let a = [1, 4, 6, 7];
        assert_eq!(CpuSDR::from(&CpuBitset::from_sdr(&a, 32)), CpuSDR::from(&a as &[u32]));
        let a = [6, 7];
        assert_eq!(CpuSDR::from(&CpuBitset::from_sdr(&a, 32)), CpuSDR::from(&a as &[u32]));
        let a = [31];
        assert_eq!(CpuSDR::from(&CpuBitset::from_sdr(&a, 32)), CpuSDR::from(&a as &[u32]));
        let a = [63];
        assert_eq!(CpuSDR::from(&CpuBitset::from_sdr(&a, 64)), CpuSDR::from(&a as &[u32]));
        let a = [4, 63];
        assert_eq!(CpuSDR::from(&CpuBitset::from_sdr(&a, 64)), CpuSDR::from(&a as &[u32]));
    }

    #[test]
    fn test23() {
        let sdr_grid = [[CpuSDR::from_slice(&[1, 2, 3])]];
        let o = CpuSDR::vote_conv2d_arr(4, 0, [1, 1], [1, 1], [1, 1], &sdr_grid);
        assert_eq!(o[0][0], sdr_grid[0][0])
    }

    #[test]
    fn test24() {
        let sdr_grid = [
            [
                CpuSDR::from_slice(&[1, 2, 3]),
                CpuSDR::from_slice(&[1, 2, 3])
            ],
            [
                CpuSDR::from_slice(&[1, 2, 3]),
                CpuSDR::from_slice(&[1, 2, 3])
            ]
        ];
        let o = CpuSDR::vote_conv2d_arr(4, 0, [1, 1], [2, 2], [2, 2], &sdr_grid);
        assert_eq!(o.len(), 1);
        assert_eq!(o[0].len(), 1);
        assert_eq!(o[0][0], sdr_grid[0][0])
    }

    #[test]
    fn test25() {
        let sdr_grid = [
            [
                CpuSDR::from_slice(&[1, 2, 3]),
                CpuSDR::from_slice(&[0, 2, 3])
            ],
            [
                CpuSDR::from_slice(&[0, 2, 3]),
                CpuSDR::from_slice(&[1, 4, 3])
            ]
        ];
        let o = CpuSDR::vote_conv2d_arr(2, 0, [1, 1], [2, 2], [2, 2], &sdr_grid);
        assert_eq!(o.len(), 1);
        assert_eq!(o[0].len(), 1);
        assert_eq!(o[0][0], CpuSDR::from_slice(&[2, 3]))
    }

    #[test]
    fn test26() {
        let sdr_grid = [[
            CpuSDR::from_slice(&[1, 2, 3]), CpuSDR::from_slice(&[0, 2, 3]), CpuSDR::from_slice(&[0, 2, 4]), ], [
            CpuSDR::from_slice(&[0, 2, 3]), CpuSDR::from_slice(&[1, 4, 3]), CpuSDR::from_slice(&[1, 4, 3]), ]
        ];
        let o = CpuSDR::vote_conv2d_arr(2, 0, [1, 1], [2, 2], [2, 3], &sdr_grid);
        assert_eq!(o.len(), 1);
        assert_eq!(o[0].len(), 2);
        assert_eq!(o[0][0], CpuSDR::from_slice(&[2, 3]));
        assert_eq!(o[0][1], CpuSDR::from_slice(&[3, 4]));
    }


    #[test]
    fn test29() {
        let sdr_grid = [[
            CpuSDR::from_slice(&[0]), CpuSDR::from_slice(&[1]), CpuSDR::from_slice(&[2]), ], [
            CpuSDR::from_slice(&[3]), CpuSDR::from_slice(&[4]), CpuSDR::from_slice(&[5]), ]
        ];
        let o = CpuSDR::vote_conv2d_transpose_arr([1, 1], [2, 2], [3, 4], &|out0, out1| &sdr_grid[out0 as usize][out1 as usize]);
        assert_eq!(o.len(), 3);
        assert_eq!(o[0].len(), 4);
        assert_eq!(o[0][0], CpuSDR::from_slice(&[0]));
        assert_eq!(o[0][1], CpuSDR::from_slice(&[0, 1]));
        assert_eq!(o[0][2], CpuSDR::from_slice(&[1, 2]));
        assert_eq!(o[0][3], CpuSDR::from_slice(&[2]));
        assert_eq!(o[1][0], CpuSDR::from_slice(&[0, 3]));
        assert_eq!(o[1][1], CpuSDR::from_slice(&[0, 1, 3, 4]));
        assert_eq!(o[1][2], CpuSDR::from_slice(&[1, 2, 4, 5]));
        assert_eq!(o[1][3], CpuSDR::from_slice(&[2, 5]));
        assert_eq!(o[2][0], CpuSDR::from_slice(&[3]));
        assert_eq!(o[2][1], CpuSDR::from_slice(&[3, 4]));
        assert_eq!(o[2][2], CpuSDR::from_slice(&[4, 5]));
        assert_eq!(o[2][3], CpuSDR::from_slice(&[5]));
    }

    // #[test]
    // fn test30() {
    //     let sdr_grid = [
    //         [CpuSDR::from_slice(&[4, 13, 50, 64, 74, 94, 115, 162, 167, 202, 203, 253]), CpuSDR::from_slice(&[4, 5, 105, 112, 117, 188, 193, 202, 212, 217, 252, 255]), CpuSDR::from_slice(&[6, 7, 11, 48, 54, 75, 85, 86, 120, 178, 248, 251]), ],
    //         [CpuSDR::from_slice(&[11, 17, 20, 26, 77, 78, 88, 105, 128, 156, 173, 187]), CpuSDR::from_slice(&[1, 2, 5, 57, 158, 165, 170, 181, 194, 196, 203, 230]), CpuSDR::from_slice(&[1, 3, 27, 40, 62, 72, 74, 105, 136, 159, 171, 204]), ]
    //     ];
    //     let o = CpuSDR::vote_conv2d_arr(12,1,(1,1),(2,2),(2,3),&sdr_grid);
    //     assert_eq!(o.len(),1);
    //     assert_eq!(o[0].len(),2);
    //     let intersection = sdr_grid[0][0]
    //         .intersection(&sdr_grid[0][1])
    //         .intersection(&sdr_grid[1][0])
    //         .intersection(&sdr_grid[1][1]);
    //     assert!(intersection.subset(&o[0][0]));
    //     assert_eq!(o[0][1],CpuSDR::from_slice(&[1, 3, 4, 5, 86, 105, 158, 203, 230, 251, 252, 255]));
    // }
    #[test]
    fn test31() {
        let mut sdr = CpuSDR::from_slice(&[5, 41, 50, 51, 125, 157, 192, 220, 225, 230, 245, 253]);
        let votes = CpuSDR::new();
        sdr.randomly_extend_from(&votes, sdr.len());
        assert_eq!(sdr, CpuSDR::from_slice(&[5, 41, 50, 51, 125, 157, 192, 220, 225, 230, 245, 253]));
    }

    #[test]
    fn test32() {
        let o = [5, 41, 50, 51, 125, 157, 192, 220, 225, 230, 245, 253];
        let mut sdr = CpuSDR::from_slice(&o);
        let votes = CpuSDR::from_slice(&[34]);
        sdr.randomly_extend_from(&votes, sdr.len());
        assert_eq!(sdr.len(), o.len());
        assert!(sdr.contains(34));
    }

    #[test]
    fn test33() {
        let o = [5, 41, 50, 51, 125, 157, 192, 220, 225, 230, 245, 253];
        let mut sdr = CpuSDR::from_slice(&o);
        let votes = CpuSDR::from_slice(&[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]);
        sdr.randomly_extend_from(&votes, sdr.len());
        assert_eq!(sdr, votes);
    }

    #[test]
    fn test34() {
        let o = [5, 21, 78, 99, 101, 150, 168, 188, 189, 211, 217, 246];
        let mut sdr = CpuSDR::from_slice(&o);
        let votes = CpuSDR::from_slice(&[97]);
        sdr.randomly_extend_from(&votes, sdr.len());
        assert_eq!(sdr.len(), o.len());
        assert!(sdr.contains(97));
    }
}
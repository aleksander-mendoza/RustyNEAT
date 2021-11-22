use ocl::{ProQue, Error, SpatialDims, flags, Platform, Device, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, Deref, DerefMut};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use ndalgebra::buffer::Buffer;
use crate::htm_program::HtmProgram;
use crate::{CpuBitset, EncoderTarget};
use std::collections::{HashMap, HashSet};
use std::borrow::Borrow;

#[derive(Clone, Eq, PartialEq)]
pub struct CpuSDR(Vec<u32>);

impl PartialEq<Vec<u32>> for CpuSDR {
    fn eq(&self, other: &Vec<u32>) -> bool {
        self.0.eq(other)
    }
}

impl From<&[u32]> for CpuSDR {
    fn from(v: &[u32]) -> Self {
        Self(Vec::from(v))
    }
}

impl From<Vec<u32>> for CpuSDR {
    fn from(v: Vec<u32>) -> Self {
        Self(v)
    }
}

impl Deref for CpuSDR {
    type Target = [u32];

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
    fn push(&mut self, neuron_index: u32) {
        self.0.push(neuron_index)
    }

    fn clear_range(&mut self, from: u32, to: u32) {
        self.0.retain(|&i| i >= to || i < from)
    }
}

impl CpuSDR {
    pub fn from_slice(s: &[u32]) -> Self {
        Self::from(s)
    }
    pub fn as_slice(&self) -> &[u32] {
        self.0.as_slice()
    }
    pub fn as_mut_slice(&mut self) -> &mut [u32] {
        self.0.as_mut_slice()
    }
    pub fn into_vec(self) -> Vec<u32> {
        let Self(v) = self;
        v
    }
    pub fn clear(&mut self) {
        self.0.clear()
    }
    pub fn set(&mut self, active_neurons: &[u32]) {
        unsafe { self.0.set_len(0) }
        self.0.extend_from_slice(active_neurons)
    }
    pub fn cardinality(&self) -> u32 {
        self.0.len() as u32
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
    pub fn overlap(&self, other: &CpuSDR) -> u32 {
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
    pub fn extend_from_slice(&mut self, other: &[u32]) {
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
    pub fn vote_conv2d_out_size(stride: (u32, u32), kernel_size: (u32, u32), grid_size: (u32, u32)) -> (u32, u32) {
        ((grid_size.0 - kernel_size.0) / stride.0 + 1, (grid_size.1 - kernel_size.1) / stride.1 + 1)
    }
    pub fn vote_conv2d_arr<T: Borrow<CpuSDR>>(n: usize, threshold: u32, stride: (u32, u32), kernel_size: (u32, u32), grid_size: (u32, u32), input_sdr_grid: &[impl AsRef<[T]>]) -> Vec<Vec<CpuSDR>> {
        Self::vote_conv2d_arr_with(n, threshold, stride, kernel_size, grid_size, &|c0, c1| input_sdr_grid[c0 as usize].as_ref()[c1 as usize].borrow(), |a| a)
    }
    pub fn vote_conv2d_arr_with<'a, O>(n: usize, threshold: u32, stride: (u32, u32), kernel_size: (u32, u32), grid_size: (u32, u32), input_sdr_grid: &'a impl Fn(u32, u32) -> &'a CpuSDR, out_sdr: impl Fn(CpuSDR) -> O) -> Vec<Vec<O>> {
        let out_grid_size = Self::vote_conv2d_out_size(stride, kernel_size, grid_size);
        let mut out_grid: Vec<Vec<O>> = (0..out_grid_size.0).map(|_| {
            let mut v = Vec::with_capacity(out_grid_size.1 as usize);
            unsafe { v.set_len(out_grid_size.1 as usize) }
            v
        }).collect();
        Self::vote_conv2d(n, threshold, stride, kernel_size, grid_size, input_sdr_grid, |c0, c1, sdr| {
            let o = &mut out_grid[c0 as usize][c1 as usize];
            let o = o as *mut O;
            unsafe { std::ptr::write(o, out_sdr(sdr)) }
        });
        out_grid
    }
    pub fn vote_conv2d<'a>(n: usize, threshold: u32, stride: (u32, u32), kernel_size: (u32, u32), grid_size: (u32, u32), input_sdr_grid: &'a impl Fn(u32, u32) -> &'a CpuSDR, mut output_sdr_grid: impl FnMut(u32, u32, CpuSDR)) {
        let out_grid_size = Self::vote_conv2d_out_size(stride, kernel_size, grid_size);
        for out0 in 0..out_grid_size.0 {
            for out1 in 0..out_grid_size.1 {
                let in_begin = (out0 * stride.0, out1 * stride.1);
                let sdr = Self::vote_over_iter((in_begin.0..in_begin.0 + kernel_size.0).flat_map(|in0| (in_begin.1..in_begin.1 + kernel_size.1).map(move |in1| input_sdr_grid(in0, in1))), n, threshold);
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
    /**Randomly picks some neurons that a present in self SDR but not in other SDR.
    Requires that both SDRs are already normalized.
    It will only add so many elements so that self.len() <= n*/
    pub fn randomly_extend_from(&mut self, other: &Self, n: usize) {
        debug_assert!(self.is_normalized());
        debug_assert!(other.is_normalized());
        self.subtract(other);
        while self.len() + other.len() <= n {
            let idx = rand::random::<usize>() % self.0.len();
            self.0.swap_remove(idx);
        }
        self.0.extend_from_slice(other.as_slice());
        self.0.sort()
    }
}


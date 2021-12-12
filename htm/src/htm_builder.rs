use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, RangeBounds};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::cpu_sdr::CpuSDR;
use crate::htm_program::HtmProgram;
use ndalgebra::buffer::Buffer;
use crate::htm2::*;
use crate::cpu_htm::CpuHTM;
use crate::cpu_bitset::CpuBitset;
use crate::rnd::{xorshift32, rand_u32_to_random_f32};
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use crate::{Shape, Shape3, Shape2, resolve_range};
use std::collections::Bound;
use crate::vector_field::{VectorFieldOne, VectorFieldDiv, VectorFieldAdd, VectorFieldMul, ArrayCast, VectorFieldSub, VectorFieldPartialOrd};

#[derive(Debug, Clone)]
pub struct Segment {
    pub synapses: Vec<u32>,
}

impl Segment {
    pub fn new() -> Self {
        Self { synapses: vec![] }
    }
    pub fn join(&mut self, other: &Segment) {
        self.synapses.extend_from_slice(other.synapses.as_slice())
    }
    pub fn len(&self) -> usize{
        self.synapses.len()
    }
    pub fn dedup(&mut self) {
        self.synapses.sort();
        self.synapses.dedup();
    }
    pub fn add_uniform_rand_inputs(&mut self, input_size: u32, synapse_count: u32, transformation: &mut impl FnMut(u32) -> u32, mut rand_seed: u32) -> u32 {
        assert!(input_size >= synapse_count, "input_size={} < synapse_count={}", input_size, synapse_count);
        let mut already_added = vec![false; input_size as usize];
        for _ in 0..synapse_count {
            rand_seed = xorshift32(rand_seed);
            let mut input_idx = rand_seed % input_size;
            while already_added[input_idx as usize] {
                input_idx += 1;
                if input_idx == input_size{
                    input_idx = 0;
                }
            }
            already_added[input_idx as usize] = true;
            self.synapses.push(transformation(input_idx));
        }
        rand_seed
    }
    pub fn add_uniform_rand_inputs_from_range(&mut self, range: Range<u32>, synapse_count: u32, mut rand_seed: u32) -> u32 {
        self.add_uniform_rand_inputs(range.len() as u32, synapse_count, &mut |x| x + range.start, rand_seed)
    }
    pub fn add_nd_uniform_rand_inputs<const DIM: usize>(&mut self, range: Range<[u32; DIM]>, synapse_count: u32, transformation: &mut impl FnMut([u32; DIM]) -> u32, mut rand_seed: u32) -> u32 {
        assert!(range.start.all_le(&range.end), "Area between points {:?} is negative", range);
        let area_dim = range.end.sub(&range.start);
        self.add_uniform_rand_inputs(area_dim.product(), synapse_count, &mut |x| transformation(area_dim.pos(x).add(&range.start)), rand_seed)
    }
    pub fn add_uniform_rand_inputs_from_area<const DIM: usize>(&mut self, input_range: Range<u32>, total_region: [u32; DIM], subregion: Range<[u32; DIM]>, synapse_count: u32, mut rand_seed: u32) -> u32 {
        assert!(subregion.end.all_le(&total_region), "Subregion {:?} exceeds total region {:?}", subregion, total_region);
        assert_eq!(input_range.len(), total_region.product() as usize, "Input range {:?} has size {} which does not match total region {:?} of volume {}", input_range, input_range.len(), total_region, total_region.product());
        self.add_nd_uniform_rand_inputs(subregion, synapse_count, &mut |coord| input_range.start + total_region.idx(coord), rand_seed)
    }
}

#[derive(Debug, Clone)]
pub struct Neuron {
    pub segments: Vec<Segment>,
}

impl Neuron {
    pub fn new(num_of_segments: usize) -> Self {
        Self { segments: vec![Segment::new(); num_of_segments] }
    }
    pub fn push(&mut self, segment: Segment) {
        self.segments.push(segment)
    }
    pub fn remove(&mut self, idx: usize) -> Segment {
        self.segments.swap_remove(idx)
    }
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
    /**Joins all segments together resulting in just one segment with all the synapses*/
    pub fn collapse(&mut self) {
        if let Some(mut joined) = self.segments.pop() {
            while let Some(seg) = self.segments.pop() {
                joined.join(&seg);
            }
            self.segments.push(joined)
        }
    }
    pub fn len(&self) -> usize {
        self.segments.len()
    }
    pub fn append(&mut self, other: &mut Neuron) {
        self.segments.append(&mut other.segments)
    }
    pub fn zip_join(&mut self, other: &Neuron) {
        assert_eq!(self.len(), other.len(), "Cannot zip join minicolumns of different len {} and {}",self.len(), other.len());
        for (a, b) in self.segments.iter_mut().zip(other.segments.iter()) {
            a.join(b)
        }
    }
    pub fn dedup_all(&mut self) {
        for seg in &mut self.segments {
            seg.dedup();
        }
    }
    pub fn add_uniform_rand_inputs(&mut self, input_size: u32, synapse_count: u32, transformation: &mut impl FnMut(u32) -> u32, mut rand_seed: u32) -> u32 {
        self.segments.iter_mut().fold(rand_seed, |r, seg| seg.add_uniform_rand_inputs(input_size, synapse_count, transformation, r))
    }
    pub fn add_uniform_rand_inputs_from_range(&mut self, range: Range<u32>, synapse_count: u32, mut rand_seed: u32) -> u32 {
        self.segments.iter_mut().fold(rand_seed, |r, seg| seg.add_uniform_rand_inputs_from_range(range.clone(), synapse_count, r))
    }
    pub fn add_nd_uniform_rand_inputs<const DIM: usize>(&mut self, range: Range<[u32; DIM]>, synapse_count: u32,  transformation: &mut impl FnMut([u32; DIM]) -> u32, mut rand_seed: u32) -> u32 {
        self.segments.iter_mut().fold(rand_seed, |r, seg| seg.add_nd_uniform_rand_inputs(range.clone(), synapse_count, transformation, r))
    }
    pub fn add_uniform_rand_inputs_from_area<const DIM: usize>(&mut self, input_range: Range<u32>, total_region: [u32; DIM], subregion: Range<[u32; DIM]>, synapse_count: u32, mut rand_seed: u32) -> u32 {
        self.segments.iter_mut().fold(rand_seed, |r, seg| seg.add_uniform_rand_inputs_from_area(input_range.clone(), total_region, subregion.clone(), synapse_count, r))
    }
}

#[derive(Debug, Clone)]
pub struct Population {
    pub neurons: Vec<Neuron>,
}

impl Population {
    pub fn new(population_size: usize, segments_per_neuron: usize) -> Self {
        Self { neurons: vec![Neuron::new(segments_per_neuron); population_size] }
    }
    pub fn new_conv<const DIM:usize>(input_size:&[u32;DIM],stride:&[u32;DIM],kernel:&[u32;DIM], segments_per_neuron: usize) -> Self {
        Self::new(input_size.conv_out_size(stride,&kernel).product() as usize,segments_per_neuron)
    }
    pub fn new_2d_column_grid_with_3d_input(neurons_per_column: u32, stride: [u32; 2], kernel: [u32; 2], input_size: [u32; 3],segments_per_neuron: usize) -> Self{
        let column_grid = [input_size.height(), input_size.width()].conv_out_size(&stride, &kernel);
        let n = column_grid.product()*neurons_per_column;
        Self::new(n as usize,segments_per_neuron)
    }

    pub fn append(&mut self, other: &mut Population) {
        self.neurons.append(&mut other.neurons);
    }
    pub fn len(&self) -> usize {
        self.neurons.len()
    }
    pub fn remove(&mut self, index:usize) -> Neuron {
        self.neurons.swap_remove(index)
    }
    pub fn push(&mut self, neuron:Neuron){
        self.neurons.push(neuron)
    }
    pub fn push_neurons(&mut self, population_size: usize, segments_per_neuron: usize){
        self.neurons.extend((0..population_size).map(|_|Neuron::new(segments_per_neuron)))
    }
    pub fn zip_join(&mut self, other: &Population) {
        assert_eq!(self.len(), other.len(), "Cannot zip join populations of different len {} and {}",self.len(), other.len());
        for (a, b) in self.neurons.iter_mut().zip(other.neurons.iter()) {
            a.zip_join(b)
        }
    }
    pub fn zip_append(&mut self, other: &mut Population) {
        assert_eq!(self.len(), other.len(), "Cannot zip append populations of different len {} and {}",self.len(), other.len());
        for (a, b) in self.neurons.iter_mut().zip(other.neurons.iter_mut()) {
            a.append( b)
        }
    }
    pub fn add_uniform_rand_inputs(&mut self, input_size: u32, synapse_count: u32, transformation: &mut impl FnMut(u32) -> u32, mut rand_seed: u32) -> u32 {
        self.neurons.iter_mut().fold(rand_seed, |r, n| n.add_uniform_rand_inputs(input_size, synapse_count, transformation, r))
    }
    pub fn add_uniform_rand_inputs_from_range(&mut self, range: Range<u32>, synapse_count: u32, mut rand_seed: u32) -> u32 {
        self.neurons.iter_mut().fold(rand_seed, |r, n| n.add_uniform_rand_inputs_from_range(range.clone(), synapse_count, r))
    }
    pub fn add_nd_uniform_rand_inputs<const DIM: usize>(&mut self, range: Range<[u32; DIM]>, synapse_count: u32, transformation: &mut impl FnMut([u32; DIM]) -> u32, mut rand_seed: u32) -> u32 {
        self.neurons.iter_mut().fold(rand_seed, |r, n| n.add_nd_uniform_rand_inputs(range.clone(), synapse_count, transformation, r))
    }
    pub fn add_uniform_rand_inputs_from_area<const DIM: usize>(&mut self, input_range: Range<u32>, total_region: [u32; DIM], subregion: Range<[u32; DIM]>, synapse_count: u32, mut rand_seed: u32) -> u32 {
        self.neurons.iter_mut().fold(rand_seed, |r, n| n.add_uniform_rand_inputs_from_area(input_range.clone(), total_region, subregion.clone(), synapse_count, r))
    }
    /** intervals holds a vector of intervals. Each interval is defined as a triple (f,c,t),
    where f is the (inclusive) beginning of the input interval, c is the number of neurons that will be chosen from this particular interval,
    t is the (exclusive) end of the input interval. */
    pub fn add_interval_uniform_prob(&mut self, intervals: &[(u32, u32, u32)], mut rand_seed: u32) -> u32 {
        let mut ranges: Vec<(u32, u32)> = intervals.iter().map(|&(f, c, t)| {
            assert!(f <= t, "Beginning of input interval {}-{} is larger than its end", f, t);
            assert!(c <= t - f, "Input interval {}-{} has size {} but requested {} connections from it", f, t, t - f, c);
            (f, t)
        }).collect();
        ranges.sort_by_key(|(f, t)| *f);
        ranges.into_iter().fold(0, |prev_t, (f, t)| {
            assert!(prev_t <= f, "There is an overlap between ranges");
            t
        });
        intervals.into_iter().fold(rand_seed, |r, &(from, synapse_count, to)| {
            self.add_uniform_rand_inputs_from_range(from..to, synapse_count, r)
        })
    }
    pub fn add_2d_column_grid_with_3d_input(&mut self, input_range: Range<u32>, neurons_per_column: u32, synapses_per_segment: u32, stride: [u32; 2], kernel: [u32; 2], input_size: [u32; 3], mut rand_seed: u32) -> u32 {
        let column_grid = [input_size.height(), input_size.width()].conv_out_size(&stride, &kernel);
        assert_eq!(self.len() as u32,column_grid.product()*neurons_per_column,"The stride {:?} and kernel {:?} produce column grid {:?} with {} neurons per column giving in total {} neurons but the population has {} neurons",stride, kernel, column_grid,neurons_per_column,column_grid.product()*neurons_per_column,self.len());
        let input_subregion_size = [input_size[0], kernel[0], kernel[1]];
        let output_grid = [neurons_per_column,column_grid[0],column_grid[1]];
        for col0 in 0..column_grid[0] {
            for col1 in 0..column_grid[1] {
                let input_offset = [0, col0 * stride[0], col1 * stride[1]];
                debug_assert!(input_offset.all_lt(&input_size));
                let input_end = input_offset.add(&input_subregion_size);
                let subregion = input_offset..input_end;
                for neuron in 0..neurons_per_column {
                    let idx = output_grid.idx([neuron,col0,col1]) as usize;
                    rand_seed = self.neurons[idx].add_uniform_rand_inputs_from_area(input_range.clone(), input_size, subregion.clone(), synapses_per_segment, rand_seed)
                }
            }
        }
        rand_seed
    }
    // pub fn add_conv(&mut self, input_range: Range<u32>, synapse_count: u32, output_shape: [u32; 3], kernel: [u32; 3], input_shape: [u32; 3], mut rand_seed: u32) -> u32 {
    //     assert_eq!(output_shape.product() as usize, self.len(), "Output shape {:?} has size {} but the population has {} neurons", output_shape, output_shape.product(), self.len());
    //     let stride: [u32; 3] = input_shape.conv_stride(&output_shape, &kernel);
    //     assert!(inputs_per_minicolumn);
    //     let input_subregion_size = [input_size[0], input_kernel[0], input_kernel[1]];
    //     for col0 in 0..output_shape[0] {
    //         for col1 in 0..column_grid[1] {
    //             let input_offset = [0, col0 * stride[0], col1 * stride[1]];
    //             debug_assert!(input_offset.all_lt(&input_size));
    //             for neuron in 0..minicolumns_per_column {
    //                 let idx = neuron as usize;
    //                 rand_seed = self.neurons[idx].add_uniform_rand_inputs_from_area(input_range.clone(), input_size, , inputs_per_minicolumn, rand_seed)
    //             }
    //         }
    //     }
    //     rand_seed
    // }
}


impl Add for Neuron{
    type Output = Neuron;

    fn add(mut self, mut rhs: Self) -> Self {
        self.append(&mut rhs);
        self
    }
}

impl AddAssign for Neuron{
    fn add_assign(&mut self, mut rhs: Self) {
        self.append(&mut rhs)
    }
}

impl Mul for Neuron{
    type Output = Neuron;

    fn mul(mut self, mut rhs: Self) -> Self {
        self.zip_join(&mut rhs);
        self
    }
}
impl MulAssign for Neuron{
    fn mul_assign(&mut self, mut rhs: Self) {
        self.zip_join(&mut rhs)
    }
}

impl Add for Population{
    type Output = Population;

    fn add(mut self, mut rhs: Self) -> Self {
        self.append(&mut rhs);
        self
    }
}

impl AddAssign for Population{
    fn add_assign(&mut self, mut rhs: Self) {
        self.append(&mut rhs)
    }
}
impl Mul for Population{
    type Output = Population;

    fn mul(mut self, mut rhs: Self) -> Self {
        self.zip_append(&mut rhs);
        self
    }
}
impl MulAssign for Population{
    fn mul_assign(&mut self, mut rhs: Self) {
        self.zip_append(&mut rhs)
    }
}
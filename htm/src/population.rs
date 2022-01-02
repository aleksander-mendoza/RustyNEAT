use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, RangeBounds};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::cpu_sdr::CpuSDR;
use crate::ecc_program::EccProgram;
use ndalgebra::buffer::Buffer;
use crate::cpu_bitset::CpuBitset;
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use crate::{Shape, resolve_range, Shape3, Shape2, from_xyz, from_xy};
use std::collections::Bound;
use crate::vector_field::{VectorFieldOne, VectorFieldDiv, VectorFieldAdd, VectorFieldMul, ArrayCast, VectorFieldSub, VectorFieldPartialOrd};
use rand::Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub input_idx:usize,
    pub weight:f32,
}
impl Synapse{
    pub fn rand(input_idx:usize, rand_seed:&mut impl Rng)->Self{
        Self::new(input_idx,rand_seed.gen())
    }
    pub fn new(input_idx:usize, weight:f32)->Self{
        Self{input_idx,weight}
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub synapses: Vec<Synapse>,
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
    pub fn set_weights_random(&mut self, rand_seed:&mut impl Rng){
        self.synapses.iter_mut().for_each(|s|s.weight=rand_seed.gen())
    }
    pub fn set_weights_uniform(&mut self){
        self.set_weights_const(1./self.synapses.len() as f32)
    }
    pub fn set_weights_const(&mut self, weight:f32){
        self.synapses.iter_mut().for_each(|s|s.weight=weight)
    }
    pub fn set_weights_scaled(&mut self, scale:f32){
        self.synapses.iter_mut().for_each(|s|s.weight*=scale)
    }
    pub fn get_weights_sum(&self)->f32{
        self.synapses.iter().map(|s|s.weight).sum()
    }
    pub fn set_weights_normalized(&mut self){
        let scale = 1./self.get_weights_sum();
        self.set_weights_scaled(scale)
    }
    pub fn get_weights_mean(&self)->f32{
        self.get_weights_sum()/self.len() as f32
    }
    pub fn dedup(&mut self) {
        self.synapses.sort_by_key(|s|s.input_idx);
        self.synapses.dedup_by_key(|s|s.input_idx);
    }
    pub fn add_uniform_rand_inputs(&mut self, input_size: usize, synapse_count: usize, transformation: &mut impl FnMut(usize) -> usize, rand_seed:&mut impl Rng)  {
        assert!(input_size >= synapse_count, "input_size={} < synapse_count={}", input_size, synapse_count);
        let mut already_added = vec![false; input_size as usize];
        for _ in 0..synapse_count {
            let mut input_idx = rand_seed.gen_range(0..input_size);
            while already_added[input_idx as usize] {
                input_idx += 1;
                if input_idx == input_size{
                    input_idx = 0;
                }
            }
            already_added[input_idx as usize] = true;
            self.synapses.push(Synapse::rand(transformation(input_idx),rand_seed));
        }
    }
    pub fn add_uniform_rand_inputs_from_range(&mut self, range: Range<usize>, synapse_count: usize, rand_seed:&mut impl Rng)  {
        self.add_uniform_rand_inputs(range.len(), synapse_count, &mut |x| x + range.start, rand_seed)
    }
    pub fn add_nd_uniform_rand_inputs<const DIM: usize>(&mut self, range: Range<[usize; DIM]>, synapse_count: usize, transformation: &mut impl FnMut([usize; DIM]) -> usize, rand_seed:&mut impl Rng)  {
        assert!(range.start.all_le(&range.end), "Area between points {:?} is negative", range);
        let area_dim = range.end.sub(&range.start);
        self.add_uniform_rand_inputs(area_dim.product(), synapse_count, &mut |x| transformation(area_dim.pos(x).add(&range.start)), rand_seed)
    }
    pub fn add_uniform_rand_inputs_from_area<const DIM: usize>(&mut self, input_range: Range<usize>, total_region: [usize; DIM], subregion: Range<[usize; DIM]>, synapse_count: usize, rand_seed:&mut impl Rng)  {
        assert!(subregion.end.all_le(&total_region), "Subregion {:?} exceeds total region {:?}", subregion, total_region);
        assert_eq!(input_range.len(), total_region.product() as usize, "Input range {:?} has size {} which does not match total region {:?} of volume {}", input_range, input_range.len(), total_region, total_region.product());
        self.add_nd_uniform_rand_inputs(subregion, synapse_count, &mut |coord| input_range.start + total_region.idx(coord), rand_seed)
    }
    pub fn add_all_inputs(&mut self, input_size: usize, transformation: &mut impl FnMut(usize) -> usize, rand_seed:&mut impl Rng)  {
        let w = 1./input_size as f32;
        (0..input_size).for_each(|input_idx|  {self.synapses.push(Synapse::rand(transformation(input_idx), rand_seed))})
    }
    pub fn add_all_inputs_from_range(&mut self, range: Range<usize>, rand_seed:&mut impl Rng)  {
        self.add_all_inputs(range.len(), &mut |x| x + range.start, rand_seed)
    }
    pub fn add_nd_all_inputs<const DIM: usize>(&mut self, range: Range<[usize; DIM]>, transformation: &mut impl FnMut([usize; DIM]) -> usize, rand_seed:&mut impl Rng)  {
        assert!(range.start.all_le(&range.end), "Area between points {:?} is negative", range);
        let area_dim = range.end.sub(&range.start);
        self.add_all_inputs(area_dim.product(), &mut |x| transformation(area_dim.pos(x).add(&range.start)),rand_seed)
    }
    pub fn add_all_inputs_from_area<const DIM: usize>(&mut self, input_range: Range<usize>, total_region: [usize; DIM], subregion: Range<[usize; DIM]>,rand_seed:&mut impl Rng)  {
        assert!(subregion.end.all_le(&total_region), "Subregion {:?} exceeds total region {:?}", subregion, total_region);
        assert_eq!(input_range.len(), total_region.product() as usize, "Input range {:?} has size {} which does not match total region {:?} of volume {}", input_range, input_range.len(), total_region, total_region.product());
        self.add_nd_all_inputs(subregion, &mut |coord| input_range.start + total_region.idx(coord),rand_seed)
    }

}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub fn add_uniform_rand_inputs(&mut self, input_size: usize, synapse_count: usize, transformation: &mut impl FnMut(usize) -> usize, rand_seed:&mut impl Rng)  {
        self.segments.iter_mut().for_each(|seg| seg.add_uniform_rand_inputs(input_size, synapse_count, transformation, rand_seed))
    }
    pub fn add_uniform_rand_inputs_from_range(&mut self, range: Range<usize>, synapse_count: usize, rand_seed:&mut impl Rng)  {
        self.segments.iter_mut().for_each(|seg| seg.add_uniform_rand_inputs_from_range(range.clone(), synapse_count, rand_seed))
    }
    pub fn add_nd_uniform_rand_inputs<const DIM: usize>(&mut self, range: Range<[usize; DIM]>, synapse_count: usize,  transformation: &mut impl FnMut([usize; DIM]) -> usize, rand_seed:&mut impl Rng)  {
        self.segments.iter_mut().for_each(|seg| seg.add_nd_uniform_rand_inputs(range.clone(), synapse_count, transformation, rand_seed))
    }
    pub fn add_uniform_rand_inputs_from_area<const DIM: usize>(&mut self, input_range: Range<usize>, total_region: [usize; DIM], subregion: Range<[usize; DIM]>, synapse_count: usize, rand_seed:&mut impl Rng)  {
        self.segments.iter_mut().for_each(|seg| seg.add_uniform_rand_inputs_from_area(input_range.clone(), total_region, subregion.clone(), synapse_count, rand_seed))
    }
    pub fn add_all_inputs(&mut self, input_size: usize, transformation: &mut impl FnMut(usize) -> usize,rand_seed:&mut impl Rng)  {
        self.segments.iter_mut().for_each(|seg| seg.add_all_inputs(input_size, transformation,rand_seed))
    }
    pub fn add_all_inputs_from_range(&mut self, range: Range<usize>,rand_seed:&mut impl Rng)  {
        self.segments.iter_mut().for_each(|seg| seg.add_all_inputs_from_range(range.clone(),rand_seed))
    }
    pub fn add_nd_all_inputs<const DIM: usize>(&mut self, range: Range<[usize; DIM]>, transformation: &mut impl FnMut([usize; DIM]) -> usize,rand_seed:&mut impl Rng)  {
        self.segments.iter_mut().for_each(|seg| seg.add_nd_all_inputs(range.clone(),transformation,rand_seed))
    }
    pub fn add_all_inputs_from_area<const DIM: usize>(&mut self, input_range: Range<usize>, total_region: [usize; DIM], subregion: Range<[usize; DIM]>,rand_seed:&mut impl Rng)  {
        self.segments.iter_mut().for_each(|seg| seg.add_all_inputs_from_area(input_range.clone(),total_region.clone(),subregion.clone(),rand_seed))
    }
    pub fn set_weights_random(&mut self, rand_seed:&mut impl Rng){
        self.segments.iter_mut().for_each(|s|s.set_weights_random(rand_seed))
    }
    pub fn set_weights_uniform(&mut self){
        self.segments.iter_mut().for_each(|s|s.set_weights_uniform())
    }
    pub fn set_weights_const(&mut self, weight:f32){
        self.segments.iter_mut().for_each(|s|s.set_weights_const(weight))
    }
    pub fn set_weights_scaled(&mut self, scale:f32){
        self.segments.iter_mut().for_each(|s|s.set_weights_scaled(scale))
    }
    pub fn get_weights_sum(&self)->f32{
        self.segments.iter().map(|s|s.get_weights_sum()).sum()
    }
    pub fn set_weights_normalized(&mut self){
        self.segments.iter_mut().for_each(|s|s.set_weights_normalized())
    }
    pub fn total_synapses(&self)->usize{
        self.segments.iter().map(|s|s.len()).sum()
    }
    pub fn get_weights_mean(&self)->f32{
        self.get_weights_sum()/self.total_synapses() as f32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population {
    pub neurons: Vec<Neuron>,
}

impl Population {
    pub fn new(population_size: usize, segments_per_neuron: usize) -> Self {
        Self { neurons: vec![Neuron::new(segments_per_neuron); population_size] }
    }
    pub fn new_conv<const DIM:usize>(input_size:&[usize;DIM],stride:&[usize;DIM],kernel:&[usize;DIM], segments_per_neuron: usize) -> Self {
        Self::new(input_size.conv_out_size(stride,&kernel).product() as usize,segments_per_neuron)
    }
    pub fn new_2d_column_grid_with_3d_input(neurons_per_column: usize, stride: [usize; 2], kernel: [usize; 2], input_size: [usize; 3],segments_per_neuron: usize) -> Self{
        let column_grid = input_size.grid().conv_out_size(&stride, &kernel);
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
    pub fn add_uniform_rand_inputs(&mut self, input_size: usize, synapse_count: usize, transformation: &mut impl FnMut(usize) -> usize, rand_seed:&mut impl Rng)  {
        self.neurons.iter_mut().for_each( |n| n.add_uniform_rand_inputs(input_size, synapse_count, transformation, rand_seed))
    }
    pub fn add_uniform_rand_inputs_from_range(&mut self, range: Range<usize>, synapse_count: usize, rand_seed:&mut impl Rng)  {
        self.neurons.iter_mut().for_each(|n| n.add_uniform_rand_inputs_from_range(range.clone(), synapse_count, rand_seed))
    }
    pub fn add_nd_uniform_rand_inputs<const DIM: usize>(&mut self, range: Range<[usize; DIM]>, synapse_count: usize, transformation: &mut impl FnMut([usize; DIM]) -> usize, rand_seed:&mut impl Rng)  {
        self.neurons.iter_mut().for_each(|n| n.add_nd_uniform_rand_inputs(range.clone(), synapse_count, transformation, rand_seed))
    }
    pub fn add_uniform_rand_inputs_from_area<const DIM: usize>(&mut self, input_range: Range<usize>, total_region: [usize; DIM], subregion: Range<[usize; DIM]>, synapse_count: usize, rand_seed:&mut impl Rng)  {
        self.neurons.iter_mut().for_each(|n| n.add_uniform_rand_inputs_from_area(input_range.clone(), total_region, subregion.clone(), synapse_count, rand_seed))
    }
    pub fn add_all_inputs(&mut self, input_size: usize, transformation: &mut impl FnMut(usize) -> usize,rand_seed:&mut impl Rng)  {
        self.neurons.iter_mut().for_each(|seg| seg.add_all_inputs(input_size, transformation,rand_seed))
    }
    pub fn add_all_inputs_from_range(&mut self, range: Range<usize>,rand_seed:&mut impl Rng)  {
        self.neurons.iter_mut().for_each(|seg| seg.add_all_inputs_from_range(range.clone(),rand_seed))
    }
    pub fn add_nd_all_inputs<const DIM: usize>(&mut self, range: Range<[usize; DIM]>, transformation: &mut impl FnMut([usize; DIM]) -> usize,rand_seed:&mut impl Rng)  {
        self.neurons.iter_mut().for_each(|seg| seg.add_nd_all_inputs(range.clone(),transformation,rand_seed))
    }
    pub fn add_all_inputs_from_area<const DIM: usize>(&mut self, input_range: Range<usize>, total_region: [usize; DIM], subregion: Range<[usize; DIM]>,rand_seed:&mut impl Rng)  {
        self.neurons.iter_mut().for_each(|seg| seg.add_all_inputs_from_area(input_range.clone(),total_region.clone(),subregion.clone(),rand_seed))
    }
    pub fn set_weights_random(&mut self, rand_seed:&mut impl Rng){
        self.neurons.iter_mut().for_each(|s|s.set_weights_random(rand_seed))
    }
    pub fn set_weights_uniform(&mut self){
        self.neurons.iter_mut().for_each(|s|s.set_weights_uniform())
    }
    pub fn set_weights_const(&mut self, weight:f32){
        self.neurons.iter_mut().for_each(|s|s.set_weights_const(weight))
    }
    pub fn set_weights_scaled(&mut self, scale:f32){
        self.neurons.iter_mut().for_each(|s|s.set_weights_scaled(scale))
    }
    pub fn get_weights_sum(&self)->f32{
        self.neurons.iter().map(|s|s.get_weights_sum()).sum()
    }
    pub fn set_weights_normalized(&mut self){
        self.neurons.iter_mut().for_each(|s|s.set_weights_normalized())
    }
    pub fn total_synapses(&self)->usize{
        self.neurons.iter().map(|s|s.total_synapses()).sum()
    }
    pub fn get_weights_mean(&self)->f32{
        self.get_weights_sum()/self.total_synapses() as f32
    }
    /** intervals holds a vector of intervals. Each interval is defined as a triple (f,c,t),
    where f is the (inclusive) beginning of the input interval, c is the number of neurons that will be chosen from this particular interval,
    t is the (exclusive) end of the input interval. */
    pub fn add_interval_uniform_prob(&mut self, intervals: &[(usize, usize, usize)], rand_seed:&mut impl Rng)  {
        let mut ranges: Vec<(usize, usize)> = intervals.iter().map(|&(f, c, t)| {
            assert!(f <= t, "Beginning of input interval {}-{} is larger than its end", f, t);
            assert!(c <= t - f, "Input interval {}-{} has size {} but requested {} connections from it", f, t, t - f, c);
            (f, t)
        }).collect();
        ranges.sort_by_key(|(f, t)| *f);
        ranges.into_iter().fold(0, |prev_t, (f, t)| {
            assert!(prev_t <= f, "There is an overlap between ranges");
            t
        });
        intervals.into_iter().for_each(| &(from, synapse_count, to)| {
            self.add_uniform_rand_inputs_from_range(from..to, synapse_count, rand_seed)
        })
    }
    pub fn add_2d_column_grid_with_3d_input(&mut self, input_range: Range<usize>, neurons_per_column: usize, synapses_per_segment: usize, stride: [usize; 2], kernel: [usize; 2], input_size: [usize; 3], rand_seed:&mut impl Rng)  {
        let column_grid = input_size.grid().conv_out_size(&stride, &kernel);
        assert_eq!(self.len() ,column_grid.product()*neurons_per_column,"The stride {:?} and kernel {:?} produce column grid {:?} with {} neurons per column giving in total {} neurons but the population has {} neurons",stride, kernel, column_grid,neurons_per_column,column_grid.product()*neurons_per_column,self.len());
        let input_subregion_size = kernel.add_channels(input_size.channels());
        let output_grid = column_grid.add_channels(neurons_per_column);
        for col0 in 0..column_grid.width() {
            for col1 in 0..column_grid.height() {
                let input_offset = from_xy(col0, col1).mul(&stride).add_channels(0);
                debug_assert!(input_offset.all_lt(&input_size));
                let input_end = input_offset.add(&input_subregion_size);
                let subregion = input_offset..input_end;
                for neuron in 0..neurons_per_column {
                    let idx = output_grid.idx(from_xyz(col0,col1,neuron)) as usize;
                    self.neurons[idx].add_uniform_rand_inputs_from_area(input_range.clone(), input_size, subregion.clone(), synapses_per_segment, rand_seed)
                }
            }
        }
    }
    // pub fn add_conv(&mut self, input_range: Range<Rand>, synapse_count: u32, output_shape: [u32; 3], kernel: [u32; 3], input_shape: [u32; 3], mut rand_seed: u32) -> u32 {
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
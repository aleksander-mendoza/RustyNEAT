// use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
// use std::mem::MaybeUninit;
// use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, RangeBounds};
// use std::fmt::{Display, Formatter, Debug};
// use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
// use crate::cpu_sdr::CpuSDR;
// use crate::htm_program::HtmProgram;
// use ndalgebra::buffer::Buffer;
// use crate::htm::*;
// use crate::cpu_bitset::CpuBitset;
// use std::cmp::Ordering;
// use serde::{Serialize, Deserialize};
// use crate::{Shape, resolve_range, EncoderTarget, Synapse, top_k_indices};
// use std::collections::Bound;
// use crate::vector_field::{VectorFieldOne, VectorFieldDiv, VectorFieldAdd, VectorFieldMul, ArrayCast, VectorFieldSub, VectorFieldPartialOrd};
// use crate::htm_builder::Population;
//
// #[derive(Copy, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
// #[repr(C)]
// pub struct AssemblyConnection {
//     /**index of the presynaptic neuron*/
//     from: usize,
//     /**normalized*/
//     nw: f32,
//     /**weight*/
//     w: f32,
// }
// impl AssemblyConnection{
//     fn new(syn:&Synapse)->Self{
//         Self{
//             from: syn.input_idx,
//             nw: 0.0,
//             w: syn.weight
//         }
//     }
// }
// #[derive(Copy, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
// #[repr(C)]
// pub struct AssemblyNeuron {
//     /**beginning (inclusive) of the range of connections*/
//     from: usize,
//     /**end (exclusive) of the range of connections*/
//     to: usize,
//     bin:u32,
//     p:f32,
// }
//
// impl AssemblyNeuron{
//     fn new(from:usize,to:usize)->Self{
//         Self{
//             from,
//             to,
//             bin: 0,
//             p: 0.0
//         }
//     }
//     fn normalize(&self, connections:&mut [AssemblyConnection]){
//         let sum = self.sum_w(connections);
//         connections[self.from..self.to].iter_mut().for_each(|c|c.nw=c.w/sum);
//     }
//     fn sum_w(&self, connections:&[AssemblyConnection])->f32{
//         connections[self.from..self.to].iter().map(|c|c.w).sum()
//     }
//     fn sum(&self, connections:&[AssemblyConnection],input:&CpuBitset)->f32{
//         connections[self.from..self.to].iter().filter(|c|input.contains(c.from as u32)).map(|c|c.nw).sum()
//     }
// }
//
// #[derive(Serialize, Deserialize, Debug, Clone)]
// pub struct CpuAssembly {
//     connections: Vec<AssemblyConnection>,
//     neurons: Vec<AssemblyNeuron>,
//     bins: Vec<usize>,
// }
//
// impl CpuAssembly {
//     pub fn connections_as_slice(&self) -> &[AssemblyConnection] {
//         self.connections.as_slice()
//     }
//     pub fn connections_as_mut_slice(&mut self) -> &mut [AssemblyConnection] {
//         self.connections.as_mut_slice()
//     }
//     pub fn normalize(&mut self) {
//         let Self{ connections, neurons, bins } = self;
//         neurons.iter().for_each(|n|n.normalize(connections))
//     }
//     pub fn neuron_connections_range(&self, idx: usize) -> Range<usize> {
//         let n = self.neurons[idx];
//         n.from..n.to
//     }
//     pub fn neuron_connections(&self, idx: usize) -> &[AssemblyConnection] {
//         &self.connections[self.neuron_connections_range(idx)]
//     }
//     pub fn neuron_connections_mut(&mut self, idx: usize) -> &mut [AssemblyConnection] {
//         let r = self.neuron_connections_range(idx);
//         &mut self.connections[r]
//     }
//
//     /**n = how many minicolumns to activate. We will always take the top n minicolumns with the greatest overlap value.*/
//     pub fn new(population: Population) -> Self {
//         let mut connections = vec![];
//         let mut neurons = vec![];
//         for neuron in &population.neurons {
//             let conn_start = connections.len();
//             for seg in &neuron.segments {
//                 for syn in &seg.synapses {
//                     connections.push(AssemblyConnection::new(syn));
//                 }
//             }
//             let conn_end = connections.len();
//             neurons.push(AssemblyNeuron::new(conn_start, conn_end));
//         }
//         let mut slf = Self {
//             connections,
//             neurons,
//             bins: vec![0; 128 + 1] //+1 because the binning range is inclusive
//         };
//         slf.normalize();
//         slf
//     }
//
//     pub fn top_n(&mut self, r: impl RangeBounds<usize>, n: u32, input: &CpuBitset, output:&mut impl EncoderTarget) {
//         if n==0{return}
//         let Self{ connections, neurons, bins } = self;
//         let r = resolve_range(neurons.len(), r);
//         assert!(r.len()>=n as usize,"The range {:?} has length {} but n=={}",r,r.len(),n);
//         let neurons = &mut neurons[r.clone()];
//         let bin_count = (bins.len()-1) as f32;
//         let max_p = neurons.iter().map(|n|n.p).reduce(f32::max).unwrap();
//         for neuron in neurons.iter_mut(){
//             let mut sum = neuron.sum(connections,input);
//             assert!(sum >= neuron.p-max_p,"{} {} {}",sum,neuron.p,max_p);
//             sum -= neuron.p-max_p;
//             //notice that sum might be equal 1. That's why we subtract 1 from bin_count
//             let bin = sum * bin_count;
//             neuron.bin = bin as u32;
//         }
//         top_k_indices(n as usize, neurons, bins, |n|n.bin as usize, |o|output.push(o as u32));
//     }
//
//     pub fn top_1(&self, r: impl RangeBounds<usize>, input: &CpuBitset) -> usize {
//         let Self{ connections, neurons, bins } = self;
//         let r = resolve_range(neurons.len(), r);
//         assert!(r.len()>0);
//         let neurons = &neurons[r.clone()];
//         let mut max_sum = -1.;
//         let mut max_neuron = 0;
//         for (neuron_idx,neuron) in neurons.iter().enumerate(){
//             let sum = neuron.sum(connections,input);
//             if sum > max_sum {
//                 max_sum = sum;
//                 max_neuron = neuron_idx;
//             }
//         }
//         r.start+max_neuron
//     }
//
//     /**repeats the top_n procedure independently m times for m consecutive regions*/
//     pub fn top_n_times_m(&mut self, offset:usize, region_size:usize, m_regions:usize, n: u32, input: &CpuBitset, output:&mut impl EncoderTarget) {
//         assert!(offset+region_size*m_regions<=self.neurons.len(),"offset+region_size*m_regions=={} but there are only {} neurons",offset+region_size*m_regions,self.neurons.len());
//         for region in 0..m_regions{
//             let from = offset+region_size*region;
//             self.top_n(from..from+region_size,n,input,output)
//         }
//     }
//
//     /**repeats the top_n procedure independently m times for m consecutive regions*/
//     pub fn top_1_times_m(&self, offset:usize, region_size:usize, m_regions:usize, input: &CpuBitset, output:&mut impl EncoderTarget) {
//         assert!(offset+region_size*m_regions<=self.neurons.len(),"offset+region_size*m_regions=={} but there are only {} neurons",offset+region_size*m_regions,self.neurons.len());
//         for region in 0..m_regions{
//             let from = offset+region_size*region;
//             output.push(self.top_1(from..from+region_size,input) as u32)
//         }
//     }
//
//     pub fn learn(&mut self, r: impl RangeBounds<usize>, plasticity:f32, input: &CpuBitset, output:&mut impl EncoderTarget){
//
//     }
//
// }
//

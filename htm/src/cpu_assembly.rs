use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, RangeBounds};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::cpu_sdr::CpuSDR;
use crate::htm_program::HtmProgram;
use ndalgebra::buffer::Buffer;
use crate::htm::*;
use crate::cpu_bitset::CpuBitset;
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use crate::{Shape, Shape3, Shape2, resolve_range, EncoderTarget};
use std::collections::Bound;
use crate::vector_field::{VectorFieldOne, VectorFieldDiv, VectorFieldAdd, VectorFieldMul, ArrayCast, VectorFieldSub, VectorFieldPartialOrd};
use crate::htm_builder::Population;

#[derive(Copy, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct AssemblyConnection {
    from: usize,
    //index of the presynaptic neuron
    weight: f32,
}

#[derive(Copy, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct AssemblyNeuron {
    from: usize,
    to: usize,
    bin:u32,
}

impl AssemblyNeuron{
    fn sum(&self, connections:&[AssemblyConnection],input:&CpuBitset)->f32{
        let mut sum = 0.;
        for conn in &connections[self.from..self.to] {
            let is_in = input.contains(conn.from as u32);
            sum += conn.weight * is_in as f32;
        }
        sum
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CpuAssembly {
    connections: Vec<AssemblyConnection>,
    neurons: Vec<AssemblyNeuron>,
    pub bins: usize,
}

impl CpuAssembly {
    pub fn connections_as_slice(&self) -> &[AssemblyConnection] {
        self.connections.as_slice()
    }
    pub fn connections_as_mut_slice(&mut self) -> &mut [AssemblyConnection] {
        self.connections.as_mut_slice()
    }
    pub fn normalise_weights(&mut self, val: f32) {}
    pub fn neuron_connections_range(&self, idx: usize) -> Range<usize> {
        let n = self.neurons[idx];
        n.from..n.to
    }
    pub fn neuron_connections(&self, idx: usize) -> &[AssemblyConnection] {
        &self.connections[self.neuron_connections_range(idx)]
    }
    pub fn neuron_connections_mut(&mut self, idx: usize) -> &mut [AssemblyConnection] {
        let r = self.neuron_connections_range(idx);
        &mut self.connections[r]
    }

    /**n = how many minicolumns to activate. We will always take the top n minicolumns with the greatest overlap value.*/
    pub fn new(population: Population) -> Self {
        let mut connections = vec![];
        let mut neurons = vec![];
        for neuron in &population.neurons {
            let conn_start = connections.len();
            let total_synapses: usize = neuron.segments.iter().map(|s| s.synapses.len()).sum();
            let weight = 1f32 / total_synapses as f32;
            for seg in &neuron.segments {
                for &syn in &seg.synapses {
                    connections.push(AssemblyConnection { from: syn, weight });
                }
            }
            let conn_end = connections.len();
            neurons.push(AssemblyNeuron { from: conn_start, to: conn_end, bin: 0 });
        }
        Self {
            connections,
            neurons,
            bins: 128
        }
    }

    pub fn top_n(&mut self, r: impl RangeBounds<usize>, n: u32, input: &CpuBitset, output:&mut impl EncoderTarget) {
        let Self{ connections, neurons, bins } = self;
        let bin_count = *bins;
        let r = resolve_range(neurons.len(), r);
        assert!(r.len()>=n as usize,"The range {:?} has length {} but n=={}",r,r.len(),n);
        let neurons = &mut neurons[r.clone()];
        let mut bins = vec![0u32; bin_count + 1];//+1 because the binning range is inclusive
        let bin_count = bin_count as f32;
        for neuron in neurons{
            let mut sum = neuron.sum(connections,input);
            //notice that sum might be equal 1. That's why the length of bins is self.bins + 1
            let bin = sum * bin_count;
            bins[bin as usize] += 1;
            neuron.bin = bin as u23;
        }
        let mut min_bin_idx = 0;
        let mut total_neurons = 0;
        for (bin_idx, bin) in bins.iter_mut().enumerate().rev() {
            let neurons_in_bin = *bin;
            total_neurons += neurons_in_bin;
            if total_neurons >= n {
                let old_total_neurons = total_neurons - neurons_in_bin;
                let remaining = n - old_total_neurons;
                *bin = remaining;
                min_bin_idx = bin_idx;
                break;
            }
        }
        for (idx,neuron) in neurons.iter().enumerate() {
            let bin_idx = neuron.bin as usize;
            if bin_idx >= min_bin_idx && bins[bin_idx]>0{
                let idx = r.start+idx;
                output.push(idx as u32);
                bins[bin_idx]-=1;
            }
        }
    }

    pub fn top_1(&self, r: impl RangeBounds<usize>, input: &CpuBitset) -> usize {
        let Self{ connections, neurons, bins } = self;
        let r = resolve_range(neurons.len(), r);
        assert!(r.len()>0);
        let neurons = &neurons[r.clone()];
        let mut max_sum = -1.;
        let mut max_neuron = r.start;
        for (neuron_idx,neuron) in neurons.iter().enumerate(){
            let sum = neuron.sum(connections,input);
            if sum > max_sum{
                max_sum = sum;
                max_neuron = neuron_idx;
            }
        }
        r.start+max_neuron
    }

    /**repeats the top_n procedure independently m times for m consecutive regions*/
    pub fn top_n_times_m(&mut self, offset:usize, region_size:usize, m_regions:usize, n: u32, input: &CpuBitset, output:&mut impl EncoderTarget) {
        assert!(offset+region_size*m_regions<=self.neurons.len(),"offset+region_size*m_regions=={} but there are only {} neurons",offset+region_size*m_regions,self.neurons.len());
        for region in 0..m_regions{
            let from = offset+region_size*region;
            self.top_n(from..from+region_size,n,input,output)
        }
    }

    /**repeats the top_n procedure independently m times for m consecutive regions*/
    pub fn top_1_times_m(&self, offset:usize, region_size:usize, m_regions:usize, input: &CpuBitset, output:&mut impl EncoderTarget) {
        assert!(offset+region_size*m_regions<=self.neurons.len(),"offset+region_size*m_regions=={} but there are only {} neurons",offset+region_size*m_regions,self.neurons.len());
        for region in 0..m_regions{
            let from = offset+region_size*region;
            output.push(self.top_1(from..from+region_size,input) as u32)
        }
    }

    pub fn update(&mut self, r: impl RangeBounds<usize>, n: u32, input: &CpuBitset, output:&mut impl EncoderTarget){
        
    }

}


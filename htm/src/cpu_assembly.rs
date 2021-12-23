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
use crate::{Shape, Shape3, Shape2, resolve_range};
use std::collections::Bound;
use crate::vector_field::{VectorFieldOne, VectorFieldDiv, VectorFieldAdd, VectorFieldMul, ArrayCast, VectorFieldSub, VectorFieldPartialOrd};
use crate::htm_builder::Population;
#[derive(Copy, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct AssemblyConnection{
    from:usize,//index of the presynaptic neuron
    weight:f32,
}
#[derive(Copy, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct AssemblyNeuron{
    from:usize,
    to:usize,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CpuAssembly {
    connections: Vec<AssemblyConnection>,
    neurons: Vec<AssemblyNeuron>,
    pub n: u32,
}

impl CpuAssembly {
    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn connections_as_slice(&self) -> &[AssemblyConnection] {
        self.connections.as_slice()
    }
    pub fn connections_as_mut_slice(&mut self) -> &mut [AssemblyConnection] {
        self.connections.as_mut_slice()
    }
    pub fn normalise_weights(&mut self, val: f32) {

    }
    pub fn neuron_connections_range(&self,idx:usize) -> Range<usize> {
        let n = self.neurons[idx];
        n.from..n.to
    }
    pub fn neuron_connections(&self,idx:usize) -> &[AssemblyConnection] {
        &self.connections[self.neuron_connections_range(idx)]
    }
    pub fn neuron_connections_mut(&mut self,idx:usize) -> &mut [AssemblyConnection] {
        let r = self.neuron_connections_range(idx);
        &mut self.connections[r]
    }

    /**n = how many minicolumns to activate. We will always take the top n minicolumns with the greatest overlap value.*/
    pub fn new(population:Population, n: u32) -> Self {
        let mut connections = vec![];
        let mut neurons = vec![];
        for neuron in &population.neurons{
            let conn_start = connections.len();
            let total_synapses:usize = neuron.segments.iter().map(|s|s.synapses.len()).sum();
            let weight = 1f32/total_synapses as f32;
            for seg in &neuron.segments{
                for &syn in &seg.synapses{
                    connections.push(AssemblyConnection { from:syn,weight });
                }
            }
            let conn_end = connections.len();
            neurons.push(AssemblyNeuron{from:conn_start,to:conn_end});
        }
        Self {
            connections,
            n,
            neurons
        }
    }

}


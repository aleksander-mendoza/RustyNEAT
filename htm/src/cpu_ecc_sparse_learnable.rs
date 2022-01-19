// use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
// use std::mem::MaybeUninit;
// use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, RangeBounds, Deref, DerefMut};
// use std::fmt::{Display, Formatter, Debug};
// use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
// use crate::cpu_sdr::CpuSDR;
// use crate::ecc_program::EccProgram;
// use ndalgebra::buffer::Buffer;
// use crate::cpu_bitset::CpuBitset;
// use std::cmp::Ordering;
// use serde::{Serialize, Deserialize};
// use crate::{Shape, resolve_range, EncoderTarget, Synapse, top_large_k_indices, top_small_k_indices, Shape3, from_xyz, Shape2, from_xy, range_contains, SparseOrDense, EccMachine, OclEccSparse, OclEccDense, CpuEccMachine, top_small_k_by_channel, Neuron, DenseWeight};
// use std::collections::{Bound, HashSet};
// use crate::vector_field::{VectorFieldOne, VectorFieldDiv, VectorFieldAdd, VectorFieldMul, ArrayCast, VectorFieldSub, VectorFieldPartialOrd};
// use crate::population::Population;
// use rand::{Rng, SeedableRng};
// use crate::xorshift::{auto_gen_seed64, xorshift64, auto_gen_seed, xorshift, xorshift32, auto_gen_seed32};
// use itertools::{Itertools, assert_equal};
// use std::iter::Sum;
// use ocl::core::DeviceInfo::MaxConstantArgs;
// use crate::ecc::{EccLayer, as_usize, Idx, as_idx, Rand, xorshift_rand};
// use crate::sdr::SDR;
// #[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
// pub struct EccNeuronSL{
//     outgoing_to_excitatory:Vec<(D,Idx)>,
//     outgoing_to_inhibitory:Vec<Idx>,
// }
// impl EccNeuronSL{
//     fn new()->Self{
//         Self{
//             outgoing_to_excitatory: vec![],
//             outgoing_to_inhibitory: vec![]
//         }
//     }
// }
// type D = f32;
// #[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
// pub struct CpuEccSparseLearnable {
//
//     connections: Vec<EccNeuronSL>,
//     //k is always 1
//     pub plasticity: D,
//     pub active_neurons:CpuSDR,
//     activity: Vec<D>,
//     pub sums: Vec<D>,
//     pub max: Vec<D>,
// }
//
// impl CpuEccSparseLearnable{
//
//
//     pub fn new(excitatory:Population,inhibitory:Population) -> Self {
//         let mut connections: Vec<EccNeuronSL> = (0..excitatory.len()).map(|_|EccNeuronSL::new()).collect();
//         for (out_idx, neuron) in excitatory.neurons.iter().enumerate() {
//             for seg in &neuron.segments {
//                 for syn in &seg.synapses {
//                     assert!(syn.input_idx<excitatory.len(),"Excitatory neuron {} connects to {} but there are only {} excitatory neurons",out_idx,syn.input_idx,excitatory.len());
//                     connections[syn.input_idx].outgoing_to_excitatory.push((syn.weight,as_idx(out_idx)));
//                 }
//             }
//         }
//         for (out_idx, neuron) in inhibitory.neurons.iter().enumerate() {
//             for seg in &neuron.segments {
//                 for syn in &seg.synapses {
//                     assert!(syn.input_idx<excitatory.len(),"Inhibitory neuron {} connects to {} but there are only {} excitatory neurons",out_idx,syn.input_idx,excitatory.len());
//                     connections[syn.input_idx].outgoing_to_inhibitory.push(as_idx(out_idx));
//                 }
//             }
//         }
//         Self {
//             connections,
//             plasticity: 0.0001,
//             activity: vec![f32::INITIAL_ACTIVITY;excitatory.len()],
//             sums: vec![0.;excitatory.len()],
//             max: vec![0.;inhibitory.len()]
//         }
//     }
//
//     pub fn set_plasticity(&mut self, plasticity: D) {
//         self.plasticity = plasticity
//     }
//     pub fn get_plasticity(&self) -> D {
//         self.plasticity
//     }
//
//     pub fn min_activity(&self) -> D {
//         self.activity.iter().cloned().reduce(D::min_w).unwrap()
//     }
//     pub fn max_activity(&self) -> D {
//         self.activity.iter().cloned().reduce(D::max_w).unwrap()
//     }
//     pub fn activity(&self, output_idx: usize) -> D {
//         self.activity[output_idx]
//     }
//     pub fn get_activity(&self) -> &[D] {
//         &self.activity
//     }
//     pub fn get_activity_mut(&mut self) -> &mut [D] {
//         &mut self.activity
//     }
//     pub fn set_initial_activity(&mut self) {
//         self.activity.fill(D::INITIAL_ACTIVITY)
//     }
//
//     pub fn step(&mut self){
//         let Self{ connections, plasticity, active_neurons, activity, sums, max } = self;
//         sums.fill(0.);
//         for &neuron in active_neurons.iter(){
//             let neuron = &mut connections[as_usize(neuron)];
//             for (weight,postsynaptic_neuron) in neuron.outgoing_to_excitatory {
//                 sums[as_usize(postsynaptic_neuron)] += weight;
//             }
//         }
//         for (neuron,&w_sum) in connections.iter().zip(sums.iter()) {
//             for inh in neuron.outgoing_to_inhibitory {
//                 let m = &mut max[as_usize(inh)];
//                 *m=m.max(w_sum);
//             }
//         }
//     }
// }
//
// #[cfg(test)]
// mod tests {
//
// }
// use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
// use std::mem::MaybeUninit;
// use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
// use std::fmt::{Display, Formatter, Debug};
// use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
// use crate::cpu_sdr::CpuSDR;
// use crate::htm_program::HtmProgram;
// use ndalgebra::buffer::Buffer;
// use crate::htm3::*;
//
// /***This implementation is the same as CpuHTM, but it additionally implements negatative
// proximal synapses.*/
// #[derive(Clone)]
// pub struct CpuHTM3 {
//     feedforward_connections: Vec<HtmFeedforwardConnection33>,
//     connection_indices: Vec<u32>,
//     inputs: Vec<HtmInput33>,
//     minicolumns: Vec<HtmMinicolumn33>,
//     pub permanence_threshold: f32,
//     pub n: u32,
//     pub permanence_decrement_increment: [f32; 2],
//     pub max_overlap: u32,
// }
//
// impl CpuHTM3 {
//     pub fn permanence_threshold(&self) -> f32{
//         self.permanence_threshold
//     }
//     pub fn n(&self) -> u32{
//         self.n
//     }
//     pub fn permanence_decrement_increment(&self) -> [f32; 2]{
//         self.permanence_decrement_increment
//     }
//     pub fn max_overlap(&self) -> u32{
//         self.max_overlap
//     }
//     pub fn feedforward_connections_as_slice(&self)->&[HtmFeedforwardConnection3]{
//         self.feedforward_connections.as_slice()
//     }
//     pub fn connection_indices_as_slice(&self)->&[u32]{
//         self.connection_indices.as_slice()
//     }
//     pub fn inputs_as_slice(&self)->&[HtmInput3]{
//         self.inputs.as_slice()
//     }
//     pub fn minicolumns_as_slice(&self)->&[HtmMinicolumn3]{
//         self.minicolumns.as_slice()
//     }
//     pub fn new_globally_uniform_prob(input_size: u32, minicolumns: u32, n: u32, permanence_threshold: f32, permanence_decrement: f32, permanence_increment: f32, inputs_per_minicolumn: u32) -> Self {
//         assert!(inputs_per_minicolumn < minicolumns);
//         Self::new(input_size, minicolumns, n, permanence_threshold, permanence_decrement, permanence_increment, |minicolumn_id| rand::random::<u32>() % minicolumns, |minicolumn_id| inputs_per_minicolumn)
//     }
//     /**n = how many minicolumns to activate. We will always take the top n minicolumns with the greatest overlap value.*/
//     pub fn new(input_size: u32, minicolumns_count: u32, n: u32, permanence_threshold: f32, permanence_decrement: f32, permanence_increment: f32, mut random_input_close_to_minicolumn: impl FnMut(u32) -> u32, mut input_count_incoming_to_minicolumn: impl FnMut(u32) -> (u32,u32)) -> Self {
//         let mut feedforward_connections: Vec<HtmFeedforwardConnection3> = vec![];
//         let mut connection_indices = vec![];
//         let mut inputs = Vec::with_capacity(input_size as usize);
//         let mut minicolumns: Vec<HtmMinicolumn3> = Vec::with_capacity(minicolumns_count as usize);
//         let mut minicolumns_per_input: Vec<(Vec<u32>,Vec<u32>)> = (0..input_size).map(|_| vec![]).collect();
//
//         let mut connected_inputs = vec![false; input_size as usize];
//         for minicolumn_id in 0..minicolumns_count as u32 {
//             let (excitatory_input_count,inhibitory_input_count) = input_count_incoming_to_minicolumn(minicolumn_id);
//             assert!(excitatory_input_count+inhibitory_input_count<=input_size);
//             let mut inputs_to_this_minicolumns: Vec<u32> = vec![];
//             for _ in 0..excitatory_input_count {
//                 let mut input_id = random_input_close_to_minicolumn(minicolumn_id);
//                 while connected_inputs[input_id as usize] { // find some input that has not been connected to this minicolumn yet
//                     input_id = random_input_close_to_minicolumn(minicolumn_id)
//                 }
//                 connected_inputs[input_id as usize] = true;
//                 minicolumns_per_input[input_id as usize].0.push(minicolumn_id);
//                 inputs_to_this_minicolumns.push(input_id);
//             }
//             for _ in 0..inhibitory_input_count {
//                 let mut input_id = random_input_close_to_minicolumn(minicolumn_id);
//                 while connected_inputs[input_id as usize] { // find some input that has not been connected to this minicolumn yet
//                     input_id = random_input_close_to_minicolumn(minicolumn_id)
//                 }
//                 connected_inputs[input_id as usize] = true;
//                 minicolumns_per_input[input_id as usize].1.push(minicolumn_id);
//                 inputs_to_this_minicolumns.push(input_id);
//             }
//             for input_id in inputs_to_this_minicolumns {
//                 connected_inputs[input_id as usize] = false;
//             }
//         }
//         let mut connections_per_minicolumn: Vec<Vec<u32>> = (0..minicolumns_count).map(|_| vec![]).collect();
//         for (input_id, (excited_minicolumn_ids,inhibited_minicolumn_ids)) in minicolumns_per_input.into_iter().enumerate() {
//             let connections_begin = feedforward_connections.len() as u32;
//             for minicolumn_id in minicolumn_ids {
//                 let connection_id = feedforward_connections.len() as u32;
//                 feedforward_connections.push(HtmFeedforwardConnection3 {
//                     minicolumn_id,
//                     permanence: rand::random::<f32>(),
//                     input_id: input_id as u32,
//                     is_inhibitory: false
//                 });
//                 connections_per_minicolumn[minicolumn_id as usize].push(connection_id);
//             }
//
//             let connections_end = feedforward_connections.len() as u32;
//             inputs.push(HtmInput3 {
//                 connection_offset: connections_begin,
//                 connection_len: connections_end - connections_begin,
//                 is_active: false,
//             });
//         }
//         let max_overlap = connections_per_minicolumn.iter().map(Vec::len).max().unwrap() as u32;
//         assert!(max_overlap > 0);
//         for connection_ids in connections_per_minicolumn.into_iter() {
//             let connection_indices_begin = connection_indices.len() as u32;
//             connection_indices.extend_from_slice(connection_ids.as_slice());
//             let connection_indices_end = connection_indices.len() as u32;
//             minicolumns.push(HtmMinicolumn3 {
//                 connection_index_offset: connection_indices_begin,
//                 connection_index_len: connection_indices_end - connection_indices_begin,
//                 overlap: 0,
//             });
//         }
//         Self {
//             feedforward_connections,
//             connection_indices,
//             inputs,
//             minicolumns,
//             permanence_threshold,
//             n,
//             permanence_decrement_increment: [permanence_decrement, permanence_increment],
//             max_overlap,
//         }
//     }
//
//     fn htm_calculate_overlap(&mut self, sdr_input: &CpuSDR){
//         for &input_neuron_idx in sdr_input.iter() {
//             self.inputs[input_neuron_idx as usize].is_active = true;
//             let input_neuron = &self.inputs[input_neuron_idx as usize];
//             for i in 0..input_neuron.connection_len {
//                 let connection_idx = input_neuron.connection_offset + i;
//                 if self.feedforward_connections[connection_idx as usize].permanence > self.permanence_threshold {
//                     let minicolumn_id = self.feedforward_connections[connection_idx as usize].minicolumn_id;
//                     self.minicolumns[minicolumn_id as usize].overlap += 1;
//                 }
//             }
//         }
//     }
//
//     fn htm_calculate_number_of_minicolumns_per_overlap(&mut self, sdr_input: &CpuSDR, number_of_minicolumns_per_overlap: &mut [i32]) {
//         for &input_neuron_idx in sdr_input.iter() {
//             let input_neuron = &self.inputs[input_neuron_idx as usize];
//             for i in 0..input_neuron.connection_len {
//                 let connection_idx = input_neuron.connection_offset + i;
//                 if self.feedforward_connections[connection_idx as usize].permanence > self.permanence_threshold {
//                     let minicolumn_idx = self.feedforward_connections[connection_idx as usize].minicolumn_id;
//                     let overlap = self.minicolumns[minicolumn_idx as usize].overlap;
//                     if overlap > 0 {
//                         self.minicolumns[minicolumn_idx as usize].overlap = -overlap;
//                         number_of_minicolumns_per_overlap[overlap as usize] += 1;
//                     }
//                 }
//             }
//         }
//     }
//
//     /**returns smallest_overlap_that_made_it_to_top_n.
//     By the end of running this function, the number_of_minicolumns_per_overlap array will become
//     number_of_minicolumns_per_overlap_that_made_it_to_top_n.
//     number_of_minicolumns_per_overlap_that_made_it_to_top_n holds rubbish for any overlap lower than smallest_overlap_that_made_it_to_top_n
//     */
//     fn htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&self, number_of_minicolumns_per_overlap: &mut [i32]) -> u32 {
//         let mut total_minicolumns = 0;
//         for overlap in (0..number_of_minicolumns_per_overlap.len()).rev() {
//             let number_of_minicolumns = number_of_minicolumns_per_overlap[overlap as usize];
//             total_minicolumns += number_of_minicolumns;
//             if total_minicolumns > self.n as i32 {
//                 number_of_minicolumns_per_overlap[overlap as usize] = self.n as i32 - (total_minicolumns - number_of_minicolumns);
//                 return overlap as u32;
//             }
//         }
//         0
//     }
//
//
//     /**
//     precodntion: current_top_n_minicolumn_idx==0 ;
//     postcondition: current_top_n_minicolumn_idx <= n ;
//     precodnition: top_n_minicolumns.len() >= n*/
//     fn htm_find_top_minicolumns(&mut self, sdr_input: &CpuSDR,
//                                 number_of_minicolumns_per_overlap_that_made_it_to_top_n: &mut [i32],
//                                 smallest_overlap_that_made_it_to_top_n: u32,
//                                 top_n_minicolumns: &mut [u32],
//                                 current_top_n_minicolumn_idx: &mut u32)  {
//         for &input_neuron_idx in sdr_input.iter() {
//             let input_neuron = &self.inputs[input_neuron_idx as usize];
//             for i in 0..input_neuron.connection_len {
//                 let connection_idx = input_neuron.connection_offset + i;
//                 if self.feedforward_connections[connection_idx as usize].permanence > self.permanence_threshold {
//                     let minicolumn_idx = self.feedforward_connections[connection_idx as usize].minicolumn_id;
//                     let overlap_negative = self.minicolumns[minicolumn_idx as usize].overlap;
//                     let overlap = -overlap_negative;
//                     if overlap >= smallest_overlap_that_made_it_to_top_n as i32 && // the array number_of_minicolumns_per_overlap_that_made_it_to_top_n holds rubbish for any overlap lower than smallest_overlap_that_made_it_to_top_n
//                         self.minicolumns[minicolumn_idx as usize].overlap != 0 { //avoid adding the same column multiple times
//                         self.minicolumns[minicolumn_idx as usize].overlap = 0;
//                         if number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap as usize] > 0 { // only add those columns that made it to top n
//                             number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap as usize] -= 1;
//                             top_n_minicolumns[*current_top_n_minicolumn_idx as usize] = minicolumn_idx;
//                             *current_top_n_minicolumn_idx += 1;
//                         }
//                     }
//                 }
//             }
//         }
//     }
//
//
//     fn htm_update_permanence(&mut self,
//                              top_n_minicolumns: &mut [u32],
//                              current_top_n_minicolumn_idx: u32) {
//         for top_minicolumn_idx in 0..current_top_n_minicolumn_idx as usize {
//             let minicolumn_idx: u32 = top_n_minicolumns[top_minicolumn_idx];
//             let connection_index_offset = self.minicolumns[minicolumn_idx as usize].connection_index_offset;
//             let connection_index_len = self.minicolumns[minicolumn_idx as usize].connection_index_len;
//             for i in 0..connection_index_len {
//                 let feedforward_connection_idx = self.connection_indices[(connection_index_offset + i) as usize];
//                 let input_id = self.feedforward_connections[feedforward_connection_idx as usize].input_id;
//                 let permanence_change = self.permanence_decrement_increment[self.inputs[input_id as usize].is_active as usize];
//                 let old_permanence = self.feedforward_connections[feedforward_connection_idx as usize].permanence;
//                 let new_permanence = (old_permanence + permanence_change).clamp(0., 1.);
//                 self.feedforward_connections[feedforward_connection_idx as usize].permanence = new_permanence;
//             }
//         }
//     }
//
//     fn htm_clean_up_active_inputs(&mut self, sdr_input: &CpuSDR) {
//         for &input_neuron_idx in sdr_input.iter() {
//             self.inputs[input_neuron_idx as usize].is_active = false;
//         }
//     }
//     fn htm_clean_up_overlap(&mut self, sdr_input: &CpuSDR) {
//         for &input_neuron_idx in sdr_input.iter() {
//             self.inputs[input_neuron_idx as usize].is_active = true;
//             let input_neuron = &self.inputs[input_neuron_idx as usize];
//             for i in 0..input_neuron.connection_len {
//                 let connection_idx = input_neuron.connection_offset + i;
//                 let minicolumn_id = self.feedforward_connections[connection_idx as usize].minicolumn_id;
//                 self.minicolumns[minicolumn_id as usize].overlap = 0;
//             }
//         }
//     }
//
//
//
//     pub fn infer(&mut self, sdr_input: &CpuSDR, learn: bool) -> CpuSDR {
//         self.htm_calculate_overlap(sdr_input);
//         let mut number_of_minicolumns_per_overlap = vec![0; self.max_overlap as usize];
//         self.htm_calculate_number_of_minicolumns_per_overlap(sdr_input, number_of_minicolumns_per_overlap.as_mut_slice());
//         let smallest_overlap_that_made_it_to_top_n = self.htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(number_of_minicolumns_per_overlap.as_mut_slice());
//         let mut top_n_minicolumns = Vec::with_capacity(self.n as usize);
//         unsafe { top_n_minicolumns.set_len(self.n as usize) }
//         let mut current_top_n_minicolumn_idx = 0;
//         self.htm_find_top_minicolumns(sdr_input, number_of_minicolumns_per_overlap.as_mut_slice(), smallest_overlap_that_made_it_to_top_n, top_n_minicolumns.as_mut_slice(), &mut current_top_n_minicolumn_idx);
//         let top_minicolumn_count = current_top_n_minicolumn_idx;
//         if learn {
//             self.htm_update_permanence(top_n_minicolumns.as_mut_slice(), top_minicolumn_count)
//         }
//         self.htm_clean_up_active_inputs(sdr_input);
//         self.htm_clean_up_overlap(sdr_input);
//         CpuSDR::from(top_n_minicolumns)
//     }
//
// }
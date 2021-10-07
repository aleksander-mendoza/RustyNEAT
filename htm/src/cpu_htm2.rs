use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::cpu_sdr::CpuSDR;
use crate::htm_program::HtmProgram;
use ndalgebra::buffer::Buffer;
use crate::htm2::*;
use crate::cpu_htm::CpuHTM;


#[derive(Clone)]
pub struct CpuHTM2 {
    feedforward_connections: Vec<HtmFeedforwardConnection2>,
    inputs: Vec<u32>,
    minicolumns: Vec<HtmMinicolumn2>,
    permanence_threshold: f32,
    n: u32,
    permanence_decrement_increment: [f32; 2],
    max_overlap: u32,
}

impl From<&CpuHTM> for CpuHTM2{
    fn from(htm: &CpuHTM) -> Self {
        Self{
            feedforward_connections: htm.connection_indices_as_slice().iter().map(| &connection_index|{
                let feedforward_connection = &htm.feedforward_connections_as_slice()[connection_index as usize];
                HtmFeedforwardConnection2{
                    permanence: feedforward_connection.permanence,
                    input_id: feedforward_connection.input_id
                }
            }).collect(),
            inputs: vec![0u32;(htm.inputs_as_slice().len()+31)/32],
            minicolumns: htm.minicolumns_as_slice().iter().map(|m|HtmMinicolumn2{
                connection_offset: m.connection_index_offset,
                connection_len: m.connection_index_len,
                overlap: 0
            }).collect(),
            permanence_threshold: htm.permanence_threshold(),
            n: htm.n(),
            permanence_decrement_increment: htm.permanence_decrement_increment(),
            max_overlap: htm.max_overlap()
        }
    }
}

impl CpuHTM2 {
    pub fn permanence_threshold(&self) -> f32{
        self.permanence_threshold
    }
    pub fn n(&self) -> u32{
        self.n
    }
    pub fn permanence_decrement_increment(&self) -> [f32; 2]{
        self.permanence_decrement_increment
    }
    pub fn max_overlap(&self) -> u32{
        self.max_overlap
    }
    pub fn feedforward_connections_as_slice(&self)->&[HtmFeedforwardConnection2]{
        self.feedforward_connections.as_slice()
    }
    pub fn inputs_as_slice(&self)->&[u32]{
        self.inputs.as_slice()
    }
    pub fn minicolumns_as_slice(&self)->&[HtmMinicolumn2]{
        self.minicolumns.as_slice()
    }
    pub fn new_globally_uniform_prob(input_size: u32, minicolumns: u32, n: u32, permanence_threshold: f32, permanence_decrement: f32, permanence_increment: f32, inputs_per_minicolumn: u32) -> Self {
        assert!(inputs_per_minicolumn < minicolumns);
        Self::new(input_size, minicolumns, n, permanence_threshold, permanence_decrement, permanence_increment, |minicolumn_id| rand::random::<u32>() % minicolumns, |minicolumn_id| inputs_per_minicolumn)
    }
    /**n = how many minicolumns to activate. We will always take the top n minicolumns with the greatest overlap value.*/
    pub fn new(input_size: u32, minicolumns_count: u32, n: u32, permanence_threshold: f32, permanence_decrement: f32, permanence_increment: f32, mut random_input_close_to_minicolumn: impl FnMut(u32) -> u32, mut input_count_incoming_to_minicolumn: impl FnMut(u32) -> u32) -> Self {
        let mut feedforward_connections: Vec<HtmFeedforwardConnection2> = vec![];
        let mut inputs = vec![0u32;(input_size as usize+31)/32];
        let mut minicolumns: Vec<HtmMinicolumn2> = Vec::with_capacity(minicolumns_count as usize);

        let mut connected_inputs = vec![false; input_size as usize];
        for minicolumn_id in 0..minicolumns_count as u32 {
            let input_count = input_count_incoming_to_minicolumn(minicolumn_id);
            let mut inputs_to_this_minicolumns: Vec<u32> = vec![];
            let connection_begin = feedforward_connections.len() as u32;
            for _ in 0..input_count {
                let mut input_id = random_input_close_to_minicolumn(minicolumn_id);
                while connected_inputs[input_id as usize] { // find some input that has not been connected to this minicolumn yet
                    input_id = random_input_close_to_minicolumn(minicolumn_id)
                }
                connected_inputs[input_id as usize] = true;
                feedforward_connections.push(HtmFeedforwardConnection2 {
                    permanence: rand::random::<f32>(),
                    input_id: input_id as u32,
                });
                inputs_to_this_minicolumns.push(input_id);
            }
            minicolumns.push(HtmMinicolumn2 {
                connection_offset: connection_begin,
                connection_len: inputs_to_this_minicolumns.len() as u32,
                overlap: 0,
            });
            for input_id in inputs_to_this_minicolumns {
                connected_inputs[input_id as usize] = false;
            }
        }

        Self {
            max_overlap:minicolumns.iter().map(|m|m.connection_len).max().unwrap(),
            feedforward_connections,
            inputs,
            minicolumns,
            permanence_threshold,
            n,
            permanence_decrement_increment: [permanence_decrement, permanence_increment],

        }
    }


    /**This function does the exact same thing as htm_calculate_overlap, but that function works
    optimally when the input is so sparse that only a tiny fraction of minicolumns has even a single
    connection to some active input. In cases where vast majority minicolumns is expected to have
    at least one connection to some active input, then htm_calculate_overlap2 will be much more optimal.
    The htm_calculate_overlap2 is implemented in two parts. First you call htm_calculate_overlap2_active_inputs
    and then you call htm_calculate_overlap2_overlap_per_minicolumn*/
    fn htm_calculate_overlap2_active_inputs(&mut self, sdr_input: &CpuSDR) {
        for &input_neuron_idx in sdr_input.iter() {
            // u32 has 32 bits
            // self.inputs stores one bit per input in form of u32 integers
            // input_neuron_idx/32 gives index of the u32 integer that contains input_neuron_idx-th bit
            // input_neuron_idx/32 == input_neuron_idx>>5
            // input_neuron_idx%32 gives us index of the input_neuron_idx-th bit but relative to the u32 that contains it
            // input_neuron_idx%32 == input_neuron_idx&31
            // we might either do  1<<(input_neuron_idx&31) or 2147483648>>(input_neuron_idx&31) . Both are equivalent. It only changes the order in which we store bits within each u32
            self.inputs[(input_neuron_idx>>5) as usize] |= 1<<(input_neuron_idx&31);
        }
    }

    fn is_input_active(&self,input_id:u32)->bool{
        (self.inputs[(input_id>>5) as usize] & (1 << (input_id & 31))) != 0
    }
    fn htm_calculate_overlap2_overlap_per_minicolumn(&mut self, number_of_minicolumns_per_overlap: &mut [i32]) {
        for minicolumn_idx in 0..self.minicolumns.len() {
            let connection_offset = self.minicolumns[minicolumn_idx].connection_offset;
            let connection_len = self.minicolumns[minicolumn_idx].connection_len;
            let mut overlap = 0;
            for feedforward_connection_idx in connection_offset..(connection_offset+connection_len) {
                if self.feedforward_connections[feedforward_connection_idx as usize].permanence > self.permanence_threshold {
                    let input_id = self.feedforward_connections[feedforward_connection_idx as usize].input_id;
                    if self.is_input_active(input_id){
                        overlap += 1;
                    }
                }
            }
            if overlap > 0 {
                number_of_minicolumns_per_overlap[overlap as usize] += 1;
            }
            self.minicolumns[minicolumn_idx].overlap = overlap;
        }
    }

    /**returns smallest_overlap_that_made_it_to_top_n.
    By the end of running this function, the number_of_minicolumns_per_overlap array will become
    number_of_minicolumns_per_overlap_that_made_it_to_top_n.
    number_of_minicolumns_per_overlap_that_made_it_to_top_n holds rubbish for any overlap lower than smallest_overlap_that_made_it_to_top_n
    */
    fn htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&self, number_of_minicolumns_per_overlap: &mut [i32]) -> u32 {
        let mut total_minicolumns = 0;
        for overlap in (0..number_of_minicolumns_per_overlap.len()).rev() {
            let number_of_minicolumns = number_of_minicolumns_per_overlap[overlap as usize];
            total_minicolumns += number_of_minicolumns;
            if total_minicolumns > self.n as i32 {
                number_of_minicolumns_per_overlap[overlap as usize] = self.n as i32 - (total_minicolumns - number_of_minicolumns);
                return overlap as u32;
            }
        }
        0
    }

    fn htm_update_permanence(&mut self,
                             top_n_minicolumns: &mut [u32],
                             current_top_n_minicolumn_idx: u32) {
        for top_minicolumn_idx in 0..current_top_n_minicolumn_idx as usize {
            let minicolumn_idx: u32 = top_n_minicolumns[top_minicolumn_idx];
            let connection_offset = self.minicolumns[minicolumn_idx as usize].connection_offset;
            let connection_len = self.minicolumns[minicolumn_idx as usize].connection_len;
            for feedforward_connection_idx in connection_offset..(connection_offset+connection_len) {
                let input_id = self.feedforward_connections[feedforward_connection_idx as usize].input_id;
                let permanence_change = self.permanence_decrement_increment[self.is_input_active(input_id) as usize];
                let old_permanence = self.feedforward_connections[feedforward_connection_idx as usize].permanence;
                let new_permanence = (old_permanence + permanence_change).clamp(0., 1.);
                self.feedforward_connections[feedforward_connection_idx as usize].permanence = new_permanence;
            }
        }
    }

    fn htm_clean_up_active_inputs(&mut self, sdr_input: &CpuSDR) {
        for &input_neuron_idx in sdr_input.iter() {
            self.inputs[(input_neuron_idx/32) as usize] = 0;
        }
    }

    /**This function does the exact same thing as htm_find_top_minicolumns, but that function works
    optimally when the input is so sparse that only a tiny fraction of minicolumns has even a single
    connection to some active input. In cases where vast majority minicolumns is expected to have
    at least one connection to some active input, then htm_find_top_minicolumns2 will be much more optimal.
    */
    fn htm_find_top_minicolumns2(&mut self,
                                 number_of_minicolumns_per_overlap_that_made_it_to_top_n: &mut [i32],
                                 smallest_overlap_that_made_it_to_top_n: u32,
                                 top_n_minicolumns: &mut [u32],
                                 current_top_n_minicolumn_idx: &mut u32) {
        for minicolumn_idx in 0..self.minicolumns.len() {
            let overlap = self.minicolumns[minicolumn_idx].overlap;
            self.minicolumns[minicolumn_idx].overlap = 0;
            if overlap >= smallest_overlap_that_made_it_to_top_n as i32 { // the array number_of_minicolumns_per_overlap_that_made_it_to_top_n holds rubbish for any overlap lower than smallest_overlap_that_made_it_to_top_n
                if number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap as usize] > 0 { // only add those columns that made it to top n
                    number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap as usize] -= 1;
                    top_n_minicolumns[*current_top_n_minicolumn_idx as usize] = minicolumn_idx as u32;
                    *current_top_n_minicolumn_idx += 1;
                }
            }
        }
    }

    pub fn infer2(&mut self, sdr_input: &CpuSDR, learn: bool) -> CpuSDR{
        self.htm_calculate_overlap2_active_inputs(sdr_input);
        let mut number_of_minicolumns_per_overlap = vec![0; self.max_overlap as usize];
        self.htm_calculate_overlap2_overlap_per_minicolumn(number_of_minicolumns_per_overlap.as_mut_slice());
        let smallest_overlap_that_made_it_to_top_n = self.htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(number_of_minicolumns_per_overlap.as_mut_slice());
        let mut top_n_minicolumns = Vec::with_capacity(self.n as usize);
        unsafe { top_n_minicolumns.set_len(self.n as usize) }
        let mut current_top_n_minicolumn_idx = 0;
        self.htm_find_top_minicolumns2(number_of_minicolumns_per_overlap.as_mut_slice(), smallest_overlap_that_made_it_to_top_n, top_n_minicolumns.as_mut_slice(), &mut current_top_n_minicolumn_idx);
        let top_minicolumn_count = current_top_n_minicolumn_idx;
        if learn {
            self.htm_update_permanence(top_n_minicolumns.as_mut_slice(), top_minicolumn_count)
        }
        self.htm_clean_up_active_inputs(sdr_input);
        CpuSDR::from(top_n_minicolumns)
    }
}

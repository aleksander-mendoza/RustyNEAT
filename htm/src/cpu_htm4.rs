use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::cpu_sdr::CpuSDR;
use crate::htm_program::HtmProgram;
use ndalgebra::buffer::Buffer;
use crate::htm4::*;
use crate::cpu_bitset::CpuBitset;
use crate::rand::{xorshift32, rand_u32_to_random_f32};

/***This implementation assumes that most of the time  vast majority of minicolumns are connected to at least one active
input. Hence instead of iterating the input and then visiting only connected minicolumns, it's better to just iterate all
minicolumns. If you are confident that your input is so sparse than only a sparse number of minicolumns
sees some active connections at any time, then use CpuHTM. It will only visit those minicolumns that are connected
to some active input.

This implementation allows some feedforward connections to be inhibitory (this is predetermined upon initialization of HTM and cannot change later)*/
#[derive(Clone)]
pub struct CpuHTM4 {
    feedforward_connections: Vec<HtmFeedforwardConnection4>,
    minicolumns: Vec<HtmMinicolumn4>,
    input_size: u32,
    pub permanence_threshold: f32,
    pub n: u32,
    pub permanence_decrement_increment: [f32; 2],
    pub max_overlap: u32,
}

impl CpuHTM4 {
    pub fn input_size(&self) -> u32 {
        self.input_size
    }
    pub fn permanence_threshold(&self) -> f32 {
        self.permanence_threshold
    }
    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn permanence_decrement_increment(&self) -> [f32; 2] {
        self.permanence_decrement_increment
    }
    pub fn max_overlap(&self) -> u32 {
        self.max_overlap
    }
    pub fn feedforward_connections_as_slice(&self) -> &[HtmFeedforwardConnection4] {
        self.feedforward_connections.as_slice()
    }
    pub fn feedforward_connections_as_mut_slice(&mut self) -> &mut [HtmFeedforwardConnection4] {
        self.feedforward_connections.as_mut_slice()
    }
    pub fn minicolumns_as_slice(&self) -> &[HtmMinicolumn4] {
        self.minicolumns.as_slice()
    }
    pub fn new_globally_uniform_prob(input_size: u32, minicolumns: u32, n: u32, inputs_per_minicolumn: u32, excitatory_connection_probability: f32, mut rand_seed: u32) -> Self {
        assert!(inputs_per_minicolumn < minicolumns);
        Self::new(input_size, minicolumns, n, |minicolumn_id, synapse_id| {
            rand_seed = xorshift32(rand_seed);
            let input_idx = rand_seed % input_size;
            rand_seed = xorshift32(rand_seed);
            let permanence = rand_u32_to_random_f32(rand_seed);
            rand_seed = xorshift32(rand_seed);
            let is_excitatory = rand_u32_to_random_f32(rand_seed) <= excitatory_connection_probability;
            (input_idx, is_excitatory, permanence)
        }, |minicolumn_id| inputs_per_minicolumn)
    }
    pub fn new_globally_uniform_prob_exact_inhibitory(input_size: u32, minicolumns: u32, n: u32, inputs_per_minicolumn: u32, inhibitory_inputs_per_minicolumn: u32, mut rand_seed: u32) -> Self {
        assert!(inputs_per_minicolumn < minicolumns);
        assert!(inhibitory_inputs_per_minicolumn <= inputs_per_minicolumn);
        Self::new(input_size, minicolumns, n, |minicolumn_id, synapse_id| {
            rand_seed = xorshift32(rand_seed);
            let input_idx = rand_seed % input_size;
            rand_seed = xorshift32(rand_seed);
            let permanence = rand_u32_to_random_f32(rand_seed);
            let is_excitatory = inhibitory_inputs_per_minicolumn <= synapse_id;
            (input_idx, is_excitatory, permanence)
        }, |minicolumn_id| inputs_per_minicolumn)
    }
    pub fn new_globally_uniform_prob_without_inhibitory(input_size: u32, minicolumns: u32, n: u32, inputs_per_minicolumn: u32, excitatory_connection_probability: f32, mut rand_seed: u32) -> Self {
        assert!(inputs_per_minicolumn < minicolumns);
        Self::new(input_size, minicolumns, n, |minicolumn_id, synapse_id| {
            rand_seed = xorshift32(rand_seed);
            let input_idx = rand_seed % input_size;
            rand_seed = xorshift32(rand_seed);
            let permanence = rand_u32_to_random_f32(rand_seed);
            rand_seed = xorshift32(rand_seed);
            let is_excitatory = rand_u32_to_random_f32(rand_seed) <= excitatory_connection_probability;
            (input_idx, is_excitatory, if is_excitatory { permanence } else { 0. })
        }, |minicolumn_id| inputs_per_minicolumn)
    }
    /**n = how many minicolumns to activate. We will always take the top n minicolumns with the greatest overlap value.*/
    pub fn new(input_size: u32, minicolumns_count: u32, n: u32, mut random_input_close_to_minicolumn: impl FnMut(u32, u32) -> (u32, bool, f32), mut input_count_incoming_to_minicolumn: impl FnMut(u32) -> u32) -> Self {
        let mut feedforward_connections: Vec<HtmFeedforwardConnection4> = vec![];
        let mut minicolumns: Vec<HtmMinicolumn4> = Vec::with_capacity(minicolumns_count as usize);

        let mut connected_inputs = vec![false; input_size as usize];
        for minicolumn_id in 0..minicolumns_count as u32 {
            let input_count = input_count_incoming_to_minicolumn(minicolumn_id);
            assert!(input_count <= input_size, "Minicolumn {} has {} input connections but there are only {} inputs", minicolumn_id, input_count, input_size);
            let mut inputs_to_this_minicolumns: Vec<u32> = vec![];
            let connection_begin = feedforward_connections.len() as u32;
            for synapse_id in 0..input_count {
                let mut input = random_input_close_to_minicolumn(minicolumn_id, synapse_id);
                while connected_inputs[input.0 as usize] { // find some input that has not been connected to this minicolumn yet
                    input.0 = (input.0 + 1) % input_size
                }
                let (input_id, is_excitatory, permanence) = input;
                connected_inputs[input_id as usize] = true;
                feedforward_connections.push(HtmFeedforwardConnection4 {
                    permanence,
                    input_id,
                    overlap_gain: if is_excitatory { 1 } else { -1 },
                });
                inputs_to_this_minicolumns.push(input_id);
            }
            minicolumns.push(HtmMinicolumn4 {
                connection_offset: connection_begin,
                connection_len: inputs_to_this_minicolumns.len() as u32,
                overlap: 0,
            });
            for input_id in inputs_to_this_minicolumns {
                connected_inputs[input_id as usize] = false;
            }
        }

        Self {
            max_overlap: minicolumns.iter().map(|m| m.connection_len).max().unwrap(),
            feedforward_connections,
            minicolumns,
            n,
            permanence_threshold: 0.7,
            permanence_decrement_increment: [-0.01, 0.02],
            input_size,
        }
    }

    fn htm_calculate_overlap4(&mut self, bitset_input: &CpuBitset, number_of_minicolumns_per_overlap: &mut [i32]) {
        for minicolumn_idx in 0..self.minicolumns.len() {
            let connection_offset = self.minicolumns[minicolumn_idx].connection_offset;
            let connection_len = self.minicolumns[minicolumn_idx].connection_len;
            let mut overlap = 0i32;
            for feedforward_connection_idx in connection_offset..(connection_offset + connection_len) {
                if self.feedforward_connections[feedforward_connection_idx as usize].permanence > self.permanence_threshold {
                    let input_id = self.feedforward_connections[feedforward_connection_idx as usize].input_id;
                    if bitset_input.is_bit_on(input_id) {
                        overlap += self.feedforward_connections[feedforward_connection_idx as usize].overlap_gain;
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
    fn htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n4(&self, number_of_minicolumns_per_overlap: &mut [i32]) -> u32 {
        let mut total_minicolumns = 0;
        for overlap in (0..number_of_minicolumns_per_overlap.len()).rev() {
            let number_of_minicolumns = number_of_minicolumns_per_overlap[overlap];
            total_minicolumns += number_of_minicolumns;
            if total_minicolumns > self.n as i32 {
                number_of_minicolumns_per_overlap[overlap] = self.n as i32 - (total_minicolumns - number_of_minicolumns);
                return overlap as u32;
            }
        }
        0
    }

    pub fn update_permanence4(&mut self,
                              top_n_minicolumns: &[u32],
                              bitset_input: &CpuBitset) {
        for &minicolumn_idx in top_n_minicolumns {
            let connection_offset = self.minicolumns[minicolumn_idx as usize].connection_offset;
            let connection_len = self.minicolumns[minicolumn_idx as usize].connection_len;
            for feedforward_connection_idx in connection_offset..(connection_offset + connection_len) {
                let input_id = self.feedforward_connections[feedforward_connection_idx as usize].input_id;
                let is_active = bitset_input.is_bit_on(input_id);
                let is_inhibitory = self.feedforward_connections[feedforward_connection_idx as usize].overlap_gain < 0;
                let permanence_change = self.permanence_decrement_increment[(is_active ^ is_inhibitory) as usize];
                let old_permanence = self.feedforward_connections[feedforward_connection_idx as usize].permanence;
                let new_permanence = (old_permanence + permanence_change).clamp(0., 1.);
                self.feedforward_connections[feedforward_connection_idx as usize].permanence = new_permanence;
            }
        }
    }

    pub fn update_permanence_ltd4(&mut self,
                                  top_n_minicolumns: &[u32],
                                  active_minicolumns: &CpuBitset,
                                  bitset_input: &CpuBitset) {
        for &minicolumn_idx in top_n_minicolumns {
            let connection_offset = self.minicolumns[minicolumn_idx as usize].connection_offset;
            let connection_len = self.minicolumns[minicolumn_idx as usize].connection_len;
            let is_col_inactive = !active_minicolumns.is_bit_on(minicolumn_idx as u32);
            for feedforward_connection_idx in connection_offset..(connection_offset + connection_len) {
                let input_id = self.feedforward_connections[feedforward_connection_idx as usize].input_id;
                let is_inp_active = bitset_input.is_bit_on(input_id);
                let is_inhibitory = self.feedforward_connections[feedforward_connection_idx as usize].overlap_gain < 0;
                let reinforce = (is_inp_active ^ is_inhibitory) ^ is_col_inactive;
                let permanence_change = self.permanence_decrement_increment[reinforce as usize];
                let old_permanence = self.feedforward_connections[feedforward_connection_idx as usize].permanence;
                let new_permanence = (old_permanence + permanence_change).clamp(0., 1.);
                self.feedforward_connections[feedforward_connection_idx as usize].permanence = new_permanence;
            }
        }
    }

    /**penalty_multiplier should be some negative number (more or less close to -1 probably). Numbers between
    -1 and 0 will make the penalties smaller. Numbers below -1 will make penalties larger. Positive numbers will
    invert the penalty and lead to greater activity of inactive columns (probably not what you want under normal circumstances)*/
    pub fn update_permanence_and_penalize4(&mut self,
                                           active_minicolumns: &CpuBitset,
                                           bitset_input: &CpuBitset,
                                           penalty_multiplier:f32) {
        for (c_idx, c) in self.minicolumns.iter().enumerate() {
            let is_col_active = active_minicolumns.is_bit_on(c_idx as u32);
            let multiplier = if is_col_active {1.} else {penalty_multiplier};
            for feedforward_connection in &mut self.feedforward_connections[c.connection_offset as usize..(c.connection_offset + c.connection_len) as usize] {
                let is_inp_active = bitset_input.is_bit_on(feedforward_connection.input_id);
                let is_inhibitory = feedforward_connection.overlap_gain < 0;
                let reinforce = (is_inp_active ^ is_inhibitory);
                let permanence_change = self.permanence_decrement_increment[reinforce as usize] * multiplier;
                let old_permanence = feedforward_connection.permanence;
                let new_permanence = (old_permanence + permanence_change).clamp(0., 1.);
                feedforward_connection.permanence = new_permanence;
            }
        }
    }


    /**This function does the exact same thing as htm_find_top_minicolumns, but that function works
    optimally when the input is so sparse that only a tiny fraction of minicolumns has even a single
    connection to some active input. In cases where vast majority minicolumns is expected to have
    at least one connection to some active input, then htm_find_top_minicolumns2 will be much more optimal.
    */
    fn htm_find_top_minicolumns4(&mut self,
                                 number_of_minicolumns_per_overlap_that_made_it_to_top_n: &mut [i32],
                                 smallest_overlap_that_made_it_to_top_n: u32,
                                 top_n_minicolumns: &mut [u32],
                                 current_top_n_minicolumn_idx: &mut u32) {
        for minicolumn_idx in 0..self.minicolumns.len() {
            let overlap = self.minicolumns[minicolumn_idx].overlap;
            if overlap >= smallest_overlap_that_made_it_to_top_n as i32 { // the array number_of_minicolumns_per_overlap_that_made_it_to_top_n holds rubbish for any overlap lower than smallest_overlap_that_made_it_to_top_n
                if number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap as usize] > 0 { // only add those columns that made it to top n
                    number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap as usize] -= 1;
                    top_n_minicolumns[*current_top_n_minicolumn_idx as usize] = minicolumn_idx as u32;
                    *current_top_n_minicolumn_idx += 1;
                }
            }
        }
    }

    pub fn compute4(&mut self, bitset_input: &CpuBitset) -> CpuSDR {
        assert!(self.input_size() <= bitset_input.size());
        let mut number_of_minicolumns_per_overlap = vec![0; self.max_overlap as usize + 1];
        self.htm_calculate_overlap4(bitset_input, &mut number_of_minicolumns_per_overlap);
        let smallest_overlap_that_made_it_to_top_n = self.htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n4(&mut number_of_minicolumns_per_overlap);
        let mut top_n_minicolumns = Vec::with_capacity(self.n as usize);
        unsafe { top_n_minicolumns.set_len(self.n as usize) }
        let mut current_top_n_minicolumn_idx = 0;
        self.htm_find_top_minicolumns4(&mut number_of_minicolumns_per_overlap, smallest_overlap_that_made_it_to_top_n, &mut top_n_minicolumns, &mut current_top_n_minicolumn_idx);
        let top_minicolumn_count = current_top_n_minicolumn_idx;
        unsafe { top_n_minicolumns.set_len(top_minicolumn_count as usize) }
        CpuSDR::from(top_n_minicolumns)
    }
    pub fn infer4(&mut self, bitset_input: &CpuBitset, learn: bool) -> CpuSDR {
        let top_n_minicolumns = self.compute4(bitset_input);
        if learn {
            self.update_permanence4(&top_n_minicolumns, bitset_input)
        }
        top_n_minicolumns
    }
}


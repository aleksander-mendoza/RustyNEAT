use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::cpu_sdr::CpuSDR;
use ndalgebra::buffer::Buffer;
use crate::htm5::*;
use crate::cpu_bitset::CpuBitset;
use crate::rand::{xorshift32, rand_u32_to_random_f32};

/***This implementation assumes that most of the time  vast majority of minicolumns are connected to at least one active
input. Hence instead of iterating the input and then visiting only connected minicolumns, it's better to just iterate all
minicolumns. If you are confident that your input is so sparse than only a sparse number of minicolumns
sees some active connections at any time, then use CpuHTM. It will only visit those minicolumns that are connected
to some active input.

This implementation allows some feedforward connections to be inhibitory (this is predetermined upon initialization of HTM and cannot change later)*/
#[derive(Clone)]
pub struct CpuHTM5 {
    feedforward_connections: Vec<HtmFeedforwardConnection5>,
    minicolumns: Vec<HtmMinicolumn5>,
    input_size: u32,
    pub permanence_threshold: f32,
    pub n: u32,
    pub acceleration_decrement_increment: [f32; 2],
    pub acceleration_gravity: f32,
    pub acceleration_attractor_point: f32,
    pub max_overlap: u32,
}

impl CpuHTM5 {
    pub fn input_size(&self) -> u32 {
        self.input_size
    }
    pub fn permanence_threshold(&self) -> f32 {
        self.permanence_threshold
    }
    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn acceleration_decrement_increment(&self) -> [f32; 2] {
        self.acceleration_decrement_increment
    }
    pub fn max_overlap(&self) -> u32 {
        self.max_overlap
    }
    pub fn feedforward_connections_as_slice(&self) -> &[HtmFeedforwardConnection5] {
        self.feedforward_connections.as_slice()
    }
    pub fn feedforward_connections_as_mut_slice(&mut self) -> &mut [HtmFeedforwardConnection5] {
        self.feedforward_connections.as_mut_slice()
    }
    pub fn minicolumns_as_slice(&self) -> &[HtmMinicolumn5] {
        self.minicolumns.as_slice()
    }
    pub fn set_all_permanences(&mut self, val:f32){
        self.feedforward_connections.iter_mut().for_each(|c|c.permanence=val)
    }
    pub fn multiply_all_permanences(&mut self, val:f32){
        self.feedforward_connections.iter_mut().for_each(|c|c.permanence*=val)
    }
    pub fn add_with_input_distribution(&mut self,input_densities: &[u32],input_neg_densities: &[u32], minicolumns: u32, inputs_per_minicolumn: u32, excitatory_connection_probability: f32, mut rand_seed:u32) {
        self.add_with_input_distribution_(input_densities,input_neg_densities,minicolumns,inputs_per_minicolumn,|rand_seed,synapse_id|{
            *rand_seed = xorshift32(*rand_seed);
            rand_u32_to_random_f32(*rand_seed) <= excitatory_connection_probability
        },rand_seed)
    }
    pub fn add_with_input_distribution_exact_inhibitory(&mut self,input_densities: &[u32],input_neg_densities: &[u32], minicolumns: u32, inputs_per_minicolumn: u32, inhibitory_inputs_per_minicolumn: u32, mut rand_seed:u32) {
        assert!(inhibitory_inputs_per_minicolumn <= inputs_per_minicolumn);
        self.add_with_input_distribution_(input_densities,input_neg_densities,minicolumns,inputs_per_minicolumn,|rand_seed,synapse_id|{
            inhibitory_inputs_per_minicolumn <= synapse_id
        },rand_seed)
    }
    fn add_with_input_distribution_(&mut self,input_densities: &[u32],input_neg_densities: &[u32], minicolumns: u32, inputs_per_minicolumn: u32, mut is_excitatory: impl FnMut(&mut u32, u32)->bool, mut rand_seed:u32) {
        assert!(input_densities.len()<=self.input_size as usize,"Input densities has length {} but there are only {} inputs",input_densities.len(),self.input_size);
        assert!(input_neg_densities.len()<=self.input_size as usize,"Negative input densities has length {} but there are only {} inputs",input_neg_densities.len(),self.input_size);
        fn f(input_densities: &[u32])->Vec<f64>{
            let total:u64 = input_densities.iter().map(|&x|x as u64).sum();
            let mut pdf = Vec::<f64>::with_capacity(input_densities.len());
            let mut sum = 0u64;
            for &den in input_densities{
                sum+=den as u64;
                pdf.push(sum as f64 / total as f64);
            }
            pdf
        }
        let pdf = [f(input_neg_densities),f(input_densities)];
        self.add_minicolumns(minicolumns, |minicolumn_id,synapse_id|{
            let is_excitatory = is_excitatory(&mut rand_seed, synapse_id);
            rand_seed = xorshift32(rand_seed);
            let rand_density = rand_u32_to_random_f32(rand_seed) as f64;
            let input_idx = match pdf[is_excitatory as usize].binary_search_by(|a|a.partial_cmp(&rand_density).unwrap()) {
                Ok(x) => x, Err(x) => x
            };
            debug_assert!(input_idx<input_densities.len(),"Last element in pdf is total/total==1. The max value returned by rand_u32_to_random_f32 is less than 1.");
            rand_seed = xorshift32(rand_seed);
            let permanence = rand_u32_to_random_f32(rand_seed);
            (input_idx as u32,is_excitatory,permanence)
        }, |minicolumn_id| inputs_per_minicolumn)
    }
    fn add_globally_uniform_prob_(&mut self, minicolumns: u32,inputs_per_minicolumn: u32, mut is_excitatory: impl FnMut(&mut u32, u32)->bool, mut rand_seed: u32){
        assert!(inputs_per_minicolumn < minicolumns);
        let input_size = self.input_size;
        self.add_minicolumns( minicolumns, |minicolumn_id, synapse_id| {
            rand_seed = xorshift32(rand_seed);
            let input_idx = rand_seed % input_size;
            rand_seed = xorshift32(rand_seed);
            let permanence = rand_u32_to_random_f32(rand_seed);
            rand_seed = xorshift32(rand_seed);
            let is_excitatory = is_excitatory(&mut rand_seed, synapse_id);
            (input_idx, is_excitatory, permanence)
        }, |minicolumn_id| inputs_per_minicolumn)
    }
    pub fn add_globally_uniform_prob(&mut self, minicolumns: u32,inputs_per_minicolumn: u32, excitatory_connection_probability: f32, mut rand_seed: u32){
        self.add_globally_uniform_prob_(minicolumns,inputs_per_minicolumn,|rand_seed,synapse_id|{
            *rand_seed = xorshift32(*rand_seed);
            rand_u32_to_random_f32(*rand_seed) <= excitatory_connection_probability
        },rand_seed)
    }
    pub fn add_globally_uniform_prob_exact_inhibitory(&mut self, minicolumns: u32, inputs_per_minicolumn: u32, inhibitory_inputs_per_minicolumn: u32, mut rand_seed: u32) {
        assert!(inhibitory_inputs_per_minicolumn <= inputs_per_minicolumn);
        self.add_globally_uniform_prob_(minicolumns,inputs_per_minicolumn,|rand_seed,synapse_id|{
            inhibitory_inputs_per_minicolumn <= synapse_id
        },rand_seed)
    }
    pub fn add_globally_uniform_prob_without_inhibitory(&mut self, minicolumns: u32, inputs_per_minicolumn: u32, excitatory_connection_probability: f32, mut rand_seed: u32) {
        assert!(inputs_per_minicolumn < minicolumns);
        let input_size = self.input_size;
        self.add_minicolumns( minicolumns, |minicolumn_id, synapse_id| {
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
    pub fn new(input_size: u32, n: u32) -> Self {
        Self {
            max_overlap: 0,
            feedforward_connections: vec![],
            minicolumns: vec![],
            n,
            permanence_threshold: 0.7,
            acceleration_decrement_increment: [-0.01, 0.02],
            acceleration_gravity: 0.0,
            input_size,
            acceleration_attractor_point: 0.0
        }
    }

    /**n = how many minicolumns to activate. We will always take the top n minicolumns with the greatest overlap value.*/
    pub fn add_minicolumns(&mut self, minicolumns_count: u32, mut random_input_close_to_minicolumn: impl FnMut(u32, u32) -> (u32, bool, f32), mut input_count_incoming_to_minicolumn: impl FnMut(u32) -> u32) {
        let Self{feedforward_connections,minicolumns,input_size,..} = self;
        minicolumns.reserve(minicolumns_count as usize);
        let original_column_count = minicolumns.len();
        let &mut input_size = input_size;
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
                feedforward_connections.push(HtmFeedforwardConnection5 {
                    permanence,
                    input_id,
                    overlap_gain: if is_excitatory { 1 } else { -1 },
                    acceleration: 0.0
                });
                inputs_to_this_minicolumns.push(input_id);
            }
            minicolumns.push(HtmMinicolumn5 {
                connection_offset: connection_begin,
                connection_len: inputs_to_this_minicolumns.len() as u32,
                overlap: 0,
            });
            for input_id in inputs_to_this_minicolumns {
                connected_inputs[input_id as usize] = false;
            }
        }

        self.max_overlap = self.max_overlap.max(minicolumns[original_column_count..].iter().map(|m| m.connection_len).max().unwrap());
    }

    fn htm_calculate_overlap(&mut self, bitset_input: &CpuBitset, number_of_minicolumns_per_overlap: &mut [i32]) {
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
    fn htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&self, number_of_minicolumns_per_overlap: &mut [i32]) -> u32 {
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

    pub fn update_permanence(&mut self,
                              top_n_minicolumns: &[u32],
                              bitset_input: &CpuBitset) {
        for &minicolumn_idx in top_n_minicolumns {
            let connection_offset = self.minicolumns[minicolumn_idx as usize].connection_offset;
            let connection_len = self.minicolumns[minicolumn_idx as usize].connection_len;
            for feedforward_connection_idx in connection_offset..(connection_offset + connection_len) {
                let input_id = self.feedforward_connections[feedforward_connection_idx as usize].input_id;
                let is_active = bitset_input.is_bit_on(input_id);
                let is_inhibitory = self.feedforward_connections[feedforward_connection_idx as usize].overlap_gain < 0;
                let permanence_change = self.acceleration_decrement_increment[(is_active ^ is_inhibitory) as usize];
                let old_permanence = self.feedforward_connections[feedforward_connection_idx as usize].permanence;
                let new_permanence = (old_permanence + permanence_change).clamp(0., 1.);
                self.feedforward_connections[feedforward_connection_idx as usize].permanence = new_permanence;
            }
        }
    }

    pub fn update_permanence_ltd(&mut self,
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
                let permanence_change = self.acceleration_decrement_increment[reinforce as usize];
                let old_permanence = self.feedforward_connections[feedforward_connection_idx as usize].permanence;
                let new_permanence = (old_permanence + permanence_change).clamp(0., 1.);
                self.feedforward_connections[feedforward_connection_idx as usize].permanence = new_permanence;
            }
        }
    }

    /**penalty_multiplier should be some negative number (more or less close to -1 probably). Numbers between
    -1 and 0 will make the penalties smaller. Numbers below -1 will make penalties larger. Positive numbers will
    invert the penalty and lead to greater activity of inactive columns (probably not what you want under normal circumstances)*/
    pub fn update_permanence_and_penalize(&mut self,
                                           active_minicolumns: &CpuBitset,
                                           bitset_input: &CpuBitset,
                                           penalty_multiplier: f32) {
        for (c_idx, c) in self.minicolumns.iter().enumerate() {
            let is_col_active = active_minicolumns.is_bit_on(c_idx as u32);
            let multiplier = if is_col_active { 1. } else { penalty_multiplier };
            for feedforward_connection in &mut self.feedforward_connections[c.connection_offset as usize..(c.connection_offset + c.connection_len) as usize] {
                let is_inp_active = bitset_input.is_bit_on(feedforward_connection.input_id);
                let is_inhibitory = feedforward_connection.overlap_gain < 0;
                let reinforce = (is_inp_active ^ is_inhibitory);
                let permanence_change = self.acceleration_decrement_increment[reinforce as usize] * multiplier;
                let old_permanence = feedforward_connection.permanence;
                let new_permanence = (old_permanence + permanence_change).clamp(0., 1.);
                feedforward_connection.permanence = new_permanence;
            }
        }
    }

    pub fn update_permanence_and_penalize_thresholded(&mut self,
                                                       active_minicolumns: &CpuBitset,
                                                       bitset_input: &CpuBitset,
                                                       activity_threshold:u32,
                                                       penalty_multiplier: f32) {
        for (c_idx, c) in self.minicolumns.iter().enumerate() {
            let is_col_active = active_minicolumns.is_bit_on(c_idx as u32);
            let multiplier = if is_col_active { 1. } else { penalty_multiplier };
            let connections = &mut self.feedforward_connections[c.connection_offset as usize..(c.connection_offset + c.connection_len) as usize];
            let activity:u32 = connections.iter().map(|c|bitset_input.is_bit_on(c.input_id) as u32).sum();
            if activity >= activity_threshold {
                for feedforward_connection in connections {
                    let is_inp_active = bitset_input.is_bit_on(feedforward_connection.input_id);
                    let is_inhibitory = feedforward_connection.overlap_gain < 0;
                    let reinforce = (is_inp_active ^ is_inhibitory);
                    let permanence_change = self.acceleration_decrement_increment[reinforce as usize] * multiplier;
                    let old_permanence = feedforward_connection.permanence;
                    let new_permanence = (old_permanence + permanence_change).clamp(0., 1.);
                    feedforward_connection.permanence = new_permanence;
                }
            }
        }
    }


    /**This function does the exact same thing as htm_find_top_minicolumns, but that function works
    optimally when the input is so sparse that only a tiny fraction of minicolumns has even a single
    connection to some active input. In cases where vast majority minicolumns is expected to have
    at least one connection to some active input, then htm_find_top_minicolumns2 will be much more optimal.
    */
    fn htm_find_top_minicolumns(&mut self,
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

    pub fn compute(&mut self, bitset_input: &CpuBitset) -> CpuSDR {
        assert!(self.input_size() <= bitset_input.size());
        let mut number_of_minicolumns_per_overlap = vec![0; self.max_overlap as usize + 1];
        self.htm_calculate_overlap(bitset_input, &mut number_of_minicolumns_per_overlap);
        let smallest_overlap_that_made_it_to_top_n = self.htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&mut number_of_minicolumns_per_overlap);
        let mut top_n_minicolumns = Vec::with_capacity(self.n as usize);
        unsafe { top_n_minicolumns.set_len(self.n as usize) }
        let mut current_top_n_minicolumn_idx = 0;
        self.htm_find_top_minicolumns(&mut number_of_minicolumns_per_overlap, smallest_overlap_that_made_it_to_top_n, &mut top_n_minicolumns, &mut current_top_n_minicolumn_idx);
        let top_minicolumn_count = current_top_n_minicolumn_idx;
        unsafe { top_n_minicolumns.set_len(top_minicolumn_count as usize) }
        CpuSDR::from(top_n_minicolumns)
    }
    pub fn infer(&mut self, bitset_input: &CpuBitset, learn: bool) -> CpuSDR {
        let top_n_minicolumns = self.compute(bitset_input);
        if learn {
            self.update_permanence(&top_n_minicolumns, bitset_input)
        }
        top_n_minicolumns
    }
}


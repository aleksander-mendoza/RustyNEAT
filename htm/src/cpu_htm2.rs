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
use crate::cpu_bitset::CpuBitset;
use crate::rand::{xorshift32, rand_u32_to_random_f32};

/***This implementation assumes that most of the time  vast majority of minicolumns are connected to at least one active
input. Hence instead of iterating the input and then visiting only connected minicolumns, it's better to just iterate all
minicolumns. If you are confident that your input is so sparse than only a sparse number of minicolumns
sees some active connections at any time, then use CpuHTM. It will only visit those minicolumns that are connected
to some active input.*/
#[derive(Clone)]
pub struct CpuHTM2 {
    feedforward_connections: Vec<HtmFeedforwardConnection2>,
    minicolumns: Vec<HtmMinicolumn2>,
    input_size: u32,
    pub permanence_threshold: f32,
    pub n: u32,
    pub permanence_decrement_increment: [f32; 2],
    pub max_overlap: u32,
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
            minicolumns: htm.minicolumns_as_slice().iter().map(|m|HtmMinicolumn2{
                connection_offset: m.connection_index_offset,
                connection_len: m.connection_index_len,
                overlap: 0
            }).collect(),
            input_size: htm.input_size(),
            permanence_threshold: htm.permanence_threshold(),
            n: htm.n(),
            permanence_decrement_increment: htm.permanence_decrement_increment(),
            max_overlap: htm.max_overlap()
        }
    }
}

impl CpuHTM2 {
    pub fn input_size(&self)->u32{
        self.input_size
    }
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
    pub fn minicolumns_as_slice(&self)->&[HtmMinicolumn2]{
        self.minicolumns.as_slice()
    }
    pub fn new_globally_uniform_prob(input_size: u32, minicolumns: u32, n: u32, inputs_per_minicolumn: u32, mut rand_seed:u32) -> Self {
        assert!(inputs_per_minicolumn < minicolumns);
        Self::new(input_size, minicolumns, n, |minicolumn_id|{
            rand_seed = xorshift32(rand_seed);
            let input_idx = rand_seed % input_size;
            rand_seed = xorshift32(rand_seed);
            let permanence = rand_u32_to_random_f32(rand_seed);
            (input_idx,permanence)
        }, |minicolumn_id| inputs_per_minicolumn)
    }
    /**n = how many minicolumns to activate. We will always take the top n minicolumns with the greatest overlap value.*/
    pub fn new(input_size: u32, minicolumns_count: u32, n: u32, mut random_input_close_to_minicolumn: impl FnMut(u32) -> (u32,f32), mut input_count_incoming_to_minicolumn: impl FnMut(u32) -> u32) -> Self {
        let mut feedforward_connections: Vec<HtmFeedforwardConnection2> = vec![];
        let mut minicolumns: Vec<HtmMinicolumn2> = Vec::with_capacity(minicolumns_count as usize);

        let mut connected_inputs = vec![false; input_size as usize];
        for minicolumn_id in 0..minicolumns_count as u32 {
            let input_count = input_count_incoming_to_minicolumn(minicolumn_id);
            assert!(input_count<=input_size,"Minicolumn {} has {} input connections but there are only {} inputs",minicolumn_id,input_count,input_size);
            let mut inputs_to_this_minicolumns: Vec<u32> = vec![];
            let connection_begin = feedforward_connections.len() as u32;
            for _ in 0..input_count {
                let mut inp_perm = random_input_close_to_minicolumn(minicolumn_id);
                while connected_inputs[inp_perm.0 as usize] { // find some input that has not been connected to this minicolumn yet
                    inp_perm.0=(inp_perm.0+1)%input_size
                }
                connected_inputs[inp_perm.0 as usize] = true;
                feedforward_connections.push(HtmFeedforwardConnection2 {
                    permanence: inp_perm.1,
                    input_id: inp_perm.0,
                });
                inputs_to_this_minicolumns.push(inp_perm.0);
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
            minicolumns,
            n,
            permanence_threshold:0.7,
            permanence_decrement_increment: [-0.01, 0.02],
            input_size
        }
    }


    fn htm_calculate_overlap2(&mut self, bitset_input:&CpuBitset, number_of_minicolumns_per_overlap: &mut [i32]) {
        for minicolumn_idx in 0..self.minicolumns.len() {
            let connection_offset = self.minicolumns[minicolumn_idx].connection_offset;
            let connection_len = self.minicolumns[minicolumn_idx].connection_len;
            let mut overlap = 0;
            for feedforward_connection_idx in connection_offset..(connection_offset+connection_len) {
                if self.feedforward_connections[feedforward_connection_idx as usize].permanence > self.permanence_threshold {
                    let input_id = self.feedforward_connections[feedforward_connection_idx as usize].input_id;
                    if bitset_input.is_bit_on(input_id){
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
    fn htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n2(&self, number_of_minicolumns_per_overlap: &mut [i32]) -> u32 {
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

    fn htm_update_permanence2(&mut self,
                              top_n_minicolumns: &[u32],
                              bitset_input:&CpuBitset,
                              current_top_n_minicolumn_idx: u32) {
        for top_minicolumn_idx in 0..current_top_n_minicolumn_idx as usize {
            let minicolumn_idx: u32 = top_n_minicolumns[top_minicolumn_idx];
            let connection_offset = self.minicolumns[minicolumn_idx as usize].connection_offset;
            let connection_len = self.minicolumns[minicolumn_idx as usize].connection_len;
            for feedforward_connection_idx in connection_offset..(connection_offset+connection_len) {
                let input_id = self.feedforward_connections[feedforward_connection_idx as usize].input_id;
                let permanence_change = self.permanence_decrement_increment[bitset_input.is_bit_on(input_id) as usize];
                let old_permanence = self.feedforward_connections[feedforward_connection_idx as usize].permanence;
                let new_permanence = (old_permanence + permanence_change).clamp(0., 1.);
                self.feedforward_connections[feedforward_connection_idx as usize].permanence = new_permanence;
            }
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

    pub fn infer2(&mut self, bitset_input: &CpuBitset, learn: bool) -> CpuSDR{
        assert!(self.input_size()<=bitset_input.size(),"HTM expects input of size {} but got {}",self.input_size(),bitset_input.size());
        let mut number_of_minicolumns_per_overlap = vec![0; self.max_overlap as usize+1];
        self.htm_calculate_overlap2(bitset_input,&mut number_of_minicolumns_per_overlap);
        let smallest_overlap_that_made_it_to_top_n = self.htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n2(&mut number_of_minicolumns_per_overlap);
        let mut top_n_minicolumns = Vec::with_capacity(self.n as usize);
        unsafe { top_n_minicolumns.set_len(self.n as usize) }
        let mut current_top_n_minicolumn_idx = 0;
        self.htm_find_top_minicolumns2(&mut number_of_minicolumns_per_overlap, smallest_overlap_that_made_it_to_top_n, &mut top_n_minicolumns, &mut current_top_n_minicolumn_idx);
        let top_minicolumn_count = current_top_n_minicolumn_idx;
        if learn {
            self.htm_update_permanence2(&top_n_minicolumns, bitset_input,top_minicolumn_count)
        }
        unsafe { top_n_minicolumns.set_len(top_minicolumn_count as usize) }
        CpuSDR::from(top_n_minicolumns)
    }
}


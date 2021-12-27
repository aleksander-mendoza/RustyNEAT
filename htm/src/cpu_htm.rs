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
use rand::Rng;

/***This implementation assumes that most of the time  vast majority of minicolumns are connected to at least one active
input. Hence instead of iterating the input and then visiting only connected minicolumns, it's better to just iterate all
minicolumns. If you are confident that your input is so sparse than only a sparse number of minicolumns
sees some active connections at any time, then use CpuHTM. It will only visit those minicolumns that are connected
to some active input.*/
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CpuHTM {
    feedforward_connections: Vec<HtmFeedforwardConnection>,
    minicolumns: Vec<HtmMinicolumn>,
    input_size: u32,
    pub permanence_threshold: f32,
    pub n: u32,
    pub permanence_decrement_increment: [f32; 2],
    pub max_overlap: u32,
}

impl CpuHTM {
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
    pub fn feedforward_connections_as_slice(&self) -> &[HtmFeedforwardConnection] {
        self.feedforward_connections.as_slice()
    }
    pub fn feedforward_connections_as_mut_slice(&mut self) -> &mut [HtmFeedforwardConnection] {
        self.feedforward_connections.as_mut_slice()
    }
    pub fn set_all_permanences(&mut self, val: f32) {
        self.feedforward_connections.iter_mut().for_each(|c| c.permanence = val)
    }
    pub fn multiply_all_permanences(&mut self, val: f32) {
        self.feedforward_connections.iter_mut().for_each(|c| c.permanence *= val)
    }
    pub fn minicolumns_as_slice(&self) -> &[HtmMinicolumn] {
        self.minicolumns.as_slice()
    }

    /**n = how many minicolumns to activate. We will always take the top n minicolumns with the greatest overlap value.*/
    pub fn new(input_size: u32, n: u32) -> Self {
        Self {
            max_overlap: 0,
            feedforward_connections:vec![],
            minicolumns:vec![],
            n,
            permanence_threshold: 0.7,
            permanence_decrement_increment: [-0.01, 0.02],
            input_size,
        }
    }
    pub fn add_globally_uniform_prob(&mut self,minicolumn_count:usize,synapses_per_column:u32,  rand_seed:&mut impl Rng){
        let mut pop = Population::new(minicolumn_count,1);
        pop.add_uniform_rand_inputs_from_range(0..self.input_size as usize,synapses_per_column as usize,rand_seed);
        self.add_population(&pop,rand_seed)
    }

    pub fn add_population(&mut self, population:&Population, rand_seed:&mut impl Rng){
        let Self{ feedforward_connections, minicolumns, max_overlap, .. } = self;
        for neuron in &population.neurons{
            let conn_start = feedforward_connections.len() as u32;
            for seg in &neuron.segments{
                for &syn in &seg.synapses{
                    let permanence = rand_seed.gen();
                    feedforward_connections.push(HtmFeedforwardConnection { permanence, input_id:syn as u32 });

                }
            }
            let conn_end = feedforward_connections.len() as u32;
            let conn_len = conn_end - conn_start;
            if conn_len > *max_overlap {
                *max_overlap = conn_len;
            }
            minicolumns.push(HtmMinicolumn {
                connection_offset: conn_start,
                connection_len: conn_len,
                overlap: 0
            });
        }
    }


    fn htm_calculate_overlap_for_minicolumn(&mut self, minicolumn_idx: usize, bitset_input: &CpuBitset, number_of_minicolumns_per_overlap: &mut [i32]) -> i32 {
        let connection_offset = self.minicolumns[minicolumn_idx].connection_offset;
        let connection_len = self.minicolumns[minicolumn_idx].connection_len;
        let mut overlap = 0;
        for feedforward_connection_idx in connection_offset..(connection_offset + connection_len) {
            if self.feedforward_connections[feedforward_connection_idx as usize].permanence > self.permanence_threshold {
                let input_id = self.feedforward_connections[feedforward_connection_idx as usize].input_id;
                if bitset_input.is_bit_on(input_id) {
                    overlap += 1;
                }
            }
        }
        if overlap > 0 {
            number_of_minicolumns_per_overlap[overlap as usize] += 1;
        }
        overlap
    }
    fn htm_calculate_overlap_and_group_into_columns(&mut self, max_overlap: usize, column_stride: usize, minicolumn_stride:usize, bitset_input: &CpuBitset, number_of_minicolumns_per_overlap: &mut [i32]) {
        for minicolumn_idx in 0..self.minicolumns.len() {
            // minicolumn_idx == column_stride * column + minicolumn_within_column_idx * minicolumn_stride
            // (column_stride == 1 && minicolumn_stride == column_count) || (column_stride == minicolumns_per_column && minicolumn_stride == 1)
            // (minicolumn_idx % minicolumn_stride == column) || (minicolumn_idx / column_stride == column)
            let column_idx = if column_stride==1{minicolumn_idx%minicolumn_stride}else{minicolumn_idx / column_stride};
            let offset = (max_overlap + 1) * column_idx;
            self.minicolumns[minicolumn_idx].overlap = self.htm_calculate_overlap_for_minicolumn(minicolumn_idx, bitset_input, &mut number_of_minicolumns_per_overlap[offset..offset + max_overlap + 1]);
        }
    }
    fn htm_calculate_overlap(&mut self, bitset_input: &CpuBitset, number_of_minicolumns_per_overlap: &mut [i32]) {
        for minicolumn_idx in 0..self.minicolumns.len() {
            self.minicolumns[minicolumn_idx].overlap = self.htm_calculate_overlap_for_minicolumn(minicolumn_idx, bitset_input, number_of_minicolumns_per_overlap);
        }
    }

    /**returns smallest_overlap_that_made_it_to_top_n.
    By the end of running this function, the number_of_minicolumns_per_overlap array will become
    number_of_minicolumns_per_overlap_that_made_it_to_top_n.
    number_of_minicolumns_per_overlap_that_made_it_to_top_n holds rubbish for any overlap lower than smallest_overlap_that_made_it_to_top_n
    */
    fn neurons_in_binhtm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&self, number_of_minicolumns_per_overlap: &mut [i32]) -> u32 {
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
    fn htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n_and_group_into_columns
    (&self, max_overlap: usize, column_count: usize, number_of_minicolumns_per_overlap: &mut [i32]) {
        for column_idx in 0..column_count {
            let offset = column_idx * (max_overlap + 1);
            let smallest_overlap_that_made_it_to_top_n = self.htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&mut number_of_minicolumns_per_overlap[offset..offset + max_overlap + 1]);
            number_of_minicolumns_per_overlap[offset] = smallest_overlap_that_made_it_to_top_n as i32;
        }
    }
    pub fn update_permanence(&mut self,
                             top_n_minicolumns: &[u32],
                             bitset_input: &CpuBitset) {
        for &minicolumn_idx in top_n_minicolumns {
            let connection_offset = self.minicolumns[minicolumn_idx as usize].connection_offset;
            let connection_len = self.minicolumns[minicolumn_idx as usize].connection_len;
            for feedforward_connection_idx in connection_offset..(connection_offset + connection_len) {
                let input_id = self.feedforward_connections[feedforward_connection_idx as usize].input_id;
                let permanence_change = self.permanence_decrement_increment[bitset_input.is_bit_on(input_id) as usize];
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
            let is_col_inactive = !active_minicolumns.is_bit_on(minicolumn_idx as u32);
            let connection_offset = self.minicolumns[minicolumn_idx as usize].connection_offset;
            let connection_len = self.minicolumns[minicolumn_idx as usize].connection_len;
            for feedforward_connection_idx in connection_offset..(connection_offset + connection_len) {
                let input_id = self.feedforward_connections[feedforward_connection_idx as usize].input_id;
                let is_inp_active = bitset_input.is_bit_on(input_id);
                let reinforce = is_inp_active ^ is_col_inactive;
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
    pub fn update_permanence_and_penalize(&mut self,
                                          active_minicolumns: &CpuBitset,
                                          bitset_input: &CpuBitset,
                                          penalty_multiplier: f32) {
        for (c_idx, c) in self.minicolumns.iter().enumerate() {
            let is_col_active = active_minicolumns.is_bit_on(c_idx as u32);
            let multiplier = if is_col_active { 1. } else { penalty_multiplier };
            for feedforward_connection in &mut self.feedforward_connections[c.connection_offset as usize..(c.connection_offset + c.connection_len) as usize] {
                let is_inp_active = bitset_input.is_bit_on(feedforward_connection.input_id);
                let permanence_change = self.permanence_decrement_increment[is_inp_active as usize] * multiplier;
                let old_permanence = feedforward_connection.permanence;
                let new_permanence = (old_permanence + permanence_change).clamp(0., 1.);
                feedforward_connection.permanence = new_permanence;
            }
        }
    }


    pub fn update_permanence_and_penalize_thresholded(&mut self,
                                                      active_minicolumns: &CpuBitset,
                                                      bitset_input: &CpuBitset,
                                                      activity_threshold: u32,
                                                      penalty_multiplier: f32) {
        for (c_idx, c) in self.minicolumns.iter().enumerate() {
            let is_col_active = active_minicolumns.is_bit_on(c_idx as u32);
            let multiplier = if is_col_active { 1. } else { penalty_multiplier };
            let connections = &mut self.feedforward_connections[c.connection_offset as usize..(c.connection_offset + c.connection_len) as usize];
            if is_col_active || connections.iter().map(|c| bitset_input.is_bit_on(c.input_id) as u32).sum::<u32>() >= activity_threshold {
                for feedforward_connection in connections {
                    let is_inp_active = bitset_input.is_bit_on(feedforward_connection.input_id);
                    let permanence_change = self.permanence_decrement_increment[is_inp_active as usize] * multiplier;
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
    at least one connection to some active input, then htm_find_top_minicolumns will be much more optimal.
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
    fn htm_find_top_minicolumns_and_group_into_columns(&mut self,
                                                       n: usize, max_overlap: usize,
                                                       column_count: usize, minicolumns_per_column: usize,
                                                       column_stride:usize, minicolumn_stride:usize,
                                                       number_of_minicolumns_per_overlap_that_made_it_to_top_n: &mut [i32],
                                                       top_n_minicolumns: &mut [u32]) {
        let mut current_top_n_minicolumn_idx = 0;
        for column_idx in 0..column_count {
            let overlap_offset = (max_overlap + 1) * column_idx;
            let smallest_overlap_that_made_it_to_top_n = number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap_offset];
            let number_of_minicolumns_per_overlap = &mut number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap_offset..overlap_offset + max_overlap + 1];
            let minicolumn_offset = column_idx * column_stride;
            for minicolumn_within_column_idx in 0..minicolumns_per_column {
                let minicolumn_idx = minicolumn_offset + minicolumn_within_column_idx*minicolumn_stride;
                let overlap = self.minicolumns[minicolumn_idx].overlap;
                if overlap >= smallest_overlap_that_made_it_to_top_n as i32 { // the array number_of_minicolumns_per_overlap_that_made_it_to_top_n holds rubbish for any overlap lower than smallest_overlap_that_made_it_to_top_n
                    if number_of_minicolumns_per_overlap[overlap as usize] > 0 { // only add those columns that made it to top n
                        number_of_minicolumns_per_overlap[overlap as usize] -= 1;
                        top_n_minicolumns[current_top_n_minicolumn_idx] = minicolumn_idx as u32;
                        current_top_n_minicolumn_idx += 1;
                    }
                }
            }
            let final_idx = (column_idx + 1) * n;
            if current_top_n_minicolumn_idx < final_idx {
                for minicolumn_idx in minicolumn_offset..minicolumn_offset + minicolumns_per_column {
                    let overlap = self.minicolumns[minicolumn_idx].overlap;
                    if overlap == 0 {
                        top_n_minicolumns[current_top_n_minicolumn_idx] = minicolumn_idx as u32;
                        current_top_n_minicolumn_idx += 1;
                        if current_top_n_minicolumn_idx >= final_idx {
                            break;
                        }
                    }
                }
            }
            assert_eq!(current_top_n_minicolumn_idx, final_idx, "Somethings wrong. Not enough columns. This shouldn't happen unless there's a bug");
        }
    }
    pub fn compute(&mut self, bitset_input: &CpuBitset) -> CpuSDR {
        assert!(self.input_size() <= bitset_input.size(), "HTM expects input of size {} but got {}", self.input_size(), bitset_input.size());
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
    pub fn compute_and_group_into_columns(&mut self, minicolumns_per_column: usize,minicolumn_stride:usize, bitset_input: &CpuBitset) -> CpuSDR {
        assert!(minicolumns_per_column>0);
        assert!(minicolumn_stride>0);
        assert!(minicolumns_per_column >= self.n as usize, "Each column activates n={} winners but there are only {} minicolumns per column", self.n, minicolumns_per_column);
        assert_eq!(self.minicolumns_as_slice().len() % minicolumns_per_column, 0, "The number of minicolumns cannot be evenly divided into columns");
        let column_count = self.minicolumns_as_slice().len() / minicolumns_per_column;
        let column_stride = if minicolumn_stride==1{minicolumns_per_column}else{1};
        assert_eq!(minicolumn_stride*(minicolumns_per_column-1)+column_stride*(column_count-1),self.minicolumns_as_slice().len()-1,"Minicolumn stride == {}, column stride == {}, minicolumns_per_column == {}, column_count == {} not compatible with total number of minicolumns {}",minicolumn_stride,column_stride,minicolumns_per_column,column_count,self.minicolumns_as_slice().len());
        assert!(self.input_size() <= bitset_input.size(), "HTM expects input of size {} but got {}", self.input_size(), bitset_input.size());
        let mut number_of_minicolumns_per_overlap = vec![0; (self.max_overlap as usize + 1) * column_count];
        self.htm_calculate_overlap_and_group_into_columns(self.max_overlap as usize, column_stride, minicolumn_stride, bitset_input, &mut number_of_minicolumns_per_overlap);
        self.htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n_and_group_into_columns(self.max_overlap as usize, column_count, &mut number_of_minicolumns_per_overlap);
        let mut top_n_minicolumns = Vec::with_capacity((self.n as usize) * column_count);
        unsafe { top_n_minicolumns.set_len((self.n as usize) * column_count) }
        self.htm_find_top_minicolumns_and_group_into_columns(self.n as usize, self.max_overlap as usize, column_count, minicolumns_per_column, column_stride,minicolumn_stride,&mut number_of_minicolumns_per_overlap, &mut top_n_minicolumns);
        CpuSDR::from(top_n_minicolumns)
    }
    pub fn infer(&mut self, bitset_input: &CpuBitset, learn: bool) -> CpuSDR {
        let top_n_minicolumns = self.compute(bitset_input);
        if learn {
            self.update_permanence(&top_n_minicolumns, bitset_input)
        }
        top_n_minicolumns
    }
    pub fn infer_and_group_into_columns(&mut self, minicolumns_per_column: usize, minicolumn_stride:usize,bitset_input: &CpuBitset, learn: bool) -> CpuSDR {
        let top_n_minicolumns = self.compute_and_group_into_columns(minicolumns_per_column, minicolumn_stride,bitset_input);
        if learn {
            self.update_permanence(&top_n_minicolumns, bitset_input)
        }
        top_n_minicolumns
    }
}


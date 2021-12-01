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
use crate::rnd::{xorshift32, rand_u32_to_random_f32};
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use crate::{Shape, Shape3, Shape2};

/***This implementation assumes that most of the time  vast majority of minicolumns are connected to at least one active
input. Hence instead of iterating the input and then visiting only connected minicolumns, it's better to just iterate all
minicolumns. If you are confident that your input is so sparse than only a sparse number of minicolumns
sees some active connections at any time, then use CpuHTM. It will only visit those minicolumns that are connected
to some active input.*/
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CpuHTM2 {
    feedforward_connections: Vec<HtmFeedforwardConnection2>,
    minicolumns: Vec<HtmMinicolumn2>,
    input_size: u32,
    pub permanence_threshold: f32,
    pub n: u32,
    pub permanence_decrement_increment: [f32; 2],
    pub max_overlap: u32,
}

impl From<&CpuHTM> for CpuHTM2 {
    fn from(htm: &CpuHTM) -> Self {
        Self {
            feedforward_connections: htm.connection_indices_as_slice().iter().map(|&connection_index| {
                let feedforward_connection = &htm.feedforward_connections_as_slice()[connection_index as usize];
                HtmFeedforwardConnection2 {
                    permanence: feedforward_connection.permanence,
                    input_id: feedforward_connection.input_id,
                }
            }).collect(),
            minicolumns: htm.minicolumns_as_slice().iter().map(|m| HtmMinicolumn2 {
                connection_offset: m.connection_index_offset,
                connection_len: m.connection_index_len,
                overlap: 0,
            }).collect(),
            input_size: htm.input_size(),
            permanence_threshold: htm.permanence_threshold(),
            n: htm.n(),
            permanence_decrement_increment: htm.permanence_decrement_increment(),
            max_overlap: htm.max_overlap(),
        }
    }
}

pub fn mod_clamp(x: f32, min: f32, max: f32) -> f32 {
    debug_assert!(min < max);
    let len = max - min;
    let y = x - f32::floor((x - min) / len) * len;
    debug_assert!(min <= y, "{} <= {}", min, y);
    debug_assert!(y <= max, "{} <= {}", y, max);
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test1() {
        assert_eq!(mod_clamp(2., 4., 6.), 4.);
        assert_eq!(mod_clamp(1., 4., 6.), 5.);
        assert_eq!(mod_clamp(1., 4., 6.5), 6.);
        assert_eq!(mod_clamp(7., 4., 6.5), 4.5);
        assert_eq!(mod_clamp(8., 4., 6.5), 5.5);
    }
}

impl CpuHTM2 {
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
    pub fn feedforward_connections_as_slice(&self) -> &[HtmFeedforwardConnection2] {
        self.feedforward_connections.as_slice()
    }
    pub fn feedforward_connections_as_mut_slice(&mut self) -> &mut [HtmFeedforwardConnection2] {
        self.feedforward_connections.as_mut_slice()
    }
    pub fn set_all_permanences(&mut self, val: f32) {
        self.feedforward_connections.iter_mut().for_each(|c| c.permanence = val)
    }
    pub fn multiply_all_permanences(&mut self, val: f32) {
        self.feedforward_connections.iter_mut().for_each(|c| c.permanence *= val)
    }
    pub fn minicolumns_as_slice(&self) -> &[HtmMinicolumn2] {
        self.minicolumns.as_slice()
    }
    pub fn new_local_2d(input_size: (u32, u32), minicolumns: (u32, u32), n: u32, inputs_per_minicolumn: u32, radius: f32, mut rand_seed: u32) -> Self {
        let mut slf = Self::new(input_size.0 * input_size.1, n);
        slf.add_local_2d(input_size, minicolumns, inputs_per_minicolumn, radius, rand_seed);
        slf
    }
    pub fn add_local_2d(&mut self, input_size: (u32,u32), minicolumns: (u32,u32), inputs_per_minicolumn: u32, radius: f32, mut rand_seed: u32) -> u32{
        assert_eq!(input_size.0 * input_size.1, self.input_size, "There are {} inputs, but requested dimensions is {}x{}=={}", self.input_size, input_size.0, input_size.1, input_size.0 * input_size.1);
        let stride = (input_size.0 as f32 / minicolumns.0 as f32, input_size.1 as f32 / minicolumns.1 as f32);
        let margin = (stride.0 / 2f32, stride.1 / 2f32);

        const EPSILON: f32 = 0.0001;
        self.add_minicolumns(minicolumns.0 * minicolumns.1, |minicolumn_id| {
            let minicolumn_pos = (minicolumn_id % minicolumns.0, minicolumn_id / minicolumns.0);
            let minicolumn_pos = (margin.0 + stride.0 * minicolumn_pos.0 as f32, margin.1 + stride.1 * minicolumn_pos.1 as f32);
            let min_bounds = ((minicolumn_pos.0 - radius).max(0.), (minicolumn_pos.1 - radius).max(0.));
            let max_bounds = ((minicolumn_pos.0 + radius).min(input_size.0 as f32 - EPSILON), (minicolumn_pos.1 + radius).min(input_size.1 as f32 - EPSILON));
            rand_seed = xorshift32(rand_seed);
            let offset_0 = rand_u32_to_random_f32(rand_seed) - 0.5f32;
            rand_seed = xorshift32(rand_seed);
            let offset_1 = rand_u32_to_random_f32(rand_seed) - 0.5f32;
            let offset = (offset_0 * radius, offset_1 * radius);
            let input_pos = (minicolumn_pos.0 + offset.0 + 0.5f32, minicolumn_pos.1 + offset.1 + 0.5f32);
            let input_pos = (mod_clamp(input_pos.0, min_bounds.0, max_bounds.0), mod_clamp(input_pos.1, min_bounds.1, max_bounds.1));
            let input_pos = (input_pos.0 as u32, input_pos.1 as u32);
            let input_idx = input_pos.0 + input_pos.1 * input_size.0;
            rand_seed = xorshift32(rand_seed);
            let permanence = rand_u32_to_random_f32(rand_seed);
            (input_idx, permanence)
        }, |minicolumn_id| inputs_per_minicolumn);
        rand_seed
    }
    pub fn add_globally_uniform_prob(&mut self, minicolumns: u32, inputs_per_minicolumn: u32, mut rand_seed: u32) -> u32 {
        let input_size = self.input_size;
        assert!(inputs_per_minicolumn <= input_size, "Column can't have {} inputs if there are only {} inputs in total", inputs_per_minicolumn, input_size);
        self.add_minicolumns(minicolumns, |minicolumn_id| {
            rand_seed = xorshift32(rand_seed);
            let input_idx = rand_seed % input_size;
            rand_seed = xorshift32(rand_seed);
            let permanence = rand_u32_to_random_f32(rand_seed);
            (input_idx, permanence)
        }, |minicolumn_id| inputs_per_minicolumn);
        rand_seed
    }
    pub fn add_with_input_distribution(&mut self, input_densities: &[u32], minicolumns: u32, inputs_per_minicolumn: u32, mut rand_seed: u32) -> u32{
        assert!(input_densities.len() <= self.input_size as usize, "Input densities has length {} but there are only {} inputs", input_densities.len(), self.input_size);
        let total: u64 = input_densities.iter().map(|&x| x as u64).sum();
        let mut pdf = Vec::<f64>::with_capacity(input_densities.len());
        let mut sum = 0u64;
        for &den in input_densities {
            sum += den as u64;
            pdf.push(sum as f64 / total as f64);
        }
        self.add_minicolumns(minicolumns, |minicolumn_id| {
            rand_seed = xorshift32(rand_seed);
            let rand_density = rand_u32_to_random_f32(rand_seed) as f64;
            let input_idx = match pdf.binary_search_by(|a| a.partial_cmp(&rand_density).unwrap()) {
                Ok(x) => x,
                Err(x) => x
            };
            debug_assert!(input_idx < input_densities.len(), "Last element in pdf is total/total==1. The max value returned by rand_u32_to_random_f32 is less than 1.");
            rand_seed = xorshift32(rand_seed);
            let permanence = rand_u32_to_random_f32(rand_seed);
            (input_idx as u32, permanence)
        }, |minicolumn_id| inputs_per_minicolumn);
        rand_seed
    }
    /**n = how many minicolumns to activate. We will always take the top n minicolumns with the greatest overlap value.*/
    pub fn new(input_size: u32, n: u32) -> Self {
        Self {
            max_overlap: 0,
            feedforward_connections: vec![],
            minicolumns: vec![],
            n,
            permanence_threshold: 0.7,
            permanence_decrement_increment: [-0.01, 0.02],
            input_size,
        }
    }
    pub fn add_2d_column_grid_with_3d_input(&mut self, minicolumns_per_column:u32, inputs_per_minicolumn: u32, input_stride: [u32;2], input_kernel: [u32;2], input_size: [u32;3], mut rand_seed: u32) -> u32{
        let column_grid = [input_size.height(),input_size.width()].conv_out_size(input_stride,input_kernel);
        let input_subregion_size = [input_size[0],input_kernel[0],input_kernel[1]];
        for col0 in 0..column_grid[0]{
            for col1 in 0..column_grid[1]{
                let input_offset = [0,col0*input_stride[0],col1*input_stride[1]];
                rand_seed = self.add_column_with_3d_input(minicolumns_per_column,inputs_per_minicolumn,input_offset,input_subregion_size,input_size,rand_seed)
            }
        }
        rand_seed
    }
    pub fn add_column_with_3d_input(&mut self, minicolumns_count: u32, inputs_per_minicolumn: u32, input_offset: [u32;3], input_subregion_size: [u32;3], input_size: [u32;3], mut rand_seed: u32) -> u32{
        assert_eq!(input_size.size(), self.input_size);
        self.add_minicolumns(minicolumns_count, |minicolumn_id| {
            let mut input_pos = [0u32;3];
            rand_seed = input_pos.rand(&input_subregion_size,rand_seed);
            let permanence = rand_u32_to_random_f32(rand_seed);
            let input_idx = input_size.idx(input_pos.add(input_offset));
            (input_idx, permanence)
        }, |minicolumn_id| inputs_per_minicolumn);
        rand_seed
    }
    /**n = how many minicolumns to activate. We will always take the top n minicolumns with the greatest overlap value.*/
    pub fn add_minicolumns(&mut self, minicolumns_count: u32, mut random_input_close_to_minicolumn: impl FnMut(u32) -> (u32, f32), mut input_count_incoming_to_minicolumn: impl FnMut(u32) -> u32) {
        let Self { feedforward_connections, minicolumns, input_size, .. } = self;
        let &mut input_size = input_size;
        minicolumns.reserve(minicolumns_count as usize);
        let original_column_count = minicolumns.len();
        let mut connected_inputs = vec![false; input_size as usize];
        for minicolumn_id in 0..minicolumns_count as u32 {
            let input_count = input_count_incoming_to_minicolumn(minicolumn_id);
            assert!(input_count <= input_size, "Minicolumn {} has {} input connections but there are only {} inputs", minicolumn_id, input_count, input_size);
            let mut inputs_to_this_minicolumns: Vec<u32> = vec![];
            let connection_begin = feedforward_connections.len() as u32;
            for _ in 0..input_count {
                let mut inp_perm = random_input_close_to_minicolumn(minicolumn_id);
                while connected_inputs[inp_perm.0 as usize] { // find some input that has not been connected to this minicolumn yet
                    inp_perm.0 = (inp_perm.0 + 1) % input_size
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
        self.max_overlap = self.max_overlap.max(minicolumns[original_column_count..].iter().map(|m| m.connection_len).max().unwrap());
    }


    fn htm_calculate_overlap_for_minicolumn(&mut self, minicolumn_idx:usize, bitset_input: &CpuBitset, number_of_minicolumns_per_overlap: &mut [i32]) -> i32{
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
    fn htm_calculate_overlap_and_group_into_columns(&mut self, max_overlap:usize, minicolumns_per_column:usize, bitset_input: &CpuBitset, number_of_minicolumns_per_overlap: &mut [i32]) {
        for minicolumn_idx in 0..self.minicolumns.len() {
            let column_idx = minicolumn_idx / minicolumns_per_column;
            let offset = (max_overlap+1)*column_idx;
            self.minicolumns[minicolumn_idx].overlap = self.htm_calculate_overlap_for_minicolumn(minicolumn_idx,bitset_input,&mut number_of_minicolumns_per_overlap[offset..offset+max_overlap+1]);
        }
    }
    fn htm_calculate_overlap(&mut self, bitset_input: &CpuBitset, number_of_minicolumns_per_overlap: &mut [i32]) {
        for minicolumn_idx in 0..self.minicolumns.len() {
            self.minicolumns[minicolumn_idx].overlap = self.htm_calculate_overlap_for_minicolumn(minicolumn_idx,bitset_input,number_of_minicolumns_per_overlap);
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
    fn htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n_and_group_into_columns
    (&self, max_overlap:usize, column_count:usize, number_of_minicolumns_per_overlap: &mut [i32]) {
        for column_idx in 0..column_count{
            let offset = column_idx*(max_overlap+1);
            let smallest_overlap_that_made_it_to_top_n = self.htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&mut number_of_minicolumns_per_overlap[offset..offset+max_overlap+1]);
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
    fn htm_find_top_minicolumns_and_group_into_columns(&mut self,
                                                       n:usize, max_overlap:usize, column_count:usize, minicolumns_per_column:usize,
                                                       number_of_minicolumns_per_overlap_that_made_it_to_top_n: &mut [i32],
                                                       top_n_minicolumns: &mut [u32]){
        let mut current_top_n_minicolumn_idx = 0;
        for column_idx in 0..column_count {
            let minicolumn_offset = column_idx*minicolumns_per_column;
            let overlap_offset = (max_overlap+1)*column_idx;
            let smallest_overlap_that_made_it_to_top_n = number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap_offset];

            let number_of_minicolumns_per_overlap = &mut number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap_offset..overlap_offset+max_overlap+1];
            for minicolumn_idx in minicolumn_offset..minicolumn_offset+minicolumns_per_column{
                let overlap = self.minicolumns[minicolumn_idx].overlap;
                if overlap >= smallest_overlap_that_made_it_to_top_n as i32 { // the array number_of_minicolumns_per_overlap_that_made_it_to_top_n holds rubbish for any overlap lower than smallest_overlap_that_made_it_to_top_n
                    if number_of_minicolumns_per_overlap[overlap as usize] > 0 { // only add those columns that made it to top n
                        number_of_minicolumns_per_overlap[overlap as usize] -= 1;
                        top_n_minicolumns[current_top_n_minicolumn_idx] = minicolumn_idx as u32;
                        current_top_n_minicolumn_idx += 1;
                    }
                }
            }
            let final_idx = (column_idx+1)*n;
            if current_top_n_minicolumn_idx < final_idx{
                for minicolumn_idx in minicolumn_offset..minicolumn_offset+minicolumns_per_column {
                    let overlap = self.minicolumns[minicolumn_idx].overlap;
                    if overlap == 0{
                        top_n_minicolumns[current_top_n_minicolumn_idx] = minicolumn_idx as u32;
                        current_top_n_minicolumn_idx += 1;
                        if current_top_n_minicolumn_idx >= final_idx{
                            return;
                        }
                    }
                }
            }
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
    pub fn compute_and_group_into_columns(&mut self, minicolumns_per_column:usize,bitset_input: &CpuBitset) -> CpuSDR {
        assert_eq!(self.minicolumns_as_slice().len()%minicolumns_per_column,0,"The number of minicolumns cannot be evenly divided into columns");
        let column_count = self.minicolumns_as_slice().len()/minicolumns_per_column;
        assert!(self.input_size() <= bitset_input.size(), "HTM expects input of size {} but got {}", self.input_size(), bitset_input.size());
        let mut number_of_minicolumns_per_overlap = vec![0; (self.max_overlap as usize + 1)*column_count];
        self.htm_calculate_overlap_and_group_into_columns(self.max_overlap as usize,minicolumns_per_column,bitset_input, &mut number_of_minicolumns_per_overlap);
        self.htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n_and_group_into_columns(self.max_overlap as usize,column_count,&mut number_of_minicolumns_per_overlap);
        let mut top_n_minicolumns = Vec::with_capacity((self.n as usize)*column_count);
        unsafe { top_n_minicolumns.set_len((self.n as usize)*column_count) }
        self.htm_find_top_minicolumns_and_group_into_columns(self.n as usize, self.max_overlap as usize,column_count,minicolumns_per_column,&mut number_of_minicolumns_per_overlap, &mut top_n_minicolumns);
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


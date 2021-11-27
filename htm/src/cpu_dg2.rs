use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::cpu_sdr::CpuSDR;
use crate::htm_program::HtmProgram;
use ndalgebra::buffer::Buffer;
use crate::dg2::*;
use crate::cpu_bitset::CpuBitset;
use crate::rand::{xorshift32, rand_u32_to_random_f32};
use std::cmp::Ordering;
use std::collections::HashSet;
use crate::cpu_bitset2d::CpuBitset2d;
use serde::{Serialize, Deserialize};

/***This implementation assumes that most of the time  vast majority of minicolumns are connected to at least one active
input. Hence instead of iterating the input and then visiting only connected minicolumns, it's better to just iterate all
minicolumns. If you are confident that your input is so sparse than only a sparse number of minicolumns
sees some active connections at any time, then use CpuHTM. It will only visit those minicolumns that are connected
to some active input.*/
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct CpuDG2<Coord: Copy> {
    feedforward_connections: Vec<Coord>,
    minicolumns: Vec<DgMinicolumn2>,
    input_size: Coord,
    span: Coord,
    pub n: u32,
    pub max_overlap: u32,
}

impl<Coord: Copy> CpuDG2<Coord> {
    pub fn input_size(&self) -> Coord {
        self.input_size
    }
    pub fn span(&self) -> Coord {
        self.span
    }
    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn max_overlap(&self) -> u32 {
        self.max_overlap
    }
    pub fn feedforward_connections_as_slice(&self) -> &[Coord] {
        self.feedforward_connections.as_slice()
    }
    pub fn feedforward_connections_as_mut_slice(&mut self) -> &mut [Coord] {
        self.feedforward_connections.as_mut_slice()
    }
    pub fn minicolumns_as_slice(&self) -> &[DgMinicolumn2] {
        self.minicolumns.as_slice()
    }

    /**n = how many minicolumns to activate. We will always take the top n minicolumns with the greatest overlap value.*/
    pub fn new(input_size: Coord, span:Coord, n: u32) -> Self {
        Self {
            max_overlap: 0,
            feedforward_connections: vec![],
            minicolumns: vec![],
            n,
            input_size,
            span
        }
    }
    /**n = how many minicolumns to activate. We will always take the top n minicolumns with the greatest overlap value.*/
    pub fn add_minicolumns(&mut self, minicolumns_count: u32, mut random_input_close_to_minicolumn: impl FnMut(u32) -> Coord, mut input_count_incoming_to_minicolumn: impl FnMut(u32) -> u32) {
        let Self { feedforward_connections, minicolumns, .. } = self;
        minicolumns.reserve(minicolumns_count as usize);
        let original_column_count = minicolumns.len();
        for minicolumn_id in 0..minicolumns_count as u32 {
            let input_count = input_count_incoming_to_minicolumn(minicolumn_id);
            let connection_begin = feedforward_connections.len() as u32;
            for _ in 0..input_count {
                let mut inp_perm = random_input_close_to_minicolumn(minicolumn_id);
                feedforward_connections.push(inp_perm);
            }
            let connection_end = feedforward_connections.len() as u32;
            minicolumns.push(DgMinicolumn2 {
                connection_offset: connection_begin,
                connection_len: connection_end - connection_begin,
                overlap: 0,
            });
        }
        self.max_overlap = self.max_overlap.max(minicolumns[original_column_count..].iter().map(|m| m.connection_len).max().unwrap());
    }


    fn dg_calculate_overlap(&mut self, number_of_minicolumns_per_overlap: &mut [i32], max_overlap: impl Fn(&[Coord]) -> u32) {
        let Self { feedforward_connections, minicolumns, .. } = self;
        for minicolumn in minicolumns {
            let input = &feedforward_connections[minicolumn.connection_offset as usize..(minicolumn.connection_offset + minicolumn.connection_len) as usize];
            let overlap = max_overlap(input);
            if overlap > 0 {
                number_of_minicolumns_per_overlap[overlap as usize] += 1;
            }
            minicolumn.overlap = overlap as i32;
        }
    }

    /**returns smallest_overlap_that_made_it_to_top_n.
    By the end of running this function, the number_of_minicolumns_per_overlap array will become
    number_of_minicolumns_per_overlap_that_made_it_to_top_n.
    number_of_minicolumns_per_overlap_that_made_it_to_top_n holds rubbish for any overlap lower than smallest_overlap_that_made_it_to_top_n
    */
    fn dg_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&self, number_of_minicolumns_per_overlap: &mut [i32]) -> u32 {
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

    /**This function does the exact same thing as dg_find_top_minicolumns, but that function works
    optimally when the input is so sparse that only a tiny fraction of minicolumns has even a single
    connection to some active input. In cases where vast majority minicolumns is expected to have
    at least one connection to some active input, then dg_find_top_minicolumns2 will be much more optimal.
    */
    fn dg_find_top_minicolumns(&mut self,
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
    pub fn compute(&mut self, max_overlap: impl Fn(&[Coord]) -> u32) -> CpuSDR {
        let mut number_of_minicolumns_per_overlap = vec![0; self.max_overlap as usize + 1];
        self.dg_calculate_overlap(&mut number_of_minicolumns_per_overlap, max_overlap);
        let smallest_overlap_that_made_it_to_top_n = self.dg_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&mut number_of_minicolumns_per_overlap);
        let mut top_n_minicolumns = Vec::with_capacity(self.n as usize);
        unsafe { top_n_minicolumns.set_len(self.n as usize) }
        let mut current_top_n_minicolumn_idx = 0;
        self.dg_find_top_minicolumns(&mut number_of_minicolumns_per_overlap, smallest_overlap_that_made_it_to_top_n, &mut top_n_minicolumns, &mut current_top_n_minicolumn_idx);
        let top_minicolumn_count = current_top_n_minicolumn_idx;
        unsafe { top_n_minicolumns.set_len(top_minicolumn_count as usize) }
        CpuSDR::from(top_n_minicolumns)
    }
}

impl CpuDG2<(u32, u32)> {
    pub fn new_2d(input_size: (u32, u32), span: (u32, u32), n: u32) -> Self {
        let (input_h, input_w) = input_size;
        let (span_h, span_w) = span;
        assert!(span_w <= input_w, "Span has width {} but total input is of width {}", span_w, input_w);
        assert!(span_h <= input_h, "Span has height  {} but total input is of height {}", span_h, input_h);
        Self::new(input_size, span, n)
    }
    pub fn compute_translation_invariant(&mut self, bitset_input: &CpuBitset2d, stride: (u32, u32)) -> CpuSDR {
        let (input_h, input_w) = self.input_size;
        let (input_h, input_w) = (input_h as i32, input_w as i32);
        let (span_h, span_w) = self.span;
        let (span_h, span_w) = (span_h as i32, span_w as i32);
        assert!(bitset_input.size() as i32 >= input_w * input_h, "Expected input of size {}x{}={} but bitset has length {}", input_w, input_h, input_w * input_h, bitset_input.size());
        let (s_y, s_x) = stride;
        let (s_y, s_x) = (s_y as i32, s_x as i32);
        self.compute(|coords| {
            let mut max_overlap = 0;
            for offset_x in (s_x-span_w..input_w).step_by(s_x as usize) {
                for offset_y in (s_y-span_h..input_h).step_by(s_y as usize) {
                    let mut overlap = 0;
                    for &(x, y) in coords {
                        let y = offset_y + y as i32;
                        let x = offset_x + x as i32;
                        if 0 <= y && y < input_h && 0 <= x && x < input_w && bitset_input.is_bit_at(y as u32, x as u32) {
                            overlap += 1;
                        }
                    }
                    if overlap > max_overlap {
                        max_overlap = overlap
                    }
                }
            }
            max_overlap
        })
    }
    pub fn add_globally_uniform_prob(&mut self, minicolumns: u32, inputs_per_granular_cell: u32, mut rand_seed: u32) {
        let (input_h, input_w) = self.input_size;
        let (span_h, span_w) = self.span;
        assert!(inputs_per_granular_cell <= span_w * span_h, "Column can't have {} inputs if there are only {} inputs in span of {}x{}", inputs_per_granular_cell, span_w * span_h, span_w, span_h);
        let mut set = HashSet::<(u32, u32)>::new();
        let mut prev_minicolumn_id = u32::MAX;
        self.add_minicolumns(minicolumns, |minicolumn_id| {
            if minicolumn_id != prev_minicolumn_id {
                prev_minicolumn_id = minicolumn_id;
                set.clear();
            }
            rand_seed = xorshift32(rand_seed);
            let mut x = rand_seed % span_w;
            rand_seed = xorshift32(rand_seed);
            let mut y = rand_seed % span_h;
            while !set.insert((y, x)) {
                x = (x + 1) % span_w;
                if x == 0 {
                    y = (y + 1) % span_h;
                }
            }
            (y, x)
        }, |minicolumn_id| inputs_per_granular_cell)
    }
    pub fn make_bitset(&self) -> CpuBitset2d {
        let (h, w) = self.input_size;
        CpuBitset2d::new(CpuBitset::new(w * h), h, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test1() {
        let mut dg = CpuDG2::new_2d((8, 8), (4, 4), 8);
        dg.add_globally_uniform_prob(64, 8, 42354);
        let mut b = dg.make_bitset();
        fn set_pattern(b: &mut CpuBitset2d, offset_x: u32, offset_y: u32) {
            b.set_bit_at(0, 1);
            b.set_bit_at(0, 3);
            b.set_bit_at(0, 6);
            b.set_bit_at(1, 0);
            b.set_bit_at(1, 2);
            b.set_bit_at(1, 6);
            b.set_bit_at(6, 0);
            b.set_bit_at(6, 5);
            b.set_bit_at(6, 6);
        }
        set_pattern(&mut b, 10, 10);
        assert!(dg.n < dg.input_size.0 * dg.input_size.1);
        let a1 = dg.compute_translation_invariant(&b, (1, 1));
        assert!(a1.cardinality() > 0);
        assert!(a1.cardinality() <= dg.n);
        b.clear_all();
        set_pattern(&mut b, 12, 10);
        let a2 = dg.compute_translation_invariant(&b, (1, 1));
        assert!(a2.cardinality() > 0);
        assert!(a2.cardinality() <= dg.n);
        assert_eq!(a1, a2);
        b.clear_all();
        set_pattern(&mut b, 12, 12);
        let a3 = dg.compute_translation_invariant(&b, (1, 1));
        assert!(a3.cardinality() > 0);
        assert!(a3.cardinality() <= dg.n);
        assert_eq!(a2, a3);
        b.clear_all();
        set_pattern(&mut b, 21, 22);
        let a4 = dg.compute_translation_invariant(&b, (1, 1));
        assert_eq!(a3, a4);
    }
}


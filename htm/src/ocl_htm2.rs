use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::ocl_sdr::OclSDR;
use crate::htm_program::HtmProgram;
use ndalgebra::buffer::Buffer;
use crate::htm2::*;
use crate::{CpuHTM2, OclBitset};

#[derive(Clone)]
pub struct OclHTM2 {
    prog: HtmProgram,
    feedforward_connections: Buffer<HtmFeedforwardConnection2>,
    minicolumns: Buffer<HtmMinicolumn2>,
    pub permanence_threshold: f32,
    pub n: u32,
    pub permanence_decrement_increment: [f32; 2],
    pub max_overlap: u32,
}

impl OclHTM2{
    pub fn new(ch:&CpuHTM2, prog:HtmProgram)->Result<Self,Error>{
        let feedforward_connections = prog.buffer_from_slice(MemFlags::READ_WRITE, ch.feedforward_connections_as_slice())?;
        let minicolumns = prog.buffer_from_slice(MemFlags::READ_WRITE, ch.minicolumns_as_slice())?;
        Ok(Self{
            prog,
            feedforward_connections,
            minicolumns,
            permanence_threshold: ch.permanence_threshold(),
            n: ch.n(),
            permanence_decrement_increment: ch.permanence_decrement_increment(),
            max_overlap: ch.max_overlap(),
        })
    }

    pub fn minicolumn_count(&self) -> usize {
        self.minicolumns.len()
    }


    fn htm_calculate_overlap(&mut self, bitset_input:&OclBitset, number_of_minicolumns_per_overlap:&Buffer<i32>) -> Result<(), Error> {
        self.prog.kernel_builder("htm_calculate_overlap2")?.
            add_num(self.permanence_threshold)?. // float permanence_threshold,
            add_buff(&self.minicolumns)?.// __global HtmMinicolumn2 * minicolumns,
            add_buff(bitset_input.buffer())?.// __global uint * inputs,
            add_buff(&self.feedforward_connections)?.// __global HtmFeedforwardConnection2 * feedforward_connections
            add_buff(number_of_minicolumns_per_overlap)?. // __global int * number_of_minicolumns_per_overlap
            enq(self.prog.queue(),&[self.minicolumn_count(),1,1]).
            map_err(Error::from)
    }

    fn htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&self, number_of_minicolumns_per_overlap: &Buffer<i32>) -> Result<u32,Error> {
        let mut total_minicolumns = 0;
        let mut number_of_minicolumns_per_overlap_vec = number_of_minicolumns_per_overlap.to_vec(self.prog.queue())?;
        for overlap in (0..number_of_minicolumns_per_overlap_vec.len()).rev() {
            let number_of_minicolumns = number_of_minicolumns_per_overlap_vec[overlap as usize];
            total_minicolumns += number_of_minicolumns;
            if total_minicolumns > self.n as i32 {
                number_of_minicolumns_per_overlap.write(self.prog.queue(),overlap,&[self.n as i32 - (total_minicolumns - number_of_minicolumns)])?;
                return Ok(overlap as u32);
            }
        }
        Ok(0)
    }



    fn htm_update_permanence(&mut self, top_n_minicolumns: &Buffer<u32>, bitset_input:&OclBitset, current_top_n_minicolumn_idx: u32) -> Result<(), Error> {
        self.prog.kernel_builder("htm_update_permanence2")?.
            add_num(self.permanence_decrement_increment[1])?. // float permanence_increment,
            add_num(self.permanence_decrement_increment[0])?. // float permanence_decrement,
            add_buff(&self.minicolumns)?.// __global HtmMinicolumn2 * minicolumns,
            add_buff(bitset_input.buffer())?.// __global uint * inputs,
            add_buff(top_n_minicolumns)?.    // __global uint * top_n_minicolumns,
            add_buff(&self.feedforward_connections)?. // __global HtmFeedforwardConnection2 * feedforward_connections
            enq(self.prog.queue(),&[current_top_n_minicolumn_idx as usize,1,1]).
            map_err(Error::from)
    }


    fn htm_find_top_minicolumns(&mut self, number_of_minicolumns_per_overlap_that_made_it_to_top_n: &Buffer<i32>, smallest_overlap_that_made_it_to_top_n: u32, top_n_minicolumns: &Buffer<u32>, current_top_n_minicolumn_idx: &Buffer<u32>)  -> Result<(), Error> {
        self.prog.kernel_builder("htm_find_top_minicolumns2")?.
            add_num(self.permanence_threshold)?. // float permanence_threshold,
            add_buff(&self.minicolumns)?.// __global HtmMinicolumn2 * minicolumns,
            add_buff(number_of_minicolumns_per_overlap_that_made_it_to_top_n)?.// __global int * number_of_minicolumns_per_overlap_that_made_it_to_top_n,
            add_num(smallest_overlap_that_made_it_to_top_n)?.// int smallest_overlap_that_made_it_to_top_n,
            add_buff(&top_n_minicolumns)?.// __global uint * top_n_minicolumns,
            add_buff(&current_top_n_minicolumn_idx)?.// __global uint * current_top_n_minicolumn_idx,
            enq(self.prog.queue(),&[self.minicolumn_count(),1,1]).
            map_err(Error::from)
    }

    pub fn infer(&mut self, bitset_input: &OclBitset, learn: bool) -> Result<OclSDR, Error> {
        let mut number_of_minicolumns_per_overlap = self.prog.buffer_filled(MemFlags::READ_WRITE,self.max_overlap as usize,0)?;
        self.htm_calculate_overlap(bitset_input, &number_of_minicolumns_per_overlap)?;
        self.prog.q.finish();
        let smallest_overlap_that_made_it_to_top_n = self.htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&number_of_minicolumns_per_overlap)?;
        let top_n_minicolumns = unsafe{self.prog.buffer_empty(MemFlags::READ_WRITE,self.n as usize)?};
        let current_top_n_minicolumn_idx = self.prog.buffer_filled(MemFlags::READ_WRITE,1,0)?;
        self.htm_find_top_minicolumns(&number_of_minicolumns_per_overlap, smallest_overlap_that_made_it_to_top_n, &top_n_minicolumns, &current_top_n_minicolumn_idx);
        let top_minicolumn_count = current_top_n_minicolumn_idx.get(self.prog.queue(),0)?;
        if learn {
            self.htm_update_permanence(&top_n_minicolumns,bitset_input, top_minicolumn_count)?
        }
        Ok(OclSDR::from_buff(self.prog.clone(),top_n_minicolumns, top_minicolumn_count))
    }
}


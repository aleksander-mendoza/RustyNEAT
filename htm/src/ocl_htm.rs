use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::ocl_sdr::OclSDR;
use crate::htm_program::HtmProgram;
use ndalgebra::buffer::Buffer;
use crate::htm::*;
use crate::{CpuHTM, OclInput, OclBitset};

#[derive(Clone)]
pub struct OclHTM {
    prog: HtmProgram,
    feedforward_connections: Buffer<HtmFeedforwardConnection>,
    connection_indices: Buffer<u32>,
    inputs: Buffer<HtmInput>,
    minicolumns: Buffer<HtmMinicolumn>,
    pub permanence_threshold: f32,
    pub n: u32,
    pub permanence_decrement_increment: [f32; 2],
    pub max_overlap: u32,
}

impl OclHTM{
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
    pub fn new(ch:&CpuHTM, prog:HtmProgram)->Result<Self,Error>{
        let feedforward_connections = prog.buffer_from_slice(MemFlags::READ_WRITE, ch.feedforward_connections_as_slice())?;
        let connection_indices = prog.buffer_from_slice(MemFlags::READ_WRITE, ch.connection_indices_as_slice())?;
        let inputs = prog.buffer_from_slice(MemFlags::READ_WRITE, ch.inputs_as_slice())?;
        assert_eq!(ch.inputs_as_slice(),inputs.to_vec(prog.queue())?.as_slice());
        let minicolumns = prog.buffer_from_slice(MemFlags::READ_WRITE, ch.minicolumns_as_slice())?;
        Ok(Self{
            prog,
            feedforward_connections,
            connection_indices,
            inputs,
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

    fn htm_calculate_overlap(&mut self, sdr_input: &OclSDR) -> Result<(), Error> {
        self.prog.kernel_builder("htm_calculate_overlap")?.
            add_num(self.permanence_threshold)?. // float permanence_threshold,
            add_buff(sdr_input.buffer())?.// __global uint * sdr_input,
            add_buff(&self.minicolumns)?.// __global HtmMinicolumn * minicolumns,
            add_buff(&self.inputs)?.// __global HtmInput * inputs,
            add_buff(&self.feedforward_connections)?.// __global HtmFeedforwardConnection * feedforward_connections;
            enq(self.prog.queue(),&[sdr_input.cardinality() as usize,1,1]).
            map_err(Error::from)
    }

    fn htm_calculate_number_of_minicolumns_per_overlap(&mut self, sdr_input: &OclSDR, number_of_minicolumns_per_overlap: &Buffer<i32>) -> Result<(), Error> {
        self.prog.kernel_builder("htm_calculate_number_of_minicolumns_per_overlap")?.
            add_num(self.permanence_threshold)?. // float permanence_threshold,
            add_buff(sdr_input.buffer())?.// __global uint * sdr_input,
            add_buff(&self.minicolumns)?.// __global HtmMinicolumn * minicolumns,
            add_buff(&self.inputs)?.// __global HtmInput * inputs,
            add_buff(number_of_minicolumns_per_overlap)?.// __global int * number_of_minicolumns_per_overlap
            add_buff(&self.feedforward_connections)?. // __global HtmFeedforwardConnection * feedforward_connections
            enq(self.prog.queue(),&[sdr_input.cardinality() as usize,1,1]).
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


    fn htm_find_top_minicolumns(&mut self, sdr_input: &OclSDR, number_of_minicolumns_per_overlap_that_made_it_to_top_n: &Buffer<i32>, smallest_overlap_that_made_it_to_top_n: u32, top_n_minicolumns: &Buffer<u32>, current_top_n_minicolumn_idx: &Buffer<u32>) -> Result<(), Error> {
        self.prog.kernel_builder("htm_find_top_minicolumns")?.
            add_num(self.permanence_threshold)?. // float permanence_threshold,
            add_buff(sdr_input.buffer())?.// __global uint * sdr_input,
            add_buff(&self.minicolumns)?.// __global HtmMinicolumn * minicolumns,
            add_buff(&self.inputs)?.// __global HtmInput * inputs,
            add_buff(number_of_minicolumns_per_overlap_that_made_it_to_top_n)?.// __global int * number_of_minicolumns_per_overlap_that_made_it_to_top_n
            add_num(smallest_overlap_that_made_it_to_top_n)?.   // int smallest_overlap_that_made_it_to_top_n,
            add_buff(top_n_minicolumns)?.    // __global uint * top_n_minicolumns,
            add_buff(current_top_n_minicolumn_idx)?.    // __global uint * current_top_n_minicolumn_idx, // precodntion: equals 0 ; postcondition: less than or equal n
            add_buff(&self.feedforward_connections)?. // __global HtmFeedforwardConnection * feedforward_connections
            enq(self.prog.queue(),&[sdr_input.cardinality() as usize,1,1]).
            map_err(Error::from)
    }

    fn htm_update_permanence(&mut self, top_n_minicolumns: &Buffer<u32>,bitset_input:&OclBitset, current_top_n_minicolumn_idx: u32) -> Result<(), Error> {
        self.prog.kernel_builder("htm_update_permanence")?.
            add_num(self.permanence_decrement_increment[1])?. // float permanence_increment,
            add_num(self.permanence_decrement_increment[0])?. // float permanence_decrement,
            add_buff(&self.connection_indices)?. // __global uint * connection_indices,
            add_buff(&self.minicolumns)?.// __global HtmMinicolumn * minicolumns,
            add_buff(bitset_input.buffer())?.// __global uint * bitset_input,
            add_buff(top_n_minicolumns)?.    // __global uint * top_n_minicolumns,
            add_buff(&self.feedforward_connections)?. // __global HtmFeedforwardConnection * feedforward_connections
            enq(self.prog.queue(),&[current_top_n_minicolumn_idx as usize,1,1]).
            map_err(Error::from)
    }

    fn htm_clean_up_active_inputs(&mut self, sdr_input: &OclSDR) -> Result<(), Error> {
        self.prog.kernel_builder("htm_clean_up_active_inputs")?.
            add_buff(sdr_input.buffer())?. // __global uint * sdr_input,
            add_buff(&self.inputs)?. // __global HtmInput * inputs
            enq(self.prog.queue(),&[sdr_input.cardinality() as usize,1,1]).
            map_err(Error::from)
    }

    fn htm_clean_up_overlap(&mut self, sdr_input: &OclSDR) -> Result<(), Error> {
        self.prog.kernel_builder("htm_clean_up_overlap")?.
            add_buff(sdr_input.buffer())?.// __global uint * sdr_input,
            add_buff(&self.minicolumns)?.// __global HtmMinicolumn * minicolumns,
            add_buff(&self.inputs)?.// __global HtmInput * inputs,
            add_buff(&self.feedforward_connections)?. // __global HtmFeedforwardConnection * feedforward_connections
            enq(self.prog.queue(),&[sdr_input.cardinality() as usize,1,1]).
            map_err(Error::from)
    }


    pub fn infer(&mut self, input: &OclInput, learn: bool) -> Result<OclSDR, Error> {
        let sdr_input = input.get_sparse();
        let bitset_input = input.get_dense();
        self.htm_calculate_overlap(sdr_input)?;
        let number_of_minicolumns_per_overlap = self.prog.buffer_filled(MemFlags::READ_WRITE,self.max_overlap as usize,0)?;
        self.htm_calculate_number_of_minicolumns_per_overlap(sdr_input, &number_of_minicolumns_per_overlap)?;
        self.prog.q.finish();
        let smallest_overlap_that_made_it_to_top_n = self.htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&number_of_minicolumns_per_overlap)?;
        let top_n_minicolumns = unsafe{self.prog.buffer_empty(MemFlags::READ_WRITE,self.n as usize)?};
        let current_top_n_minicolumn_idx = self.prog.buffer_filled(MemFlags::READ_WRITE,1,0)?;
        self.htm_find_top_minicolumns(sdr_input, &number_of_minicolumns_per_overlap, smallest_overlap_that_made_it_to_top_n, &top_n_minicolumns, &current_top_n_minicolumn_idx)?;
        let top_minicolumn_count = current_top_n_minicolumn_idx.get(self.prog.queue(),0)?;
        if learn {
            self.htm_update_permanence(&top_n_minicolumns, bitset_input,top_minicolumn_count)?
        }
        self.htm_clean_up_overlap(sdr_input)?;
        Ok(OclSDR::from_buff(self.prog.clone(),top_n_minicolumns, top_minicolumn_count))
    }

}


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
    input_size:u32,
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
            input_size:ch.input_size(),
            prog,
            feedforward_connections,
            minicolumns,
            permanence_threshold: ch.permanence_threshold(),
            n: ch.n(),
            permanence_decrement_increment: ch.permanence_decrement_increment(),
            max_overlap: ch.max_overlap(),
        })
    }
    pub fn input_size(&self)->u32{
        self.input_size
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

    fn htm_calculate_overlap_and_group_into_columns(&mut self, max_overlap:usize,minicolumns_per_column:usize, bitset_input:&OclBitset, number_of_minicolumns_per_overlap:&Buffer<i32>) -> Result<(), Error> {
        self.prog.kernel_builder("htm_calculate_overlap_and_group_into_columns2")?.
            add_num(max_overlap)?. // size_t max_overlap,
            add_num(minicolumns_per_column)?. // size_t minicolumns_per_column,
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

    fn htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n_and_group_into_columns(&self, n:usize, max_overlap:usize, column_cont:usize, number_of_minicolumns_per_overlap: &Buffer<i32>) -> Result<(), Error> {
        self.prog.kernel_builder("htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n_and_group_into_columns2")?.
            add_num(n)?. // const size_t n,
            add_num(max_overlap)?. //const size_t max_overlap,
            add_buff(number_of_minicolumns_per_overlap)?. // __global int * number_of_minicolumns_per_overlap,
            enq(self.prog.queue(),&[column_cont,1,1]).
            map_err(Error::from)
    }



    pub fn update_permanence(&mut self, active_minicolumns: &OclSDR, bitset_input:&OclBitset) -> Result<(), Error> {
        self.prog.kernel_builder("htm_update_permanence2")?.
            add_num(self.permanence_decrement_increment[1])?. // float permanence_increment,
            add_num(self.permanence_decrement_increment[0])?. // float permanence_decrement,
            add_buff(&self.minicolumns)?.// __global HtmMinicolumn2 * minicolumns,
            add_buff(bitset_input.buffer())?.// __global uint * inputs,
            add_buff(active_minicolumns.buffer())?.    // __global uint * top_n_minicolumns,
            add_buff(&self.feedforward_connections)?. // __global HtmFeedforwardConnection2 * feedforward_connections
            enq(self.prog.queue(),&[active_minicolumns.cardinality() as usize,1,1]).
            map_err(Error::from)
    }


    fn htm_find_top_minicolumns(&mut self, number_of_minicolumns_per_overlap_that_made_it_to_top_n: &Buffer<i32>, smallest_overlap_that_made_it_to_top_n: u32, top_n_minicolumns: &Buffer<u32>, current_top_n_minicolumn_idx: &Buffer<u32>)  -> Result<(), Error> {
        self.prog.kernel_builder("htm_find_top_minicolumns2")?.
            add_buff(&self.minicolumns)?.// __global HtmMinicolumn2 * minicolumns,
            add_buff(number_of_minicolumns_per_overlap_that_made_it_to_top_n)?.// __global int * number_of_minicolumns_per_overlap_that_made_it_to_top_n,
            add_num(smallest_overlap_that_made_it_to_top_n)?.// int smallest_overlap_that_made_it_to_top_n,
            add_buff(top_n_minicolumns)?.// __global uint * top_n_minicolumns,
            add_buff(current_top_n_minicolumn_idx)?.// __global uint * current_top_n_minicolumn_idx,
            enq(self.prog.queue(),&[self.minicolumn_count(),1,1]).
            map_err(Error::from)
    }
    fn htm_find_top_minicolumns_and_group_into_columns(&mut self, minicolumns_per_column:usize,
                                                       number_of_minicolumns_per_overlap_that_made_it_to_top_n: &Buffer<i32>, top_n_minicolumns: &Buffer<u32>, current_top_n_minicolumn_idx: &Buffer<u32>) -> Result<(), Error> {
        self.prog.kernel_builder("htm_find_top_minicolumns_and_group_into_columns2")?.
            add_num(self.n)?. // size_t n,
            add_num(self.max_overlap)?. // size_t max_overlap,
            add_num(minicolumns_per_column)?. // size_t minicolumns_per_column,
            add_buff(&self.minicolumns)?.// __global HtmMinicolumn2  * minicolumns,
            add_buff(number_of_minicolumns_per_overlap_that_made_it_to_top_n)?.// __global int * number_of_minicolumns_per_overlap_that_made_it_to_top_n,
            add_buff(top_n_minicolumns)?.// __global uint * top_n_minicolumns,
            add_buff(current_top_n_minicolumn_idx)?.// __global uint * current_top_n_minicolumn_idx,
            enq(self.prog.queue(),&[self.minicolumn_count(),1,1]).
            map_err(Error::from)
    }
    pub fn compute(&mut self, bitset_input: &OclBitset) -> Result<OclSDR, Error> {
        let mut number_of_minicolumns_per_overlap = self.prog.buffer_filled(MemFlags::READ_WRITE,self.max_overlap as usize+1,0)?;
        self.htm_calculate_overlap(bitset_input, &number_of_minicolumns_per_overlap)?;
        self.prog.q.finish();
        let smallest_overlap_that_made_it_to_top_n = self.htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&number_of_minicolumns_per_overlap)?;
        let top_n_minicolumns = unsafe{self.prog.buffer_empty(MemFlags::READ_WRITE,self.n as usize)?};
        let current_top_n_minicolumn_idx = self.prog.buffer_filled(MemFlags::READ_WRITE,1,0)?;
        self.htm_find_top_minicolumns(&number_of_minicolumns_per_overlap, smallest_overlap_that_made_it_to_top_n, &top_n_minicolumns, &current_top_n_minicolumn_idx);
        let top_minicolumn_count = current_top_n_minicolumn_idx.get(self.prog.queue(),0)?;
        Ok(OclSDR::from_buff(self.prog.clone(),top_n_minicolumns, top_minicolumn_count))
    }
    pub fn infer(&mut self, bitset_input: &OclBitset, learn: bool) -> Result<OclSDR, Error> {
        let top_n_minicolumns = self.compute(bitset_input)?;
        if learn {
            self.update_permanence(&top_n_minicolumns, bitset_input)?
        }
        Ok(top_n_minicolumns)
    }

    pub fn compute_and_group_into_columns(&mut self, minicolumns_per_column:usize,bitset_input: &OclBitset) -> Result<OclSDR,Error> {
        assert!(minicolumns_per_column>=self.n as usize,"Each column activates n={} winners but there are only {} minicolumns per column",self.n,minicolumns_per_column);
        assert_eq!(self.minicolumn_count()%minicolumns_per_column,0,"The number of minicolumns cannot be evenly divided into columns");
        let column_count = self.minicolumn_count()/minicolumns_per_column;
        assert!(self.input_size() as usize <= bitset_input.size(), "HTM expects input of size {} but got {}", self.input_size(), bitset_input.size());
        let mut number_of_minicolumns_per_overlap = self.prog.buffer_filled(MemFlags::READ_WRITE,(self.max_overlap as usize + 1)*column_count,0)?;
        self.htm_calculate_overlap_and_group_into_columns(self.max_overlap as usize,minicolumns_per_column,bitset_input, &number_of_minicolumns_per_overlap);
        self.htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n_and_group_into_columns(self.n as usize,self.max_overlap as usize,column_count,&number_of_minicolumns_per_overlap);
        let top_n_minicolumns = unsafe{self.prog.buffer_empty(MemFlags::READ_WRITE,(self.n as usize)*column_count)?};
        let current_top_n_minicolumn_idx = self.prog.buffer_filled(MemFlags::READ_WRITE,column_count,0)?;
        self.htm_find_top_minicolumns_and_group_into_columns(minicolumns_per_column,&number_of_minicolumns_per_overlap,&top_n_minicolumns, &current_top_n_minicolumn_idx);
        Ok(OclSDR::from_buff(self.prog.clone(),top_n_minicolumns,self.n*column_count as u32))
    }
    pub fn infer_and_group_into_columns(&mut self, minicolumns_per_column:usize,bitset_input: &OclBitset, learn: bool) -> Result<OclSDR,Error> {
        let top_n_minicolumns = self.compute_and_group_into_columns(minicolumns_per_column,bitset_input)?;
        if learn {
            self.update_permanence(&top_n_minicolumns, bitset_input)?
        }
        Ok(top_n_minicolumns)
    }
}


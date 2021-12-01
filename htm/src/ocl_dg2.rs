use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::ocl_sdr::OclSDR;
use crate::htm_program::HtmProgram;
use ndalgebra::buffer::Buffer;
use crate::dg2::*;
use crate::{CpuDG2, OclBitset};

#[derive(Clone)]
pub struct OclDG2 {
    prog: HtmProgram,
    feedforward_connections: Buffer<DgCoord2d>,
    minicolumns: Buffer<DgMinicolumn2>,
    input_size: DgCoord2d,
    span: DgCoord2d,
    pub n: u32,
    pub max_overlap: u32,
}

impl OclDG2{
    pub fn input_size(&self) -> DgCoord2d {
        self.input_size
    }
    pub fn span(&self) -> DgCoord2d {
        self.span
    }
    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn max_overlap(&self) -> u32 {
        self.max_overlap
    }

    pub fn new(ch:&CpuDG2<DgCoord2d>, prog:HtmProgram)->Result<Self,Error>{
        let feedforward_connections = prog.buffer_from_slice(MemFlags::READ_WRITE, ch.feedforward_connections_as_slice())?;
        let minicolumns = prog.buffer_from_slice(MemFlags::READ_WRITE, ch.minicolumns_as_slice())?;
        Ok(Self{
            prog,
            feedforward_connections,
            minicolumns,
            input_size: ch.input_size(),
            span: ch.span(),
            n: ch.n(),
            max_overlap: ch.max_overlap(),
        })
    }

    pub fn minicolumn_count(&self) -> usize {
        self.minicolumns.len()
    }


    fn dg_calculate_overlap(&mut self, bitset_input:&OclBitset, number_of_minicolumns_per_overlap:&Buffer<i32>, stride: (u32, u32)) -> Result<(), Error> {
        self.prog.kernel_builder("dg_calculate_overlap2")?.
            add_num(self.input_size.y)?.     // int input_h,
            add_num(self.input_size.x)?.      // int input_w,
            add_num(self.span.y)?.      // int span_h,
            add_num(self.span.x)?.      // int span_w,
            add_num(stride.0)?.      // int stride_y,
            add_num(stride.1)?.      // int stride_x,
            add_buff(&self.minicolumns)?.      // __global DgMinicolumn2 * minicolumns,
            add_buff(bitset_input.buffer())?.      // __global uint * bitset_input,
            add_buff(&self.feedforward_connections)?.      // __global DgCoord2d * feedforward_connections,
            add_buff(number_of_minicolumns_per_overlap)?.      // __global int * number_of_minicolumns_per_overlap
            enq(self.prog.queue(),&[self.minicolumn_count(),1,1]).
            map_err(Error::from)
    }

    fn dg_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&self, number_of_minicolumns_per_overlap: &Buffer<i32>) -> Result<u32,Error> {
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

    fn dg_find_top_minicolumns(&mut self, number_of_minicolumns_per_overlap_that_made_it_to_top_n: &Buffer<i32>, smallest_overlap_that_made_it_to_top_n: u32, top_n_minicolumns: &Buffer<u32>, current_top_n_minicolumn_idx: &Buffer<u32>)  -> Result<(), Error> {
        self.prog.kernel_builder("dg_find_top_minicolumns2")?.
            add_buff(&self.minicolumns)?.// __global HtmMinicolumn2 * minicolumns,
            add_buff(number_of_minicolumns_per_overlap_that_made_it_to_top_n)?.// __global int * number_of_minicolumns_per_overlap_that_made_it_to_top_n,
            add_num(smallest_overlap_that_made_it_to_top_n)?.// int smallest_overlap_that_made_it_to_top_n,
            add_buff(&top_n_minicolumns)?.// __global uint * top_n_minicolumns,
            add_buff(&current_top_n_minicolumn_idx)?.// __global uint * current_top_n_minicolumn_idx,
            enq(self.prog.queue(),&[self.minicolumn_count(),1,1]).
            map_err(Error::from)
    }
    pub fn compute_translation_invariant(&mut self, bitset_input: &OclBitset, stride: (u32,u32)) -> Result<OclSDR, Error> {
        let mut number_of_minicolumns_per_overlap = self.prog.buffer_filled(MemFlags::READ_WRITE,self.max_overlap as usize,0)?;
        self.dg_calculate_overlap(bitset_input, &number_of_minicolumns_per_overlap, stride)?;
        self.prog.q.finish();
        let smallest_overlap_that_made_it_to_top_n = self.dg_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n(&number_of_minicolumns_per_overlap)?;
        let top_n_minicolumns = unsafe{self.prog.buffer_empty(MemFlags::READ_WRITE,self.n as usize)?};
        let current_top_n_minicolumn_idx = self.prog.buffer_filled(MemFlags::READ_WRITE,1,0)?;
        self.dg_find_top_minicolumns(&number_of_minicolumns_per_overlap, smallest_overlap_that_made_it_to_top_n, &top_n_minicolumns, &current_top_n_minicolumn_idx);
        let top_minicolumn_count = current_top_n_minicolumn_idx.get(self.prog.queue(),0)?;
        Ok(OclSDR::from_buff(self.prog.clone(),top_n_minicolumns, top_minicolumn_count))
    }
}


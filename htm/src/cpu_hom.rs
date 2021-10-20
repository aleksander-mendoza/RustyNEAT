use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::cpu_sdr::CpuSDR;
use crate::htm_program::HtmProgram;
use ndalgebra::buffer::Buffer;
use std::cmp::Ordering;
use itertools::Itertools;
use crate::EncoderTarget;

#[derive(Clone)]
pub struct HomHyperparams{
    cells_per_minicolumn:u32,
    minicolumn_count:u32,
    current_iteration:u32,
    pub max_synapses_per_segment:usize,
    pub max_segments_per_cell: usize,
    pub max_new_synapse_count:usize,
    pub initial_permanence: f32,
    pub min_permanence_to_keep: f32,
    pub activation_threshold: u32,
    pub learning_threshold: u32,
    pub permanence_threshold: f32,
    pub predicted_decrement:f32,
    pub permanence_decrement_increment: [f32; 2],
}

impl HomHyperparams{
    fn new(cells_per_minicolumn:u32,
           minicolumn_count:u32)->Self{
        Self{
            cells_per_minicolumn,
            max_synapses_per_segment: 255,
            minicolumn_count,
            current_iteration: 0,
            max_segments_per_cell: 255,
            max_new_synapse_count: 20,
            initial_permanence: 0.21,
            min_permanence_to_keep: 0.00001,
            activation_threshold: 13,
            learning_threshold: 10,
            permanence_threshold: 0.0,
            predicted_decrement: -0.05,
            permanence_decrement_increment: [-0.05, 0.10]
        }
    }
    pub fn encode_cell_id(&self,minicolumn_idx:u32, cell_idx:u32)->u32{
        minicolumn_idx*self.cells_per_minicolumn+cell_idx
    }
    pub fn encode_segment_id(&self,minicolumn_idx:u32, segment_idx:u32)->u32{
        minicolumn_idx+segment_idx*self.minicolumn_count
    }
    pub fn decode_cell_id(&self,cell_id:u32)->(u32,u32){
        let minicolumn_idx = cell_id/self.cells_per_minicolumn;
        let segment_idx = cell_id%self.cells_per_minicolumn;
        (minicolumn_idx, segment_idx)
    }
    pub fn decode_segment_id(&self,segment_id:u32)->(u32,u32){
        let minicolumn_idx = segment_id%self.minicolumn_count;
        let segment_idx = segment_id/self.minicolumn_count;
        (minicolumn_idx, segment_idx)
    }
}

#[derive(Clone, Copy)]
pub struct HomDistalSynapse {
    pub cell_id: u32,
    pub permanence: f32,
}
#[derive(Clone)]
pub struct HomSegment {
    pub synapses: Vec<HomDistalSynapse>,
    pub cell_idx:u8, // cell index within the minicolumn
    pub last_used_in_iteration: u32,
    pub is_active:bool,
    pub num_active_potential: u32,
}

impl HomSegment{
    fn is_matching(&self, hyp:&HomHyperparams)->bool{
        self.num_active_potential >= hyp.learning_threshold
    }
    fn update_segment(&mut self, hyp:&HomHyperparams, permanence_decrement_increment:[f32;2], prev_active_cells:&Vec<u32>){
        let mut writing_idx = 0;
        for synapse_idx in 0..self.synapses.len(){
            let mut synapse = self.synapses[synapse_idx];
            let is_active = prev_active_cells.binary_search(&synapse.cell_id).is_ok();
            let permanence_change = permanence_decrement_increment[is_active as usize];
            synapse.permanence = if is_active{
                (synapse.permanence + permanence_change).min(1.)
            }else{
                let new_perm = synapse.permanence + permanence_change;
                if new_perm < hyp.min_permanence_to_keep{
                    continue;
                }
                new_perm
            };
            self.synapses[writing_idx] = synapse;
        }
        unsafe{self.synapses.set_len(writing_idx)}
    }

    fn destroy_min_permanence_synapses(&mut self, new_synapse_count:usize, prev_winner_cells:&Vec<u32>){
        if self.synapses.len() <= new_synapse_count{return}
        self.synapses.sort_by(|a,b|b.permanence.partial_cmp(&a.permanence).unwrap_or(Ordering::Equal));
        for i in (0..self.synapses.len()).rev(){
            if prev_winner_cells.binary_search(&self.synapses[i].cell_id).is_err(){
                let last_idx = self.synapses.len()-1;
                self.synapses[i] = self.synapses[last_idx];
                unsafe{self.synapses.set_len(last_idx)}
                if self.synapses.len() <= new_synapse_count{return}
            }
        }
    }
    fn grow_synapses(&mut self, hyp:&HomHyperparams, added_synapse_count:usize, prev_winner_cells:&Vec<u32>){
        let mut candidates = prev_winner_cells.clone();
        debug_assert!(prev_winner_cells.windows(2).all(|a|a[0]<a[1]));
        for synapse in &self.synapses{
            if let Ok(already_connected_candidate_idx) = candidates.binary_search(&synapse.cell_id){
                candidates.remove(already_connected_candidate_idx);
            }
        }
        let added_synapse_count = added_synapse_count.min(candidates.len());
        self.destroy_min_permanence_synapses(hyp.max_synapses_per_segment-added_synapse_count,prev_winner_cells);
        for _ in 0..added_synapse_count{
            let winner_cell = candidates.swap_remove(rand::random::<usize>() % candidates.len());
            self.synapses.push(HomDistalSynapse{ cell_id: winner_cell, permanence: hyp.initial_permanence })
        }
    }
}

#[derive(Clone)]
pub struct HomMinicolumn {
    pub segments: Vec<HomSegment>,
}

impl HomMinicolumn{
    fn punish_segment(&mut self, hyp:&HomHyperparams, segment_idx:usize, prev_active_cells:&Vec<u32>){
        self.segments[segment_idx].update_segment(hyp,[hyp.predicted_decrement,0.], prev_active_cells);
        if self.segments[segment_idx].synapses.is_empty(){
            self.segments.swap_remove(segment_idx);
        }
    }
    fn adapt_segment(&mut self, hyp:&HomHyperparams, segment_idx:usize, prev_active_cells:&Vec<u32>, prev_winner_cells:&Vec<u32>){
        self.segments[segment_idx].update_segment(hyp,hyp.permanence_decrement_increment, prev_active_cells);
        let num_active_potential = self.segments[segment_idx].num_active_potential as usize;
        if hyp.max_new_synapse_count > num_active_potential{
            let added_synapse_count = hyp.max_new_synapse_count - num_active_potential;
            self.segments[segment_idx].grow_synapses(hyp,added_synapse_count,prev_winner_cells);
        }
        if self.segments[segment_idx].synapses.is_empty(){
            self.segments.swap_remove(segment_idx);
        }
    }
    fn activate_predicted_column(&mut self, learn:bool, hyp:&HomHyperparams,
                                 this_minicolumn_idx:u32,
                                 active_cells:&mut Vec<u32>,
                                 prev_active_cells:&Vec<u32>,
                                 winner_cells:&mut Vec<u32>,
                                 prev_winner_cells:&Vec<u32>)->bool{
        let mut has_cell_been_added = vec![false;hyp.cells_per_minicolumn as usize];
        let mut has_any_active_cells = false;
        for segment_idx in 0..self.segments.len(){
            if self.segments[segment_idx].is_active {
                has_any_active_cells = true;
                let cell_idx= self.segments[segment_idx].cell_idx ;
                if !has_cell_been_added[cell_idx as usize]{
                    has_cell_been_added[cell_idx as usize] = true;
                    let cell_id = hyp.encode_cell_id(this_minicolumn_idx,cell_idx as u32);
                    active_cells.push(cell_id);
                    winner_cells.push(cell_id);
                }
                if learn {
                    self.adapt_segment(hyp,segment_idx,prev_active_cells,prev_winner_cells)
                }
            }
        }
        has_any_active_cells
    }
    fn best_matching_segment(&self, hyp:&HomHyperparams)->u32{
        let mut num_active_potential = hyp.learning_threshold as i32 - 1;
        let mut segment_idx = self.segments.len();
        for (i,segment) in self.segments.iter().enumerate(){
            if segment.num_active_potential as i32 > num_active_potential{
                num_active_potential = segment.num_active_potential as i32;
                segment_idx = i ;
            }
        }
        segment_idx as u32
    }
    fn least_used_cells(&self,hyp:&HomHyperparams)->u32{
        let mut segments_per_cell = vec![0;hyp.cells_per_minicolumn as usize];
        for segment in &self.segments{
            segments_per_cell[segment.cell_idx as usize]+=1;
        }
        let fewest_segments = segments_per_cell.iter().cloned().min().unwrap_or(0);
        let least_used_cells = segments_per_cell.iter().enumerate().filter(|(_,&segment_num)|segment_num==fewest_segments).map(|(i,_)|i as u32).collect::<Vec<u32>>();
        least_used_cells[rand::random::<usize>() % least_used_cells.len()]
    }
    fn create_segment(&mut self, hyp:&HomHyperparams,
                      winner_cell:u32,
                      prev_winner_cells:&Vec<u32>){
        while self.segments.len() >= hyp.max_segments_per_cell{
            let least_used_segment = self.segments.iter().position_min_by_key(|&s|s.last_used_in_iteration).unwrap();
            self.segments.swap_remove(least_used_segment);
        }
        let added_synapse_count = hyp.max_new_synapse_count.min(prev_winner_cells.len());
        if added_synapse_count>0 {
            let mut new_grown_segment = HomSegment{
                synapses: vec![],
                cell_idx: winner_cell as u8,
                last_used_in_iteration: hyp.current_iteration,
                is_active: false,
                num_active_potential: 0
            };
            new_grown_segment.grow_synapses(hyp, added_synapse_count, prev_winner_cells);
            self.segments.push(new_grown_segment);
        }
    }
    fn burst_column(&mut self, learn: bool,
                    hyp:&HomHyperparams,
                    this_minicolumn_idx:u32,
                    active_cells:&mut Vec<u32>,
                    prev_active_cells:&Vec<u32>,
                    winner_cells:&mut Vec<u32>,
                    prev_winner_cells:&Vec<u32>,){
        for cell_idx in 0..hyp.cells_per_minicolumn{
            let cell_id = hyp.encode_cell_id(this_minicolumn_idx,cell_idx);
            active_cells.push(cell_id);
        }
        let learning_segment = self.best_matching_segment(hyp) as usize;
        let winner_cell = if learning_segment == self.segments.len(){
            let winner_cell = self.least_used_cells(hyp);
            if learn{ // Grow new segment.
                // Notice that learning_segment will now be pointing to the newest segment.
                self.create_segment(hyp,winner_cell,prev_winner_cells)
            }
            winner_cell
        }else{
            let winner_cell = self.segments[learning_segment].cell_idx as u32;
            if learn {
                self.adapt_segment(hyp,learning_segment,prev_active_cells,prev_winner_cells)
            }
            winner_cell
        };
        let winner_cell = hyp.encode_cell_id(this_minicolumn_idx,winner_cell);
        winner_cells.push(winner_cell);
    }
    fn punish_predicted_column(&mut self, hyp:&HomHyperparams,
                               prev_active_cells:&Vec<u32>){
        for segment_idx in 0..self.segments.len(){
            if self.segments[segment_idx].is_matching(hyp){
                self.punish_segment(hyp,segment_idx,prev_active_cells);
            }
        }
    }
}
#[derive(Clone)]
pub struct CpuHOM {
    pub hyp: HomHyperparams,
    pub minicolumns: Vec<HomMinicolumn>,
    pub active_cells: Vec<u32>,
    pub winner_cells: Vec<u32>,
}

impl CpuHOM {

    pub fn reset(&mut self){
        self.active_cells.clear();
        self.winner_cells.clear();
        for minicolumn in &mut self.minicolumns{
            for segment in &mut minicolumn.segments{
                segment.is_active = false;
                segment.num_active_potential = 0;
                segment.last_used_in_iteration = 0;
            }
        }
        self.hyp.current_iteration = 0;
    }
    pub fn new(cells_per_minicolumn:u32,
               minicolumn_count:u32)->Self{
        Self{
            hyp: HomHyperparams::new(cells_per_minicolumn, minicolumn_count),
            minicolumns: (0..minicolumn_count).map(|_|HomMinicolumn{ segments: vec![] }).collect(),
            active_cells: vec![],
            winner_cells: vec![]
        }
    }

    fn activate_dendrites(&mut self, learn:bool) -> CpuSDR{
        let Self{ hyp, minicolumns,active_cells,.. } = self;
        let mut predicted_minicolumns = CpuSDR::new();
        for (minicolumn_idx, column) in minicolumns.iter_mut().enumerate() {
            let mut num_active_connected = 0;
            let mut num_active_potential = 0;
            let mut had_active_segment = false;
            for segment in &mut column.segments {
                for synapse in &segment.synapses {
                    let is_active = active_cells.binary_search(&synapse.cell_id).is_ok();
                    if is_active {
                        if synapse.permanence > hyp.permanence_threshold {
                            num_active_connected += 1
                        }
                        if synapse.permanence >= 0. {
                            num_active_connected += 1
                        }
                    }
                }
                segment.num_active_potential = num_active_potential;
                segment.is_active = num_active_connected >= hyp.activation_threshold;
                if segment.is_active{
                    had_active_segment = true;
                    if learn {
                        segment.last_used_in_iteration = hyp.current_iteration;
                    }
                }
            }
            if had_active_segment{
                predicted_minicolumns.push(minicolumn_idx as u32);
            }
        }
        predicted_minicolumns
    }

    /**active_minicolumns should be normalized!*/
    fn activate_cells(&mut self, active_minicolumns:&CpuSDR, learn:bool){
        debug_assert!(active_minicolumns.is_normalized());
        let mut active_cells:Vec<u32> = vec![];
        let mut winner_cells:Vec<u32> = vec![];
        let Self{hyp, minicolumns, active_cells:prev_active_cells,winner_cells:prev_winner_cells,..} =  self;
        for (column_idx, column) in minicolumns.iter_mut().enumerate(){
            let column_idx = column_idx as u32;
            let is_active = active_minicolumns.binary_search(column_idx);
            if is_active{
                if !column.activate_predicted_column(learn,hyp,column_idx,&mut active_cells,prev_active_cells,&mut winner_cells, prev_winner_cells){ // burst column
                    column.burst_column(learn,hyp,column_idx,&mut active_cells,prev_active_cells,&mut winner_cells,prev_winner_cells)
                }
            }else if learn && hyp.predicted_decrement<0. { // punishPredictedColumn
                column.punish_predicted_column(hyp,prev_active_cells);
            }
        }
        active_cells.sort();
        winner_cells.sort();
        std::mem::swap(&mut active_cells,  prev_active_cells);
        std::mem::swap(&mut winner_cells,  prev_winner_cells);
    }

    pub fn infer(&mut self, active_minicolumns:&CpuSDR, learn:bool)->CpuSDR{
        self.activate_cells(active_minicolumns,learn);
        let predicted_minicolumns = self.activate_dendrites(learn);
        self.hyp.current_iteration+=1;
        predicted_minicolumns
    }

}


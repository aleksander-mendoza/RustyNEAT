// use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
// use std::mem::MaybeUninit;
// use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
// use std::fmt::{Display, Formatter, Debug};
// use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
// use crate::cpu_sdr::CpuSDR;
// use crate::htm_program::HtmProgram;
// use ndalgebra::buffer::Buffer;
// use std::cmp::Ordering;
//
// pub struct HomHyperparams{
//     cells_per_minicolumn:u32,
//     max_synapses_per_segment:usize,
//     minicolumn_count:u32,
//     pub max_new_synapse_count_during_bursting:usize,
//     pub initial_permanence: f32,
//     pub min_permanence_to_keep: f32,
//     pub activation_threshold: u32,
//     pub learning_threshold: u32,
//     pub permanence_threshold: f32,
//     pub synapse_sample_size: u32,
//     pub predicted_decrement:f32,
//     pub permanence_decrement_increment: [f32; 2],
// }
//
// impl HomHyperparams{
//     pub fn encode_cell_id(&self,minicolumn_idx:u32, cell_idx:u32)->u32{
//         minicolumn_idx*self.cells_per_minicolumn+cell_idx
//     }
//     pub fn encode_segment_id(&self,minicolumn_idx:u32, segment_idx:u32)->u32{
//         minicolumn_idx+segment_idx*self.minicolumn_count
//     }
//     pub fn decode_cell_id(&self,cell_id:u32)->(u32,u32){
//         let minicolumn_idx = cell_id/self.cells_per_minicolumn;
//         let segment_idx = cell_id%self.cells_per_minicolumn;
//         (minicolumn_idx, segment_idx)
//     }
//     pub fn decode_segment_id(&self,segment_id:u32)->(u32,u32){
//         let minicolumn_idx = segment_id%self.minicolumn_count;
//         let segment_idx = segment_id/self.minicolumn_count;
//         (minicolumn_idx, segment_idx)
//     }
// }
//
// #[derive(Clone, Copy)]
// pub struct HomDistalSynapse {
//     pub cell_id: u32,
//     pub permanence: f32,
// }
// #[derive(Clone)]
// pub struct HomSegment {
//     pub synapses: Vec<HomDistalSynapse>,
//     pub cell_idx:u8, // cell index within the minicolumn
//     pub is_active:bool,
//     pub num_active_potential: u32,
// }
//
// impl HomSegment{
//     fn is_matching(&self, hyp:&HomHyperparams)->bool{
//         self.num_active_potential >= hyp.learning_threshold
//     }
//     fn update_segment(&mut self, hyp:&HomHyperparams, permanence_decrement_increment:[f32;2], prev_active_cells:&Vec<u32>){
//         let mut writing_idx = 0;
//         for synapse_idx in 0..self.synapses.len(){
//             let mut synapse = self.synapses[synapse_idx];
//             let is_active = prev_active_cells.binary_search(&synapse.cell_id).is_ok();
//             let permanence_change = permanence_decrement_increment[is_active as usize];
//             synapse.permanence = if is_active{
//                 (synapse.permanence + permanence_change).min(1.)
//             }else{
//                 let new_perm = synapse.permanence + permanence_change;
//                 if new_perm < hyp.min_permanence_to_keep{
//                     continue;
//                 }
//                 new_perm
//             };
//             self.synapses[writing_idx] = synapse;
//         }
//         unsafe{self.synapses.set_len(writing_idx)}
//     }
//
//     fn destroy_min_permanence_synapses(&mut self, new_synapse_count:usize, prev_winner_cells:&Vec<u32>){
//         if self.synapses.len() <= new_synapse_count{return}
//         self.synapses.sort_by(|a,b|b.permanence.partial_cmp(&a.permanence).unwrap_or(Ordering::Equal));
//         for i in (0..self.synapses.len()).rev(){
//             if prev_winner_cells.binary_search(&self.synapses[i].cell_id).is_err(){
//                 let last_idx = self.synapses.len()-1;
//                 self.synapses[i] = self.synapses[last_idx];
//                 unsafe{self.synapses.set_len(last_idx)}
//                 if self.synapses.len() <= new_synapse_count{return}
//             }
//         }
//     }
//     fn grow_synapses(&mut self, hyp:&HomHyperparams, added_synapse_count:usize, prev_winner_cells:&Vec<u32>){
//         let mut candidates = prev_winner_cells.clone();
//         debug_assert!(prev_winner_cells.windows(2).all(|a|a[0]<a[1]));
//         for synapse in self.synapses{
//             if let Ok(already_connected_candidate_idx) = candidates.binary_search(&synapse.cell_id){
//                 candidates.remove(already_connected_candidate_idx)
//             }
//         }
//         let added_synapse_count = added_synapse_count.min(candidates.len());
//         self.destroy_min_permanence_synapses(hyp.max_synapses_per_segment-added_synapse_count,prev_winner_cells);
//         for _ in 0..added_synapse_count{
//             let winner_cell = candidates.swap_remove(rand::random::<usize>() % candidates.len());
//             self.synapses.push(HomDistalSynapse{ cell_id: winner_cell, permanence: hyp.initial_permanence })
//         }
//     }
// }
//
// #[derive(Clone)]
// pub struct HomMinicolumn {
//     pub segments: Vec<HomSegment>,
// }
//
// impl HomMinicolumn{
//     fn punish_segment(&mut self, hyp:&HomHyperparams, segment_idx:usize, prev_active_cells:&Vec<u32>){
//         self.segments[segment_idx].update_segment(hyp,[hyp.predicted_decrement,0.], prev_active_cells);
//         if self.segments[segment_idx].is_empty(){
//             self.segments.swap_remove(segment_idx);
//         }
//     }
//     fn adapt_segment(&mut self, hyp:&HomHyperparams, segment_idx:usize, prev_active_cells:&Vec<u32>, prev_winner_cells:&Vec<u32>){
//         self.segments[segment_idx].update_segment(hyp,hyp.permanence_decrement_increment, prev_active_cells);
//         let num_active_potential = self.segments[segment_idx].num_active_potential;
//         if hyp.synapse_sample_size > num_active_potential{
//             let added_synapse_count = (hyp.synapse_sample_size - num_active_potential) as usize;
//             self.segments[segment_idx].grow_synapses(hyp,added_synapse_count,prev_winner_cells);
//         }
//         if self.segments[segment_idx].is_empty(){
//             self.segments.swap_remove(segment_idx);
//         }
//     }
//     fn activate_predicted_column(&mut self, learn:bool, hyp:&HomHyperparams,
//                                  this_minicolumn_idx:u32,
//                                  active_cells:&mut Vec<u32>,
//                                  prev_active_cells:&Vec<u32>,
//                                  winner_cells:&mut Vec<u32>,
//                                  prev_winner_cells:&Vec<u32>,){
//         let mut has_cell_been_added = vec![false;hyp.cells_per_minicolumn as usize];
//         for segment_idx in 0..self.segments.len(){
//             if self.segments[segment_idx].is_active {
//                 let cell_idx= self.segments[segment_idx].cell_idx ;
//                 if !has_cell_been_added[cell_idx as usize]{
//                     has_cell_been_added[cell_idx as usize] = true;
//                     let cell_id = hyp.encode_cell_id(this_minicolumn_idx,cell_idx as u32);
//                     active_cells.push(cell_id);
//                     winner_cells.push(cell_id);
//                 }
//                 if learn {
//                     self.adapt_segment(hyp,segment_idx,prev_active_cells,prev_winner_cells)
//                 }
//             }
//         }
//     }
//     fn best_matching_segment(&self, hyp:&HomHyperparams)->u32{
//         let mut num_active_potential = hyp.learning_threshold as i32 - 1;
//         let mut segment_idx = self.segments.len();
//         for (i,segment) in self.segments.iter().enumerate(){
//             if segment.num_active_potential as i32 > num_active_potential{
//                 num_active_potential = segment.num_active_potential as i32;
//                 segment_idx = i ;
//             }
//         }
//         segment_idx as u32
//     }
//     fn least_used_cells(&self,hyp:&HomHyperparams)->u32{
//         let mut segments_per_cell = vec![0;hyp.cells_per_minicolumn as usize];
//         for segment in &self.segments{
//             segments_per_cell[segment.cell_idx as usize]+=1;
//         }
//         let fewest_segments = segments_per_cell.iter().cloned().min().unwrap_or(0);
//         let least_used_cells = segments_per_cell.iter().enumerate().filter(|(_,segment_num)|segment_num==fewest_segments).map(|(i,_)|i as u32).collect::<Vec<u32>>();
//         least_used_cells[rand::random::<usize>() % least_used_cells.len()]
//     }
//     fn create_segment(&mut self, hyp:&HomHyperparams,
//                       winner_cell:u32,
//                       prev_winner_cells:&Vec<u32>){
//         let added_synapse_count = hyp.max_new_synapse_count_during_bursting.min(prev_winner_cells.len());
//         if added_synapse_count>0 {
//             let mut new_grown_segment = HomSegment{
//                 synapses: vec![],
//                 cell_idx: winner_cell as u8,
//                 is_active: false,
//                 num_active_potential: 0
//             };
//             new_grown_segment.grow_synapses(hyp, added_synapse_count, prev_winner_cells);
//             self.segments.push(new_grown_segment);
//         }
//     }
//     fn burst_column(&mut self, learn: bool,
//                     hyp:&HomHyperparams,
//                     this_minicolumn_idx:u32,
//                     active_cells:&mut Vec<u32>,
//                     prev_active_cells:&Vec<u32>,
//                     winner_cells:&mut Vec<u32>,
//                     prev_winner_cells:&Vec<u32>,){
//         for cell_idx in 0..hyp.cells_per_minicolumn{
//             let cell_id = hyp.encode_cell_id(this_minicolumn_idx,cell_idx);
//             active_cells.push(cell_id);
//         }
//         let learning_segment = self.best_matching_segment(hyp) as usize;
//         let winner_cell = if learning_segment == self.segments.len(){
//             let winner_cell = self.least_used_cells(hyp);
//             if learn{ // Grow new segment.
//                 // Notice that learning_segment will now be pointing to the newest segment.
//                 self.create_segment(hyp,winner_cell,prev_winner_cells)
//             }
//             winner_cell
//         }else{
//             let winner_cell = self.segments[learning_segment].cell_idx;
//             if learn {
//                 self.adapt_segment(hyp,learning_segment,prev_active_cells,prev_winner_cells)
//             }
//             winner_cell
//         };
//         let winner_cell = hyp.encode_cell_id(this_minicolumn_idx,winner_cell);
//         winner_cells.push(winner_cell);
//     }
//     fn punish_predicted_column(&mut self, hyp:&HomHyperparams,
//                                prev_active_cells:&Vec<u32>){
//         for segment_idx in 0..self.segments.len(){
//             if self.segments[segment_idx].is_matching(hyp){
//                 self.punish_segment(hyp,segment_idx,prev_active_cells);
//             }
//         }
//     }
// }
// #[derive(Clone)]
// pub struct CpuHOM {
//     minicolumns: Vec<HomMinicolumn>,
//     active_cells: Vec<u32>,
//     winner_cells: Vec<u32>,
// }
//
// impl CpuHOM {
//
//     fn activate_dendrites(&mut self, learn:bool){
//         for column in self.minicolumns.iter_mut() {
//             let mut num_active_connected = 0;
//             let mut num_active_potential = 0;
//             for segment in &mut column.segments {
//                 for synapse in &segment.synapses {
//                     let is_active = self.active_cells.binary_search(&synapse.cell_id).is_ok();
//                     if is_active {
//                         if synapse.permanence > self.permanence_threshold {
//                             num_active_connected += 1
//                         }
//                         if synapse.permanence >= 0. {
//                             num_active_connected += 1
//                         }
//                     }
//                 }
//                 segment.num_active_potential = num_active_potential;
//                 segment.is_active = num_active_connected >= self.activation_threshold;
//             }
//         }
//     }
//
//     /**active_minicolumns should be normalized!*/
//     fn activate_cells(&mut self, active_minicolumns:&CpuSDR, learn:bool){
//         debug_assert!(active_minicolumns.is_normalized());
//         let mut active_cells:Vec<u32> = vec![];
//         let mut winner_cells:Vec<u32> = vec![];
//         let learning_threshold = self.learning_threshold;
//         let predicted_decrement = self.predicted_decrement;
//         let Self{minicolumns, active_cells:prev_active_cells,winner_cells:prev_winner_cells,..} =  self;
//         for (column_idx, column) in minicolumns.iter_mut().enumerate(){
//             let is_active = active_minicolumns.binary_search(column_idx as u32);
//             if is_active{
//                 if column.segments.find(){ // burst column
//                     column.burst_column(learn,&mut active_cells,&mut winner_cells)
//                 } else { // activatePredictedColumn
//                     column.activate_predicted_column()
//                 }
//             }else if learn && predicted_decrement<0. && column.contains_matching_segment(self.learning_threshold){ // punishPredictedColumn
//                 column.punish_predicted_column(learning_threshold,prev_active_cells, predicted_decrement);
//             }
//         }
//         std::mem::swap(&mut active_cells,  prev_active_cells);
//         std::mem::swap(&mut winner_cells,  prev_winner_cells);
//     }
//
//
//     pub fn compute(&mut self, active_minicolumns:&CpuSDR, learn:bool){
//         self.activate_cells(active_minicolumns,learn);
//         self.activate_dendrites(learn)
//     }
//
// }
//

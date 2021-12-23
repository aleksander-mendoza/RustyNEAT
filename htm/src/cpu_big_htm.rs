// use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
// use std::mem::MaybeUninit;
// use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
// use std::fmt::{Display, Formatter, Debug};
// use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
// use crate::cpu_sdr::CpuSDR;
// use crate::htm_program::HtmProgram;
// use ndalgebra::buffer::Buffer;
// use std::cmp::Ordering;
// use itertools::Itertools;
// use crate::{EncoderTarget, CpuBitset, CpuInput};
// use std::collections::HashMap;
// use serde::{Serialize, Deserialize};
//
// #[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
// pub struct BigHtmHyperparams {
//     pub n: usize,
//     pub max_synapses_per_segment: usize,
//     pub max_segments_per_minicolumn: u32,
//     pub max_new_synapse_count: usize,
//     pub initial_permanence: f32,
//     pub permanence_threshold: f32,
//     /**Connection is removed from the segment if it falls below this threshold*/
//     pub removal_permanence_threshold: f32,
//     pub permanence_decrement_increment: [f32; 2],
//     pub rand_seed: u32,
//     /**The segment is activated if it is connected to at least this many synapses to active inputs*/
//     pub activation_threshold: u32,
// }
//
// impl BigHtmHyperparams {
//     fn new(n:usize,rand_seed: u32) -> Self {
//         Self {
//             rand_seed,
//             max_synapses_per_segment: 255,
//             max_new_synapse_count: 20,
//             initial_permanence: 0.5,
//             permanence_threshold: 0.5,
//             removal_permanence_threshold: 0.2,
//             permanence_decrement_increment: [-0.05, 0.10],
//             n,
//             max_segments_per_minicolumn: 8,
//             activation_threshold: 1,
//         }
//     }
// }
//
// #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
// pub struct BigHtmSynapse {
//     pub input_id: u32,
//     pub permanence: f32,
// }
//
// #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
// pub struct BigHtmSegment {
//     pub synapses: Vec<BigHtmSynapse>,
// }
//
// impl BigHtmSegment{
//     fn overlap(&self,active_inputs:&CpuInput,hyperparams:&BigHtmHyperparams)->u32{
//         self.synapses.iter().filter(|s| s.permanence >= hyperparams.permanence_threshold && active_inputs.contains(s.input_id)).count() as u32
//     }
// }
//
// #[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
// pub struct BigHtmMinicolumn {
//     pub segments: Vec<BigHtmSegment>,
// }
//
//
// #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
// pub struct CpuBigHTM {
//     hyperparams: BigHtmHyperparams,
//     is_connected: Vec<bool>,
//     minicolumns: Vec<BigHtmMinicolumn>,
//
// }
//
// impl CpuBigHTM {
//     pub fn input_size(&self) -> u32 {
//         self.is_connected.len() as u32
//     }
//     pub fn params(&self) -> &BigHtmHyperparams {
//         &self.hyperparams
//     }
//     pub fn params_mut(&mut self) -> &mut BigHtmHyperparams {
//         &mut self.hyperparams
//     }
//     pub fn minicolumns_as_slice(&self) -> &[BigHtmMinicolumn] {
//         self.minicolumns.as_slice()
//     }
//
//     pub fn new(input_size: u32, minicolumn_count: u32, n:usize, rand_seed: u32) -> Self {
//         Self {
//             hyperparams: BigHtmHyperparams::new(n,rand_seed),
//             is_connected: vec![false;input_size as usize],
//             minicolumns: (0..minicolumn_count).map(|_| BigHtmMinicolumn { segments: vec![] }).collect(),
//
//         }
//     }
//
//     fn learn_segment(segment: &mut BigHtmSegment,
//                      hyperparams: &mut BigHtmHyperparams,
//                      is_connected: &mut Vec<bool>,
//                      active_inputs: &CpuInput) -> bool {
//         let mut synapse_idx = 0;
//         let mut already_connected_active_inputs = 0;
//         let mut connected_and_above_threshold = 0;
//         while synapse_idx < segment.synapses.len() {
//             let mut synapse = &mut segment.synapses[synapse_idx as usize];
//             let is_input_active = active_inputs.contains(synapse.input_id);
//             synapse.permanence = (synapse.permanence + hyperparams.permanence_decrement_increment[is_input_active as usize]).min(1.);
//             if synapse.permanence <= hyperparams.removal_permanence_threshold {
//                 segment.synapses.swap_remove(synapse_idx as usize);
//             } else {
//                 already_connected_active_inputs += is_input_active as u32;
//                 is_connected[synapse.input_id as usize] = true;
//                 synapse_idx += 1;
//                 if synapse.permanence >= hyperparams.permanence_threshold{
//                     connected_and_above_threshold +=1
//                 }
//             }
//         }
//         let card = active_inputs.cardinality();
//         let remaining_unconnected_inputs = card  - already_connected_active_inputs;
//         let synapses_to_grow = (hyperparams.max_synapses_per_segment - segment.synapses.len()).min(hyperparams.max_new_synapse_count).min(remaining_unconnected_inputs as usize);
//         let mut rand_seed = hyperparams.rand_seed;
//         for _ in 0..synapses_to_grow {
//             rand_seed = xorshift32(rand_seed);
//             let active_input = active_inputs.get_sparse().as_slice()[(rand_seed % card) as usize];
//             let is_connected = &mut is_connected[active_input as usize];
//             if !*is_connected {
//                 *is_connected = true;
//                 segment.synapses.push(BigHtmSynapse {
//                     input_id: active_input,
//                     permanence: hyperparams.initial_permanence,
//                 });
//                 if hyperparams.initial_permanence >= hyperparams.permanence_threshold{
//                     connected_and_above_threshold += 1
//                 }
//             }
//         }
//         hyperparams.rand_seed = rand_seed;
//         segment.synapses.iter().for_each(|syn| is_connected[syn.input_id as usize] = false);
//         debug_assert_eq!(connected_and_above_threshold as usize,segment.synapses.iter().filter(|s|s.permanence>=hyperparams.permanence_threshold).count());
//         connected_and_above_threshold < hyperparams.activation_threshold
//     }
//     /**If there is not enough input activity to determine n winner cells, then return only those we could find*/
//     pub fn infer_less_than_n(&mut self, active_inputs: &CpuInput, learn: bool) -> CpuSDR{
//         self.infer_and_postprocess(active_inputs,learn,|_,_,_,_,_,rand_seed|rand_seed)
//     }
//     /**If there is not enough input activity to determine n winner cells, then the missing cells will be chosen at random from the provided sticky activity SDR. It is necessary that sticky_activity contains no duplicates*/
//     pub fn infer_sticky(&mut self, active_inputs: &CpuInput, learn: bool,sticky_activity:&[u32]) -> CpuSDR{
//         assert!(sticky_activity.len() >= self.params().n,"This HTM needs to select n=={} winner cells but the provided sticky activity has only {} cells",self.params().n,sticky_activity.len());
//         self.infer_and_postprocess(active_inputs,learn,|predicted_minicolumns,overlap_per_minicolumn,_,hyperparams,min_overlap,mut rand_seed|{
//             while predicted_minicolumns.len() < hyperparams.n{
//                 let mut rand_idx = rand_seed as usize % sticky_activity.len();
//                 while overlap_per_minicolumn[sticky_activity[rand_idx] as usize] as usize > min_overlap{
//                     rand_seed = xorshift32(rand_seed);
//                     rand_idx = rand_seed as usize % sticky_activity.len();
//                 }
//                 predicted_minicolumns.push(sticky_activity[rand_idx]);
//                 overlap_per_minicolumn[sticky_activity[rand_idx] as usize] = u32::MAX;
//             }
//             rand_seed
//         })
//     }
//     /**If there is not enough input activity to determine n winner cells, then the missing cells will be chosen at random from the provided SDR. It is necessary that whitelist contains no duplicates. If
//      there are any winner cells which are not on the whitelist, then they will be removed and replaced with some cells from the list.*/
//     pub fn infer_from_whitelist(&mut self, active_inputs: &CpuInput, learn: bool,whitelist:&CpuInput) -> CpuSDR{
//         assert!(whitelist.cardinality() as usize >= self.params().n,"This HTM needs to select n=={} winner cells but the provided whitelist has only {} cells",self.params().n,whitelist.cardinality());
//         self.infer_and_postprocess(active_inputs,learn,|predicted_minicolumns,overlap_per_minicolumn,_,hyperparams,min_overlap,mut rand_seed|{
//             predicted_minicolumns.retain(|x|whitelist.contains(x));
//             while predicted_minicolumns.len() < hyperparams.n{
//                 let mut rand_idx = (rand_seed % whitelist.cardinality())as usize;
//                 while overlap_per_minicolumn[whitelist.get_sparse()[rand_idx] as usize] as usize > min_overlap{
//                     rand_seed = xorshift32(rand_seed);
//                     rand_idx = (rand_seed  % whitelist.cardinality())as usize;
//                 }
//                 predicted_minicolumns.push(whitelist.get_sparse()[rand_idx]);
//                 overlap_per_minicolumn[whitelist.get_sparse()[rand_idx] as usize] = u32::MAX;
//             }
//             rand_seed
//         })
//     }
//     /**If there is not enough input activity to determine n winner cells, then the missing cells will be chosen at random.*/
//     pub fn infer(&mut self, active_inputs: &CpuInput, learn: bool) -> CpuSDR{
//         self.infer_and_postprocess(active_inputs,learn, |predicted_minicolumns,overlap_per_minicolumn,minicolumns,hyperparams,min_overlap,mut rand_seed|{
//             while predicted_minicolumns.len() < hyperparams.n{
//                 let mut rand_idx = rand_seed % minicolumns.len() as u32;
//                 while overlap_per_minicolumn[rand_idx as usize] as usize > min_overlap{
//                     rand_seed = xorshift32(rand_seed);
//                     rand_idx = rand_seed % minicolumns.len() as u32;
//                 }
//                 predicted_minicolumns.push(rand_idx);
//                 overlap_per_minicolumn[rand_idx as usize] = u32::MAX;
//             }
//             rand_seed
//         })
//     }
//     fn infer_and_postprocess(&mut self, active_inputs: &CpuInput, learn: bool, mut activity_postprocessing:impl FnMut(&mut CpuSDR,&mut [u32],&[BigHtmMinicolumn],&BigHtmHyperparams, usize, u32)->u32) -> CpuSDR {
//         let Self { minicolumns, is_connected, hyperparams, .. } = self;
//         assert!(hyperparams.removal_permanence_threshold <= hyperparams.permanence_threshold,"removal_permanence_threshold is higher than permanence_threshold");
//         assert!(0. <= hyperparams.removal_permanence_threshold ,"removal_permanence_threshold is negative");
//         assert!(hyperparams.permanence_threshold < 1. ,"permanence_threshold must be less than 1");
//         assert!(0 < hyperparams.activation_threshold, "activation_threshold can't be 0");
//         let mut predicted_minicolumns = CpuSDR::with_capacity(hyperparams.n);
//         let mut overlap_per_minicolumn = Vec::with_capacity(minicolumns.len());
//         let mut best_segment_per_minicolumn = Vec::with_capacity(minicolumns.len());
//         let mut count_per_overlap = vec![0usize;hyperparams.max_synapses_per_segment+1];
//         for column in minicolumns.iter() {
//             let mut max_overlap = 0u32;
//             let mut max_segment = u16::MAX;
//             for (i,segment) in column.segments.iter().enumerate(){
//                 let overlap = segment.overlap(active_inputs, hyperparams);
//                 if overlap > max_overlap{
//                     max_overlap = overlap;
//                     max_segment = i as u16;
//                 }
//             }
//             overlap_per_minicolumn.push(max_overlap);
//             best_segment_per_minicolumn.push(max_segment);
//             if max_overlap >= hyperparams.activation_threshold {
//                 count_per_overlap[max_overlap as usize]+=1;
//             }
//         }
//         let mut remaining = hyperparams.n;
//         let mut min_overlap = hyperparams.max_synapses_per_segment;
//         while min_overlap > 0{
//             let count = count_per_overlap[min_overlap];
//             if count < remaining{
//                 remaining -= count;
//                 min_overlap -= 1;
//             }else{
//                 count_per_overlap[min_overlap] = remaining;
//                 min_overlap -= 1;
//                 break;
//             }
//         }
//         for (minicolumn_idx, (&best_overlap, &best_segment)) in overlap_per_minicolumn.iter().zip(best_segment_per_minicolumn.iter()).enumerate(){
//             let best_overlap = best_overlap as usize;
//             if best_overlap > min_overlap{
//                 let count = &mut count_per_overlap[best_overlap];
//                 if *count > 0 {
//                     *count -= 1;
//                     predicted_minicolumns.push(minicolumn_idx as u32);
//                 }
//             }
//         }
//         hyperparams.rand_seed = activity_postprocessing(&mut predicted_minicolumns,&mut overlap_per_minicolumn,minicolumns,hyperparams,min_overlap,hyperparams.rand_seed);
//
//         if learn{
//             for &predicted_minicolumn in predicted_minicolumns.iter() {
//                 //We already have the best-matching segment. No need to recompute it
//                 let mut best_segment = best_segment_per_minicolumn[predicted_minicolumn as usize] as usize;
//                 let segments = &mut minicolumns[predicted_minicolumn as usize].segments;
//                 if best_segment == u16::MAX as usize{ // no segment, let's create a new one
//                     if segments.len() < hyperparams.max_segments_per_minicolumn as usize {
//                         best_segment = segments.len();
//                         segments.push(BigHtmSegment { synapses: vec![] });
//                     } else {
//                         continue;
//                     }
//                 }
//                 if Self::learn_segment(&mut segments[best_segment], hyperparams, is_connected, active_inputs) {
//                     segments.swap_remove(best_segment);
//                 }
//             }
//         }
//         predicted_minicolumns
//     }
//     /**Notice that self.infer(active_inputs, true) is equivalent to but much faster than self.update_permanence(active_inputs, self.infer(active_inputs, false))*/
//     pub fn update_permanence(&mut self, active_inputs: &CpuInput, active_minicolumns: &CpuSDR) {
//         let Self { minicolumns, is_connected, hyperparams, .. } = self;
//         for &active_minicolumn in active_minicolumns.as_slice() {
//             let mut max_overlap = 0;
//             let mut max_position = 9999999; // we need to find the best-matching segment
//             let segments = &mut minicolumns[active_minicolumn as usize].segments;
//             for (pos, seg) in segments.iter().enumerate() {
//                 let overlap = seg.overlap(active_inputs, hyperparams);
//                 if overlap > max_overlap {
//                     max_position = pos;
//                     max_overlap = overlap;
//                 }
//             }
//             if max_overlap < hyperparams.activation_threshold || max_position >= segments.len(){
//                 if segments.len() < hyperparams.max_segments_per_minicolumn as usize {
//                     max_position = segments.len();
//                     segments.push(BigHtmSegment { synapses: vec![] });
//                 } else {
//                     continue;
//                 }
//             }
//             if Self::learn_segment(&mut segments[max_position], hyperparams, is_connected, active_inputs) {
//                 segments.swap_remove(max_position);
//             }
//         }
//     }
// }
//

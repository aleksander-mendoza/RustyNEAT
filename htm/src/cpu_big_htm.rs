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
use crate::{EncoderTarget, CpuBitset, CpuInput};
use crate::rand::xorshift32;
use std::collections::HashMap;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BigHtmHyperparams {
    pub n: usize,
    pub max_synapses_per_segment: usize,
    pub max_segments_per_minicolumn: u32,
    pub max_new_synapse_count: usize,
    pub initial_permanence: f32,
    pub permanence_threshold: f32,
    pub permanence_decrement_increment: [f32; 2],
    pub rand_seed: u32,
    pub activation_threshold: u32,
}

impl BigHtmHyperparams {
    fn new(n:usize,rand_seed: u32) -> Self {
        Self {
            rand_seed,
            max_synapses_per_segment: 255,
            max_new_synapse_count: 20,
            initial_permanence: 0.5,
            permanence_threshold: 0.5,
            permanence_decrement_increment: [-0.05, 0.10],
            n,
            max_segments_per_minicolumn: 8,
            activation_threshold: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BigHtmSynapse {
    pub input_id: u32,
    pub permanence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BigHtmSegment {
    pub synapses: Vec<BigHtmSynapse>,
}

impl BigHtmSegment{
    fn overlap(&self,active_inputs:&CpuInput)->u32{
        self.synapses.iter().filter(|s| active_inputs.contains(s.input_id)).count() as u32
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BigHtmMinicolumn {
    pub segments: Vec<BigHtmSegment>,
}


#[derive(Debug, Clone, PartialEq)]
pub struct CpuBigHTM {
    hyperparams: BigHtmHyperparams,
    is_connected: Vec<bool>,
    minicolumns: Vec<BigHtmMinicolumn>,

}

impl CpuBigHTM {
    pub fn input_size(&self) -> u32 {
        self.is_connected.len() as u32
    }
    pub fn params(&self) -> &BigHtmHyperparams {
        &self.hyperparams
    }
    pub fn params_mut(&mut self) -> &mut BigHtmHyperparams {
        &mut self.hyperparams
    }
    pub fn minicolumns_as_slice(&self) -> &[BigHtmMinicolumn] {
        self.minicolumns.as_slice()
    }

    pub fn new(input_size: u32, minicolumn_count: u32, n:usize, rand_seed: u32) -> Self {
        Self {
            hyperparams: BigHtmHyperparams::new(n,rand_seed),
            is_connected: vec![false;input_size as usize],
            minicolumns: (0..minicolumn_count).map(|_| BigHtmMinicolumn { segments: vec![] }).collect(),

        }
    }

    fn learn_segment(segment: &mut BigHtmSegment,
                     hyperparams: &mut BigHtmHyperparams,
                     is_connected: &mut Vec<bool>,
                     active_inputs: &CpuInput) -> bool {
        let mut synapse_idx = 0;
        let mut already_connected = 0;
        while synapse_idx < segment.synapses.len() {
            let mut synapse = &mut segment.synapses[synapse_idx as usize];
            let is_input_active = active_inputs.contains(synapse.input_id);
            synapse.permanence = (synapse.permanence + hyperparams.permanence_decrement_increment[is_input_active as usize]).clamp(0., 1.);
            if synapse.permanence < hyperparams.permanence_threshold {
                segment.synapses.swap_remove(synapse_idx as usize);
            } else {
                already_connected += 1;
                is_connected[synapse.input_id as usize] = true;
                synapse_idx += 1;
            }
        }
        let card = active_inputs.cardinality();
        let remaining_unconnected_inputs = card as usize - already_connected;
        let synapses_to_grow = (hyperparams.max_synapses_per_segment - segment.synapses.len()).min(hyperparams.max_new_synapse_count).min(remaining_unconnected_inputs);
        let mut rand_seed = hyperparams.rand_seed;
        for _ in 0..synapses_to_grow {
            rand_seed = xorshift32(rand_seed);
            let active_input = active_inputs.get_sparse().as_slice()[(rand_seed % card) as usize];
            let is_connected = &mut is_connected[active_input as usize];
            if !*is_connected {
                *is_connected = true;
                segment.synapses.push(BigHtmSynapse {
                    input_id: active_input,
                    permanence: hyperparams.initial_permanence,
                });
            }
        }
        hyperparams.rand_seed = rand_seed;
        segment.synapses.iter().for_each(|syn| is_connected[syn.input_id as usize] = false);
        (segment.synapses.len() as u32) < hyperparams.activation_threshold
    }
    pub fn infer(&mut self, active_inputs: &CpuInput, learn: bool) -> CpuSDR {
        let Self { minicolumns, is_connected, hyperparams, .. } = self;
        let mut predicted_minicolumns = CpuSDR::with_capacity(hyperparams.n);
        let mut overlap_per_minicolumn = Vec::with_capacity(minicolumns.len());
        let mut best_segment_per_minicolumn = Vec::with_capacity(minicolumns.len());
        let mut count_per_overlap = vec![0usize;hyperparams.max_synapses_per_segment+1];
        for column in minicolumns.iter() {
            let mut max_overlap = 0u32;
            let mut max_segment = u16::MAX;
            for (i,segment) in column.segments.iter().enumerate(){
                let overlap = segment.overlap(active_inputs);
                if overlap > max_overlap{
                    max_overlap = overlap;
                    max_segment = i as u16;
                }
            }
            overlap_per_minicolumn.push(max_overlap);
            best_segment_per_minicolumn.push(max_segment);
            if max_overlap >= hyperparams.activation_threshold {
                count_per_overlap[max_overlap as usize]+=1;
            }
        }
        let mut remaining = hyperparams.n;
        let mut min_overlap = hyperparams.max_synapses_per_segment;
        while min_overlap > 0{
            let count = count_per_overlap[min_overlap];
            if count < remaining{
                remaining -= count;
                min_overlap -= 1;
            }else{
                count_per_overlap[min_overlap] = remaining;
                min_overlap -= 1;
                break;
            }
        }
        for (minicolumn_idx, (&best_overlap, &best_segment)) in overlap_per_minicolumn.iter().zip(best_segment_per_minicolumn.iter()).enumerate(){
            let best_overlap = best_overlap as usize;
            if best_overlap > min_overlap{
                let count = &mut count_per_overlap[best_overlap];
                if *count > 0 {
                    *count -= 1;
                    predicted_minicolumns.push(minicolumn_idx as u32);
                }
            }
        }
        let mut rand_seed = hyperparams.rand_seed;
        while predicted_minicolumns.len() < hyperparams.n{
            let mut rand_idx = rand_seed % minicolumns.len() as u32;
            while overlap_per_minicolumn[rand_idx as usize] as usize > min_overlap{
                rand_seed = xorshift32(rand_seed);
                rand_idx = rand_seed % minicolumns.len() as u32;
            }
            predicted_minicolumns.push(rand_idx);
            overlap_per_minicolumn[rand_idx as usize] = u32::MAX;
        }
        hyperparams.rand_seed = rand_seed;
        if learn{
            for (&predicted_minicolumn,&best_segment) in predicted_minicolumns.iter().zip(best_segment_per_minicolumn.iter()) {
                let mut best_segment = best_segment as usize;
                let segments = &mut minicolumns[predicted_minicolumn as usize].segments;
                if best_segment == u16::MAX as usize{
                    if segments.len() < hyperparams.max_segments_per_minicolumn as usize {
                        best_segment = segments.len();
                        segments.push(BigHtmSegment { synapses: vec![] });
                    } else {
                        continue;
                    }
                }
                if Self::learn_segment(&mut segments[best_segment], hyperparams, is_connected, active_inputs) {
                    segments.swap_remove(best_segment);
                }
            }
        }
        predicted_minicolumns
    }
    pub fn update_permanence(&mut self, active_inputs: &CpuInput, active_minicolumns: &CpuSDR) {
        let Self { minicolumns, is_connected, hyperparams, .. } = self;
        for &active_minicolumn in active_minicolumns.as_slice() {
            let mut max_overlap = 0;
            let mut max_position = 9999999;
            let segments = &mut minicolumns[active_minicolumn as usize].segments;
            for (pos, seg) in segments.iter().enumerate() {
                let overlap = seg.overlap(active_inputs);
                if overlap > max_overlap {
                    max_position = pos;
                    max_overlap = overlap;
                }
            }
            if max_overlap < hyperparams.activation_threshold || max_position >= segments.len(){
                if segments.len() < hyperparams.max_segments_per_minicolumn as usize {
                    max_position = segments.len();
                    segments.push(BigHtmSegment { synapses: vec![] });
                } else {
                    continue;
                }
            }
            if Self::learn_segment(&mut segments[max_position], hyperparams, is_connected, active_inputs) {
                segments.swap_remove(max_position);
            }
        }
    }
}


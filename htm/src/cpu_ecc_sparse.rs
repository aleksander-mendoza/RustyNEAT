use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, RangeBounds, Deref, DerefMut};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::cpu_sdr::CpuSDR;
use crate::ecc_program::EccProgram;
use ndalgebra::buffer::Buffer;
use crate::cpu_bitset::CpuBitset;
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use crate::{Shape, resolve_range, EncoderTarget, Synapse, top_large_k_indices, top_small_k_indices, Shape3, from_xyz, Shape2, from_xy, range_contains, SparseOrDense, EccMachine, OclEccSparse, OclEccDense, top_small_k_by_channel, DenseWeight, CpuEccDense, EccSepConMachine, EccLayerD};
use std::collections::{Bound, HashSet};
use crate::vector_field::{VectorFieldOne, VectorFieldDiv, VectorFieldAdd, VectorFieldMul, ArrayCast, VectorFieldSub, VectorFieldPartialOrd};
use crate::population::Population;
use rand::{Rng, SeedableRng};
use crate::xorshift::{auto_gen_seed64, xorshift64, auto_gen_seed, xorshift, xorshift32, auto_gen_seed32};
use itertools::{Itertools, assert_equal};
use std::iter::Sum;
use ocl::core::DeviceInfo::MaxConstantArgs;
use crate::ecc::{EccLayer, as_usize, Idx, as_idx, Rand, xorshift_rand};
use crate::sdr::SDR;

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct CpuEccSparse {
    /**connections[input_idx]==vector_of_output_indices*/
    connections: Vec<Vec<Idx>>,
    max_incoming_synapses: Idx,
    input_shape: [Idx; 3],
    //[height, width, channels]
    output_shape: [Idx; 3],
    //[height, width, channels]
    kernel: [Idx; 2],
    stride: [Idx; 2],
    k: Idx,
    pub threshold: u16,
    pub sums: Vec<u16>,
}

impl CpuEccSparse {
    pub fn into_machine<D:DenseWeight>(self) -> CpuEccMachine<D> {
        CpuEccMachine::new_singleton(SparseOrDense::Sparse(self))
    }
    pub fn to_ocl(&self, prog: EccProgram) -> Result<OclEccSparse, Error> {
        OclEccSparse::new(self, prog)
    }

    pub fn connections(&self) -> &Vec<Vec<Idx>> {
        &self.connections
    }

    pub fn new(output: [Idx; 2], kernel: [Idx; 2], stride: [Idx; 2], in_channels: Idx, out_channels: Idx, k: Idx, connections_per_output: Idx, rng: &mut impl Rng) -> Self {
        let input = output.conv_in_size(&stride, &kernel);
        let output = [output[0], output[1], out_channels];
        let input = [input[0], input[1], in_channels];
        let in_size = input.product();
        let out_size = output.product();
        let mut pop = Population::new(as_usize(out_size), 1);
        pop.add_2d_column_grid_with_3d_input(0..as_usize(in_size),
                                             as_usize(out_channels),
                                             as_usize(connections_per_output),
                                             stride.map(as_usize),
                                             kernel.map(as_usize),
                                             input.map(as_usize),
                                             rng);
        let slf = Self::new_from_pop(k, input, output, kernel, stride, &pop);
        debug_assert_eq!(slf.max_incoming_synapses, connections_per_output);
        slf
    }


    fn new_from_pop(k: Idx, input_shape: [Idx; 3], output_shape: [Idx; 3], kernel: [Idx; 2], stride: [Idx; 2], population: &Population) -> Self {
        let mut connections: Vec<Vec<Idx>> = (0..input_shape.product()).map(|_| vec![]).collect();
        let mut max_incoming_synapses = as_idx(population.neurons.iter().map(|n| n.total_synapses()).max().unwrap());
        for (out_idx, neuron) in population.neurons.iter().enumerate() {
            for seg in &neuron.segments {
                for syn in &seg.synapses {
                    connections[syn.input_idx].push(as_idx(out_idx));
                }
            }
        }
        assert!(k <= output_shape.channels(), "k is larger than layer output");
        Self {
            threshold: 1,
            k,
            input_shape,
            output_shape,
            connections,
            max_incoming_synapses,
            kernel,
            stride,
            sums: vec![0u16; as_usize(output_shape.product())],
        }
    }
}
impl EccLayerD for CpuEccSparse {
    type D = u16;
    fn get_threshold(&self) -> Self::D {
        self.threshold
    }

    fn set_threshold(&mut self, threshold: Self::D) {
        self.threshold = threshold
    }

    fn get_plasticity(&self) -> Self::D {
        0
    }

    fn set_plasticity(&mut self, threshold: Self::D) {
    }
}
impl EccLayer for CpuEccSparse {
    type A = CpuSDR;


    fn k(&self) -> Idx { self.k }
    fn set_k(&mut self, k: Idx) {
        assert!(k <= self.out_channels(), "k is larger than layer output!");
        self.k = k;
    }



    fn out_shape(&self) -> &[Idx; 3] { &self.output_shape }

    fn in_shape(&self) -> &[Idx; 3] { &self.input_shape }

    fn kernel(&self) -> &[Idx; 2] { &self.kernel }

    fn stride(&self) -> &[Idx; 2] { &self.stride }

    fn learnable_parameters(&self) -> usize {
        0
    }

    fn get_max_incoming_synapses(&self) -> Idx {
        self.max_incoming_synapses
    }
    fn get_threshold_f32(&self) -> f32 {
        self.threshold as f32 / self.max_incoming_synapses as f32
    }
    fn set_threshold_f32(&mut self, threshold: f32) {
        assert!(threshold > 0., "Negative threshold!");
        self.threshold = (self.max_incoming_synapses as f32 * threshold).round() as u16
    }

    fn set_plasticity_f32(&mut self, fractional: f32) {}

    fn get_plasticity_f32(&self) -> f32 { 0. }

    fn new_empty_sdr(&self, capacity: Idx) -> Self::A {
        CpuSDR::new()
    }

    fn infer_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR) {
        self.sums.fill(0);
        for &input_idx in input.as_slice() {
            for &c in &self.connections[as_usize(input_idx)] {
                self.sums[as_usize(c)] += 1;
            }
        }
        let t = self.threshold;
        top_small_k_by_channel(self, |i| self.sums[i], |i, v| v >= t, |a, b| a > b, output)
    }

    fn decrement_activities(&mut self, output: &CpuSDR) {}

    fn learn(&mut self, input: &CpuSDR, output: &CpuSDR) {}
}

impl OclEccSparse {

    pub fn to_cpu(&self) -> CpuEccSparse {
        let connection_ranges = self.get_connection_ranges().to_vec(self.prog().queue()).unwrap();
        let connections = self.get_connections().to_vec(self.prog().queue()).unwrap();
        CpuEccSparse {
            connections: connection_ranges.into_iter().map(|range| connections[range[0] as usize..(range[0] + range[1]) as usize].to_vec()).collect(),
            max_incoming_synapses: self.get_max_incoming_synapses(),
            input_shape: *self.in_shape(),
            output_shape: *self.out_shape(),
            kernel: *self.kernel(),
            stride: *self.stride(),
            k: self.k,
            threshold: self.get_threshold(),
            sums: vec![0; self.sums.len()],
        }
    }
}

pub type CpuEccMachine<D: DenseWeight> = EccSepConMachine<CpuSDR, CpuEccSparse, CpuEccDense<D>>;

impl<D: DenseWeight> CpuEccMachine<D> {
    pub fn new_cpu(output: [Idx; 2], kernels: &[[Idx; 2]], strides: &[[Idx; 2]], channels: &[Idx], k: &[Idx], connections_per_output: &[Option<Idx>], rng: &mut impl Rng) -> Self {
        Self::new(output, kernels, strides, channels, k, connections_per_output, rng, |output, in_channels, out_channels, k, kernel, stride, conn, rng|
            if let Some(connections_per_output) = conn {
                SparseOrDense::Sparse(CpuEccSparse::new(output, kernel, stride, in_channels, out_channels, k, connections_per_output, rng))
            } else {
                SparseOrDense::Dense(CpuEccDense::new(output, kernel, stride, in_channels, out_channels, k, rng))
            })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use std::cmp::Ordering::{Greater, Less};

    #[test]
    fn test1() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 8;
        let mut a = CpuEccSparse::new([4, 4], [2, 2], [1, 1], 3, 4, 1, 4, &mut rng);
        for _ in 0..64 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            let o = a.run(&input);
            assert_ne!(o.len(), 0);
        }
        Ok(())
    }


}
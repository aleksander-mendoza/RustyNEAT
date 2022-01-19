use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::ocl_sdr::OclSDR;
use crate::ecc_program::EccProgram;
use ndalgebra::buffer::Buffer;
use crate::{OclBitset, CpuEccDense, Shape2, Shape3, Shape, VectorFieldOne, CpuEccSparse, EccMachine, SparseOrDense, DenseWeight, CpuEccMachine, EccSepConMachine, EccLayerD};
use ocl::prm::{Uint3, Uint2};
use crate::ecc::{as_usize, EccLayer, Idx};
use crate::sdr::SDR;
use rand::Rng;

#[derive(Clone)]
pub struct OclEccDense {
    prog: EccProgram,
    /**The layout is w[output_idx+input_idx_relative_to_kernel_column*output_volume]
    where kernel column has shape [kernel[0],kernel[1],in_channels]*/
    w: Buffer<u32>,
    // instead of f32 we use u32 but they are equivalent. Just imagine that you divide
    // the u32 value by some large constant and the obtain f32. Addition and subtraction factor out
    //during division (4u32/1000f32)+(8u32/1000f32) == (4u32+8u32)/1000f32
    input_shape: [Idx; 3],
    //[height, width, channels]
    output_shape: [Idx; 3],
    //[height, width, channels]
    kernel: [Idx; 2],
    //[height, width]
    stride: [Idx; 2],
    //[height, width]
    // pub k: Idx,
    pub threshold: u32,
    pub plasticity: u32,
    activity: Buffer<u32>,
    pub sums: Buffer<u32>,
    pub top_values: Buffer<u32>,
}

impl EccLayerD for OclEccDense {
    type D = u32;
    fn get_threshold(&self) -> u32 {
        self.threshold
    }

    fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold
    }

    fn get_plasticity(&self) -> u32 {
        self.plasticity
    }
    fn set_plasticity(&mut self, plasticity: u32) {
        self.plasticity = plasticity
    }
}

impl EccLayer for OclEccDense {
    type A = OclSDR;

    fn k(&self) -> Idx {
        1
    }

    fn set_k(&mut self, k: Idx) {}


    fn out_shape(&self) -> &[Idx; 3] {
        &self.output_shape
    }

    fn in_shape(&self) -> &[Idx; 3] {
        &self.input_shape
    }

    fn kernel(&self) -> &[Idx; 2] {
        &self.kernel
    }

    fn stride(&self) -> &[Idx; 2] {
        &self.stride
    }

    fn learnable_parameters(&self) -> usize {
        self.w.len()
    }

    fn get_max_incoming_synapses(&self) -> u32 {
        self.kernel_column().product()
    }

    fn get_threshold_f32(&self) -> f32 {
        u32::w_to_f32(self.threshold)
    }

    fn set_threshold_f32(&mut self, threshold: f32) {}

    fn set_plasticity_f32(&mut self, fractional: f32) {
        self.plasticity = u32::f32_to_w(fractional)
    }

    fn get_plasticity_f32(&self) -> f32 {
        u32::w_to_f32(self.plasticity)
    }

    fn new_empty_sdr(&self, capacity: u32) -> Self::A {
        OclSDR::new(self.prog.clone(), capacity).unwrap()
    }

    fn infer_in_place(&mut self, input: &OclSDR, output: &mut OclSDR) {
        self.zero_out_all_sums().unwrap();
        self.sums(input).unwrap();
        self.zero_out_all_top_values().unwrap();
        self.max_r().unwrap();
        self.top_1(output).unwrap();
        self.prog.q.finish().unwrap();
        unsafe {
            output.set_cardinality(self.get_output_len().unwrap());
        }
    }

    fn decrement_activities(&mut self, output: &Self::A) {
        self.decrement_activities_for_sdr(output).unwrap()
    }

    fn learn(&mut self, input: &OclSDR, output: &OclSDR) {
        self.increment_weights(input, output).unwrap();
        self.zero_out_sums_for_sdr(input).unwrap();
        self.incoming_weights_sum(output).unwrap();
        self.normalize(output).unwrap();
    }
}

impl OclEccDense {
    pub fn activity(&self) -> &Buffer<u32> {
        &self.activity
    }
    pub fn w(&self) -> &Buffer<u32> {
        &self.w
    }
    pub fn prog(&self) -> &EccProgram {
        &self.prog
    }
    pub fn new(ecc: &CpuEccDense<u32>, prog: EccProgram) -> Result<Self, Error> {
        let sums = prog.buffer_from_slice(MemFlags::READ_WRITE, &ecc.sums)?;
        let activity = prog.buffer_from_slice(MemFlags::READ_WRITE, ecc.get_activity())?;
        let w = prog.buffer_from_slice(MemFlags::READ_WRITE, ecc.get_weights())?;
        let top_values = prog.buffer_filled(MemFlags::READ_WRITE, as_usize(ecc.out_area()) + 1, 0)?;
        assert_eq!(ecc.k(), 1, "k must be 1");
        Ok(Self {
            prog,
            w,
            input_shape: *ecc.in_shape(),
            output_shape: *ecc.out_shape(),
            kernel: *ecc.kernel(),
            stride: *ecc.stride(),
            threshold: ecc.threshold,
            plasticity: ecc.plasticity,
            activity,
            sums,
            top_values,
        })
    }

    fn zero_out_all_sums(&mut self) -> Result<(), Error> {
        let Self { prog, sums, .. } = self;
        sums.fill(prog.queue(), 0)
    }
    fn zero_out_all_top_values(&mut self) -> Result<(), Error> {
        let Self { prog, top_values, .. } = self;
        top_values.fill(prog.queue(), 0)
    }
    fn get_output_len(&mut self) -> Result<u32, Error> {
        let Self { prog, top_values, output_shape, .. } = self;
        top_values.get(prog.queue(), as_usize(output_shape.grid().product()))
    }
    fn sums(&mut self, input: &OclSDR) -> Result<(), Error> {
        let Self { prog, w, input_shape, output_shape, kernel, stride, sums, .. } = self;
        let output_kernel_column = kernel.conv_out_transpose_kernel(stride).add_channels(output_shape.channels());
        let output_volume = output_shape.product();
        let output_kernel_column_volume = output_kernel_column.product();
        prog.kernel_builder("ecc_dense_sums")?.
            add_vec(Uint3::from(output_kernel_column))?.// const uint3 output_kernel_column,
            add_vec(Uint3::from(*output_shape))?.// const uint3 output_shape,
            add_vec(Uint3::from(*input_shape))?.// const uint3 input_shape,
            add_vec(Uint2::from(*stride))?.// const uint2 stride,
            add_vec(Uint2::from(*kernel))?.// const uint2 kernel,
            add_num(output_volume)?.// const uint v,
            add_buff(input.buffer())?.// __global uint  * input,
            add_buff(sums)?.// __global uint  * sums,
            add_buff(w)?.// __global uint  * w
            enq(prog.queue(), &[as_usize(input.cardinality()), as_usize(output_kernel_column_volume), 1]).
            map_err(Error::from)
    }
    fn zero_out_sums_for_sdr(&mut self, sdr: &OclSDR) -> Result<(), Error> {
        let Self { prog, activity, .. } = self;
        prog.kernel_builder("ecc_dense_zero_out_sums_for_sdr")?.
            add_buff(sdr.buffer())?.// __global uint  * sdr
            add_buff(activity)?.// __global uint  * activity,
            enq(prog.queue(), &[as_usize(sdr.cardinality()), 1, 1]).
            map_err(Error::from)
    }
    fn incoming_weights_sum(&mut self, output: &OclSDR) -> Result<(), Error> {
        let kv = self.kernel_column().product();
        let Self { prog, w, output_shape, sums, .. } = self;
        let output_volume = output_shape.product();
        prog.kernel_builder("ecc_dense_incoming_weights_sum")?.
            add_num(output_volume)?. // const uint v,
            add_buff(output.buffer())?. // __global uint * output_sdr,
            add_buff(sums)?. // __global uint * sums
            add_buff(w)?. // __global uint * w
            enq(prog.queue(), &[as_usize(kv), as_usize(output.cardinality()), 1]).
            map_err(Error::from)
    }
    fn normalize(&mut self, output: &OclSDR) -> Result<(), Error> {
        let kv = self.kernel_column().product();
        let Self { prog, w, output_shape, sums, .. } = self;
        let output_volume = output_shape.product();
        prog.kernel_builder("ecc_dense_normalize")?.
            add_num(output_volume)?. // const uint v,
            add_buff(output.buffer())?. // __global uint * output_sdr,
            add_buff(sums)?. // __global uint * sums
            add_buff(w)?. // __global uint * w
            enq(prog.queue(), &[as_usize(kv), as_usize(output.cardinality()), 1]).
            map_err(Error::from)
    }
    fn increment_weights(&mut self, input: &OclSDR, output: &OclSDR) -> Result<(), Error> {
        let kernel_column = self.kernel_column();
        let Self { prog, w, output_shape, input_shape, plasticity, stride, .. } = self;
        let output_volume = output_shape.product();
        prog.kernel_builder("ecc_dense_increment_weights")?.
            add_num(output_volume)?.// const uint v,
            add_vec(Uint3::from(*input_shape))?.// const uint3 input_shape,
            add_vec(Uint3::from(*output_shape))?.// const uint3 output_shape,
            add_vec(Uint3::from(kernel_column))?.// const uint3 kernel_column,
            add_vec(Uint2::from(*stride))?.// const uint2 stride,
            add_num(*plasticity)?.// const uint plasticity,
            add_buff(input.buffer())?.// __global uint * output_sdr,
            add_buff(output.buffer())?.// __global uint * input_sdr,
            add_buff(w)?. // __global uint * w
            enq(prog.queue(), &[as_usize(input.cardinality()), as_usize(output.cardinality()), 1]).
            map_err(Error::from)
    }
    fn max_r(&mut self) -> Result<(), Error> {
        let Self { prog, top_values, activity, threshold, output_shape, sums, .. } = self;
        let output_grid_area = output_shape.grid().product();
        prog.kernel_builder("ecc_dense_max_r")?.
            add_num(*threshold)?. // const uint threshold,
            add_buff(activity)?.// __global uint * activity,
            add_buff(top_values)?.// __global uint * top_values,
            add_buff(sums)?.// __global uint * sums
            enq(prog.queue(), &[as_usize(output_shape.channels()), as_usize(output_grid_area), 1]).
            map_err(Error::from)
    }
    fn top_1(&mut self, output: &mut OclSDR) -> Result<(), Error> {
        let Self { prog, top_values, activity, threshold, output_shape, sums, .. } = self;
        let output_grid_area = output_shape.grid().product();
        prog.kernel_builder("ecc_dense_top_1")?.
            add_num(*threshold)?. // const uint threshold,
            add_buff(activity)?.// __global uint * activity,
            add_buff(top_values)?.// __global uint * top_values,
            add_buff(output.buffer())?.//__global uint * output_sdr,
            add_buff(sums)?.// __global uint * sums
            enq(prog.queue(), &[as_usize(output_shape.channels()), as_usize(output_grid_area), 1]).
            map_err(Error::from)
    }
    fn decrement_activities_for_sdr(&mut self, sdr: &OclSDR) -> Result<(), Error> {
        let Self { prog, activity, .. } = self;
        prog.kernel_builder("ecc_dense_decrement_activities_for_sdr")?.
            add_buff(sdr.buffer())?.// __global uint  * sdr
            add_buff(activity)?.// __global uint  * activity,
            enq(prog.queue(), &[as_usize(sdr.cardinality()), 1, 1]).
            map_err(Error::from)
    }
}


#[derive(Clone)]
pub struct OclEccSparse {
    prog: EccProgram,
    connections: Buffer<u32>,
    connection_ranges: Buffer<Uint2>,
    max_incoming_synapses: Idx,
    max_outgoing_synapses: Idx,
    input_shape: [Idx; 3],
    //[height, width, channels]
    output_shape: [Idx; 3],
    //[height, width, channels]
    kernel: [Idx; 2],
    //[height, width]
    stride: [Idx; 2],
    //[height, width]
    pub k: Idx,
    threshold: u16,
    pub sums: Buffer<u32>,
    pub candidates_per_sum: Buffer<u32>,
}

impl EccLayerD for OclEccSparse {
    type D = u16;
    fn get_threshold(&self) -> u16 {
        self.threshold
    }
    fn set_threshold(&mut self, threshold: u16) {
        //nothing
    }
    fn get_plasticity(&self) -> u16 {
        0
    }
    fn set_plasticity(&mut self, plasticity: u16) {}
}

impl EccLayer for OclEccSparse {
    type A = OclSDR;

    fn k(&self) -> u32 {
        self.k
    }

    fn set_k(&mut self, k: u32) {
        self.k = k
    }

    fn out_shape(&self) -> &[u32; 3] {
        &self.output_shape
    }

    fn in_shape(&self) -> &[u32; 3] {
        &self.input_shape
    }

    fn kernel(&self) -> &[u32; 2] {
        &self.kernel
    }

    fn stride(&self) -> &[u32; 2] {
        &self.stride
    }
    fn learnable_parameters(&self) -> usize {
        0
    }

    fn get_max_incoming_synapses(&self) -> u32 {
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

    fn new_empty_sdr(&self, capacity: u32) -> Self::A {
        OclSDR::new(self.prog.clone(), capacity).unwrap()
    }

    fn infer_in_place(&mut self, input: &OclSDR, output_sdr: &mut OclSDR) {
        self.reset_sums().unwrap();
        self.sums(input).unwrap();
        self.elements_per_sum().unwrap();
        self.retain_top_k_candidates().unwrap();
        self.top_k(output_sdr).unwrap();
        self.prog.queue().finish().unwrap();
        unsafe {
            output_sdr.set_cardinality(self.get_cardinality().unwrap());
        }
    }

    fn decrement_activities(&mut self, sdr: &OclSDR) {}

    fn learn(&mut self, input: &OclSDR, output: &OclSDR) {}
}

impl OclEccSparse {
    pub fn prog(&self) -> &EccProgram {
        &self.prog
    }
    pub fn get_connections(&self) -> &Buffer<u32> {
        &self.connections
    }
    pub fn get_connection_ranges(&self) -> &Buffer<Uint2> {
        &self.connection_ranges
    }
    pub fn new(ecc: &CpuEccSparse, prog: EccProgram) -> Result<Self, Error> {
        let mut connections = vec![];
        let mut connection_ranges = vec![];
        for conn in ecc.connections() {
            debug_assert_eq!(connection_ranges.last().map(|a: &Uint2| a.iter().sum()).unwrap_or(0) as usize, connections.len());
            let range = Uint2::from([connections.len() as u32, conn.len() as u32]);
            connection_ranges.push(range);
            connections.extend_from_slice(conn.as_slice());
            debug_assert!(conn.iter().all(|&a| a < ecc.out_volume()), "{:?}<{}", conn, ecc.out_volume());
            debug_assert_eq!(connections.len() as u32, range[0] + range[1]);
        }

        let max_outgoing_synapses = connection_ranges.iter().map(|a| a[1]).max().unwrap();
        let connections = prog.buffer_from_slice(MemFlags::READ_WRITE, &connections)?;
        let connection_ranges = prog.buffer_from_slice(MemFlags::READ_WRITE, &connection_ranges)?;
        let sums = prog.buffer_filled(MemFlags::READ_WRITE, ecc.sums.len(), 0)?;
        let sum_range = ecc.get_max_incoming_synapses() - ecc.threshold as u32 + 1;
        let candidates_per_sum = prog.buffer_filled(MemFlags::READ_WRITE, as_usize(ecc.out_area() * sum_range + 1), 0)?;
        Ok(Self {
            prog,
            connections,
            connection_ranges,
            max_outgoing_synapses,
            max_incoming_synapses: ecc.get_max_incoming_synapses(),
            input_shape: *ecc.in_shape(),
            output_shape: *ecc.out_shape(),
            kernel: *ecc.kernel(),
            stride: *ecc.stride(),
            k: ecc.k(),
            threshold: ecc.threshold,
            sums,
            candidates_per_sum,
        })
    }

    fn sums(&mut self, input: &OclSDR) -> Result<(), Error> {
        let Self { prog, connection_ranges, sums, connections, max_outgoing_synapses, .. } = self;
        prog.kernel_builder("ecc_sparse_sums")?.
            add_buff(input.buffer())?.// __global uint * input_sdr,
            add_buff(connections)?.// __global uint * connections,
            add_buff(connection_ranges)?.// __global uint2 * connection_ranges,
            add_buff(sums)?.// __global uint * sums
            enq(prog.queue(), &[as_usize(input.cardinality()), as_usize(*max_outgoing_synapses), 1]).
            map_err(Error::from)
    }


    fn elements_per_sum(&mut self) -> Result<(), Error> {
        let Self { prog, sums, threshold, output_shape, candidates_per_sum, .. } = self;
        let threshold = *threshold as u32;
        let channels = output_shape.channels();
        prog.kernel_builder("ecc_sparse_elements_per_sum")?.
            add_num(threshold)?.// const uint threshold,
            add_buff(candidates_per_sum)?.// __global int * candidates_per_sum
            add_buff(sums)?.// __global uint * sums
            enq(prog.queue(), &[as_usize(output_shape.grid().product()), as_usize(channels), 1]).
            map_err(Error::from)
    }

    fn retain_top_k_candidates(&mut self) -> Result<(), Error> {
        let Self { prog, sums, max_incoming_synapses, k, threshold, output_shape, candidates_per_sum, .. } = self;
        let threshold = *threshold as u32;
        prog.kernel_builder("ecc_sparse_retain_top_k_candidates")?.
            add_num(*k)?.// const uint k,
            add_num(threshold)?.// const uint threshold,
            add_num(*max_incoming_synapses)?.// const uint max_value,
            add_buff(candidates_per_sum)?.// __global int * candidates_per_sum
            enq(prog.queue(), &[as_usize(output_shape.grid().product()), 1, 1]).
            map_err(Error::from)
    }

    fn top_k(&mut self, output_sdr: &mut OclSDR) -> Result<(), Error> {
        let Self { prog, sums, threshold, output_shape, candidates_per_sum, .. } = self;
        let threshold = *threshold as u32;
        let channels = output_shape.channels();
        prog.kernel_builder("ecc_sparse_top_k")?.
            add_num(threshold)?.// const uint threshold,
            add_buff(candidates_per_sum)?.// __global int * candidates_per_sum
            add_buff(sums)?.// __global uint * sums
            add_buff(output_sdr.buffer())?.// __global uint * output_sdr
            enq(prog.queue(), &[as_usize(output_shape.grid().product()), as_usize(channels), 1]).
            map_err(Error::from)
    }

    fn get_cardinality(&self) -> Result<u32, Error> {
        let Self { prog, candidates_per_sum, .. } = self;
        candidates_per_sum.get(prog.queue(), 0)
    }

    fn reset_sums(&mut self) -> Result<(), Error> {
        let Self { sums, prog, .. } = self;
        sums.fill(prog.queue(), 0)
    }
}

pub type OclEccMachine = EccSepConMachine<OclSDR, OclEccSparse, OclEccDense>;

impl OclEccMachine {
    pub fn new_gpu(prog: EccProgram, output: [Idx; 2], kernels: &[[Idx; 2]], strides: &[[Idx; 2]], channels: &[Idx], k: &[Idx], connections_per_output: &[Option<Idx>], rng: &mut impl Rng) -> Self {
        Self::new(output, kernels, strides, channels, k, connections_per_output, rng, |output, in_channels, out_channels, k, kernel, stride, conn, rng|
            if let Some(connections_per_output) = conn {
                SparseOrDense::Sparse(OclEccSparse::new(&CpuEccSparse::new(output, kernel, stride, in_channels, out_channels, k, connections_per_output, rng), prog.clone()).unwrap())
            } else {
                SparseOrDense::Dense(OclEccDense::new(&CpuEccDense::new(output, kernel, stride, in_channels, out_channels, k, rng), prog.clone()).unwrap())
            })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use std::cmp::Ordering::{Greater, Less};
    use crate::{CpuSDR, from_xyz, from_xy, DenseWeight};

    #[test]
    fn test1() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut ecc = CpuEccDense::new([2, 2], [2, 2], [1, 1], 3, 4, 1, &mut rng);

        let mut input = CpuSDR::new();
        input.add_unique_random(4, 0..ecc.in_volume());

        let p = EccProgram::default()?;
        let mut ecc2 = OclEccDense::new(&ecc, p.clone())?;

        let o = ecc.run(&input);
        let input2 = OclSDR::from_cpu(p.clone(), &input, input.cardinality())?;
        ecc2.zero_out_all_sums();
        ecc2.sums(&input2);

        let sums2 = ecc2.sums.to_vec(p.queue())?;
        assert_eq!(&sums2, &ecc.sums);
        Ok(())
    }

    #[test]
    fn test2() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut ecc = CpuEccDense::new([6, 6], [2, 2], [2, 2], 16, 8, 1, &mut rng);

        let mut input = CpuSDR::new();
        input.add_unique_random(16, 0..ecc.in_volume());

        let p = EccProgram::default()?;
        let mut ecc2 = OclEccDense::new(&ecc, p.clone())?;

        let o = ecc.run(&input);
        let input2 = OclSDR::from_cpu(p.clone(), &input, input.cardinality())?;
        ecc2.zero_out_all_sums();
        ecc2.sums(&input2);

        let sums2 = ecc2.sums.to_vec(p.queue())?;
        assert_eq!(&sums2, &ecc.sums);
        Ok(())
    }

    #[test]
    fn test3() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut ecc = CpuEccDense::new([6, 6], [2, 2], [2, 2], 16, 8, 1, &mut rng);
        let p = EccProgram::default()?;
        let mut ecc2 = OclEccDense::new(&ecc, p.clone())?;
        assert_eq!(ecc.get_activity(), ecc2.activity.to_vec(p.queue()).unwrap().as_slice(), "activity before");
        for i in 0..32 {
            let mut input = CpuSDR::new();
            input.add_unique_random(ecc.in_volume(), 0..ecc.in_volume());
            let mut output = ecc.infer(&input);
            let input2 = OclSDR::from_cpu(p.clone(), &input, input.cardinality()).unwrap();
            let max_card = ecc2.output_shape.grid().product();
            let mut output2 = OclSDR::new(p.clone(), max_card).unwrap();
            assert_eq!(ecc.get_activity(), ecc2.activity.to_vec(p.queue()).unwrap().as_slice(), "activity at i={}", i);
            ecc2.zero_out_all_sums().unwrap();
            ecc2.sums(&input2).unwrap();
            p.queue().finish().unwrap();
            assert_eq!(ecc.sums.len(), ecc2.sums.len());
            let s = ecc2.sums.to_vec(p.queue()).unwrap();
            assert_eq!(ecc.sums, s.as_slice(), "sums at i={}", i);
            ecc2.zero_out_all_top_values().unwrap();
            ecc2.max_r().unwrap();
            p.q.finish().unwrap();
            let r: Vec<u32> = ecc.sums.iter().zip(ecc.get_activity().iter()).map(|(&a, &b)| a + b).collect();
            let a = ecc2.activity.to_vec(p.queue()).unwrap();
            let r2: Vec<u32> = s.iter().zip(a.iter()).map(|(&a, &b)| a + b).collect();
            assert_eq!(r, r2, "r at i={}", i);
            let mut max_r = vec![0; as_usize(ecc.out_area())];
            for x in 0..ecc.out_width() {
                for y in 0..ecc.out_height() {
                    let output_column_idx = ecc.out_shape().grid().idx(from_xy(x, y));
                    max_r[as_usize(output_column_idx)] = (0..ecc.out_channels()).map(|c| {
                        let i = ecc.out_shape().idx(from_xyz(x, y, c));
                        assert_eq!(i, output_column_idx * ecc.out_channels() + c);
                        r2[as_usize(i)]
                    }).max().unwrap();
                }
            }
            let max_r2 = ecc2.top_values.to_vec(p.queue()).unwrap();
            assert_eq!(max_r.as_slice(), &max_r2[0..max_r2.len() - 1], "max_r2 at i={}", i);
            ecc2.top_1(&mut output2).unwrap();
            p.q.finish().unwrap();
            unsafe {
                output2.set_cardinality(ecc2.get_output_len()?);
            }
            output.sort();
            let mut cpu_output2 = output2.to_cpu().unwrap();
            cpu_output2.sort();
            assert_eq!(output, cpu_output2, "output at i={}", i);
            ecc.decrement_activities(&output);
            ecc2.decrement_activities_for_sdr(&output2)?;
            p.queue().finish()?;
            let decremented_a = ecc2.activity.to_vec(p.queue()).unwrap();
            let mut a = a;
            for &o in cpu_output2.iter() {
                a[as_usize(o)] -= u32::ACTIVITY_PENALTY;
            }
            assert_eq!(ecc.get_activity(), a.as_slice(), "decremented activity at i={} {:?}", i, cpu_output2);
            assert_eq!(decremented_a, a, "decremented activity2 at i={} {:?}", i, cpu_output2);
            assert_eq!(ecc.get_activity(), decremented_a.as_slice(), "decremented activity3 at i={} {:?}", i, cpu_output2);
        }
        Ok(())
    }

    #[test]
    fn test4() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut ecc = CpuEccDense::new([6, 6], [2, 2], [2, 2], 16, 8, 1, &mut rng);
        let p = EccProgram::default()?;
        let mut ecc2 = OclEccDense::new(&ecc, p.clone())?;
        for i in 0..32 {
            let mut input = CpuSDR::new();
            input.add_unique_random(ecc.in_volume(), 0..ecc.in_volume());
            let mut output = ecc.run(&input);
            let input2 = OclSDR::from_cpu(p.clone(), &input, input.cardinality()).unwrap();
            let output2 = ecc2.run(&input2);
            p.queue().finish()?;
            let mut output2 = output2.to_cpu().unwrap();
            assert_eq!(ecc.get_activity(), ecc2.activity.to_vec(p.queue()).unwrap().as_slice(), "i={}", i);
            output.sort();
            output2.sort();
            assert_eq!(output, output2);
        }
        Ok(())
    }

    #[test]
    fn test5() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut ecc = CpuEccSparse::new([6, 6], [2, 2], [2, 2], 16, 8, 1, 6, &mut rng);
        let p = EccProgram::default()?;
        let mut ecc2 = OclEccSparse::new(&ecc, p.clone())?;
        for i in 0..32 {
            let mut input = CpuSDR::new();
            input.add_unique_random(ecc.in_volume(), 0..ecc.in_volume());
            let mut output = ecc.run(&input);
            let input2 = OclSDR::from_cpu(p.clone(), &input, input.cardinality()).unwrap();
            let mut output2 = OclSDR::new(ecc2.prog.clone(), ecc2.output_shape.grid().product() * ecc2.k)?;
            ecc2.reset_sums()?;
            ecc2.sums(&input2)?;
            let mut sums0 = vec![0u32; ecc2.sums.len()];
            let input0 = input2.to_cpu()?;
            let connections = ecc2.connections.to_vec(p.queue()).unwrap();
            let connection_ranges = ecc2.connection_ranges.to_vec(p.queue()).unwrap();
            assert_eq!(input0, input, "input {:?}", i);
            assert_eq!(connection_ranges.len() as u32, ecc2.input_shape.product());
            assert_eq!(sums0.len() as u32, ecc2.output_shape.product());
            for &input_idx in input0.iter() {
                let connection_range = connection_ranges[input_idx as usize];
                assert!(input_idx + 1 == ecc2.input_shape.product() || connection_range[0] + connection_range[1] == connection_ranges[1 + input_idx as usize][0]);
                for i in 0..connection_range[1] {
                    sums0[connections[connection_range[0] as usize + i as usize] as usize] += 1;
                }
            }
            ecc2.prog.queue().finish()?;
            let sums2 = ecc2.sums.to_vec(ecc2.prog.queue())?;
            let sums: Vec<u32> = ecc.sums.iter().map(|&s| s as u32).collect();
            assert_eq!(sums0, sums, "sums0 {}", i);
            assert_eq!(sums, sums2, "sums2 {}", i);
            ecc2.elements_per_sum()?;
            ecc2.prog.queue().finish()?;
            let candidates_per_sum = ecc2.candidates_per_sum.to_vec(p.queue())?;
            let mut candidates_per_sum0 = vec![0; candidates_per_sum.len()];
            let area = ecc2.output_shape.grid().product() as usize;
            for output_column_idx in 0..area {
                for c in 0..ecc2.output_shape.channels() as usize {
                    let sum = sums[output_column_idx * ecc2.output_shape.channels() as usize + c] as usize;
                    if ecc2.threshold as usize <= sum {
                        candidates_per_sum0[output_column_idx + (sum - ecc2.threshold as usize) * area + 1] += 1;
                    }
                }
            }
            assert_eq!(candidates_per_sum, candidates_per_sum0, "candidates_per_sum {}", i);
            ecc2.retain_top_k_candidates()?;
            ecc2.top_k(&mut output2)?;
            ecc2.prog.queue().finish()?;
            unsafe {
                output2.set_cardinality(ecc2.get_cardinality()?);
            }
            p.queue().finish()?;
            let mut output2 = output2.to_cpu().unwrap();
            output.sort();
            output2.sort();
            assert_eq!(output, output2, "output at {}", i);
        }
        Ok(())
    }

    #[test]
    fn test6() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut ecc = CpuEccSparse::new([6, 6], [2, 2], [2, 2], 16, 8, 1, 6, &mut rng);
        let p = EccProgram::default()?;
        let mut ecc2 = OclEccSparse::new(&ecc, p.clone())?;
        for i in 0..32 {
            let mut input = CpuSDR::new();
            input.add_unique_random(ecc.in_volume(), 0..ecc.in_volume());
            let mut output = ecc.run(&input);
            let input2 = OclSDR::from_cpu(p.clone(), &input, input.cardinality()).unwrap();
            let output2 = ecc2.infer(&input2);
            p.queue().finish()?;
            let mut output2 = output2.to_cpu().unwrap();
            output.sort();
            output2.sort();
            assert_eq!(output, output2);
        }
        Ok(())
    }
}
use crate::{VectorFieldOne, Shape2, Shape3, VectorFieldPartialOrd};
use crate::xorshift::xorshift32;
use std::ops::{Deref, DerefMut};
use itertools::Itertools;
use rand::Rng;
use serde::{Serialize,Deserialize};
use crate::sdr::SDR;

pub type Idx = u32;
pub type Rand = u32;
pub fn xorshift_rand(rand:Rand)->Rand{
    xorshift32(rand)
}
pub fn as_usize(i:Idx)->usize{
    i as usize
}
pub fn as_idx(i:usize)->Idx{
    i as u32
}

pub trait EccLayer {
    type A:SDR;
    fn k(&self) -> Idx;
    fn set_k(&mut self, k: Idx);
    fn out_shape(&self) -> &[Idx; 3];
    fn in_shape(&self) -> &[Idx; 3];
    fn kernel(&self) -> &[Idx; 2];
    fn stride(&self) -> &[Idx; 2];
    fn learnable_parameters(&self) -> usize;
    fn kernel_column(&self) -> [Idx; 3] {
        self.kernel().add_channels(self.in_channels())
    }
    fn get_max_incoming_synapses(&self) -> Idx;
    fn get_threshold_f32(&self) -> f32;
    fn set_threshold_f32(&mut self, threshold: f32);
    fn set_plasticity(&mut self, fractional: f32);
    fn get_plasticity(&self) -> f32;
    fn in_grid(&self) -> &[Idx; 2] {
        self.in_shape().grid()
    }
    fn out_grid(&self) -> &[Idx; 2] {
        self.out_shape().grid()
    }
    fn out_width(&self) -> Idx {
        self.out_shape()[1]
    }
    fn out_height(&self) -> Idx {
        self.out_shape()[0]
    }
    fn out_channels(&self) -> Idx {
        self.out_shape()[2]
    }
    fn in_width(&self) -> Idx {
        self.in_shape()[1]
    }
    fn in_height(&self) -> Idx {
        self.in_shape()[0]
    }
    fn in_channels(&self) -> Idx {
        self.in_shape()[2]
    }
    fn out_area(&self) -> Idx {
        self.out_grid().product()
    }
    fn in_area(&self) -> Idx {
        self.in_grid().product()
    }
    fn out_volume(&self) -> Idx {
        self.out_shape().product()
    }
    fn in_volume(&self) -> Idx {
        self.in_shape().product()
    }
    fn run(&mut self, input: &Self::A) -> Self::A {
        let output = self.infer(input);
        self.decrement_activities(&output);
        output
    }
    fn new_empty_sdr(&self,capacity:Idx)->Self::A;
    fn new_empty_output_sdr(&self)->Self::A {
        let k = self.k();
        let a = self.out_area();
        self.new_empty_sdr(a*k)
    }
    fn infer(&mut self, input: &Self::A) -> Self::A {
        let mut output = self.new_empty_output_sdr();
        self.infer_in_place(input, &mut output);
        output
    }
    fn run_in_place(&mut self, input: &Self::A, output: &mut Self::A){
        self.infer_in_place(input,output);
        self.decrement_activities(output)
    }
    fn infer_in_place(&mut self, input: &Self::A, output: &mut Self::A);
    fn decrement_activities(&mut self,output: &Self::A);
    fn learn(&mut self, input: &Self::A, output: &Self::A);
    // fn top_large_k_by_channel<T>(&self, sums: &[T], candidates_per_value: &mut [Idx], f: fn(&T) -> Idx, threshold: impl Fn(Idx) -> bool) -> CpuSDR {
    //     let a = as_usize(self.out_area());
    //     let c = as_usize(self.out_channels());
    //     let k = as_usize(self.k());
    //     let mut top_k = CpuSDR::with_capacity(k * a);
    //     for column_idx in 0..a {
    //         let offset = c * column_idx;
    //         let range = offset..offset + c;
    //         top_large_k_indices(k, &sums[range], candidates_per_value, f, |t| if threshold(t) { top_k.push(offset + t) });
    //     }
    //     top_k
    // }

}



#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum SparseOrDense<A:SDR,S:EccLayer<A=A>,D:EccLayer<A=A>> {
    Sparse(S),
    Dense(D),
}

impl<A:SDR,S:EccLayer<A=A>,D:EccLayer<A=A>> SparseOrDense<A,S,D> {
    pub fn is_sparse(&self)->bool{
        match self {
            SparseOrDense::Sparse(a) => true,
            SparseOrDense::Dense(a) => false
        }
    }
}
impl<A:SDR,S:EccLayer<A=A>,D:EccLayer<A=A>> EccLayer for SparseOrDense<A,S,D> {
    type A = A;

    fn k(&self) -> Idx {
        match self {
            SparseOrDense::Sparse(a) => a.k(),
            SparseOrDense::Dense(a) => a.k()
        }
    }

    fn set_k(&mut self, k: Idx) {
        match self {
            SparseOrDense::Sparse(a) => a.set_k(k),
            SparseOrDense::Dense(a) => a.set_k(k)
        }
    }

    fn out_shape(&self) -> &[Idx; 3] {
        match self {
            SparseOrDense::Sparse(a) => a.out_shape(),
            SparseOrDense::Dense(a) => a.out_shape()
        }
    }

    fn in_shape(&self) -> &[Idx; 3] {
        match self {
            SparseOrDense::Sparse(a) => a.in_shape(),
            SparseOrDense::Dense(a) => a.in_shape()
        }
    }

    fn kernel(&self) -> &[Idx; 2] {
        match self {
            SparseOrDense::Sparse(a) => a.kernel(),
            SparseOrDense::Dense(a) => a.kernel()
        }
    }

    fn stride(&self) -> &[Idx; 2] {
        match self {
            SparseOrDense::Sparse(a) => a.stride(),
            SparseOrDense::Dense(a) => a.stride()
        }
    }

    fn learnable_parameters(&self) -> usize {
        match self {
            SparseOrDense::Sparse(a) => a.learnable_parameters(),
            SparseOrDense::Dense(a) => a.learnable_parameters()
        }
    }

    fn get_max_incoming_synapses(&self) -> Idx {
        match self{
            SparseOrDense::Sparse(a) => a.get_max_incoming_synapses(),
            SparseOrDense::Dense(a) => a.get_max_incoming_synapses()
        }
    }

    fn get_threshold_f32(&self) -> f32 {
        match self{
            SparseOrDense::Sparse(a) => a.get_threshold_f32(),
            SparseOrDense::Dense(a) => a.get_threshold_f32()
        }
    }

    fn set_threshold_f32(&mut self, threshold: f32) {
        match self{
            SparseOrDense::Sparse(a) => a.set_threshold_f32(threshold),
            SparseOrDense::Dense(a) => a.set_threshold_f32(threshold)
        }
    }

    fn set_plasticity(&mut self, fractional: f32) {
        match self{
            SparseOrDense::Sparse(a) => a.set_plasticity(fractional),
            SparseOrDense::Dense(a) => a.set_plasticity(fractional)
        }
    }

    fn get_plasticity(&self) -> f32 {
        match self{
            SparseOrDense::Sparse(a) => a.get_plasticity(),
            SparseOrDense::Dense(a) => a.get_plasticity()
        }
    }

    fn new_empty_sdr(&self, capacity: u32) -> Self::A {
        match self {
            SparseOrDense::Sparse(a) => a.new_empty_sdr(capacity),
            SparseOrDense::Dense(a) => a.new_empty_sdr(capacity)
        }
    }

    fn run_in_place(&mut self, input: &Self::A, output: &mut Self::A) {
        match self {
            SparseOrDense::Sparse(a) => a.run_in_place(input, output),
            SparseOrDense::Dense(a) => a.run_in_place(input, output)
        }
    }

    fn infer_in_place(&mut self, input: &Self::A, output: &mut Self::A) {
        match self {
            SparseOrDense::Sparse(a) => a.infer_in_place(input, output),
            SparseOrDense::Dense(a) => a.infer_in_place(input, output)
        }
    }

    fn decrement_activities(&mut self, output: &Self::A) {
        match self {
            SparseOrDense::Sparse(a) => a.decrement_activities(output),
            SparseOrDense::Dense(a) => a.decrement_activities(output)
        }
    }

    fn learn(&mut self, input: &Self::A, output: &Self::A) {
        match self {
            SparseOrDense::Sparse(a) => a.learn(input, output),
            SparseOrDense::Dense(a) => a.learn(input, output)
        }
    }
}


#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EccMachine<A:SDR,S:EccLayer<A=A>,D:EccLayer<A=A>> {
    ecc: Vec<SparseOrDense<A,S,D>>,
    inputs: Vec<A>,
}

impl<A:SDR,S:EccLayer<A=A>,D:EccLayer<A=A>> Deref for EccMachine<A,S,D> {
    type Target = Vec<SparseOrDense<A,S,D>>;

    fn deref(&self) -> &Self::Target {
        &self.ecc
    }
}

impl<A:SDR,S:EccLayer<A=A>,D:EccLayer<A=A>> DerefMut for EccMachine<A,S,D>{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ecc
    }
}


impl<A:SDR,S:EccLayer<A=A>,D:EccLayer<A=A>> EccMachine<A,S,D> {
    pub fn new<R:Rng>(output: [Idx; 2], kernels: &[[Idx; 2]], strides: &[[Idx; 2]], channels: &[Idx],
               k: &[Idx], connections_per_output: &[Option<Idx>], rng: &mut R,
               mut new_layer:impl FnMut([Idx;2],Idx,Idx,Idx,[Idx;2],[Idx;2],Option<Idx>,&mut R)->SparseOrDense<A,S,D>) -> Self {
        let layers = kernels.len();

        assert!(layers > 0);
        assert_eq!(layers, strides.len());
        assert_eq!(layers, k.len());
        assert_eq!(layers, connections_per_output.len());
        assert_eq!(layers + 1, channels.len());
        let mut layers_vec = Vec::<SparseOrDense<A,S,D>>::with_capacity(layers);
        let mut prev_output = output;
        for i in (0..layers).rev() {
            let in_channels = channels[i];
            let out_channels = channels[i + 1];
            let k = k[i];
            let kernel = kernels[i];
            let stride = strides[i];
            let conn =  connections_per_output[i];
            let l = new_layer(prev_output,in_channels,out_channels,k,kernel,stride,conn,rng);
            prev_output = *l.in_shape().grid();
            layers_vec.push(l);
        }
        layers_vec.reverse();
        #[cfg(debug_assertions)] {
            let last = layers_vec.last().unwrap().out_shape();
            debug_assert!(last.grid().all_eq(&output), "{:?}=={:?}", last.grid(), output);
            debug_assert_eq!(last.channels(), *channels.last().unwrap());
            debug_assert_eq!(layers_vec[0].in_channels(), channels[0]);
            for (prev, next) in layers_vec.iter().tuple_windows() {
                debug_assert!(prev.out_shape().all_eq(next.in_shape()), "{:?}=={:?}", prev.out_shape(), next.in_shape());
            }
        }
        let mut inputs = vec![layers_vec.first().unwrap().new_empty_sdr(prev_output.product())];
        inputs.extend(layers_vec.iter().map(|l|l.new_empty_output_sdr()));
        assert_eq!(inputs.len(),channels.len());
        Self { ecc: layers_vec, inputs }
    }
    pub fn learnable_parameters(&self) -> usize {
        self.ecc.iter().map(|w| w.learnable_parameters()).sum()
    }
    pub fn input_sdr(&self, layer_index: usize) -> &A {
        &self.inputs[layer_index]
    }
    pub fn input_sdr_mut(&mut self, layer_index: usize) -> &mut A {
        &mut self.inputs[layer_index]
    }
    pub fn output_sdr(&self, layer_index: usize) -> &A {
        &self.inputs[layer_index + 1]
    }
    pub fn output_sdr_mut(&mut self, layer_index: usize) -> &mut A {
        &mut self.inputs[layer_index + 1]
    }
    pub fn last_output_sdr(&self) -> &A {
        self.inputs.last().unwrap()
    }
    pub fn last_output_sdr_mut(&mut self) -> &mut A {
        self.inputs.last_mut().unwrap()
    }
    pub fn learn(&mut self) {
        let Self { ecc, inputs } = self;
        for (i, layer) in ecc.iter_mut().enumerate() {
            let (prev, next) = inputs.as_slice().split_at(i + 1);
            layer.learn(&prev[i], &next[0]);
        }
    }
    pub fn run(&mut self, input: &A) -> &A {
        let Self { ecc, inputs } = self;
        inputs[0].set_from_sdr(input);
        for (i, layer) in ecc.iter_mut().enumerate() {
            let (prev, next) = inputs.as_mut_slice().split_at_mut(i + 1);
            layer.run_in_place(&prev[i], &mut next[0]);
        }
        self.last_output_sdr()
    }
    pub fn infer(&mut self, input: &A) -> &A {
        let Self { ecc, inputs } = self;
        inputs[0].set_from_sdr(input);
        for (i, layer) in ecc.iter_mut().enumerate() {
            let (prev, next) = inputs.as_mut_slice().split_at_mut(i + 1);
            layer.infer_in_place(&prev[i], &mut next[0]);
        }
        self.last_output_sdr()
    }
    pub fn in_shape(&self) -> &[Idx; 3] {
        self.ecc[0].in_shape()
    }
    pub fn in_channels(&self) -> Idx {
        self.ecc[0].in_channels()
    }
    pub fn in_volume(&self) -> Idx {
        self.ecc[0].in_volume()
    }
    pub fn out_shape(&self) -> &[Idx; 3] {
        self.ecc.last().unwrap().out_shape()
    }
    pub fn out_channels(&self) -> Idx {
        self.ecc.last().unwrap().out_channels()
    }
    pub fn out_volume(&self) -> Idx {
        self.ecc.last().unwrap().out_volume()
    }
}
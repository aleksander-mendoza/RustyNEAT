use crate::{SDR, EccLayer, CpuSDR, CpuEccDense, Idx, DenseWeight, ConvShape, VectorFieldOne, Shape3, VectorFieldPartialOrd};
use std::ops::{Index, IndexMut};
use serde::{Serialize, Deserialize};
use rand::Rng;
use itertools::Itertools;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EccMachine<A:SDR,L:EccLayer<A=A>> {
    ecc: Vec<L>,
    inputs: Vec<A>,
}
pub type CpuEccMachine<D> = EccMachine<CpuSDR,CpuEccDense<D>>;
impl<A:SDR,L:EccLayer<A=A>> Index<usize> for EccMachine<A,L>{
    type Output = L;

    fn index(&self, index: usize) -> &Self::Output {
        &self.ecc[index]
    }
}
impl<A:SDR,L:EccLayer<A=A>> IndexMut<usize> for EccMachine<A,L>{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.ecc[index]
    }
}

impl<A:SDR,L:EccLayer<A=A>> EccMachine<A,L> {
    pub fn len(&self)->usize{
        self.ecc.len()
    }
    pub fn last(&self)->&L{
        self.ecc.last().unwrap()
    }
    pub fn last_mut (&mut self)->&mut L{
        self.ecc.last_mut().unwrap()
    }
    pub fn layer_mut(&mut self,idx:usize)->&mut L{
        &mut self.ecc[idx]
    }
    pub fn push(&mut self,top:L){
        assert_eq!(self.out_shape(),top.shape().in_shape());
        self.inputs.push(top.new_empty_output_sdr());
        self.ecc.push(top);
    }
    pub fn pop(&mut self) -> Option<L> {
        if self.len()<=1{return None}
        self.inputs.pop();
        self.ecc.pop()
    }
    pub fn new_singleton(layer:L)->Self{
        Self{
            inputs: vec![layer.new_empty_sdr(layer.shape().in_volume()), layer.new_empty_output_sdr()],
            ecc: vec![layer],
        }
    }
    pub fn new<R:Rng>(output: [Idx; 2], kernels: &[[Idx; 2]], strides: &[[Idx; 2]], channels: &[Idx],
                      k: &[Idx], rng: &mut R,
                      mut new_layer:impl FnMut([Idx;2],Idx,Idx,Idx,[Idx;2],[Idx;2],&mut R)->L) -> Self {
        let layers = kernels.len();

        assert!(layers > 0);
        assert_eq!(layers, strides.len());
        assert_eq!(layers, k.len());
        assert_eq!(layers + 1, channels.len());
        let mut layers_vec = Vec::<L>::with_capacity(layers);
        let mut prev_output = output;
        for i in (0..layers).rev() {
            let in_channels = channels[i];
            let out_channels = channels[i + 1];
            let k = k[i];
            let kernel = kernels[i];
            let stride = strides[i];
            let l = new_layer(prev_output,in_channels,out_channels,k,kernel,stride,rng);
            prev_output = *l.shape().in_grid();
            layers_vec.push(l);
        }
        layers_vec.reverse();
        #[cfg(debug_assertions)] {
            let last = layers_vec.last().unwrap().shape().out_shape();
            debug_assert!(last.grid().all_eq(&output), "{:?}=={:?}", last.grid(), output);
            debug_assert_eq!(last.channels(), *channels.last().unwrap());
            debug_assert_eq!(layers_vec[0].shape().in_channels(), channels[0]);
            for (prev, next) in layers_vec.iter().tuple_windows() {
                debug_assert!(prev.shape().out_shape().all_eq(next.shape().in_shape()), "{:?}=={:?}", prev.shape().out_shape(), next.shape().in_shape());
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
    /**input_sdr(self.len()) returns the final output*/
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
    pub fn set_plasticity_f32_everywhere(&mut self, plasticity:f32){
        self.ecc.iter_mut().for_each(|l|l.set_plasticity_f32(plasticity))
    }
    pub fn last_output_sdr(&self) -> &A {
        self.inputs.last().unwrap()
    }
    pub fn last_output_sdr_mut(&mut self) -> &mut A {
        self.inputs.last_mut().unwrap()
    }
    pub fn learn(&mut self) {
        for i in 0..self.len(){
            self.learn_one_layer(i)
        }
    }
    pub fn learn_one_layer(&mut self,layer:usize) {
        let Self { ecc, inputs } = self;
        let (prev, next) = inputs.as_slice().split_at(layer + 1);
        ecc[layer].learn(&prev[layer], &next[0]);
    }
    /**last_layer to evaluate (exclusive)*/
    pub fn run_up_to_layer(&mut self, last_layer:usize,input: &A) -> &A {
        let Self { ecc, inputs } = self;
        inputs[0].set_from_sdr(input);
        for (i, layer) in ecc.iter_mut().take(last_layer).enumerate() {
            let (prev, next) = inputs.as_mut_slice().split_at_mut(i + 1);
            layer.run_in_place(&prev[i], &mut next[0]);
        }
        self.input_sdr(last_layer)
    }
    pub fn run(&mut self, input: &A) -> &A {
        self.run_up_to_layer(self.len(),input)
    }
    pub fn infer_up_to_layer(&mut self,last_layer:usize, input: &A) -> &A {
        let Self { ecc, inputs } = self;
        inputs[0].set_from_sdr(input);
        for (i, layer) in ecc.iter_mut().take(last_layer).enumerate() {
            let (prev, next) = inputs.as_mut_slice().split_at_mut(i + 1);
            layer.infer_in_place(&prev[i], &mut next[0]);
        }
        self.input_sdr(last_layer)
    }
    pub fn infer(&mut self, input: &A) -> &A {
        self.infer_up_to_layer(self.len(),input)
    }
    pub fn in_shape(&self) -> &[Idx; 3] {
        self.ecc[0].shape().in_shape()
    }
    pub fn in_channels(&self) -> Idx {
        self.ecc[0].shape().in_channels()
    }
    pub fn in_volume(&self) -> Idx {
        self.ecc[0].shape().in_volume()
    }
    pub fn out_shape(&self) -> &[Idx; 3] {
        self.ecc.last().unwrap().shape().out_shape()
    }
    pub fn out_channels(&self) -> Idx {
        self.ecc.last().unwrap().shape().out_channels()
    }
    pub fn out_volume(&self) -> Idx {
        self.ecc.last().unwrap().shape().out_volume()
    }
}

impl<D: DenseWeight> CpuEccMachine<D> {
    pub fn new_cpu(output: [Idx; 2], kernels: &[[Idx; 2]], strides: &[[Idx; 2]], channels: &[Idx], k: &[Idx], rng: &mut impl Rng) -> Self {
        Self::new(output, kernels, strides, channels, k, rng,
                  |output, in_channels, out_channels, k, kernel, stride, rng|
                CpuEccDense::new(ConvShape::new(output, kernel, stride, in_channels, out_channels), k, rng)
            )
    }
}
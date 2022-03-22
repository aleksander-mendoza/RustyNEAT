use crate::{HasEccConfig, SDR, EccLayer, CpuSDR, Idx, ConvShape, VectorFieldOne, Shape3, VectorFieldPartialOrd, Shape, ConvShapeTrait, HasConvShape, D, ConvTensorTrait, EccConfig, EccLayerTrait, Weight, HasEccConfigMut, WNorm, Activity};
use std::ops::{Index, IndexMut};
use serde::{Serialize, Deserialize};
use rand::Rng;
use itertools::Itertools;
use std::mem::MaybeUninit;
use crate::conv_tensor::ConvTensor;
use std::slice::{Iter, IterMut};
use num_traits::{Num, NumAssign};
use crate::vector_field_norm::Sqrt;

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct EccNet {
    layers: Vec<EccLayer>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct EccNetSDRs {
    pub layers: Vec<CpuSDR>,
}

impl EccNetSDRs {
    pub fn len(&self) -> usize {
        self.layers.len()
    }
    pub fn clear(&mut self) {
        self.layers.iter_mut().for_each(|l| l.clear())
    }
    pub fn new(length: usize) -> Self {
        Self {
            layers: (0..length).map(|_| CpuSDR::new()).collect()
        }
    }
}

impl EccNet {
    pub fn cfg(&self) -> Option<&EccConfig<D>> {
        self.first().map(|l|l.cfg())
    }
    pub fn get_cfg_bias(&mut self) -> Option<bool> {
        self.first().map(|l|l.cfg_biased())
    }
    pub fn get_cfg_entropy_maximisation(&mut self) -> Option<D> {
        self.first().map(|l|l.cfg_entropy_maximisation())
    }
    pub fn get_cfg_w_norm(&mut self) -> Option<WNorm> {
        self.first().map(|l|l.cfg_w_norm())
    }
    pub fn get_cfg_activity(&self) -> Option<Activity> {
        self.first().map(|l|l.cfg_activity())
    }
    pub fn set_cfg_bias(&mut self,bias:bool){
        self.layers.iter_mut().for_each(|l|*l.cfg_biased_mut()=bias)
    }
    pub fn set_cfg_entropy_maximisation(&mut self,v:D){
        self.layers.iter_mut().for_each(|l|*l.cfg_entropy_maximisation_mut()=v)
    }
    pub fn set_cfg_w_norm(&mut self,v:WNorm){
        self.layers.iter_mut().for_each(|l|*l.cfg_w_norm_mut()=v)
    }
    pub fn set_cfg_activity(&mut self,v:Activity){
        self.layers.iter_mut().for_each(|l|*l.cfg_activity_mut()=v)
    }
    pub fn layers(&self) -> &[EccLayer] {
        &self.layers
    }
    pub fn layers_mut(&mut self) -> &mut [EccLayer] {
        &mut self.layers
    }
    pub fn len(&self) -> usize {
        self.layers().len()
    }
    pub fn iter(&self) -> Iter<EccLayer> {
        self.layers().iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<EccLayer> {
        self.layers_mut().iter_mut()
    }
    pub fn last(&self) -> Option<&EccLayer> {
        self.layers().last()
    }
    pub fn first(&self) -> Option<&EccLayer> {
        self.layers().first()
    }
    pub fn first_mut(&mut self) -> Option<&mut EccLayer> {
        self.layers_mut().first_mut()
    }
    pub fn last_mut(&mut self) -> Option<&mut EccLayer> {
        self.layers_mut().last_mut()
    }
    pub fn layer_mut(&mut self, idx: usize) -> &mut EccLayer {
        &mut self.layers_mut()[idx]
    }
    pub fn set_plasticity_everywhere(&mut self,plasticity:D){
        self.layers_mut().iter_mut().for_each(|l|l.set_plasticity(plasticity))
    }
    pub fn layer(& self, idx: usize) -> &EccLayer {
        &self.layers()[idx]
    }
    pub fn composed_kernel_and_stride(&self) -> ([Idx; 2], [Idx; 2]) {
        self.composed_kernel_and_stride_up_to(self.len())
    }
    pub fn composed_kernel_and_stride_up_to(&self, idx: usize) -> ([Idx; 2], [Idx; 2]) {
        let (mut kernel, mut stride) = ([1; 2], [1; 2]);
        for l in self.layers().iter().take(idx) {
            (stride, kernel) = stride.conv_compose(&kernel, l.stride(), l.kernel());
        }
        (kernel, stride)
    }
    pub fn in_shape(&self) -> Option<&[Idx; 3]> {
        self.first().map(|f| f.in_shape())
    }
    pub fn in_channels(&self) -> Option<Idx> {
        self.first().map(|f| f.in_channels())
    }
    pub fn in_volume(&self) -> Option<Idx> {
        self.first().map(|f| f.in_volume())
    }
    pub fn in_grid(&self) -> Option<&[Idx; 2]> {
        self.first().map(|f| f.in_grid())
    }
    pub fn out_grid(&self) -> Option<&[Idx; 2]> {
        self.last().map(|f| f.out_grid())
    }
    pub fn out_shape(&self) -> Option<&[Idx; 3]> {
        self.last().map(|f| f.out_shape())
    }
    pub fn out_channels(&self) -> Option<Idx> {
        self.last().map(|f| f.out_channels())
    }
    pub fn out_volume(&self) -> Option<Idx> {
        self.last().map(|f| f.out_volume())
    }
    pub fn learnable_parameters(&self) -> usize {
        self.iter().map(|w| w.len()).sum()
    }
    pub fn learn(&mut self, input: &CpuSDR, output: &EccNetSDRs) {
        assert!(output.layers.len() >= self.len(), "Number of layers doesn't match number of SDRs");
        let mut iter = self.iter_mut().enumerate();
        if let Some((i, layer)) = iter.next() {
            layer.learn(input, &output.layers[i]);
            for (i, layer) in iter {
                layer.learn(&output.layers[i - 1], &output.layers[i]);
            }
        }
    }
    pub fn infer(&mut self, input: &CpuSDR, output: &mut EccNetSDRs, learn: bool) {
        assert!(output.layers.len() >= self.len(), "Number of layers doesn't match number of SDRs");
        let mut iter = self.iter_mut().enumerate();
        if let Some((i, layer)) = iter.next() {
            layer.infer(input, &mut output.layers[i], learn);
            for (i, layer) in iter {
                let (pre, post) = output.layers.split_at_mut(i);
                layer.infer(&pre[i - 1], &mut post[0], learn);
            }
        }
    }
    pub fn infer_rotating(&mut self, input: &CpuSDR, output: &mut EccNetSDRs, learn: bool) -> usize {
        assert!(output.len() >= 2, "At least two SDRs are needed");
        let mut iter = self.iter_mut();
        if let Some(layer) = iter.next() {
            let (sdr0, sdr1) = output.layers.split_at_mut(1);
            let mut sdr0 = &mut sdr0[0];
            let mut sdr1 = &mut sdr1[0];
            layer.infer(input, sdr0, learn);
            for layer in iter {
                layer.infer(sdr0, sdr1, learn);
                std::mem::swap(&mut sdr0, &mut sdr1);
            }
            (self.len() - 1) % 2
        } else {
            output.len()
        }
    }
    pub fn layers_vec(&self) -> &Vec<EccLayer> {
        &self.layers
    }
    pub fn layers_vec_mut(&mut self) -> &mut Vec<EccLayer> {
        &mut self.layers
    }
    pub fn cfg_compatible(&self, other:&impl HasEccConfig<D>)->bool{
        self.cfg().map(|c|c.compatible(other.cfg())).unwrap_or(true)
    }
    pub fn push(&mut self, top: EccLayer) {
        assert!(self.cfg_compatible(&top),"Config not compatible");
        if let Some(l) = self.out_shape() {
            assert_eq!(l, top.in_shape());
        }
        self.layers_vec_mut().push(top);
    }
    pub fn prepend(&mut self, bottom: EccLayer) {
        assert!(self.cfg_compatible(&bottom),"Config not compatible");
        if let Some(l) = self.in_shape() {
            assert_eq!(l, bottom.out_shape());
        }
        self.layers_vec_mut().insert(0, bottom);
    }
    pub fn pop(&mut self) -> Option<EccLayer> {
        self.layers_vec_mut().pop()
    }
    pub fn pop_front(&mut self) -> Option<EccLayer> {
        if self.len() > 0 {
            Some(self.layers_vec_mut().remove(0))
        } else {
            None
        }
    }
    pub fn push_repeated_column(&mut self, top: &EccLayer, column_pos: [Idx; 2]) where EccLayer: Clone {
        assert!(self.cfg_compatible(top),"Config not compatible");
        self.push(if let Some(input) = self.out_shape() {
            let output = input.grid().conv_out_size(top.stride(), top.kernel());
            top.repeat_column(output, column_pos)
        } else {
            top.clone()
        })
    }
    pub fn prepend_repeated_column(&mut self, bottom: &EccLayer, column_pos: [Idx; 2]) where EccLayer: Clone {
        assert!(self.cfg_compatible(bottom),"Config not compatible");
        self.prepend(if let Some(output) = self.in_shape() {
            bottom.repeat_column(*output.grid(), column_pos)
        } else {
            bottom.clone()
        })
    }

    pub fn empty() -> Self {
        Self::default()
    }
    pub fn new(cfg: EccConfig<D>, mut output: [Idx; 2], kernels: &[[Idx; 2]], strides: &[[Idx; 2]], channels: &[Idx], k: &[Idx], rng: &mut impl Rng) -> Self {
        let layers_count = kernels.len();
        assert!(layers_count > 0);
        assert_eq!(layers_count, strides.len());
        assert_eq!(layers_count, k.len());
        assert_eq!(layers_count + 1, channels.len());
        let mut layers = Vec::<EccLayer>::with_capacity(layers_count);
        for i in (0..layers_count).rev() {
            let in_channels = channels[i];
            let out_channels = channels[i + 1];
            let k = k[i];
            let kernel = kernels[i];
            let stride = strides[i];
            let shape = ConvShape::new(output, kernel, stride, in_channels, out_channels);
            let l = EccLayer::new(cfg.clone(), shape, k, rng);
            output = *l.in_grid();
            layers.push(l);
        }
        layers.reverse();
        Self { layers }
    }
    pub fn from_repeated_column(mut final_column_grid: [Idx; 2], pretrained: &Self) -> Self {
        let mut layers = vec![];
        for layer in pretrained.iter().rev() {
            let l = layer.repeat_column(final_column_grid, [0, 0]);
            final_column_grid = *l.in_grid();
            layers.push(l);
        }
        layers.reverse();
        Self { layers }
    }
}


impl From<EccLayer> for EccNet {
    fn from(l: EccLayer) -> Self {
        let mut slf = Self::empty();
        slf.push(l);
        slf
    }
}

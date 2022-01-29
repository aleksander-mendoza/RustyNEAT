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
use crate::{Shape, resolve_range, EncoderTarget, Synapse, top_large_k_indices, top_small_k_indices, Shape3, from_xyz, Shape2, from_xy, range_contains, DenseWeight, w_idx, kernel_column_weight_sum, debug_assert_approx_eq_weight, EccLayerD, kernel_column_dropped_weights_count, ConvShape, ConvWeights, CpuEccMachine};
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
use rand::prelude::SliceRandom;
use crate::cpu_ecc_population::CpuEccPopulation;


#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct CpuEccDense<D: DenseWeight> {
    w: ConvWeights<D>,
    p: CpuEccPopulation<D>,
}

impl<D: DenseWeight> Deref for CpuEccDense<D> {
    type Target = ConvWeights<D>;

    fn deref(&self) -> &Self::Target {
        self.weights()
    }
}

impl<D: DenseWeight> DerefMut for CpuEccDense<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.weights_mut()
    }
}
impl<D: DenseWeight + Send + Sync> CpuEccDense<D> {
    pub fn parallel_run(&mut self, input: &CpuSDR) -> CpuSDR {
        let Self{w,p} = self;
        w.parallel_run(input, p)
    }
    pub fn parallel_run_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR) {
        let Self{w,p} = self;
        w.parallel_run_in_place(input, output, p)
    }
    pub fn parallel_infer(&mut self, input: &CpuSDR) -> CpuSDR {
        let Self{w,p} = self;
        w.parallel_infer(input, p)
    }
    pub fn parallel_infer_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR) {
        let Self{w,p} = self;
        w.parallel_infer_in_place(input, output,p)
    }
}
impl<D: DenseWeight> CpuEccDense<D> {
    pub fn into_machine(self) -> CpuEccMachine<D> {
        CpuEccMachine::new_singleton(self)
    }
    pub fn weights(&self) -> &ConvWeights<D> {
        &self.w
    }
    pub fn population(&self) -> &CpuEccPopulation<D> {
        &self.p
    }
    pub fn weights_mut(&mut self) -> &mut ConvWeights<D> {
        &mut self.w
    }
    pub fn population_mut(&mut self) -> &mut CpuEccPopulation<D> {
        &mut self.p
    }
    pub fn from_repeated_column(output: [Idx; 2], pretrained: &Self, pretrained_column_pos: [Idx; 2]) -> Self {
        let w = ConvWeights::from_repeated_column(output, pretrained.weights(), pretrained_column_pos);
        let p = CpuEccPopulation::from_repeated_column(output, pretrained.population(), pretrained_column_pos);
        Self { w, p }
    }
    pub fn new(shape: ConvShape, k: Idx, rng: &mut impl Rng) -> Self {
        Self { p: CpuEccPopulation::new(shape.output_shape(), k), w: ConvWeights::new(shape, rng) }
    }
    pub fn from(weights:ConvWeights<D>, pop:CpuEccPopulation<D>) -> Self {
        assert_eq!(weights.out_shape(),pop.shape());
        Self { p:pop, w:weights}
    }
    pub fn concat<T>(layers: &[T], f: impl Fn(&T) -> &Self) -> Self {
        let w = ConvWeights::concat(layers, |p| f(p).weights());
        let p = CpuEccPopulation::concat(layers, |p| f(p).population());
        Self { w, p }
    }
}

impl<D: DenseWeight> EccLayerD for CpuEccDense<D> {
    type D = D;
    fn get_threshold(&self) -> D {
        self.population().get_threshold()
    }

    fn set_threshold(&mut self, threshold: D) {
        self.population_mut().set_threshold(threshold)
    }

    fn get_plasticity(&self) -> D {
        ConvWeights::get_plasticity(self)
    }
    fn set_plasticity(&mut self, plasticity: D) {
        ConvWeights::set_plasticity(self,plasticity)
    }
}
impl<D: DenseWeight> EccLayer for CpuEccDense<D> {
    type A = CpuSDR;

    fn k(&self) -> Idx { self.population().k() }

    fn set_k(&mut self, k: Idx) {
        self.population_mut().set_k(k)
    }

    fn shape(&self) -> &ConvShape {
        self.weights().shape()
    }

    fn shape_mut(&mut self) -> &mut ConvShape {
        self.weights_mut().shape_mut()
    }

    fn learnable_parameters(&self) -> usize {
        self.weights().len()
    }
    fn get_max_incoming_synapses(&self) -> Idx {
        self.kernel_column_volume()
    }
    fn get_threshold_f32(&self) -> f32 {
        self.population().get_threshold_f32()
    }

    fn set_threshold_f32(&mut self, fractional: f32) {
        self.population_mut().set_threshold_f32(fractional)
    }

    fn set_plasticity_f32(&mut self, fractional: f32) {
        self.plasticity = D::f32_to_w(fractional)
    }
    fn get_plasticity_f32(&self) -> f32 {
        D::w_to_f32(self.plasticity)
    }

    fn new_empty_sdr(&self, capacity: Idx) -> Self::A {
        CpuSDR::new()
    }

    fn infer_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR) {
        let Self{w,p} = self;
        w.infer_in_place(input, output,p)
    }

    fn decrement_activities(&mut self, output: &CpuSDR) {
        self.population_mut().decrement_activities(output)
    }

    fn learn(&mut self, input: &CpuSDR, output: &CpuSDR) {
        self.weights_mut().learn(input, output);
        #[cfg(debug_assertions)] {
            let min_acc = self.population().min_activity();
            for output_idx in 0..self.out_volume() {
                debug_assert!(self.population().activity[as_usize(output_idx)].lt(min_acc + D::TOTAL_SUM), "{} @ {} < {}", output_idx, self.population().activity[as_usize(output_idx)], min_acc)
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use std::cmp::Ordering::{Greater, Less};

    #[test]
    fn test2() -> Result<(), String> {
        test2_::<u32>()
    }

    #[test]
    fn test3() -> Result<(), String> {
        test3_::<u32>()
    }

    #[test]
    fn test4() -> Result<(), String> {
        test4_::<u32>()
    }

    #[test]
    fn test5() -> Result<(), String> {
        test5_::<u32>()
    }

    #[test]
    fn test6() -> Result<(), String> {
        test6_::<u32>()
    }

    #[test]
    fn test7() -> Result<(), String> {
        test7_::<u32>()
    }

    #[test]
    fn test2f() -> Result<(), String> {
        test2_::<f32>()
    }

    #[test]
    fn test3f() -> Result<(), String> {
        test3_::<f32>()
    }

    #[test]
    fn test4f() -> Result<(), String> {
        test4_::<f32>()
    }

    #[test]
    fn test5f() -> Result<(), String> {
        test5_::<f32>()
    }

    #[test]
    fn test6f() -> Result<(), String> {
        test6_::<f32>()
    }

    #[test]
    fn test7f() -> Result<(), String> {
        test7_::<f32>()
    }

    fn test2_<D: DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 8;
        let mut a = CpuEccDense::<D>::new(ConvShape::new([4, 4], [2, 2], [1, 1], 3, 4), 1, &mut rng);
        for _ in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            let mut o = a.run(&input);
            a.learn(&input, &o);
            o.sort();
            assert!(o.is_normalized(), "{:?}", o);
        }
        Ok(())
    }

    fn test3_<D: DenseWeight>() -> Result<(), String> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(634634636);//rand::thread_rng();
        let k = 8;
        let mut a = CpuEccDense::<D>::new(ConvShape::new([4, 4], [2, 2], [1, 1], 3, 4), 1, &mut rng);
        a.set_plasticity_f32(0.1);//let's see if this breaks anything
        for i in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            let mut o = a.run(&input);
            a.learn(&input, &o);
            o.sort();
            assert!(o.is_normalized(), "{:?}", o);
        }
        Ok(())
    }

    fn test4_<D: DenseWeight>() -> Result<(), String> {
        let s = auto_gen_seed64();
        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
        let k = 16;
        let mut a = CpuEccDense::<D>::new(ConvShape::new([1, 1], [4, 4], [1, 1], 3, 4), 1, &mut rng);
        a.set_threshold_f32(0.2);
        for i in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            let mut o = a.run(&input);
            a.learn(&input, &o);
            if o.len() == 0 {
                assert!(a.population().sums.iter().all(|&x| x.lt(a.population().threshold)), "{:?}", a.population().sums);
                // println!("(a[{}]=={} < {}) + {}",argmax,a.sums[argmax],a.threshold,a.activity[argmax]);
                // println!("{:?}",a.sums);
                // println!("{:?}",a.activity);
                // println!("{:?}",a.sums.iter().zip(a.activity.iter()).map(|(&a,&b)|a+b).collect::<Vec<D>>());
            } else {
                o.sort();
                assert!(o.is_normalized(), "{:?}", o);
            }
        }
        Ok(())
    }

    fn test5_<D: DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 16;
        let mut a = CpuEccDense::<D>::new(ConvShape::new([1, 1], [4, 4], [1, 1], 3, 4), 1, &mut rng);
        a.set_threshold_f32(0.99);//let's see if this breaks anything
        for _ in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            let mut o = a.run(&input);
            a.learn(&input, &o);
            assert_eq!(o.len(), 0);
            o.sort();
            assert!(o.is_normalized(), "{:?}", o);
        }
        Ok(())
    }

    fn test6_<D: DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 16;
        let mut a = CpuEccMachine::<D>::new_cpu([1, 1],
                                                &[[5, 5], [3, 3], [3, 3]],
                                                &[[2, 2], [1, 1], [1, 1]],
                                                &[1, 20, 20, 20],
                                                &[1, 1, 1],
                                                &mut rng);
        for _ in 0..16 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume().unwrap() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            a.run(&input);
            a.learn();
            let o = a.last_output_sdr_mut();
            assert_ne!(o.len(), 0);
            o.sort();
            assert!(o.is_normalized(), "{:?}", o);
        }
        Ok(())
    }


    fn test7_<D: DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 1;
        let mut a = CpuEccMachine::<D>::new_cpu([1, 1],
                                                &[[5, 5], [3, 3], [3, 3]],
                                                &[[2, 2], [1, 1], [1, 1]],
                                                &[1, 5, 2, 2],
                                                &[1, 1, 1],
                                                &mut rng);
        let mut number_of_empty_outputs = 0;
        for _ in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume().unwrap() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            a.run(&input);
            a.learn();
            let o = a.last_output_sdr_mut();
            if o.is_empty() {
                number_of_empty_outputs += 1;
            }
            o.sort();
            assert!(o.is_normalized(), "{:?}", o);
        }
        assert!(number_of_empty_outputs < 54, "{}", number_of_empty_outputs);
        Ok(())
    }

    #[test]
    fn test8() -> Result<(), String> {
        test8_::<u32>()
    }

    #[test]
    fn test8f() -> Result<(), String> {
        test8_::<f32>()
    }

    fn test8_<D: DenseWeight>() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 1;
        let kernel = [3, 3];
        let stride = [1, 1];
        let in_channels = 4;
        let out_channels = [2, 3, 6];
        let mut a: Vec<CpuEccDense<D>> = out_channels.iter().map(|&out_channels|
            CpuEccDense::new(ConvShape::new([1, 1], kernel, stride, in_channels, out_channels), k, &mut rng)).collect();
        let mut c = CpuEccDense::concat(&a, |a| a);

        let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a[0].in_volume() as u32)).collect();
        let mut input = CpuSDR::from(input);
        input.normalize();
        assert_ne!(input.len(), 0);
        let mut output = CpuSDR::new();
        for a in a.iter_mut().rev() {
            a.run_in_place(&input, &mut output);
            output.shift(a.out_volume() as i32);
        }
        output.sort();
        let mut output2 = c.run(&input);
        output2.sort();
        assert_eq!(output, output2);
        Ok(())
    }


    #[test]
    fn test9() -> Result<(), String> {
        test9_::<u32>()
    }

    #[test]
    fn test9f() -> Result<(), String> {
        test9_::<f32>()
    }

    fn test9_<D: DenseWeight>() -> Result<(), String> {
        let s = auto_gen_seed64();
        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
        let k = 16;
        let mut a = CpuEccDense::<D>::new(ConvShape::new([1, 1], [4, 4], [1, 1], 3, 4), 1, &mut rng);
        a.set_threshold_f32(0.2);
        let mut a2 = a.clone();
        println!("seed=={}", s);
        assert_eq!(a2.k(), 1);
        assert_eq!(a.k(), 1);
        assert!(a.get_threshold().eq(a2.get_threshold()), "threshold {:?}!={:?}", a.get_threshold(), a2.get_threshold());
        for i in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            assert!(a.weights().weight_slice().iter().zip(a2.weights().weight_slice().iter()).all(|(a, b)| a.eq(*b)), "w {:?}!={:?}", a.weight_slice(), a2.weight_slice());
            assert!(a.population().sums.iter().zip(a2.population().sums.iter()).all(|(a, b)| a.eq(*b)), "sums {:?}!={:?}", a.population().sums, a2.population().sums);
            assert!(a.population().activity.iter().zip(a2.population().activity.iter()).all(|(a, b)| a.eq(*b)), "activity {:?}!={:?}", a.population().activity, a2.population().activity);
            let mut o = a.run(&input);
            let mut o2 = a2.run(&input);
            assert_eq!(o, o2, "outputs i=={}", i);
            a.learn(&input, &o);
            a2.learn(&input, &o2);
        }
        Ok(())
    }


    #[test]
    fn test10() -> Result<(), String> {
        test10_::<u32>()
    }

    #[test]
    fn test10f() -> Result<(), String> {
        test10_::<f32>()
    }

    fn test10_<D: DenseWeight>() -> Result<(), String> {
        let s = auto_gen_seed64();
        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
        let k = 16;
        let mut a = CpuEccDense::<D>::new(ConvShape::new([2, 2], [4, 4], [1, 1], 3, 4), 1, &mut rng);
        a.set_threshold_f32(0.2);
        let mut a2 = a.clone();
        println!("seed=={}", s);
        assert_eq!(a2.k(), 1);
        assert_eq!(a.k(),1);
        assert!(a.get_threshold().eq(a2.get_threshold()), "threshold {:?}!={:?}", a.get_threshold(), a2.get_threshold());
        for i in 0..1024 {
            let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
            let mut input = CpuSDR::from(input);
            input.normalize();
            assert_ne!(input.len(), 0);
            assert!(a.weights().weight_slice().iter().zip(a2.weights().weight_slice().iter()).all(|(a, b)| a.eq(*b)), "w {:?}!={:?}", a.weights().weight_slice(), a2.weights().weight_slice());
            assert!(a.population().sums.iter().zip(a2.population().sums.iter()).all(|(a, b)| a.eq(*b)), "sums {:?}!={:?}", a.population().sums, a2.population().sums);
            assert!(a.population().activity.iter().zip(a2.population().activity.iter()).all(|(a, b)| a.eq(*b)), "activity {:?}!={:?}", a.population().activity, a2.population().activity);
            let mut o = a.run(&input);
            let mut o2 = a2.run(&input);
            assert_eq!(o, o2, "outputs i=={}", i);
            a.learn(&input, &o);
            a2.learn(&input, &o2);
        }
        Ok(())
    }
}
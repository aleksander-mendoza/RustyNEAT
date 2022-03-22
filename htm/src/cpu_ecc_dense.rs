// use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
// use std::mem::MaybeUninit;
// use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, RangeBounds, Deref, DerefMut};
// use std::fmt::{Display, Formatter, Debug};
// use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
// use crate::cpu_sdr::CpuSDR;
// use crate::ecc_program::EccProgram;
// use ndalgebra::buffer::Buffer;
// use crate::cpu_bitset::CpuBitset;
// use std::cmp::Ordering;
// use serde::{Serialize, Deserialize};
// use crate::{Shape, resolve_range, EncoderTarget, Synapse, top_large_k_indices, top_small_k_indices, Shape3, from_xyz, Shape2, from_xy, range_contains, DenseWeight, w_idx, debug_assert_approx_eq_weight, kernel_column_dropped_weights_count, ConvShape, ConvTensor, CpuEccMachine, ConvShapeTrait, HasConvShape, HasConvShapeMut, DenseWeightL2, Metric, D, HasShape, HasThreshold, HasPlasticity, HasK, ForwardToTarget, HasActivity};
// use std::collections::{Bound, HashSet};
// use crate::vector_field::{VectorFieldOne, VectorFieldDiv, VectorFieldAdd, VectorFieldMul, ArrayCast, VectorFieldSub, VectorFieldPartialOrd};
// use crate::population::Population;
// use rand::{Rng, SeedableRng};
// use crate::xorshift::{auto_gen_seed64, xorshift64, auto_gen_seed, xorshift, xorshift32, auto_gen_seed32};
// use itertools::{Itertools, assert_equal};
// use std::iter::Sum;
// use ocl::core::DeviceInfo::MaxConstantArgs;
// use crate::ecc::{EccLayer, Idx, as_idx, Rand, xorshift_rand};
// use crate::sdr::SDR;
// use rand::prelude::SliceRandom;
// use crate::cpu_ecc_population::CpuEccPopulation;
// use crate::as_usize::AsUsize;
//
//
// #[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
// pub struct CpuEccDense<M:?Sized + Metric<D>> {
//     w: ConvTensor<M>,
//     p: CpuEccPopulation<M>,
// }
//
// impl<M:Metric<D>> Deref for CpuEccDense<M> {
//     type Target = ConvTensor<M>;
//
//     fn deref(&self) -> &Self::Target {
//         self.weights()
//     }
// }
//
// impl<M:Metric<D>> DerefMut for CpuEccDense<M> {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         self.weights_mut()
//     }
// }
// impl<M:Metric<D>> CpuEccDense<M> {
//     pub fn batch_infer<T,O:Send>(&self, input: &[T], f:impl Fn(&T)->&CpuSDR+Send+Sync, of:impl Fn(CpuSDR)->O+Sync) -> Vec<O>{
//         let Self{w,p} = self;
//         w.batch_infer(input, f,p.clone(),of)
//     }
//     pub fn batch_infer_and_measure_s_expectation<T,O:Send>(&self, input: &[T], f:impl Fn(&T)->&CpuSDR+Send+Sync, of:impl Fn(CpuSDR)->O+Sync) -> (Vec<O>, D, u32) {
//         let Self{w,p} = self;
//         w.batch_infer_and_measure_s_expectation(input, f,p.clone(),of)
//     }
//     pub fn into_machine(self) -> CpuEccMachine<M> {
//         CpuEccMachine::new_singleton(self)
//     }
//     pub fn weights(&self) -> &ConvTensor<M> {
//         &self.w
//     }
//     pub fn population(&self) -> &CpuEccPopulation<M> {
//         &self.p
//     }
//     pub fn weights_mut(&mut self) -> &mut ConvTensor<M> {
//         &mut self.w
//     }
//     pub fn population_mut(&mut self) -> &mut CpuEccPopulation<M> {
//         &mut self.p
//     }
//     pub fn from_repeated_column(output: [Idx; 2], pretrained: &Self, pretrained_column_pos: [Idx; 2]) -> Self {
//         let w = ConvTensor::from_repeated_column(output, pretrained.weights(), pretrained_column_pos);
//         let p = CpuEccPopulation::from_repeated_column(output, pretrained.population(), pretrained_column_pos);
//         Self { w, p }
//     }
//     pub fn new(shape: ConvShape, k: Idx, rng: &mut impl Rng) -> Self {
//         Self { p: CpuEccPopulation::new(shape.output_shape(), k), w: ConvTensor::new(shape, rng) }
//     }
//     pub fn from(weights: ConvTensor<M>, pop: CpuEccPopulation<M>) -> Self {
//         assert_eq!(weights.out_shape(), pop.shape());
//         Self { p: pop, w: weights }
//     }
// }
// impl<'a, M:Metric<D> + 'a> CpuEccDense<M> {
//     pub fn concat< T,F:'a + Fn(&'a T) -> &'a Self>(layers: &'a [T], f:  F) -> Self {
//         let w = ConvTensor::concat(layers, |p| f(p).weights());
//         let p = CpuEccPopulation::concat(layers, |p| f(p).population());
//         Self { w, p }
//     }
// }
//
// impl<M: Metric<D>> HasShape for CpuEccDense<M> {
//     fn shape(&self) -> &[u32; 3] {
//         self.weights().shape()
//     }
// }
// impl<M:Metric<D>> HasConvShape for CpuEccDense<M>{
//     fn cshape(&self) -> &ConvShape {
//         self.weights().cshape()
//     }
// }
// impl<M:Metric<D>> HasConvShapeMut for CpuEccDense<M>{
//     fn cshape_mut(&mut self) -> &mut ConvShape {
//         self.weights_mut().cshape_mut()
//     }
// }
// impl<M:Metric<D>> EccLayer for CpuEccDense<M> {
//     type A = CpuSDR;
//     type D = D;
//     fn get_threshold(&self) -> D {
//         self.population().get_threshold()
//     }
//
//     fn set_threshold(&mut self, threshold: D) {
//         self.population_mut().set_threshold(threshold)
//     }
//
//     fn get_plasticity(&self) -> D {
//         ConvTensor::get_plasticity(self)
//     }
//     fn set_plasticity(&mut self, plasticity: D) {
//         ConvTensor::set_plasticity(self, plasticity)
//     }
//
//     fn k(&self) -> Idx { self.population().k() }
//
//     fn set_k(&mut self, k: Idx) {
//         self.population_mut().set_k(k)
//     }
//
//     fn learnable_parameters(&self) -> usize {
//         self.weights().len()
//     }
//     fn get_max_incoming_synapses(&self) -> Idx {
//         self.kernel_column_volume()
//     }
//
//     fn new_empty_sdr(&self, capacity: Idx) -> Self::A {
//         CpuSDR::new()
//     }
//
//     fn infer_in_place(&mut self, input: &CpuSDR, output: &mut CpuSDR) {
//         let Self{w,p} = self;
//         w.infer_in_place(input, output,p)
//     }
//
//     fn decrement_activities(&mut self, output: &CpuSDR) {
//         self.population_mut().decrement_activities(output)
//     }
//     fn learn(&mut self, input: &CpuSDR, output: &CpuSDR) {
//         self.weights_mut().learn(input, output);
//         #[cfg(debug_assertions)] {
//             let min_acc = self.population().min_activity();
//             for output_idx in 0..self.out_volume() {
//                 debug_assert!(self.population().activity[output_idx.as_usize()]<min_acc + D::ONE, "{} @ {} < {}", output_idx, self.population().activity[output_idx.as_usize()], min_acc)
//             }
//         }
//     }
//
// }
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use rand::SeedableRng;
//     use std::cmp::Ordering::{Greater, Less};
//     use crate::{MetricL1, MetricL2};
//
//     #[test]
//     fn test2() -> Result<(), String> {
//         test2_::<MetricL2>()
//     }
//
//     #[test]
//     fn test3() -> Result<(), String> {
//         test3_::<MetricL2>()
//     }
//
//     #[test]
//     fn test4() -> Result<(), String> {
//         test4_::<MetricL2>()
//     }
//
//     #[test]
//     fn test5() -> Result<(), String> {
//         test5_::<MetricL2>()
//     }
//
//     #[test]
//     fn test6() -> Result<(), String> {
//         test6_::<MetricL2>()
//     }
//
//     #[test]
//     fn test7() -> Result<(), String> {
//         test7_::<MetricL2>()
//     }
//     #[test]
//     fn test8() -> Result<(), String> {
//         test8_::<MetricL2>()
//     }
//
//     #[test]
//     fn test9() -> Result<(), String> {
//         test9_::<MetricL2>()
//     }
//     #[test]
//     fn test2f() -> Result<(), String> {
//         test2_::<MetricL2>()
//     }
//
//     #[test]
//     fn test3f() -> Result<(), String> {
//         test3_::<MetricL2>()
//     }
//
//     #[test]
//     fn test4f() -> Result<(), String> {
//         test4_::<MetricL1>()
//     }
//
//     #[test]
//     fn test5f() -> Result<(), String> {
//         test5_::<MetricL1>()
//     }
//
//     #[test]
//     fn test6f() -> Result<(), String> {
//         test6_::<MetricL1>()
//     }
//
//     #[test]
//     fn test7f() -> Result<(), String> {
//         test7_::<MetricL1>()
//     }
//     #[test]
//     fn test8f() -> Result<(), String> {
//         test8_::<MetricL1>()
//     }
//
//     #[test]
//     fn test9f() -> Result<(), String> {
//         test9_::<MetricL1>()
//     }
//     #[test]
//     fn test10f() -> Result<(), String> {
//         test10_::<MetricL1>()
//     }
//
//     fn test2_<M:Metric<D>>() -> Result<(), String> {
//         let mut rng = rand::thread_rng();
//         let k = 8;
//         let mut a = CpuEccDense::<M>::new(ConvShape::new([4, 4], [2, 2], [1, 1], 3, 4), 1, &mut rng);
//         for _ in 0..1024 {
//             let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
//             let mut input = CpuSDR::from(input);
//             input.normalize();
//             assert_ne!(input.len(), 0);
//             let mut o = a.run(&input);
//             a.learn(&input, &o);
//             o.sort();
//             assert!(o.is_normalized(), "{:?}", o);
//         }
//         Ok(())
//     }
//
//     fn test3_<M:Metric<D>>() -> Result<(), String> {
//         let mut rng = rand::rngs::StdRng::seed_from_u64(634634636);//rand::thread_rng();
//         let k = 8;
//         let mut a = CpuEccDense::<M>::new(ConvShape::new([4, 4], [2, 2], [1, 1], 3, 4), 1, &mut rng);
//         a.set_plasticity(0.1);//let's see if this breaks anything
//         for i in 0..1024 {
//             let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
//             let mut input = CpuSDR::from(input);
//             input.normalize();
//             assert_ne!(input.len(), 0);
//             let mut o = a.run(&input);
//             a.learn(&input, &o);
//             o.sort();
//             assert!(o.is_normalized(), "{:?}", o);
//         }
//         Ok(())
//     }
//
//     fn test4_<M:Metric<D>>() -> Result<(), String> {
//         let s = auto_gen_seed64();
//         let mut rng = rand::rngs::StdRng::seed_from_u64(s);
//         let k = 16;
//         let mut a = CpuEccDense::<M>::new(ConvShape::new([1, 1], [4, 4], [1, 1], 3, 4), 1, &mut rng);
//         a.set_threshold(0.2);
//         for i in 0..1024 {
//             let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
//             let mut input = CpuSDR::from(input);
//             input.normalize();
//             assert_ne!(input.len(), 0);
//             let mut o = a.run(&input);
//             a.learn(&input, &o);
//             if o.len() == 0 {
//                 assert!(a.population().sums.iter().all(|&x| x <a.population().threshold), "{:?}", a.population().sums);
//                 // println!("(a[{}]=={} < {}) + {}",argmax,a.sums[argmax],a.threshold,a.activity[argmax]);
//                 // println!("{:?}",a.sums);
//                 // println!("{:?}",a.activity);
//                 // println!("{:?}",a.sums.iter().zip(a.activity.iter()).map(|(&a,&b)|a+b).collect::<Vec<D>>());
//             } else {
//                 o.sort();
//                 assert!(o.is_normalized(), "{:?}", o);
//             }
//         }
//         Ok(())
//     }
//
//     fn test5_<M:Metric<D>>() -> Result<(), String> {
//         let mut rng = rand::thread_rng();
//         let k = 16;
//         let mut a = CpuEccDense::<M>::new(ConvShape::new([1, 1], [4, 4], [1, 1], 3, 4), 1, &mut rng);
//         a.set_threshold(0.99);//let's see if this breaks anything
//         for _ in 0..1024 {
//             let mut input = CpuSDR::rand(k,a.in_volume());
//             input.normalize();
//             assert_ne!(input.len(), 0);
//             let mut o = a.run(&input);
//             a.learn(&input, &o);
//             assert_eq!(o.len(), 0);
//             o.sort();
//             assert!(o.is_normalized(), "{:?}", o);
//         }
//         Ok(())
//     }
//
//     fn test6_<M:Metric<D>>() -> Result<(), String> {
//         let mut rng = rand::thread_rng();
//         let k = 16;
//         let mut a = CpuEccMachine::<M>::new_cpu([1, 1],
//                                                 &[[5, 5], [3, 3], [3, 3]],
//                                                 &[[2, 2], [1, 1], [1, 1]],
//                                                 &[1, 20, 20, 20],
//                                                 &[1, 1, 1],
//                                                 &mut rng);
//         for _ in 0..16 {
//             let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume().unwrap() as u32)).collect();
//             let mut input = CpuSDR::from(input);
//             input.normalize();
//             assert_ne!(input.len(), 0);
//             a.run(&input);
//             a.learn();
//             let o = a.last_output_sdr_mut();
//             assert_ne!(o.len(), 0);
//             o.sort();
//             assert!(o.is_normalized(), "{:?}", o);
//         }
//         Ok(())
//     }
//
//
//     fn test7_<M:Metric<D>>() -> Result<(), String> {
//         let mut rng = rand::thread_rng();
//         let k = 1;
//         let mut a = CpuEccMachine::<M>::new_cpu([1, 1],
//                                                 &[[5, 5], [3, 3], [3, 3]],
//                                                 &[[2, 2], [1, 1], [1, 1]],
//                                                 &[1, 5, 2, 2],
//                                                 &[1, 1, 1],
//                                                 &mut rng);
//         let mut number_of_empty_outputs = 0;
//         for _ in 0..1024 {
//             let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume().unwrap() as u32)).collect();
//             let mut input = CpuSDR::from(input);
//             input.normalize();
//             assert_ne!(input.len(), 0);
//             a.run(&input);
//             a.learn();
//             let o = a.last_output_sdr_mut();
//             if o.is_empty() {
//                 number_of_empty_outputs += 1;
//             }
//             o.sort();
//             assert!(o.is_normalized(), "{:?}", o);
//         }
//         assert!(number_of_empty_outputs < 54, "{}", number_of_empty_outputs);
//         Ok(())
//     }
//
//
//
//     fn test8_<M:Metric<D>>() -> Result<(), String> {
//         let mut rng = rand::thread_rng();
//         let k = 1;
//         let kernel = [3, 3];
//         let stride = [1, 1];
//         let in_channels = 4;
//         let out_channels = [2, 3, 6];
//         let mut a: Vec<CpuEccDense<M>> = out_channels.iter().map(|&out_channels|
//             CpuEccDense::new(ConvShape::new([1, 1], kernel, stride, in_channels, out_channels), k, &mut rng)).collect();
//         let mut c = CpuEccDense::concat(&a, |a| a);
//
//         let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a[0].in_volume() as u32)).collect();
//         let mut input = CpuSDR::from(input);
//         input.normalize();
//         assert_ne!(input.len(), 0);
//         let mut output = CpuSDR::new();
//         for a in a.iter_mut().rev() {
//             a.run_in_place(&input, &mut output);
//             output.shift(a.out_volume() as i32);
//         }
//         output.sort();
//         let mut output2 = c.run(&input);
//         output2.sort();
//         assert_eq!(output, output2);
//         Ok(())
//     }
//
//
//
//
//     fn test9_<M:Metric<D>>() -> Result<(), String> {
//         let s = auto_gen_seed64();
//         let mut rng = rand::rngs::StdRng::seed_from_u64(s);
//         let k = 16;
//         let mut a = CpuEccDense::<M>::new(ConvShape::new([1, 1], [4, 4], [1, 1], 3, 4), 1, &mut rng);
//         a.set_threshold(0.2);
//         let mut a2 = a.clone();
//         println!("seed=={}", s);
//         assert_eq!(a2.k(), 1);
//         assert_eq!(a.k(), 1);
//         assert_eq!(a.get_threshold(), a2.get_threshold(), "threshold {:?}!={:?}", a.get_threshold(), a2.get_threshold());
//         for i in 0..1024 {
//             let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
//             let mut input = CpuSDR::from(input);
//             input.normalize();
//             assert_ne!(input.len(), 0);
//             assert!(a.weights().as_slice().iter().zip(a2.weights().as_slice().iter()).all(|(a, b)| a==b), "w {:?}!={:?}", a.as_slice(), a2.as_slice());
//             assert!(a.population().sums.iter().zip(a2.population().sums.iter()).all(|(a, b)| a ==b), "sums {:?}!={:?}", a.population().sums, a2.population().sums);
//             assert!(a.population().activity.iter().zip(a2.population().activity.iter()).all(|(a, b)| a ==b), "activity {:?}!={:?}", a.population().activity, a2.population().activity);
//             let mut o = a.run(&input);
//             let mut o2 = a2.run(&input);
//             assert_eq!(o, o2, "outputs i=={}", i);
//             a.learn(&input, &o);
//             a2.learn(&input, &o2);
//         }
//         Ok(())
//     }
//
//
//     // #[test]
//     // fn test10() -> Result<(), String> {
//     //     test10_::<u32>()
//     // }
//
//     fn test10_<M:Metric<D>>() -> Result<(), String> {
//         let s = auto_gen_seed64();
//         let mut rng = rand::rngs::StdRng::seed_from_u64(s);
//         let k = 16;
//         let mut a = CpuEccDense::<M>::new(ConvShape::new([2, 2], [4, 4], [1, 1], 3, 4), 1, &mut rng);
//         a.set_threshold(0.2);
//         let mut a2 = a.clone();
//         println!("seed=={}", s);
//         assert_eq!(a2.k(), 1);
//         assert_eq!(a.k(),1);
//         assert_eq!(a.get_threshold(), a2.get_threshold(), "threshold {:?}!={:?}", a.get_threshold(), a2.get_threshold());
//         for i in 0..1024 {
//             let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
//             let mut input = CpuSDR::from(input);
//             input.normalize();
//             assert_ne!(input.len(), 0);
//             assert!(a.weights().as_slice().iter().zip(a2.weights().as_slice().iter()).all(|(a, b)| a==b), "w {:?}!={:?}", a.weights().as_slice(), a2.weights().as_slice());
//             assert!(a.population().sums.iter().zip(a2.population().sums.iter()).all(|(a, b)| a==b), "sums {:?}!={:?}", a.population().sums, a2.population().sums);
//             assert!(a.population().activity.iter().zip(a2.population().activity.iter()).all(|(a, b)| a==b), "activity {:?}!={:?}", a.population().activity, a2.population().activity);
//             let mut o = a.run(&input);
//             let mut o2 = a2.run(&input);
//             assert_eq!(o, o2, "outputs i=={}", i);
//             a.learn(&input, &o);
//             a2.learn(&input, &o2);
//         }
//         Ok(())
//     }
// }
// use crate::{DenseWeight, ConvShape, Idx, as_idx, as_usize, w_idx, Shape, VectorFieldOne, Shape2, Shape3, from_xyz, debug_assert_approx_eq_weight, kernel_column_weight_sum, kernel_column_dropped_weights_count, VectorFieldPartialOrd, CpuSDR, from_xy, SDR, kernel_column_weight_copy, range_contains, ConvShapeTrait, copy_repeated_column, ConvWeightVec, HasShape, HasShapeMut};
// use std::ops::{Deref, DerefMut, Index, IndexMut, Range};
// use rand::Rng;
// use rand::prelude::SliceRandom;
// use crate::cpu_ecc_population::CpuEccPopulation;
// use ndalgebra::mat::AsShape;
// use itertools::Itertools;
// use std::collections::HashSet;
// use serde::{Serialize, Deserialize};
// use rayon::prelude::*;
// use std::thread::JoinHandle;
// use crate::parallel::{parallel_map_vector, parallel_map_collect};
// use crate::conv_spiking_shape::{ConvSpikingShape, TIdx};
// use crate::cpu_sdr_spiking::CpuSdrSpiking;
// use crate::as_usize::AsUsize;
//
// #[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
// pub struct ConvSpikingWeights<D: DenseWeight> {
//     /**The layout is w[output_idx+input_idx_relative_to_kernel_column*output_volume]
//     where kernel column has shape [kernel[0],kernel[1],in_channels]*/
//     w: Vec<D>,
//     t: Vec<TIdx>,
//     shape: ConvShape,
//     time: TIdx,
//     pub plasticity: D,
// }
//
// impl <D: DenseWeight> ConvSpikingShape for ConvSpikingWeights<D> {
//
//     fn time(&self) -> TIdx {
//         self.time
//     }
// }
// impl <D: DenseWeight> HasShape for ConvSpikingWeights<D> {
//     fn shape(&self) -> &ConvShape{
//         &self.shape
//     }
// }
// impl <D: DenseWeight> HasShapeMut for ConvSpikingWeights<D> {
//     fn shape_mut(&mut self) -> &mut ConvShape{
//         &mut self.shape
//     }
// }
// impl <D: DenseWeight> ConvWeightVec<D> for ConvSpikingWeights<D>{
//     fn weight_slice(&self) -> &[D] {
//         self.w.as_slice()
//     }
//
//     fn weight_slice_mut(&mut self) -> &mut [D] {
//         self.w.as_mut_slice()
//     }
// }
// impl<D: DenseWeight> ConvSpikingWeights<D> {
//
//     pub fn weight_slice(&self) ->&[D]{
//         &self.w
//     }
//     pub fn time_slice(&self) ->&[u8]{
//         &self.t
//     }
//
//     pub fn from_repeated_column(output: [Idx; 2], pretrained: &Self, pretrained_column_pos: [Idx; 2]) -> Self {
//         let output = output.add_channels(pretrained.out_channels());
//         let shape = ConvShape::new_out(pretrained.in_channels(),output,*pretrained.kernel(),*pretrained.stride());
//         let kv = pretrained.kernel_column_volume();
//         let new_v = output.product();
//         let mut w = vec![D::ZERO; (kv * new_v).as_usize()];
//         let mut t = vec![0; (kv * new_v).as_usize()];
//         let mut slf = Self {
//             plasticity: pretrained.plasticity,
//             shape,
//             w,
//             t,
//             time: pretrained.time
//         };
//         slf.copy_repeated_column(pretrained,pretrained_column_pos);
//         copy_repeated_column(&mut slf.t,output,&pretrained.t,pretrained.shape(),pretrained_column_pos);
//         slf
//     }
//     pub fn new(shape:ConvShape, time:TIdx, rng: &mut impl Rng) -> Self {
//         let v = shape.out_volume();
//         let kv = shape.kernel_column_volume();
//         let wf: Vec<f32> = (0..kv * v).map(|_| rng.gen()).collect();
//         let w = D::initialise_weight_matrix(kv, v, wf);
//         let slf = Self {
//             w,
//             t: (0..kv * v).map(|_| rng.gen::<TIdx>()%time).collect(),
//             shape,
//             time,
//             plasticity: D::DEFAULT_PLASTICITY,
//         };
//         #[cfg(debug_assertions)] {
//             for output_idx in 0..v {
//                 debug_assert_approx_eq_weight(slf.incoming_weight_sum(output_idx), D::TOTAL_SUM);
//             }
//         }
//         slf
//     }
//     pub fn get_plasticity(&self) -> D {
//         self.plasticity
//     }
//     pub fn set_plasticity(&mut self, plasticity: D) {
//         self.plasticity = plasticity
//     }
//     pub fn t(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> TIdx {
//         self.t[self.w_index(input_pos, output_pos).as_usize()]
//     }
//
//     pub fn reset_and_forward(&self, input: &CpuSdrSpiking, target: &mut CpuEccPopulation<D>){
//         assert_eq!(target.shape(),self.out_shape(),"Shapes don't match");
//         target.reset_sums();
//         self.forward(input,&mut target.sums)
//     }
//     pub fn forward(&self, input: &CpuSdrSpiking, target: &mut [D]) {
//         assert_eq!(target.len(), self.out_volume().as_usize(), "Shapes don't match!");
//         self.forward_with(input,|output_idx,w|{
//             target[output_idx.as_usize()] += w;
//             debug_assert!(target[output_idx.as_usize()].le(D::TOTAL_SUM));
//         })
//     }
//     pub fn forward_with(&self, input: &CpuSdrSpiking, mut target:impl FnMut(Idx,D)) {
//         let kernel_column = self.kernel_column();
//         let v = self.out_volume();
//         #[cfg(debug_assertions)] {
//             let mut i = input.clone();
//             i.sort();
//             debug_assert!(i.is_normalized());
//             for output_idx in 0..v {
//                 debug_assert_approx_eq_weight(self.incoming_weight_sum(output_idx), D::TOTAL_SUM);
//             }
//         }
//         let mut used_w = HashSet::new();
//         for &input_idx in input.as_slice() {
//             let input_pos: [Idx; 3] = self.in_shape().pos(input_idx);
//             let r = input_pos.grid().conv_out_range_clipped(self.stride(), self.kernel());
//             for p0 in r.start.width()..r.end.width().min(self.out_shape().width()) {
//                 for p1 in r.start.height()..r.end.height().min(self.out_shape().height()) {
//                     let kernel_offset = from_xy(p0, p1).conv_in_range_begin(self.stride());
//                     for p2 in 0..self.out_channels() {
//                         let output_pos = from_xyz(p0, p1, p2);
//                         let output_idx = self.out_shape().idx(output_pos);
//                         let w_index = ConvShape::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
//                         debug_assert_eq!(w_index, self.w_index(&input_pos, &output_pos));
//                         debug_assert!(used_w.insert(w_index), "{}", w_index);
//                         let w = self.w[w_index.as_usize()];
//                         if w.is_valid() { // the connection is disabled
//                             target(output_idx,w);
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     pub fn infer_in_place(&self, input: &CpuSdrSpiking, output:&mut CpuSDR, target:&mut CpuEccPopulation<D>){
//         self.reset_and_forward(input, target);
//         target.determine_winners_top1_per_region(output);
//     }
//
//     pub fn infer(&self, input: &CpuSdrSpiking, target:&mut CpuEccPopulation<D>)->CpuSDR{
//         let mut sdr = CpuSDR::new();
//         self.infer_in_place(input,&mut sdr,target);
//         sdr
//     }
//     pub fn run_in_place(&self, input: &CpuSdrSpiking, output:&mut CpuSDR, target:&mut CpuEccPopulation<D>){
//         self.infer_in_place(input,output,target);
//         target.decrement_activities(output)
//     }
//     pub fn run(&self, input: &CpuSdrSpiking, target:&mut CpuEccPopulation<D>)->CpuSDR{
//         let mut sdr = CpuSDR::new();
//         self.run_in_place(input,&mut sdr,target);
//         sdr
//     }
//     pub fn learn_and_store_sums(&mut self, input: &CpuSdrSpiking, output: &CpuSDR, weight_sums:&mut [D]) {
//         assert_eq!(weight_sums.len(), self.out_volume().as_usize(), "Shapes don't match!");
//         self.learn_with(input,output,|w, active_inputs, output_idx, p, kv, v|
//             weight_sums[output_idx.as_usize()] += p.mul(active_inputs)
//         );
//     }
//     pub fn learn(&mut self, input: &CpuSdrSpiking, output: &CpuSDR) {
//
//         self.learn_with(input,output,|w, active_inputs, output_idx, p, kv, v|
//             D::normalize_recommended(w, active_inputs, output_idx, p, kv, v)
//         );
//         #[cfg(debug_assertions)] {
//             let v = self.out_volume();
//             for output_idx in 0..v {
//                 debug_assert_approx_eq_weight(self.incoming_weight_sum(output_idx), D::TOTAL_SUM)
//             }
//             debug_assert!(self.w.iter().all(|&w| w.ge(D::ZERO)));
//             debug_assert!(self.w.iter().all(|&w| w.le(D::TOTAL_SUM)));
//         }
//     }
//     pub fn learn_with(&mut self, input: &CpuSdrSpiking, output: &CpuSDR,mut normalization:impl FnMut(&mut [D],Idx,Idx,D,Idx,Idx)) {
//         debug_assert!(output.clone().sorted().is_normalized());
//         let v = self.out_volume();
//         let p = self.plasticity;
//         let kernel_column = self.kernel_column();
//         let kv = kernel_column.product();
//         let input_pos: Vec<[Idx; 3]> = input.iter().map(|&i| self.in_shape().pos(i)).collect();
//         for &output_idx in output.as_slice() {
//             let mut active_inputs = self._increment_incoming_weights(input,&input_pos,output_idx);
//             normalization(&mut self.w, active_inputs, output_idx, p, kv, v)
//         }
//
//     }
//     fn _increment_incoming_weights(&mut self, input: &CpuSdrSpiking, input_pos: &Vec<[u32; 3]>, output_idx: Idx) -> u32 {
//         let Self{ w, t, shape, time, plasticity } = self;
//         Self::increment_incoming_weights(shape,*plasticity,input,input_pos,output_idx,w.as_mut_slice())
//     }
//     fn increment_incoming_weights(shape:&ConvShape,plasticity:D, input: &CpuSdrSpiking, input_pos: &Vec<[u32; 3]>, output_idx: Idx, w_slice: &mut [D]) -> u32 {
//         let v = shape.out_volume();
//         let kernel_column = shape.kernel_column();
//         let output_pos = shape.out_shape().pos(output_idx);
//         let kernel_offset = shape.kernel_offset(&output_pos);
//         let input_range = output_pos.grid().conv_in_range(shape.stride(), shape.kernel());
//         let mut active_inputs = 0;
//         for (&input_idx, input_pos) in input.iter().zip(input_pos.iter()) {
//             if input_range.start.all_le(input_pos.grid()) && input_pos.grid().all_lt(&input_range.end) {
//                 let w_index = ConvShape::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
//                 debug_assert_eq!(w_index, shape.w_index(input_pos, &output_pos));
//                 let w = w_slice[w_index.as_usize()];
//                 if w.is_valid() {
//                     w_slice[w_index.as_usize()] = w + plasticity;
//                     active_inputs += 1
//                 }
//             }
//         }
//         active_inputs
//     }
//
// }
// impl <'a, D:DenseWeight+Send+Sync> ConvSpikingWeights<D>{
//     pub fn parallel_reset_and_store_all_incoming_weight_sums(&self, sums:&mut [D]) {
//         sums.par_iter_mut().enumerate().for_each(|(i,w)|*w+=self.incoming_weight_sum(as_idx(i)))
//     }
//     pub fn parallel_store_all_incoming_weight_sums(&self, sums:&mut [D]) {
//         sums.par_iter_mut().enumerate().for_each(|(i,w)|*w+=self.incoming_weight_sum(as_idx(i)))
//     }
//     pub fn parallel_reset_and_forward(&self, input: &CpuSdrSpiking, target: &'a mut [D]){
//         self.parallel_forward_with(input, target,|r,w|*r=w);
//     }
//     pub fn parallel_forward(&self, input: &CpuSdrSpiking, target:&'a mut [D]) {
//         self.parallel_forward_with(input,target,|r,w|*r+=w)
//     }
//     pub fn parallel_forward_with(&self, input: &CpuSdrSpiking, target:&'a mut [D], update_weight:impl Fn(&mut D, D)+Send+Sync) {
//         assert_eq!(target.len(), self.out_volume().as_usize(), "Shapes don't match!");
//         let kernel_column = self.kernel_column();
//         let v = self.out_volume();
//         debug_assert!(input.clone().sorted().is_normalized());
//         #[cfg(debug_assertions)] {
//             for output_idx in 0..v {
//                 debug_assert_approx_eq_weight(self.incoming_weight_sum(output_idx), D::TOTAL_SUM);
//             }
//         }
//         let input_positions:Vec<[Idx; 3] > = input.iter().map(|&input_idx|self.in_shape().pos(input_idx)).collect();
//         let ranges:Vec<Range<[Idx;2]>> = input_positions.iter().map(|input_pos|input_pos.grid().conv_out_range_clipped_both_sides(self.stride(), self.kernel(),self.out_grid())).collect();
//         target.par_iter_mut().enumerate().for_each(|(output_idx,weight_sum)|{
//             let output_idx = as_idx(output_idx);
//             let output_pos = self.out_shape().pos(output_idx);
//             let kernel_offset = output_pos.grid().conv_in_range_begin(self.stride());
//             let mut w_sum = D::ZERO;
//             for (input_pos,r) in input_positions.iter().zip(ranges.iter()){
//                 if range_contains(r,output_pos.grid()){
//                     let w_index = ConvShape::w_index_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
//                     let w = self.w[w_index.as_usize()];
//                     if w.is_valid() { // the connection is disabled
//                         w_sum += w
//                     }
//                 }
//             }
//             update_weight(weight_sum, w_sum)
//         });
//     }
//     pub fn parallel_infer_in_place(&self, input: &CpuSDR, output:&mut CpuSDR, target:&'a mut CpuEccPopulation<D>){
//         assert_eq!(target.shape(),self.out_shape(),"Shapes don't match");
//         self.parallel_forward_with(input, &mut target.sums,|r,w|*r=w);
//         target.parallel_determine_winners_top1_per_region(output);
//     }
//     pub fn parallel_infer(&self, input: &CpuSDR, target:&mut CpuEccPopulation<D>)->CpuSDR{
//         assert_eq!(target.shape(),self.out_shape(),"Shapes don't match");
//         let mut sdr = CpuSDR::new();
//         self.parallel_infer_in_place(input,&mut sdr,target);
//         sdr
//     }
//     pub fn parallel_run_in_place(&self, input: &CpuSDR, output:&mut CpuSDR, target:&mut CpuEccPopulation<D>){
//         assert_eq!(target.shape(),self.out_shape(),"Shapes don't match");
//         self.parallel_infer_in_place(input,output,target);
//         target.decrement_activities(output)
//     }
//     pub fn parallel_run(&self, input: &CpuSDR, target:&mut CpuEccPopulation<D>)->CpuSDR{
//         assert_eq!(target.shape(),self.out_shape(),"Shapes don't match");
//         let mut sdr = CpuSDR::new();
//         self.parallel_run_in_place(input,&mut sdr,target);
//         sdr
//     }
//
//     pub fn parallel_learn_and_store_sums(&mut self, input: &CpuSDR, output: &CpuSDR,sums:&mut [D]) {
//         debug_assert!(output.clone().sorted().is_normalized());
//         let w_len = self.w.len();
//         let w_ptr = self.w.as_mut_ptr() as usize;
//         let sums_len = sums.len();
//         let sums_ptr = sums.as_mut_ptr() as usize;
//         let p = self.plasticity;
//         let input_pos: Vec<[Idx; 3]> = input.iter().map(|&i| self.in_shape().pos(i)).collect();
//         output.par_iter().for_each(|&output_idx|{
//             let w_slice = unsafe{std::slice::from_raw_parts_mut(w_ptr as *mut D,w_len)};
//             let weight_sums = unsafe{std::slice::from_raw_parts_mut(sums_ptr as *mut D,sums_len)};
//             let active_inputs = Self::increment_incoming_weights(self.shape(),p,input, &input_pos, output_idx, w_slice);
//             weight_sums[output_idx.as_usize()] += p.mul(active_inputs);
//         });
//     }
//
//     pub fn parallel_learn(&mut self, input: &CpuSDR, output: &CpuSDR) {
//         debug_assert!(output.clone().sorted().is_normalized());
//         let w_len = self.w.len();
//         let w_ptr = self.w.as_mut_ptr() as usize;
//         let kv = self.kernel_column_volume();
//         let v = self.out_volume();
//         let p = self.plasticity;
//         let input_pos: Vec<[Idx; 3]> = input.iter().map(|&i| self.in_shape().pos(i)).collect();
//         output.par_iter().for_each(|&output_idx|{
//             let w_slice = unsafe{std::slice::from_raw_parts_mut(w_ptr as *mut D,w_len)};
//             let active_inputs = Self::increment_incoming_weights(self.shape(),p,input, &input_pos, output_idx, w_slice);
//             D::normalize_recommended(w_slice,active_inputs,output_idx,p,kv,v)
//         });
//     }
//
//     pub fn parallel_normalize_with_stored_sums(&mut self, output: &CpuSDR, weight_sums:&[D]) {
//         assert_eq!(weight_sums.len(),self.out_volume().as_usize());
//         let v = self.out_volume();
//         let kv = self.kernel_column_volume();
//         let w_len = self.w.len();
//         let w_ptr = self.w.as_mut_ptr() as usize;
//         output.par_iter().for_each(|&output_idx| {
//             let w_slice = unsafe{std::slice::from_raw_parts_mut(w_ptr as *mut D,w_len)};
//             D::normalize(w_slice, weight_sums[output_idx.as_usize()], output_idx,  kv, v)
//         })
//     }
//
//     pub fn batch_infer<T,O:Send>(&self, input: &[T], f:impl Fn(&T)->&CpuSDR+Send+Sync, target: CpuEccPopulation<D>, of:impl Fn(CpuSDR)->O+Sync) -> Vec<O> {
//         assert_eq!(target.shape(),self.out_shape(),"Shapes don't match");
//         let mut targets:Vec<CpuEccPopulation<D>> = (1..num_cpus::get()).map(|_|target.clone()).collect();
//         targets.push(target);
//         parallel_map_collect(input,&mut targets,|i,t|of(self.infer(f(i), t)))
//     }
//
//     pub fn batch_infer_and_measure_s_expectation<T,O:Send>(&self, input: &[T], f:impl Fn(&T)->&CpuSdrSpiking+Send+Sync, target: CpuEccPopulation<D>, of:impl Fn(CpuSDR)->O+Sync) -> (Vec<O>,D,u32) {
//         assert_eq!(target.shape(),self.out_shape(),"Shapes don't match");
//         let mut targets:Vec<(D,u32,CpuEccPopulation<D>)> = (1..num_cpus::get()).map(|_|(D::ZERO,0,target.clone())).collect();
//         targets.push((D::ZERO,0,target));
//         let o_vec = parallel_map_collect(input,&mut targets,|i,(s,missed,t)|{
//             let o_sdr = self.infer(f(i), t);
//             *s+=o_sdr.iter().map(|&k|t.sums[k.as_usize()]).sum();
//             if o_sdr.is_empty(){
//                 *missed+=1
//             }
//             of(o_sdr)
//         });
//         (o_vec,targets.iter().map(|(s,_,_)|*s).sum(),targets.iter().map(|(_,s,_)|*s).sum())
//     }
// }
// impl ConvSpikingWeights<f32>{
//     pub fn forward_with_multiplier(&self, input: &CpuSdrSpiking, target: &mut CpuEccPopulation<f32>, multiplier:f32) {
//         assert_eq!(target.shape(), self.out_shape(), "Shapes don't match!");
//         self.forward_with(input,|output_idx,w|{
//             target.sums[output_idx.as_usize()] += multiplier*w;
//         })
//     }
// }
//
// #[cfg(test)]
// mod tests{
//     use crate::{DenseWeight, ConvWeights, ConvShape, CpuEccDense, EccLayer, EccLayerD, CpuSDR};
//     use crate::xorshift::auto_gen_seed64;
//     use rand::{SeedableRng, Rng};
//     #[test]
//     fn test10() -> Result<(), String> {
//         test10_::<u32>()
//     }
//
//     #[test]
//     fn test10f() -> Result<(), String> {
//         test10_::<f32>()
//     }
//
//     fn test10_<D: DenseWeight+Send+Sync>() -> Result<(), String> {
//         let s = auto_gen_seed64();
//         let mut rng = rand::rngs::StdRng::seed_from_u64(s);
//         let k = 16;
//         let mut a = CpuEccDense::<D>::new(ConvShape::new([2, 2], [4, 4], [1, 1], 3, 4), 1,  &mut rng);
//         a.set_threshold_f32(0.2);
//         let mut a2 = a.clone();
//         println!("seed=={}", s);
//         assert_eq!(a2.k(), 1);
//         assert_eq!(a.k(),1);
//         assert!(a.get_threshold().eq(a2.get_threshold()), "threshold {:?}!={:?}", a.get_threshold(), a2.get_threshold());
//         for i in 0..1024 {
//             let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
//             let mut input = CpuSDR::from(input);
//             input.normalize();
//             assert_ne!(input.len(), 0);
//             assert!(a.weight_slice().iter().zip(a2.weight_slice().iter()).all(|(a, b)| a.eq(*b)), "w {:?}!={:?}", a.weights().weight_slice(), a2.weights().weight_slice());
//             assert!(a.population().sums.iter().zip(a2.population().sums.iter()).all(|(a, b)| a.eq(*b)), "sums {:?}!={:?}", a.population().sums, a2.population().sums);
//             assert!(a.population().activity.iter().zip(a2.population().activity.iter()).all(|(a, b)| a.eq(*b)), "activity {:?}!={:?}", a.population().activity, a2.population().activity);
//             let mut o = a.run(&input);
//             let mut o2 = a2.parallel_run(&input);
//             assert_eq!(o, o2, "outputs i=={}", i);
//             a.learn(&input, &o);
//             a2.parallel_learn(&input, &o2);
//         }
//         Ok(())
//     }
// }
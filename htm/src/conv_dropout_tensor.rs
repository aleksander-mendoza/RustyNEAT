// use crate::{ConvShape, Idx, as_idx, as_usize, w_idx, Shape, VectorFieldOne, Shape2, Shape3, from_xyz, debug_assert_approx_eq_weight, VectorFieldPartialOrd, CpuSDR, from_xy, SDR, kernel_column_weight_copy, range_contains, ConvShapeTrait, HasConvShape, HasConvShapeMut, DenseWeightL2, Metric, range_foreach2d, D, ShapedArrayTrait, HasShape, ForwardTarget, ConvTensor};
// use std::ops::{Deref, DerefMut, Index, IndexMut, Range};
// use rand::Rng;
// use rand::prelude::SliceRandom;
// use ndalgebra::mat::AsShape;
// use itertools::Itertools;
// use std::collections::HashSet;
// use serde::{Serialize, Deserialize};
// use rayon::prelude::*;
// use std::thread::JoinHandle;
// use crate::parallel::{parallel_map_vector, parallel_map_collect};
// use crate::as_usize::AsUsize;
// use std::marker::PhantomData;
//
// pub trait DroppedScalar{
//     fn is_dropped(&self)->bool;
//     fn is_valid(&self)->bool{
//         !self.is_dropped()
//     }
//     fn drop(&mut self);
//     fn undrop(&mut self);
// }
//
// pub trait ConvDropoutTensor<D:DroppedScalar>: ConvTensor<D> {
//     fn restore_dropped_out_weights(&mut self) {
//         self.weight_slice_mut().iter_mut().filter(|w| w.is_droped()).for_each(|w| w.se)
//     }
//     fn dropout_f32(&mut self, number_of_connections_to_drop: f32, rng: &mut impl Rng) {
//         self.dropout((self.out_volume() as f32 * number_of_connections_to_drop) as usize, rng)
//     }
//     fn dropout(&mut self, number_of_connections_to_drop: usize, rng: &mut impl Rng) {
//         assert!(number_of_connections_to_drop <= self.len(), "number_of_connections_to_drop={} > number_of_connections=={}", number_of_connections_to_drop, self.len());
//         let mut indices: Vec<Idx> = (0..as_idx(self.len())).collect();
//         indices.shuffle(rng);
//         for i in 0..number_of_connections_to_drop {
//             self.weight_slice_mut()[indices[i].as_usize()] = D::IMPOSSIBLE_WEIGHT;
//         }
//     }
//     fn dropout_per_kernel_f32(&mut self, number_of_connections_to_drop_per_kernel_column: f32, rng: &mut impl Rng) {
//         self.dropout_per_kernel((self.kernel_column_volume() as f32 * number_of_connections_to_drop_per_kernel_column) as usize, rng)
//     }
//     fn dropout_per_kernel(&mut self, number_of_connections_to_drop_per_kernel_column: usize, rng: &mut impl Rng) {
//         let kv = self.kernel_column_volume();
//         let v = self.out_volume();
//         assert!(number_of_connections_to_drop_per_kernel_column <= kv.as_usize(), "number_of_connections_to_drop_per_kernel_column={} > kernel_column_volume=={}", number_of_connections_to_drop_per_kernel_column, kv);
//         for out_idx in 0..v {
//             let mut indices: Vec<Idx> = (0..kv).collect();
//             indices.shuffle(rng);
//             for i in 0..number_of_connections_to_drop_per_kernel_column {
//                 let idx_within_kernel = indices[i];
//                 let w_idx = w_idx(out_idx, idx_within_kernel, v);
//                 self.weight_slice_mut()[w_idx.as_usize()] = D::IMPOSSIBLE_WEIGHT;
//             }
//         }
//     }
//     fn get_dropped_weights_count(&self) -> usize {
//         self.weight_slice().iter().filter(|&w| !w.is_valid()).count()
//     }
//     fn get_dropped_weights_of_kernel_column_count(&self, output_neuron_idx: Idx) -> usize {
//         kernel_column_dropped_weights_count(self.kernel_column_volume(), self.out_volume(), output_neuron_idx, self.weight_slice())
//     }
// }

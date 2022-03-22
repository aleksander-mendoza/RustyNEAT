use crate::{ConvShape, ConvTensorTrait, HasShape, HasConvShape, HasConvShapeMut, ConvShapeTrait, Idx, Shape2, VectorFieldOne, AsUsize, Weight};
use rand::Rng;
use serde::{Serialize, Deserialize};
use rand::distributions::{Standard, Distribution};

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct ConvTensor<D> {
    /**The layout is w[output_idx+input_idx_relative_to_kernel_column*output_volume]
    where kernel column has shape [kernel[0],kernel[1],in_channels]*/
    w: Vec<D>,
    shape: ConvShape,
}

impl <D:Weight> ConvTensorTrait<D> for ConvTensor<D> {
    fn as_slice(&self) -> &[D] {
        self.w.as_slice()
    }

    fn unpack_mut(&mut self) -> (&mut [D], &mut ConvShape) {
        let Self{ w, shape } = self;
        (w,shape)
    }
    fn repeat_column(&self, column_grid: [Idx; 2], column_pos: [Idx; 2]) -> Self {
        let output = column_grid.add_channels(self.out_channels());
        let shape = ConvShape::new_out(self.in_channels(), output, *self.kernel(), *self.stride());
        let mut slf = unsafe{Self::empty(shape)};
        slf.copy_repeated_column(self, column_pos);
        slf
    }
}

impl<D> HasShape for ConvTensor<D> {
    fn shape(&self) -> &[u32; 3] {
        self.shape.out_shape()
    }
}

impl<D> HasConvShape for ConvTensor<D> {
    fn cshape(&self) -> &ConvShape {
        &self.shape
    }
}

impl<D> HasConvShapeMut for ConvTensor<D> {
    fn cshape_mut(&mut self) -> &mut ConvShape {
        &mut self.shape
    }
}


impl<D:Copy> ConvTensor<D> {
    pub unsafe fn empty(shape: ConvShape) -> Self {
        let v = shape.out_volume();
        let kv = shape.kernel_column_volume();
        let l = (kv * v).as_usize();
        let mut w = Vec::with_capacity(l);
        w.set_len(l);
        Self {
            w,
            shape,
        }
    }
    pub fn rand(shape: ConvShape, rng: &mut impl Rng) -> Self where Standard: Distribution<D>{
        let mut slf = unsafe{Self::empty(shape)};
        slf.w.fill_with(|| rng.gen());
        slf
    }
    pub fn new(shape: ConvShape, constant:D) -> Self {
        let mut slf = unsafe{Self::empty(shape)};
        slf.w.fill(constant);
        slf
    }
    // pub fn concat<'a, T>(layers: &'a [T], f: impl Fn(&'a T) -> &'a Self) -> Self {
    //     let shape = ConvShape::concat(layers, |l| f(l).cshape());
    //     let new_v = shape.out_volume();
    //     let kv = shape.kernel_column_volume();
    //
    //     let mut slf = Self {
    //         w: vec![D::IMPOSSIBLE_WEIGHT; (kv * new_v).as_usize()],
    //         shape,
    //         plasticity: f(&layers[0]).plasticity,
    //         _d: Default::default(),
    //     };
    //     #[cfg(debug_assertions)]
    //         let mut w_written_to = vec![false; slf.w.len()];
    //
    //     let mut channel_offset = 0;
    //     for l in 0..layers.len() {
    //         let l = f(&layers[l]);
    //         let v = l.out_volume();
    //         for w in 0..l.out_width() {
    //             for h in 0..l.out_height() {
    //                 for c in 0..l.out_channels() {
    //                     let original_output_idx = l.out_shape().idx(from_xyz(w, h, c));
    //                     let new_output_idx = slf.out_shape().idx(from_xyz(w, h, channel_offset + c));
    //                     for idx_within_kernel_column in 0..kv {
    //                         let original_w_idx = w_idx(original_output_idx, idx_within_kernel_column, v);
    //                         let new_w_idx = w_idx(new_output_idx, idx_within_kernel_column, new_v);
    //                         #[cfg(debug_assertions)]
    //                         debug_assert!(!w_written_to[new_w_idx.as_usize()]);
    //                         slf.w[new_w_idx.as_usize()] = l.w[original_w_idx.as_usize()];
    //                         #[cfg(debug_assertions)] {
    //                             w_written_to[new_w_idx.as_usize()] = true;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         channel_offset += l.out_channels();
    //     }
    //     #[cfg(debug_assertions)]
    //     debug_assert!(w_written_to.into_iter().all(|a| a));
    //     slf
    // }
}

#[cfg(test)]
mod tests {
    use crate::xorshift::auto_gen_seed64;
    use rand::{SeedableRng, Rng};
    // #[test]
    // fn test10() -> Result<(), String> {
    //     test10_::<u32>()
    // }

    // #[test]
    // fn test10f() -> Result<(), String> {
    //     test10_::<f32, MetricL1<f32>>()
    // }
    // #[test]
    // fn test10fl2() -> Result<(), String> {
    //     test10_::<f32, MetricL2<f32>>()
    // }
    //
    // fn test10_<D: DenseWeight + Send + Sync, M: Metric<D> + Send + Sync>() -> Result<(), String> {
    //     let s = 1646318184253;//auto_gen_seed64();
    //     let mut rng = rand::rngs::StdRng::seed_from_u64(s);
    //     let k = 16;
    //     let mut a = CpuEccDense::<D, M>::new(ConvShape::new([2, 2], [4, 4], [1, 1], 3, 4), 1, &mut rng);
    //     a.set_threshold(D::f32_to_w(0.2));
    //     let mut a2 = a.clone();
    //     println!("seed=={}", s);
    //     assert_eq!(a2.k(), 1);
    //     assert_eq!(a.k(), 1);
    //     assert!(a.get_threshold().eq(a2.get_threshold()), "threshold {:?}!={:?}", a.get_threshold(), a2.get_threshold());
    //     for i in 0..1024 {
    //         let input: Vec<u32> = (0..k).map(|_| rng.gen_range(0..a.in_volume() as u32)).collect();
    //         let mut input = CpuSDR::from(input);
    //         input.normalize();
    //         assert_ne!(input.len(), 0);
    //         assert!(a.weight_slice().iter().zip(a2.weight_slice().iter()).all(|(a, b)| a.eq(*b)), "w {:?}!={:?}", a.weights().weight_slice(), a2.weights().weight_slice());
    //         assert!(a.population().sums.iter().zip(a2.population().sums.iter()).all(|(a, b)| a.eq(*b)), "sums {:?}!={:?}", a.population().sums, a2.population().sums);
    //         assert!(a.population().activity.iter().zip(a2.population().activity.iter()).all(|(a, b)| a.eq(*b)), "activity {:?}!={:?}", a.population().activity, a2.population().activity);
    //         let mut o = a.run(&input);
    //         // let mut o2 = a2.parallel_run(&input);
    //         assert_eq!(o, o2, "outputs i=={}", i);
    //         a.learn(&input, &o);
    //         a2.parallel_learn(&input, &o2);
    //     }
    //     Ok(())
    // }
}
// use crate::{D, Tensor, Idx, TensorTrait, CpuSDR, HasShape, Shape3, AsUsize};
// use rand::Rng;
//
// pub struct EccInhibition{
//     exc:Tensor<D>,
//     inh:Tensor<D>,
//     s:Tensor<D>,
//     threshold:D,
//     epsilon:D
// }
//
// impl EccInhibition{
//     pub fn new(n:Idx,m:Idx,threshold:D,epsilon:D,rng:&mut impl Rng)->Self{
//         let mut slf = Self{
//             exc: Tensor::rand([m,n,1],rng),
//             inh: Tensor::rand([m,n,1],rng),
//             s: unsafe{Tensor::empty([m,1,1])},
//             threshold,
//             epsilon
//         };
//         slf.exc.mat_norm_assign_columnwise();
//         slf.inh.mat_norm_assign_rowwise();
//         slf
//     }
//
//     pub fn run(&mut self,x:&CpuSDR, learn:bool)->CpuSDR{
//         let t = self.threshold;
//         let e = self.epsilon;
//         let m = self.exc.shape().width();
//         let Self{ exc, inh, s,.. } = self;
//         exc.mat_sparse_dot_lhs_vec(x,s);
//         inh.mat_sparse_dot_lhs_vec_sub_assign(x,s);
//         let mut y = CpuSDR::new();
//         s.find_sparse_gt(t,&mut y);
//         if learn{
//             exc.mat_sparse_add_assign_scalar_to_area(&y,&x,e/x.len() as D);
//             exc.mat_sparse_norm_assign_column(&y);
//             for j in 0..m{
//                 if s[j.as_usize()]<t{
//                     inh.
//                 }
//             }
//         }
//         y
//     }
// }
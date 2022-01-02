use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg, RangeBounds};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::cpu_sdr::CpuSDR;
use crate::ecc_program::EccProgram;
use ndalgebra::buffer::Buffer;
use crate::cpu_bitset::CpuBitset;
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use crate::{Shape, resolve_range, EncoderTarget, Synapse, top_large_k_indices, top_small_k_indices, Shape3, from_xyz, Shape2, from_xy};
use std::collections::Bound;
use crate::vector_field::{VectorFieldOne, VectorFieldDiv, VectorFieldAdd, VectorFieldMul, ArrayCast, VectorFieldSub, VectorFieldPartialOrd};
use crate::population::Population;
use rand::{Rng, SeedableRng};
use crate::xorshift::{ auto_gen_seed64, xorshift64, auto_gen_seed, xorshift};

pub trait EccLayer{
    fn k(&self)->usize;
    fn out_shape(&self)->&[usize;3];
    fn in_shape(&self)->&[usize;3];
    fn in_grid(&self)->&[usize;2]{
        let [ref grid@ ..,_] = self.in_shape();
        grid
    }
    fn out_width(&self)->usize{
        self.out_shape()[1]
    }
    fn out_height(&self)->usize{
        self.out_shape()[0]
    }
    fn out_channels(&self)->usize{
        self.out_shape()[2]
    }
    fn in_width(&self)->usize{
        self.in_shape()[1]
    }
    fn in_height(&self)->usize{
        self.in_shape()[0]
    }
    fn in_channels(&self)->usize{
        self.in_shape()[2]
    }
    fn out_area(&self)->usize{
        self.out_width()*self.out_height()
    }
    fn out_volume(&self)->usize{
        self.out_shape().product()
    }
    fn in_volume(&self)->usize{
        self.in_shape().product()
    }
    fn top_large_k_by_channel<T>(&self,sums:&[T],candidates_per_value:&mut[usize], f:fn(&T) ->usize,threshold:impl Fn(usize)->bool)->CpuSDR{
        let a = self.out_area();
        let c = self.out_channels();
        let k = self.k();
        let mut top_k = CpuSDR::with_capacity(k*a);
        for column_idx in 0..a {
            let r = c*column_idx;
            let r = r..r+c;
            top_large_k_indices(k, &sums[r], candidates_per_value,f, |t| if threshold(t){top_k.push(t as u32)});
        }
        top_k
    }
    fn top_small_k_by_channel<V:Copy>(&self, f:impl Fn(usize) -> V,filter:impl Fn(usize,V)->bool,gt:fn(V,V)->bool)->CpuSDR{
        let a = self.out_area();
        let c = self.out_channels();
        let k = self.k();
        let mut top_k = CpuSDR::with_capacity(k*a);
        for column_idx in 0..a {
            let r = c*column_idx;
            for (i,v) in top_small_k_indices(k,c,|i|f(i+r),gt){
                if filter(i,v){
                    top_k.push(i as u32);
                }
            }
        }
        top_k
    }
}
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct EccSparse {
    /**connections[input_idx]==vector_of_output_indices*/
    connections: Vec<Vec<usize>>,
    max_incoming_synapses:usize,
    input_shape:[usize;3], //[height, width, channels]
    output_shape:[usize;3], //[height, width, channels]
    pub k:usize,
    pub threshold:u16,
}
impl EccSparse{
    pub fn new(output:[usize;2],kernel:[usize;2],stride:[usize;2],in_channels:usize,out_channels:usize,k:usize,connections_per_output:usize,rng:&mut impl Rng)->Self{
        let input = output.conv_in_size(&stride,&kernel);
        let output = [output[0],output[1], out_channels];
        let input = [input[0],input[1], in_channels];
        let in_size = input.product();
        let out_size = output.product();
        let mut pop = Population::new(out_size,1);
        pop.add_2d_column_grid_with_3d_input(0..in_size,out_channels,connections_per_output,stride,kernel,input,rng);
        let slf = Self::new_from_pop(k,input,output,&pop);
        debug_assert_eq!(slf.max_incoming_synapses,connections_per_output);
        slf
    }
    pub fn new_from_pop(k:usize,input_shape:[usize;3],output_shape:[usize;3], population:&Population)->Self{
        let mut connections:Vec<Vec<usize>> = (0..input_shape.product()).map(|_|vec![]).collect();
        let mut max_incoming_synapses = population.neurons.iter().map(|n|n.total_synapses()).max().unwrap();
        for (out_idx,neuron) in population.neurons.iter().enumerate() {
            for seg in &neuron.segments {
                for syn in &seg.synapses {
                    connections[syn.input_idx].push(out_idx);
                }
            }
        }
        Self {threshold:1,k,input_shape, output_shape, connections, max_incoming_synapses }
    }

    pub fn run(&self, input:&CpuSDR)->CpuSDR{
        let mut sums = vec![0u16;self.out_volume()];
        for &input_idx in input.as_slice(){
            for &c in &self.connections[input_idx as usize]{
                sums[c]+=1;
            }
        }
        let t = self.threshold;
        self.top_small_k_by_channel(|i| sums[i],|i,v|v>=t,|a,b|a>b)

    }

}
impl EccLayer for EccSparse{

    fn k(&self)->usize{self.k}

    fn out_shape(&self) -> &[usize; 3] { &self.output_shape }

    fn in_shape(&self) -> &[usize; 3] { &self.input_shape }
}
pub const ACTIVITY_PENALTY:f32 = 0.0001;

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct EccDense {
    /**The layout is w[output_idx+input_idx_relative_to_kernel_column*output_volume]
    where kernel column has shape [kernel[0],kernel[1],in_channels]*/
    w:Vec<f32>,
    input_shape:[usize;3], //[height, width, channels]
    output_shape:[usize;3], //[height, width, channels]
    kernel:[usize;2], //[height, width]
    stride:[usize;2], //[height, width]
    pub k:usize,
    pub threshold:f32,
    pub plasticity:f32,
    activity:Vec<f32>,
    pub rand_seed:usize,
}

impl EccDense{
    pub fn new(output:[usize;2],kernel:[usize;2],stride:[usize;2],in_channels:usize,out_channels:usize,k:usize,rng:&mut impl Rng)->Self{
        let input = output.conv_in_size(&stride,&kernel);
        let output = [output[0],output[1],out_channels];
        let input = [input[0],input[1],in_channels];
        let kernel_column = [kernel[0],kernel[1],in_channels];
        let mut slf = Self{
            w:(0..kernel_column.product()*output.product()).map(|_|rng.gen()).collect(),
            input_shape:input,
            output_shape:output,
            kernel,
            stride,
            k,
            threshold:1./out_channels as f32,
            plasticity: 0.0001,
            activity: vec![0.;output.product()],
            rand_seed: auto_gen_seed()
        };
        slf.normalise_all_weights();
        slf
    }
    pub fn kernel_column(&self) ->[usize;3]{
        self.kernel.add_channels(self.in_channels())
    }
    pub fn kernel_offset(&self, output_pos:&[usize;3])->[usize;2]{
        output_pos.grid().conv_in_range_begin(&self.stride)
    }
    pub fn pos_within_kernel(&self, input_pos:&[usize;3], output_pos:&[usize;3])->[usize;3]{
        Self::sub_kernel_offset(input_pos,&self.kernel_offset(output_pos))
    }
    fn sub_kernel_offset(input_pos:&[usize;3],offset:&[usize;2])->[usize;3]{
        from_xyz(input_pos.width()-offset.width(),input_pos.height()-offset.height(),input_pos.channels())
    }
    #[inline]
    fn w_idx(output_idx:usize,idx_within_kernel_column:usize,output_volume:usize)->usize{
        output_idx + idx_within_kernel_column*output_volume
    }
    #[inline]
    fn w_index_(input_pos:&[usize;3],kernel_offset:&[usize;2],output_idx:usize,kernel_column:&[usize;3],output_volume:usize)->usize{
        let position_within_kernel_column = Self::sub_kernel_offset(input_pos,kernel_offset);
        Self::w_idx(output_idx, kernel_column.idx(position_within_kernel_column),output_volume)
    }
    pub fn idx_within_kernel(&self, input_pos:&[usize;3], output_pos:&[usize;3])->usize{
        self.kernel_column().idx(self.pos_within_kernel(input_pos,output_pos))
    }
    pub fn w_index(&self, input_pos:&[usize;3], output_pos:&[usize;3])->usize{
        Self::w_idx(self.out_shape().idx(*output_pos),self.idx_within_kernel(input_pos,output_pos),self.out_volume())
    }
    pub fn w(&self, input_pos:&[usize;3], output_pos:&[usize;3])->f32{
        self.w[self.w_index(input_pos,output_pos)]
    }
    pub fn run(&mut self, input:&CpuSDR)->CpuSDR{
        fn min(a:f32,b:f32)->f32{
            if a<b{a}else{b}
        }
        let mut sums = vec![0f32;self.out_volume()];
        let kernel_column = self.kernel_column();
        let v = self.out_volume();
        for &input_idx in input.as_slice(){
            let input_pos:[usize;3] = self.input_shape.pos(input_idx as usize);
            let r = input_pos.grid().conv_out_range_clipped(&self.stride,&self.kernel);
            for p0 in r.start.width()..r.end.width().min(self.output_shape.width()){
                for p1 in r.start.height()..r.end.height().min(self.output_shape.height()){
                    let kernel_offset = from_xy(p0,p1).conv_in_range_begin(&self.stride);
                    for p2 in 0..self.out_channels(){
                        let output_pos = from_xyz(p0,p1,p2);
                        let output_idx = self.output_shape.idx(output_pos);
                        let w_index = Self::w_index_(&input_pos,&kernel_offset,output_idx,&kernel_column,v);
                        debug_assert_eq!(w_index,self.w_index(&input_pos,&output_pos));
                        sums[output_idx]+=self.w[w_index];
                    }
                }
            }
        }
        // The difference between min_activity and max_activity cannot be greater than 1,
        // due to the obvious nature of entropy maximisation.
        // Hence 0<=self.activity[i]-min_activity<=1 for all i
        // The weights must sum up to 1, therefore
        // 0<=sums[i]<=1
        // 0<=sums[i]+self.activity[i]-min_activity<=2
        // We need to divide by 2 in order to return to 0-1 range
        //0<=(sums[i]+self.activity[i]-min_activity)/2<=1
        let t = self.threshold;
        let top = self.top_small_k_by_channel(|i|sums[i]+self.activity[i], |i,v|sums[i]>=t,|a,b|a>b);
        for &winner in top.iter(){
            self.activity[winner as usize] -= ACTIVITY_PENALTY;
        }

        top
    }

    pub fn learn(&mut self, input:&CpuSDR, output:&CpuSDR){
        let v = self.out_volume();
        let p = self.plasticity;
        let one_minus_p = 1.-p;
        let kernel_column = self.kernel_column();
        let kv = kernel_column.product();
        let input_pos:Vec<[usize;3]> = input.iter().map(|&i|self.input_shape.pos(i as usize)).collect();
        let mut rand_seed = xorshift(self.rand_seed);
        for &output_idx in output.as_slice(){
            let output_idx = output_idx as usize;
            let output_pos = self.output_shape.pos(output_idx);
            let kernel_offset = self.kernel_offset(&output_pos);
            let input_range = output_pos.grid().conv_in_range(&self.stride,&self.kernel);
            let mut subtracted = 0;
            for (&input_idx,input_pos) in input.iter().zip(input_pos.iter()){
                if input_range.start.all_le(input_pos.grid()) && input_pos.grid().all_lt( &input_range.end) {
                    let w_index = Self::w_index_(&input_pos,&kernel_offset,output_idx,&kernel_column,v);
                    debug_assert_eq!(w_index,self.w_index(input_pos,&output_pos));
                    if self.w[w_index] <= one_minus_p {
                        self.w[w_index] += p;
                        subtracted += 1;
                    }
                }
            }
            let mut fallback_input_idx = rand_seed%kv;
            for _ in 0..subtracted{
                rand_seed = xorshift(rand_seed);
                let input_idx_within_kernel_column = rand_seed%kv;
                let w_index = Self::w_idx( output_idx, input_idx_within_kernel_column, v);
                if self.w[w_index] >= p {
                    self.w[w_index] -= p;
                }else {
                    loop {
                        let w_index = Self::w_idx( output_idx, fallback_input_idx, v);
                        fallback_input_idx += 1;
                        if fallback_input_idx == kv {
                            fallback_input_idx = 0
                        }
                        if self.w[w_index] >= p {
                            self.w[w_index] -= p;
                            break;
                        }
                    }
                }
            }
        }
        #[cfg(debug_assertions)]{
            for output_idx in 0..v{
                float_eq(self.incoming_weight_sum(output_idx),1.,0.000001)
            }
            let min_acc = self.min_activity();
            for output_idx in 0..v{
                assert!(self.activity[output_idx] < min_acc+1.,"{} @ {} < {}",output_idx,self.activity[output_idx],min_acc)
            }
        }
        debug_assert!(self.w.iter().all(|&w|w>=0.));
        debug_assert!(self.w.iter().all(|&w|w<=1.));
        self.rand_seed = rand_seed;
    }
    pub fn incoming_weight_sum(&self,output_neuron_idx:usize)->f32{
        let kv = self.kernel_column().product();
        let v = self.out_volume();
        assert!(output_neuron_idx<v);
        (0..kv).map(|i|self.w[Self::w_idx( output_neuron_idx, i, v)]).sum()
    }
    pub fn normalise_weights(&mut self, output_neuron_idx:usize){
        let kv = self.kernel_column().product();
        let v = self.out_volume();
        let sum = self.incoming_weight_sum(output_neuron_idx);
        for i in 0..kv{
            self.w[Self::w_idx( output_neuron_idx, i, v)]/=sum;
        }
    }
    pub fn min_activity(&self)->f32{
        self.activity.iter().cloned().reduce(|a,b|if a<b{a}else{b}).unwrap()
    }
    pub fn activity(&self,output_idx:usize)->f32{
        self.activity[output_idx]
    }
    pub fn normalise_all_weights(&mut self){
        let v = self.out_volume();
        for o in 0..v{
            self.normalise_weights(o)
        }
    }
}
fn float_eq(a:f32,b:f32,eps:f32){
    debug_assert!((a-b).abs() < eps,"{}!={}",a,b);
}
impl EccLayer for EccDense{
    fn k(&self)->usize{self.k}

    fn out_shape(&self) -> &[usize; 3] { &self.output_shape }

    fn in_shape(&self) -> &[usize; 3] { &self.input_shape }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test1() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 8;
        let mut a = EccSparse::new([4,4],[2,2],[1,1],3,4,1,4,&mut rng);
        for _ in 0..64{
            let input:Vec<u32> = (0..k).map(|_|rng.gen_range(0..a.in_volume() as u32)).collect();
            let input = CpuSDR::from(input);
            let o = a.run(&input);
        }
        Ok(())
    }

    #[test]
    fn test2() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 8;
        let mut a = EccDense::new([4,4],[2,2],[1,1],3,4,1,&mut rng);
        for _ in 0..1024{
            let input:Vec<u32> = (0..k).map(|_|rng.gen_range(0..a.in_volume() as u32)).collect();
            let input = CpuSDR::from(input);
            let o = a.run(&input);
            a.learn(&input, &o);
        }
        Ok(())
    }

    #[test]
    fn test3() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let k = 8;
        let mut a = EccDense::new([4,4],[2,2],[1,1],3,4,1,&mut rng);
        a.plasticity = 0.1;//let's see if this breaks anything
        for _ in 0..1024{
            let input:Vec<u32> = (0..k).map(|_|rng.gen_range(0..a.in_volume() as u32)).collect();
            let input = CpuSDR::from(input);
            let o = a.run(&input);
            a.learn(&input, &o);
        }
        Ok(())
    }

}
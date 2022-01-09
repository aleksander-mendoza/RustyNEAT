use ocl::{ProQue, SpatialDims, flags, Platform, Device, Error, Queue, MemFlags};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use ocl::core::{MemInfo, MemInfoResult, BufferRegion, Mem, ArgVal};
use crate::ocl_sdr::OclSDR;
use crate::ecc_program::EccProgram;
use ndalgebra::buffer::Buffer;
use crate::{OclBitset, CpuEccDense, EccLayer, Shape2, Shape3, Idx, Shape, VectorFieldOne, as_usize};
use ocl::prm::{Uint3, Uint2};

#[derive(Clone)]
pub struct OclEccDense {
    prog: EccProgram,
    /**The layout is w[output_idx+input_idx_relative_to_kernel_column*output_volume]
    where kernel column has shape [kernel[0],kernel[1],in_channels]*/
    w: Buffer<u32>,
    // instead of f32 we use u32 but they are equivalent. Just imagine that you divide
    // the u32 value by some large constant and the obtain f32. Addition and subtraction factor out
    //during division (4u32/1000f32)+(8u32/1000f32) == (4u32+8u32)/1000f32
    input_shape: [Idx; 3],
    //[height, width, channels]
    output_shape: [Idx; 3],
    //[height, width, channels]
    kernel: [Idx; 2],
    //[height, width]
    stride: [Idx; 2],
    //[height, width]
    pub k: Idx,
    pub threshold: u32,
    pub plasticity: u32,
    activity: Buffer<u32>,
    pub sums: Buffer<u32>,
    pub top_values: Buffer<u32>,
}

impl OclEccDense {
    pub fn new(ecc:&CpuEccDense<u32>, prog: EccProgram) ->Result<Self,Error>{
        let sums = prog.buffer_from_slice(MemFlags::READ_WRITE, &ecc.sums)?;
        let activity = prog.buffer_from_slice(MemFlags::READ_WRITE, ecc.get_activity())?;
        let w = prog.buffer_from_slice(MemFlags::READ_WRITE, ecc.get_weights())?;
        let top_values = prog.buffer_filled(MemFlags::READ_WRITE, as_usize(ecc.out_area())+1,0)?;
        Ok(Self{
            prog,
            w,
            input_shape: *ecc.in_shape(),
            output_shape: *ecc.out_shape(),
            kernel: *ecc.kernel(),
            stride: *ecc.stride(),
            k: ecc.k(),
            threshold: ecc.threshold,
            plasticity: ecc.plasticity,
            activity,
            sums,
            top_values
        })
    }
    pub fn kernel_column(&self) -> [Idx; 3] {
        self.kernel.add_channels(self.input_shape.channels())
    }
    fn zero_out_all_sums(&mut self) -> Result<(), Error> {
        let Self{ prog, sums, .. } = self;
        sums.fill(prog.queue(),0)
    }
    fn zero_out_all_top_values(&mut self) -> Result<(), Error> {
        let Self{ prog, top_values, .. } = self;
        top_values.fill(prog.queue(),0)
    }
    fn get_output_len(&mut self) -> Result<u32, Error> {
        let Self{ prog, top_values,output_shape, .. } = self;
        top_values.get(prog.queue(),as_usize(output_shape.grid().product()))
    }
    fn sums(&mut self, input:&OclSDR) -> Result<(), Error> {
        let Self{ prog, w, input_shape, output_shape, kernel, stride, sums, ..  } = self;
        let output_kernel_column = kernel.conv_out_transpose_kernel(stride).add_channels(output_shape.channels());
        let output_volume = output_shape.product();
        let output_kernel_column_volume = output_kernel_column.product();
        prog.kernel_builder("ecc_dense_sums")?.
            add_vec(Uint3::from(output_kernel_column))?.// const uint3 output_kernel_column,
            add_vec(Uint3::from(*output_shape))?.// const uint3 output_shape,
            add_vec(Uint3::from(*input_shape))?.// const uint3 input_shape,
            add_vec(Uint2::from(*stride))?.// const uint2 stride,
            add_vec(Uint2::from(*kernel))?.// const uint2 kernel,
            add_num(output_volume)?.// const uint v,
            add_buff(input.buffer())?.// __global uint  * input,
            add_buff(sums)?.// __global uint  * sums,
            add_buff(w)?.// __global uint  * w
            enq(prog.queue(),&[as_usize(input.cardinality()),as_usize(output_kernel_column_volume),1]).
            map_err(Error::from)
    }
    fn zero_out_sums_for_sdr(&mut self, sdr:&OclSDR)-> Result<(), Error> {
        let Self{ prog, activity,.. } = self;
        prog.kernel_builder("ecc_dense_zero_out_sums_for_sdr")?.
            add_buff(sdr.buffer())?.// __global uint  * sdr
            add_buff(activity)?.// __global uint  * activity,
            enq(prog.queue(),&[as_usize(sdr.cardinality()),1,1]).
            map_err(Error::from)
    }
    fn decrement_activities_for_sdr(&mut self, sdr:&OclSDR)-> Result<(), Error> {
        let Self{ prog, activity,.. } = self;
        prog.kernel_builder("ecc_dense_decrement_activities_for_sdr")?.
            add_buff(sdr.buffer())?.// __global uint  * sdr
            add_buff(activity)?.// __global uint  * activity,
            enq(prog.queue(),&[as_usize(sdr.cardinality()),1,1]).
            map_err(Error::from)
    }
    fn incoming_weights_sum(&mut self, output:&OclSDR) -> Result<(), Error> {
        let kv = self.kernel_column().product();
        let Self{ prog,w, output_shape, sums,.. } = self;
        let output_volume = output_shape.product();
        prog.kernel_builder("ecc_dense_incoming_weights_sum")?.
            add_num(output_volume)?. // const uint v,
            add_buff(output.buffer())?. // __global uint * output_sdr,
            add_buff(sums)?. // __global uint * sums
            add_buff(w)?. // __global uint * w
            enq(prog.queue(),&[as_usize(kv),as_usize(output.cardinality()),1]).
            map_err(Error::from)
    }
    fn normalize(&mut self, output:&OclSDR) -> Result<(), Error> {
        let kv = self.kernel_column().product();
        let Self{ prog,w, output_shape, sums,.. } = self;
        let output_volume = output_shape.product();
        prog.kernel_builder("ecc_dense_normalize")?.
            add_num(output_volume)?. // const uint v,
            add_buff(output.buffer())?. // __global uint * output_sdr,
            add_buff(sums)?. // __global uint * sums
            add_buff(w)?. // __global uint * w
            enq(prog.queue(),&[as_usize(kv),as_usize(output.cardinality()),1]).
            map_err(Error::from)
    }
    fn increment_weights(&mut self, input:&OclSDR,output:&OclSDR) -> Result<(), Error> {
        let kernel_column = self.kernel_column();
        let Self{ prog,w, output_shape,input_shape, plasticity,stride,.. } = self;
        let output_volume = output_shape.product();
        prog.kernel_builder("ecc_dense_increment_weights")?.
            add_num(output_volume)?.// const uint v,
            add_vec(Uint3::from(*input_shape))?.// const uint3 input_shape,
            add_vec(Uint3::from(*output_shape))?.// const uint3 output_shape,
            add_vec(Uint3::from(kernel_column))?.// const uint3 kernel_column,
            add_vec(Uint2::from(*stride))?.// const uint2 stride,
            add_num(*plasticity)?.// const uint plasticity,
            add_buff(input.buffer())?.// __global uint * output_sdr,
            add_buff(output.buffer())?.// __global uint * input_sdr,
            add_buff(w)?. // __global uint * w
            enq(prog.queue(),&[as_usize(input.cardinality()),as_usize(output.cardinality()),1]).
            map_err(Error::from)
    }
    fn max_r(&mut self) -> Result<(), Error> {
        let Self{ prog,top_values,activity,threshold, output_shape,sums,.. } = self;
        let output_grid_area = output_shape.grid().product();
        prog.kernel_builder("ecc_dense_max_r")?.
            add_num(*threshold)?. // const uint threshold,
            add_buff(activity)?.// __global uint * activity,
            add_buff(top_values)?.// __global uint * top_values,
            add_buff(sums)?.// __global uint * sums
            enq(prog.queue(),&[as_usize(output_shape.channels()),as_usize(output_grid_area),1]).
            map_err(Error::from)
    }
    fn top_1(&mut self,output:&mut OclSDR) -> Result<(), Error> {
        let Self{ prog,top_values,activity,threshold, output_shape,sums,.. } = self;
        let output_grid_area = output_shape.grid().product();
        prog.kernel_builder("ecc_dense_top_1")?.
            add_num(*threshold)?. // const uint threshold,
            add_buff(activity)?.// __global uint * activity,
            add_buff(top_values)?.// __global uint * top_values,
            add_buff(output.buffer())?.//__global uint * output_sdr,
            add_buff(sums)?.// __global uint * sums
            enq(prog.queue(),&[as_usize(output_shape.channels()),as_usize(output_grid_area),1]).
            map_err(Error::from)
    }
    pub fn run_in_place(&mut self, input:&OclSDR, output:&mut OclSDR) -> Result<(), Error> {
        self.infer_in_place(input,output);
        self.decrement_activities_for_sdr(output)
    }
    pub fn infer_in_place(&mut self, input:&OclSDR, output:&mut OclSDR) -> Result<(), Error> {
        self.zero_out_all_sums()?;
        self.sums(input)?;
        self.zero_out_all_top_values()?;
        self.max_r()?;
        self.top_1(output)?;
        self.prog.q.finish()?;
        unsafe{
            output.set_cardinality(self.get_output_len()?);
        }
        Ok(())
    }
    pub fn run(&mut self, input:&OclSDR) -> Result<OclSDR, Error> {
        let output = self.infer(input)?;
        self.decrement_activities_for_sdr(&output)?;
        Ok(output)
    }
    pub fn infer(&mut self, input:&OclSDR) -> Result<OclSDR, Error> {
        let max_card = self.output_shape.grid().product();
        let mut output = OclSDR::new(self.prog.clone(),max_card)?;
        self.infer_in_place(input,&mut output)?;
        Ok(output)
    }
    pub fn learn(&mut self, input:&OclSDR,output:&OclSDR) -> Result<(), Error> {
        self.increment_weights(input,output)?;
        self.zero_out_sums_for_sdr(input)?;
        self.incoming_weights_sum(output)?;
        self.normalize(output)?;
        Ok(())

    }

}


#[derive(Clone)]
pub struct OclEccSparse {
    prog: EccProgram,
    connections: Buffer<u32>,
    connection_ranges: Buffer<Uint2>,
    input_shape: [Idx; 3],
    //[height, width, channels]
    output_shape: [Idx; 3],
    //[height, width, channels]
    kernel: [Idx; 2],
    //[height, width]
    stride: [Idx; 2],
    //[height, width]
    pub k: Idx,
    pub threshold: u32,
    activity: Buffer<u32>,
    pub sums: Buffer<u32>,
    pub top_values: Buffer<u32>,
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use std::cmp::Ordering::{Greater, Less};
    use crate::{CpuSDR, from_xyz, from_xy, DenseWeight};

    #[test]
    fn test1() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut ecc = CpuEccDense::new([2,2],[2,2],[1,1],3,4,1,&mut rng);

        let mut input = CpuSDR::new();
        input.add_unique_random(4,0..ecc.in_volume());

        let p = EccProgram::default()?;
        let mut ecc2 = OclEccDense::new(&ecc,p.clone())?;

        let o = ecc.run(&input);
        let input2 = OclSDR::from_cpu(p.clone(),&input, input.cardinality())?;
        ecc2.zero_out_all_sums();
        ecc2.sums(&input2);

        let sums2 = ecc2.sums.to_vec(p.queue())?;
        assert_eq!(&sums2,&ecc.sums);
        Ok(())
    }

    #[test]
    fn test2() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut ecc = CpuEccDense::new([6,6],[2,2],[2,2],16,8,1,&mut rng);

        let mut input = CpuSDR::new();
        input.add_unique_random(16,0..ecc.in_volume());

        let p = EccProgram::default()?;
        let mut ecc2 = OclEccDense::new(&ecc,p.clone())?;

        let o = ecc.run(&input);
        let input2 = OclSDR::from_cpu(p.clone(),&input, input.cardinality())?;
        ecc2.zero_out_all_sums();
        ecc2.sums(&input2);

        let sums2 = ecc2.sums.to_vec(p.queue())?;
        assert_eq!(&sums2,&ecc.sums);
        Ok(())
    }
    #[test]
    fn test3() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut ecc = CpuEccDense::new([6,6],[2,2],[2,2],16,8,1,&mut rng);
        let p = EccProgram::default()?;
        let mut ecc2 = OclEccDense::new(&ecc,p.clone())?;
        assert_eq!(ecc.get_activity(), ecc2.activity.to_vec(p.queue()).unwrap().as_slice(),"activity before");
        for i in 0..32 {
            let mut input = CpuSDR::new();
            input.add_unique_random(ecc.in_volume(), 0..ecc.in_volume());
            let mut output = ecc.infer(&input);
            let input2 = OclSDR::from_cpu(p.clone(), &input, input.cardinality()).unwrap();
            let max_card = ecc2.output_shape.grid().product();
            let mut output2 = OclSDR::new(p.clone(),max_card).unwrap();
            assert_eq!(ecc.get_activity(), ecc2.activity.to_vec(p.queue()).unwrap().as_slice(),"activity at i={}",i);
            ecc2.zero_out_all_sums().unwrap();
            ecc2.sums(&input2).unwrap();
            p.queue().finish().unwrap();
            assert_eq!(ecc.sums.len(), ecc2.sums.len());
            let s = ecc2.sums.to_vec(p.queue()).unwrap();
            assert_eq!(ecc.sums, s.as_slice(),"sums at i={}",i);
            ecc2.zero_out_all_top_values().unwrap();
            ecc2.max_r().unwrap();
            p.q.finish().unwrap();
            let r:Vec<u32> = ecc.sums.iter().zip(ecc.get_activity().iter()).map(|(&a,&b)|a+b).collect();
            let a = ecc2.activity.to_vec(p.queue()).unwrap();
            let r2:Vec<u32> = s.iter().zip(a.iter()).map(|(&a,&b)|a+b).collect();
            assert_eq!(r, r2,"r at i={}",i);
            let mut max_r = vec![0;as_usize(ecc.out_area())];
            for x in 0..ecc.out_width(){
                for y in 0..ecc.out_height(){
                    let output_column_idx = ecc.out_shape().grid().idx(from_xy(x,y));
                    max_r[as_usize(output_column_idx)] = (0..ecc.out_channels()).map(|c|{
                        let i = ecc.out_shape().idx(from_xyz(x,y,c));
                        assert_eq!(i,output_column_idx*ecc.out_channels()+c);
                        r2[as_usize(i)]
                    }).max().unwrap();
                }
            }
            let max_r2 = ecc2.top_values.to_vec(p.queue()).unwrap();
            assert_eq!(max_r.as_slice(),&max_r2[0..max_r2.len()-1],"max_r2 at i={}",i);
            ecc2.top_1(&mut output2).unwrap();
            p.q.finish().unwrap();
            unsafe{
                output2.set_cardinality(ecc2.get_output_len()?);
            }
            output.sort();
            let mut cpu_output2 = output2.to_cpu().unwrap();
            cpu_output2.sort();
            assert_eq!(output, cpu_output2,"output at i={}",i);
            ecc.decrement_activities(&output);
            ecc2.decrement_activities_for_sdr(&output2)?;
            p.queue().finish()?;
            let decremented_a = ecc2.activity.to_vec(p.queue()).unwrap();
            let mut a = a;
            for &o in cpu_output2.iter(){
                a[as_usize(o)] -= u32::ACTIVITY_PENALTY;
            }
            assert_eq!(ecc.get_activity(),a.as_slice(),"decremented activity at i={} {:?}",i,cpu_output2);
            assert_eq!(decremented_a,a,"decremented activity2 at i={} {:?}",i,cpu_output2);
            assert_eq!(ecc.get_activity(), decremented_a.as_slice(),"decremented activity3 at i={} {:?}",i,cpu_output2);


        }
        Ok(())
    }
    #[test]
    fn test4() -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let mut ecc = CpuEccDense::new([6,6],[2,2],[2,2],16,8,1,&mut rng);
        let p = EccProgram::default()?;
        let mut ecc2 = OclEccDense::new(&ecc,p.clone())?;
        for i in 0..32 {
            let mut input = CpuSDR::new();
            input.add_unique_random(ecc.in_volume(), 0..ecc.in_volume());
            let mut output = ecc.run(&input);
            let input2 = OclSDR::from_cpu(p.clone(), &input, input.cardinality()).unwrap();
            let output2 = ecc2.run(&input2).unwrap();
            p.queue().finish()?;
            let mut output2 = output2.to_cpu().unwrap();
            assert_eq!(ecc.get_activity(), ecc2.activity.to_vec(p.queue()).unwrap().as_slice(),"i={}",i);
            output.sort();
            output2.sort();
            assert_eq!(output, output2);

        }
        Ok(())
    }
}
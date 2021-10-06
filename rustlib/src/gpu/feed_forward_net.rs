use crate::cppn::FeedForwardNet;
use ocl::{ProQue, SpatialDims, Device, Error, Platform, flags, Program, Kernel, Queue};
use crate::gpu::PREAMBLE;
use std::fmt::Write;
use ndalgebra::lin_alg_program::LinAlgProgram;
use ndalgebra::mat::{MatError, Mat, AsShape};
use crate::context::NeatContext;
use ndalgebra::kernel_builder::KernelBuilder;

pub struct FeedForwardNetOpenCL {
    in_columns: usize,
    out_columns: usize,
    program: Program,
    lin_alg: LinAlgProgram,
}

impl FeedForwardNetOpenCL {
    pub fn new(context:&NeatContext, net: &FeedForwardNet<f32>) -> Result<Self, Error> {
        let mut src = String::from(PREAMBLE);
        write!(src, "{}", net.opencl_view());
        let program = Program::builder()
            .devices(context.device().clone())
            .src(src)
            .build(context.lin_alg().pro_que.context())?;
        Ok(FeedForwardNetOpenCL {
            in_columns: net.get_input_size(),
            out_columns: net.get_output_size(),
            program,
            lin_alg:context.lin_alg().clone()
        })
    }
    pub fn get_input_size(&self) -> usize {
        self.in_columns
    }
    pub fn lin_alg(&self) -> &LinAlgProgram {
        &self.lin_alg
    }
    pub fn queue(&self) -> &Queue {
        self.lin_alg.pro_que.queue()
    }
    pub fn get_output_size(&self) -> usize {
        self.out_columns
    }

    pub fn run(&self, input: &Mat<f32>) -> Result<Mat<f32>, MatError> {
        let rows = input.shape()[0];
        if input.ndim()!=2{
            return Err(Error::from(format!("Input of shape {} was expected to be 2-dimensional", input.shape().as_shape())).into());
        }
        if input.shape()[1] != self.in_columns{
            return Err(Error::from(format!("Input of shape {} was expected to have {} columns, one for each input neuron", input.shape().as_shape(), self.in_columns)).into());
        }

        let out = unsafe{Mat::empty(&self.lin_alg, &[rows, self.out_columns ])?};
        KernelBuilder::new(&self.program, "feedforward")?
            .add_buff(input.buffer().unwrap())?
            .add_buff(out.buffer().unwrap())?
            .add_num(input.strides()[0])?
            .add_num(input.strides()[1])?
            .add_num(out.strides()[0])?
            .add_num(out.strides()[1])?
            .enq(self.queue(), &[rows])?;
        Ok(out)
    }
}
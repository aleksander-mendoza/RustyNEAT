use crate::cppn::FeedForwardNet;
use ocl::{ProQue, SpatialDims, Device, Error, Platform, flags, Program, Kernel};
use crate::gpu::PREAMBLE;
use std::fmt::Write;
use ndalgebra::kernel::LinAlgProgram;
use ndalgebra::mat::{MatError, Mat, AsShape};
use crate::context::NeatContext;

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

        let out_len = self.out_columns * rows;

        let out_buffer = self.lin_alg.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_WRITE)
            .len(out_len)
            .build()?;
        let out = Mat::from_buffer(&self.lin_alg, out_buffer, &[rows, self.out_columns ])?;
        let kernel = Kernel::builder()
            .name("feedforward")
            .program(&self.program)
            .queue(self.lin_alg.pro_que.queue().clone())
            .arg(input.buffer())
            .arg(out.buffer())
            .arg(input.strides()[0])
            .arg(input.strides()[1])
            .arg(out.strides()[0])
            .arg(out.strides()[1])
            .global_work_size(rows)
            .build()?;
        unsafe {
            kernel.cmd().enq()?;
        }
        Ok(out)
    }
}
use crate::cppn::FeedForwardNet;
use ocl::{ProQue, SpatialDims, Device, Error, Platform, flags, Program, Kernel};
use crate::gpu::PREAMBLE;
use std::fmt::Write;
use ndalgebra::kernel::LinAlgProgram;
use ndalgebra::mat::{Mat, MatError, AsShape};
use crate::context::NeatContext;

pub struct FeedForwardNetSubstrate {
    in_dimensions: usize,
    out_dimensions: usize,
    weight_dimensions: usize,
    program: Program,
    lin_alg: LinAlgProgram
}

impl FeedForwardNetSubstrate {
    pub fn new(context:&NeatContext, net: &FeedForwardNet<f32>, input_dimensions: usize, output_dimensions: usize) -> Result<Self, Error> {
        let mut src = String::from(PREAMBLE);
        write!(src, "{}", net.substrate_view(input_dimensions, output_dimensions).map_err(Error::from)?);
        let program = Program::builder()
            .devices(context.device().clone())
            .src(src)
            .build(context.lin_alg().pro_que.context())?;

        Ok(FeedForwardNetSubstrate {
            in_dimensions: input_dimensions,
            out_dimensions: output_dimensions,
            weight_dimensions: net.get_output_size(),
            program,
            lin_alg:context.lin_alg().clone()
        })
    }
    pub fn get_input_size(&self) -> usize {
        self.in_dimensions
    }
    pub fn lin_alg(&self) -> &LinAlgProgram {
        &self.lin_alg
    }
    pub fn get_output_size(&self) -> usize {
        self.out_dimensions
    }
    pub fn get_weight_size(&self) -> usize {
        self.weight_dimensions
    }
    pub fn run(&self, input_neurons: &Mat<f32>, output_neurons: &Mat<f32>) -> Result<Mat<f32>, MatError> {
        let in_neuron_count = input_neurons.shape()[0];
        let out_neuron_count = output_neurons.shape()[0];
        let in_dim = self.in_dimensions;
        let out_dim = self.out_dimensions;
        if input_neurons.shape()[1] != in_dim {
            return Err(Error::from(format!("Shape of input neuron tensor is {} but expected {} columns", input_neurons.shape().as_shape(),in_dim)).into());
        }
        if output_neurons.shape()[1] != out_dim {
            return Err(Error::from(format!("Shape of output neuron tensor is {} but expected {} columns", output_neurons.shape().as_shape(),out_dim)).into());
        }
        /*
Weights shape=(3, 3, 1)
Weights strides=(3, 1, 1)
Input shape=(3, 2)
Input strides=(2, 1)
Output shape=(3, 2)
Output strides=(2, 1)
Weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Inputs=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
Outputs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
         */
        let kernel_dim = SpatialDims::Two(in_neuron_count,out_neuron_count);
        let weights = unsafe{Mat::empty(&self.lin_alg, &[in_neuron_count,out_neuron_count,self.weight_dimensions])}?;
        let kernel = Kernel::builder()
            .name("substrate")
            .program(&self.program)
            .queue(self.lin_alg.pro_que.queue().clone())
            .arg(input_neurons.buffer().unwrap())
            .arg(output_neurons.buffer().unwrap())
            .arg(weights.buffer().unwrap())
            .arg(&weights.strides()[0])
            .arg(&weights.strides()[1])
            .arg(&weights.strides()[2])
            .arg(&input_neurons.strides()[1])
            .arg(&output_neurons.strides()[1])
            .arg(&input_neurons.strides()[0])
            .arg(&output_neurons.strides()[0])
            .global_work_size(kernel_dim)
            .build()?;
        unsafe {
            kernel.cmd().enq()?;
        }
        Ok(weights)
    }
}


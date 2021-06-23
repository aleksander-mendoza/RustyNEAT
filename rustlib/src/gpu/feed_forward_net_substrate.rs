use crate::cppn::FeedForwardNet;
use ocl::{ProQue, SpatialDims, Device, Error, Platform, flags, Program, Kernel, Queue};
use crate::gpu::PREAMBLE;
use std::fmt::Write;
use ndalgebra::kernel::LinAlgProgram;
use ndalgebra::mat::{Mat, MatError, AsShape};
use crate::context::NeatContext;
use ndalgebra::kernel_builder::KernelBuilder;

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
    pub fn queue(&self) -> &Queue {
        &self.lin_alg.pro_que.queue()
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
        let weights = unsafe{Mat::empty(&self.lin_alg, &[in_neuron_count,out_neuron_count,self.weight_dimensions])}?;
        KernelBuilder::new(&self.program,"substrate")?
            .add_buff(input_neurons.buffer().unwrap())?
            .add_buff(output_neurons.buffer().unwrap())?
            .add_buff(weights.buffer().unwrap())?
            .add_num(weights.strides()[0])?
            .add_num(weights.strides()[1])?
            .add_num(weights.strides()[2])?
            .add_num(input_neurons.strides()[1])?
            .add_num(output_neurons.strides()[1])?
            .add_num(input_neurons.strides()[0])?
            .add_num(output_neurons.strides()[0])?
            .enq(self.queue(),&[in_neuron_count,out_neuron_count])?;
        Ok(weights)
    }
}


use crate::cppn::FeedForwardNet;
use ocl::{ProQue, SpatialDims, Device, Error, Platform, flags, Program, Kernel, Queue};
use crate::gpu::PREAMBLE;
use std::fmt::Write;
use ndalgebra::mat::{Mat, MatError};
use ndalgebra::lin_alg_program::LinAlgProgram;
use ndalgebra::buffer::Buffer;
use ndalgebra::kernel_builder::KernelBuilder;
use ndalgebra::context::Context;

pub struct FeedForwardNetPicbreeder {
    in_dimensions: usize,
    out_dimensions: usize,
    program: Program,
    lin_alg: LinAlgProgram,
}

impl FeedForwardNetPicbreeder {
    pub fn new(ling_alg:&LinAlgProgram  , net: &FeedForwardNet<f32>, with_distance_from_center: Option<&[f32]>, with_bias: bool) -> Result<Self, Error> {
        let mut src = String::from(PREAMBLE);
        write!(src, "{}", net.picbreeder_view(with_distance_from_center, with_bias).map_err(Error::from)?);
        let program = Program::builder()
            .src(src)
            .build(ling_alg.context())?;
        let spacial_dimensions = net.get_input_size() - if with_distance_from_center.is_some() { 1 } else { 0 } - if with_bias { 1 } else { 0 };
        Ok(FeedForwardNetPicbreeder {
            in_dimensions: spacial_dimensions,
            out_dimensions: net.get_output_size(),
            program,
            lin_alg:ling_alg.clone()
        })
    }
    pub fn get_input_size(&self) -> usize {
        self.in_dimensions
    }
    pub fn lin_alg(&self) -> &LinAlgProgram {
        &self.lin_alg
    }
    pub fn queue(&self) -> &Queue {
        self.lin_alg.queue()
    }
    pub fn get_output_size(&self) -> usize {
        self.out_dimensions
    }
    pub fn run(&self, shape: &[usize], pixel_size: &[f32], pixel_offset: &[f32]) -> Result<Mat<f32>, MatError> {
        let pixels = shape.iter().fold(1,|a,&b|a*b);
        if shape.len() % self.in_dimensions != 0 {
            return Err(Error::from(format!("Input shape {:?} has {} dimensions but expected {} dimensions", shape, shape.len(), self.in_dimensions)).into());
        }
        if pixel_size.len() % self.in_dimensions != 0 {
            return Err(Error::from(format!("Input pixel_size {:?} has {} dimensions but expected {} dimensions", pixel_size, pixel_size.len(), self.in_dimensions)).into());
        }
        if pixel_offset.len() % self.in_dimensions != 0 {
            return Err(Error::from(format!("Input pixel_offset {:?} has {} dimensions but expected {} dimensions", pixel_offset, pixel_offset.len(), self.in_dimensions)).into());
        }
        let out_len = self.out_dimensions * pixels;

        let pixel_size_per_dimension = self.lin_alg.buffer_from_slice(flags::MEM_READ_ONLY,pixel_size)?;
        let offset_per_dimension = self.lin_alg.buffer_from_slice(flags::MEM_READ_ONLY, pixel_offset)?;
        let out_buffer = unsafe{self.lin_alg.buffer_empty(flags::MEM_READ_WRITE,out_len)}?;
        let dimensions = self.lin_alg.buffer_from_slice(flags::MEM_READ_ONLY,shape)?;

        KernelBuilder::new(&self.program, "picbreeder")?
            .add_buff(&out_buffer)?
            .add_buff(&dimensions)?
            .add_buff(&pixel_size_per_dimension)?
            .add_buff(&offset_per_dimension)?
            .enq(self.queue(), &[pixels])?;
        let mut mat_shape = Vec::with_capacity(shape.len()+1);
        mat_shape.extend_from_slice(shape);
        mat_shape.push(self.out_dimensions);
        Mat::from_buffer_boxed(&self.lin_alg,out_buffer,mat_shape.into_boxed_slice())
    }
}


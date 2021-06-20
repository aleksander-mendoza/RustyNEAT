use crate::cppn::FeedForwardNet;
use ocl::{ProQue, SpatialDims, Device, Error, Platform, flags, Program, Kernel};
use crate::gpu::PREAMBLE;
use std::fmt::Write;
use ndalgebra::mat::{Mat, MatError};
use ndalgebra::kernel::LinAlgProgram;
use crate::context::NeatContext;

pub struct FeedForwardNetPicbreeder {
    in_dimensions: usize,
    out_dimensions: usize,
    program: Program,
    lin_alg: LinAlgProgram,
}

impl FeedForwardNetPicbreeder {
    pub fn new(context:&NeatContext, net: &FeedForwardNet<f32>, with_distance_from_center: Option<&[f32]>, with_bias: bool) -> Result<Self, Error> {
        let mut src = String::from(PREAMBLE);
        write!(src, "{}", net.picbreeder_view(with_distance_from_center, with_bias).map_err(Error::from)?);
        let program = Program::builder()
            .devices(context.device().clone())
            .src(src)
            .build(context.lin_alg().pro_que.context())?;
        let spacial_dimensions = net.get_input_size() - if with_distance_from_center.is_some() { 1 } else { 0 } - if with_bias { 1 } else { 0 };
        Ok(FeedForwardNetPicbreeder {
            in_dimensions: spacial_dimensions,
            out_dimensions: net.get_output_size(),
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
    pub fn run(&self, shape: &[usize], pixel_size: &[f32], pixel_offset: &[f32]) -> Result<Mat<f32>, MatError> {
        let pixels = shape.iter().fold(1, |a, b| a * b);
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

        let buffer = self.lin_alg.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_WRITE)
            .len(pixel_size.len() + pixel_offset.len())
            .build()?;

        let pixel_size_per_dimension = buffer.create_sub_buffer(Some(flags::MEM_READ_ONLY), SpatialDims::Unspecified, pixel_size.len())?;
        let offset_per_dimension = buffer.create_sub_buffer(Some(flags::MEM_READ_ONLY), pixel_size.len(), pixel_offset.len())?;
        let out_buffer = self.lin_alg.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_WRITE)
            .len(out_len)
            .build()?;

        let dimensions = self.lin_alg.pro_que.buffer_builder::<usize>()
            .flags(flags::MEM_READ_ONLY)
            .len(shape.len())
            .copy_host_slice(shape)
            .build()?;

        let kernel = Kernel::builder()
            .name("picbreeder")
            .program(&self.program)
            .queue(self.lin_alg.pro_que.queue().clone())
            .arg(&out_buffer)
            .arg(&dimensions)
            .arg(&pixel_size_per_dimension)
            .arg(&offset_per_dimension)
            .global_work_size(pixels)
            .build()?;
        unsafe {
            pixel_size_per_dimension.cmd()
                .queue(&self.lin_alg.pro_que.queue())
                .offset(0)
                .write(pixel_size)
                .enq()?;
            offset_per_dimension.cmd()
                .queue(&self.lin_alg.pro_que.queue())
                .offset(0)
                .write(pixel_offset)
                .enq()?;
            kernel.cmd().enq()?;
        }
        let mut mat_shape = Vec::with_capacity(shape.len()+1);
        mat_shape.extend_from_slice(shape);
        mat_shape.push(self.out_dimensions);
        Mat::from_buffer_boxed(&self.lin_alg,out_buffer,mat_shape.into_boxed_slice())
    }
}


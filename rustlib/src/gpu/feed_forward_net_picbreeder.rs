use crate::cppn::FeedForwardNet;
use ocl::{ProQue, SpatialDims, Device, Error, Platform, flags};
use crate::gpu::PREAMBLE;
use std::fmt::Write;

pub struct FeedForwardNetPicbreeder {
    in_dimensions: usize,
    out_dimensions: usize,
    pro_que: ProQue,
}

impl FeedForwardNetPicbreeder {
    pub fn new(net: &FeedForwardNet<f32>, with_distance_from_center: Option<&[f32]>, with_bias: bool, platform: Platform, device: Device) -> Result<Self, Error> {
        let mut src = String::from(PREAMBLE);
        write!(src, "{}", net.picbreeder_view(with_distance_from_center, with_bias).map_err(Error::from)?);
        let pro_que = ProQue::builder()
            .platform(platform)
            .device(device)
            .src(src)
            .dims(SpatialDims::Unspecified)
            .build()?;
        let spacial_dimensions = net.get_input_size() - if with_distance_from_center.is_some() { 1 } else { 0 } - if with_bias { 1 } else { 0 };
        Ok(FeedForwardNetPicbreeder { in_dimensions: spacial_dimensions, out_dimensions: net.get_output_size(), pro_que })
    }
    pub fn get_input_size(&self) -> usize {
        self.in_dimensions
    }
    pub fn get_device(&self) -> Device {
        self.pro_que.device()
    }
    pub fn get_output_size(&self) -> usize {
        self.out_dimensions
    }
    pub fn run(&self, shape: &[usize], pixel_size: &[f32], pixel_offset: &[f32]) -> Result<Vec<f32>, Error> {
        let pixels = shape.iter().fold(1, |a, b| a * b);
        if shape.len() % self.in_dimensions != 0 {
            return Err(Error::from(format!("Input shape {:?} has {} dimensions but expected {} dimensions", shape, shape.len(), self.in_dimensions)));
        }
        if pixel_size.len() % self.in_dimensions != 0 {
            return Err(Error::from(format!("Input pixel_size {:?} has {} dimensions but expected {} dimensions", pixel_size, pixel_size.len(), self.in_dimensions)));
        }
        if pixel_offset.len() % self.in_dimensions != 0 {
            return Err(Error::from(format!("Input pixel_offset {:?} has {} dimensions but expected {} dimensions", pixel_offset, pixel_offset.len(), self.in_dimensions)));
        }
        let out_len = self.out_dimensions * pixels;

        let buffer = self.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_WRITE)
            .len(pixel_size.len() + pixel_offset.len() + out_len)
            .build()?;

        let pixel_size_per_dimension = buffer.create_sub_buffer(Some(flags::MEM_READ_ONLY), SpatialDims::Unspecified, pixel_size.len())?;
        let offset_per_dimension = buffer.create_sub_buffer(Some(flags::MEM_READ_ONLY), pixel_size.len(), pixel_offset.len())?;
        let out_buffer = buffer.create_sub_buffer(None, pixel_size.len() + pixel_offset.len(), out_len)?;

        let dimensions = self.pro_que.buffer_builder::<usize>()
            .flags(flags::MEM_READ_ONLY)
            .len(shape.len())
            .copy_host_slice(shape)
            .build()?;

        let kernel = self.pro_que.kernel_builder("picbreeder")
            .arg(&out_buffer)
            .arg(&dimensions)
            .arg(&pixel_size_per_dimension)
            .arg(&offset_per_dimension)
            .global_work_size(pixels)
            .build()?;
        unsafe {
            pixel_size_per_dimension.cmd()
                .queue(&self.pro_que.queue())
                .offset(0)
                .write(pixel_size)
                .enq()?;
            offset_per_dimension.cmd()
                .queue(&self.pro_que.queue())
                .offset(0)
                .write(pixel_offset)
                .enq()?;
            kernel.cmd()
                .queue(&self.pro_que.queue())
                .global_work_offset(kernel.default_global_work_offset())
                .global_work_size(pixels)
                .local_work_size(kernel.default_local_work_size())
                .enq()?;
        }
        let mut output = Vec::with_capacity(out_len);
        unsafe {
            output.set_len(out_len);
            out_buffer.cmd()
                .queue(&self.pro_que.queue())
                .offset(0)
                .read(&mut output)
                .enq()?;
        }
        Ok(output)
    }
}


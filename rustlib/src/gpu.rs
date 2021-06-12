use crate::cppn::FeedForwardNet;
use std::fmt::Write;
use ocl::{ProQue, Kernel, Buffer, flags, SpatialDims, Platform, Device};
use ocl::Error;
use ocl::core::ClVersions;


pub const PREAMBLE: &'static str = r#"
        float relu32(float x) {
            return max(x, 0.f);
        }
        float sigmoid32(float x) {
            return 1.0 / (1.0 + exp(-x));
        }
        float square32(float x) {
            return x*x;
        }
        float step32(float x) {
            return x <= 0 ? 0 : 1;
        }
        float const_1_32(float x) {
            return 1.;
        }
        float exp32(float x) {
            return exp(x);
        }
        float const_e_32(float x) {
            return 2.71828182845904523536028747135266250;
        }
        float const_pi_32(float x) {
            return 3.14159265358979323846264338327950288;
        }
        float const_neg1_32(float x) {
            return -1.;
        }
        float gaussian32(float z) {
            return 1./(const_pi_32(0)*0.5*0.5)*exp32(-z*z/(2.*0.5*0.5));
        }
        float fraction32(float z) {
            return z - (long)z;
        }
"#;

pub struct FeedForwardNetOpenCL {
    in_columns: usize,
    out_columns: usize,
    pro_que: ProQue,
}

impl FeedForwardNetOpenCL {
    pub fn new(net: &FeedForwardNet<f32>, platform: Platform, device: Device) -> Result<Self, Error> {
        let mut src = String::from(PREAMBLE);
        write!(src, "{}", net.opencl_view());
        let pro_que = ProQue::builder()
            .platform(platform)
            .device(device)
            .src(src)
            .dims(SpatialDims::Unspecified)
            .build()?;
        Ok(FeedForwardNetOpenCL { in_columns: net.get_input_size(), out_columns: net.get_output_size(), pro_que })
    }
    pub fn get_input_size(&self) -> usize {
        self.in_columns
    }
    pub fn get_device(&self) -> Device {
        self.pro_que.device()
    }
    pub fn get_output_size(&self) -> usize {
        self.out_columns
    }
    pub fn run(&self, input: &[f32], row_major: bool) -> Result<Vec<f32>, Error> {
        let rows = input.len() / self.in_columns;
        let (in_col_stride, in_row_stride) = if row_major { (1, self.in_columns) } else { (rows, 1) };
        let (out_col_stride, out_row_stride) = if row_major { (1, self.out_columns) } else { (rows, 1) };
        self.run_with_strides(input, in_col_stride, in_row_stride, out_col_stride, out_row_stride)
    }
    pub fn run_with_strides(&self, input: &[f32],
                            in_col_stride: usize, in_row_stride: usize,
                            out_col_stride: usize, out_row_stride: usize) -> Result<Vec<f32>, Error> {
        let rows = input.len() / self.in_columns;
        if input.len() % self.in_columns != 0 {
            return Err(Error::from(format!("Input buffer has length {} which is not divisible by number of expected input nodes {}", input.len(), self.in_columns)));
        }
        let in_buffer = self.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_ONLY)
            .len(input.len())
            .build()?;

        let out_len = self.out_columns * rows;

        let out_buffer = self.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_WRITE)
            .len(out_len)
            .build()?;

        let kernel = self.pro_que.kernel_builder("feedforward")
            .arg(&in_buffer)
            .arg(&out_buffer)
            .arg(in_row_stride)
            .arg(in_col_stride)
            .arg(out_row_stride)
            .arg(out_col_stride)
            .global_work_size(rows)
            .build()?;
        unsafe {
            in_buffer.cmd()
                .queue(&self.pro_que.queue())
                .offset(0)
                .write(input)
                .enq()?;
        }
        unsafe {
            kernel.cmd()
                .queue(&self.pro_que.queue())
                .global_work_offset(kernel.default_global_work_offset())
                .global_work_size(rows)
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


pub struct FeedForwardNetSubstrate {
    in_dimensions: usize,
    out_dimensions: usize,
    weight_dimensions: usize,
    pro_que: ProQue,
}

impl FeedForwardNetSubstrate {
    pub fn new(net: &FeedForwardNet<f32>, input_dimensions: usize, output_dimensions: usize, platform: Platform, device: Device) -> Result<Self, Error> {
        let mut src = String::from(PREAMBLE);
        write!(src, "{}", net.substrate_view(input_dimensions, output_dimensions).map_err(Error::from)?);
        let pro_que = ProQue::builder()
            .platform(platform)
            .device(device)
            .src(src)
            .dims(SpatialDims::Unspecified)
            .build()?;
        Ok(FeedForwardNetSubstrate {
            in_dimensions: input_dimensions,
            out_dimensions: output_dimensions,
            weight_dimensions: net.get_output_size(),
            pro_que,
        })
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
    pub fn get_weight_size(&self) -> usize {
        self.weight_dimensions
    }
    pub fn run(&self, input_neurons: &[f32], output_neurons: &[f32],
               out_row_stride: usize, out_col_stride: usize, out_depth_stride: usize,
               in_neuron_col_stride: usize, out_neuron_col_stride: usize,
               in_neuron_row_stride: usize, out_neuron_row_stride: usize) -> Result<Vec<f32>, Error> {
        let in_neuron_count = input_neurons.len() / in_neuron_row_stride;
        let out_neuron_count = output_neurons.len() / out_neuron_row_stride;
        let in_dim = self.in_dimensions;
        let out_dim = self.out_dimensions;
        if (in_neuron_count - 1) * in_neuron_row_stride + (in_dim - 1) * in_neuron_col_stride != input_neurons.len() - 1 {
            return Err(Error::from(format!("Input length of {} is not compatible with row/column strides {}/{}", input_neurons.len(), in_neuron_row_stride, in_neuron_col_stride)));
        }
        if (out_neuron_count - 1) * out_neuron_row_stride + (out_dim - 1) * out_neuron_col_stride != output_neurons.len() - 1 {
            return Err(Error::from(format!("Output length of {} is not compatible with row/column strides {}/{}", output_neurons.len(), out_neuron_row_stride, out_neuron_col_stride)));
        }
        let total_weights = in_neuron_count * out_neuron_count;
        let kernel_dim = SpatialDims::Two(in_neuron_count,out_neuron_count);
        let weight_buff_len = self.weight_dimensions * total_weights;

        let weight_buff = self.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_WRITE)
            .len(weight_buff_len)
            .build()?;

        let input_buff = self.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_ONLY)
            .len(input_neurons.len())
            .copy_host_slice(input_neurons)
            .build()?;

        let output_buff = self.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_ONLY)
            .len(output_neurons.len())
            .copy_host_slice(output_neurons)
            .build()?;

        let kernel = self.pro_que.kernel_builder("substrate")
            .arg(&input_buff)
            .arg(&output_buff)
            .arg(&weight_buff)
            .arg(&out_row_stride)
            .arg(&out_col_stride)
            .arg(&out_depth_stride)
            .arg(&in_neuron_col_stride)
            .arg(&out_neuron_col_stride)
            .arg(&in_neuron_row_stride)
            .arg(&out_neuron_row_stride)
            .global_work_size(kernel_dim)
            .build()?;
        unsafe {
            kernel.cmd()
                .queue(&self.pro_que.queue())
                .enq()?;
        }
        let mut weights = Vec::with_capacity(weight_buff_len);
        unsafe {
            weights.set_len(weight_buff_len);
            weight_buff.cmd()
                .queue(&self.pro_que.queue())
                .offset(0)
                .read(&mut weights)
                .enq()?;
        }
        Ok(weights)
    }
}



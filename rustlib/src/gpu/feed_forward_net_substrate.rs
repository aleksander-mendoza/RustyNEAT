use crate::cppn::FeedForwardNet;
use ocl::{ProQue, SpatialDims, Device, Error, Platform, flags};
use crate::gpu::PREAMBLE;
use std::fmt::Write;

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


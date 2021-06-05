use crate::cppn::FeedForwardNet;
use std::io::Write;
use ocl::{ProQue, Kernel, Buffer, flags, SpatialDims};
use ocl::Error;


pub struct FeedForwardNetOpenCL {
    in_columns:usize,
    out_columns:usize,
    pro_que:ProQue
}

impl FeedForwardNetOpenCL {
    pub fn new(net: FeedForwardNet<f32>) -> Result<Self, Error> {
        let src = format!(r#"
        float relu32(float x) {{
            return max(x, 0.f);
        }}
        float sigmoid32(float x) {{
            return 1.0 / (1.0 + exp(-x));
        }}
        float square32(float x) {{
            return x*x;
        }}
        {}
"#,net);
        let pro_que = ProQue::builder()
            .src(src)
            .dims(SpatialDims::Unspecified)
            .build()?;
        Ok(FeedForwardNetOpenCL {in_columns:net.get_input_size(),out_columns:net.get_output_size(),pro_que})
    }
    pub fn run(&self,input:&[f32],row_major:bool)->Result<Vec<f32>, Error>{
        if input.len() % self.in_columns != 0{
            return Err(Error::from(format!("Input buffer has length {} which is not divisible by number of expected input nodes {}",input.len(), self.in_columns)));
        }
        let rows = input.len()/self.in_columns;
        let (in_col_stride,in_row_stride) = if row_major{(1, self.in_columns)}else{(rows,1)};
        let (out_col_stride,out_row_stride) = if row_major{(1, self.out_columns)}else{(rows,1)};

        let in_buffer = self.pro_que.buffer_builder::<f32>()
            .flags(flags::MEM_READ_ONLY)
            .len(input.len())
            .build()?;

        let out_len = self.out_columns*rows;

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
                .write( input)
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



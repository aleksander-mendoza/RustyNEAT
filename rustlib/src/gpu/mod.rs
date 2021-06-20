mod feed_forward_net_substrate;
mod feed_forward_net_picbreeder;
mod feed_forward_net;


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

pub use feed_forward_net::FeedForwardNetOpenCL;
pub use feed_forward_net_picbreeder::FeedForwardNetPicbreeder;
pub use feed_forward_net_substrate::FeedForwardNetSubstrate;




#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::real::Real;
    use rand::Rng;
    use ocl::{SpatialDims, Platform, Device, Context, Program, Queue, Buffer, flags, Kernel, ProQue};
    use ocl::core::{BufferRegion};
    use std::ffi::CString;

    #[test]
    fn test_tch() -> ocl::Result<()> {
        let src = r#"
        __kernel void sin32(__global float * buffer) {
            buffer[get_global_id(0)] = sin(buffer[get_global_id(0)]);
        }
        __kernel void cos32(__global float * buffer) {
            buffer[get_global_id(0)] = cos(buffer[get_global_id(0)]);
        }
        __kernel void tan32(__global float * buffer) {
            buffer[get_global_id(0)] = tan(buffer[get_global_id(0)]);
        }
        __kernel void tanh32(__global float * buffer) {
            buffer[get_global_id(0)] = tanh(buffer[get_global_id(0)]);
        }
        __kernel void relu32(__global float * buffer) {
            buffer[get_global_id(0)] = max(buffer[get_global_id(0)], 0.f);
        }
        __kernel void sigmoid32(__global float * buffer) {
            buffer[get_global_id(0)] = 1.0 / (1.0 + exp(-buffer[get_global_id(0)]));
        }
        __kernel void abs32(__global float * buffer) {
            buffer[get_global_id(0)] = fabs(buffer[get_global_id(0)]);
        }
        __kernel void square32(__global float * buffer) {
            float f = buffer[get_global_id(0)];
            buffer[get_global_id(0)] = f*f;
        }
        __kernel void const32(__global float * buffer, float val) {
            buffer[get_global_id(0)] = val;
        }
        __kernel void identity32(__global float * buffer) {
        }
        __kernel void add32(__global float * in_buffer,__global float * out_buffer) {
            out_buffer[get_global_id(0)] += in_buffer[get_global_id(0)];
        }

    "#;

        let pro_que = ProQue::builder()
            .src(src)
            .dims(64 * 64)
            .build()?;

        let buffer = pro_que.buffer_builder::<f32>().len(64 * 64 * 2).build()?;
        let in_buffer = buffer.create_sub_buffer(None, SpatialDims::Unspecified, 64 * 64)?;
        let out_buffer = buffer.create_sub_buffer(None, 64 * 64, 64 * 64)?;
        println!("{}", buffer.len());
        println!("{:?}", buffer);
        println!("{}", in_buffer.len());
        println!("{}", out_buffer.len());
        let const32 = pro_que.kernel_builder("const32")
            .arg(&in_buffer)
            .arg(3f32)
            .build()?;
        let add32 = pro_que.kernel_builder("add32")
            .arg(&in_buffer)
            .arg(&out_buffer)
            .build()?;
        let square32 = pro_que.kernel_builder("square32")
            .arg(&out_buffer)
            .build()?;

        unsafe { const32.enq()?; }
        unsafe { add32.enq()?; }
        unsafe { square32.enq()?; }

        let mut vec = vec![0.0f32; out_buffer.len()];
        out_buffer.read(&mut vec).enq()?;

        println!("The value at index [{}] is now '{}'!", 4, vec[4]);
        Ok(())
    }


    #[test]
    fn test_tch_share_buff_between_progs() -> ocl::Result<()> {
        use std::ffi::CString;
        use ocl::{core, flags};
        use ocl::enums::ArgVal;
        use ocl::builders::ContextProperties;

        let src = r#"
        __kernel void fill(__global float * buff) {
            buff[get_global_id(0)] = 1.0;
        }
    "#;

        let platform_id = core::default_platform()?;
        let device_ids = core::get_device_ids(&platform_id, None, None)?;
        let device_id = device_ids[0];
        let context_properties = ContextProperties::new().platform(platform_id);
        let context = core::create_context(Some(&context_properties), &[device_id], None, None)?;
        let src_cstring = CString::new(src)?;
        let program1 = core::create_program_with_source(&context, &[src_cstring])?;
        core::build_program(&program1, Some(&[device_id]), &CString::new("")?,
                            None, None)?;
        let src = r#"
        __kernel void fill(__global float * buff) {
            buff[get_global_id(0)] = 2.0;
        }
    "#;
        let src_cstring = CString::new(src)?;
        let program2 = core::create_program_with_source(&context, &[src_cstring])?;
        core::build_program(&program2, Some(&[device_id]), &CString::new("")?,
                            None, None)?;
        let queue = core::create_command_queue(&context, &device_id, None)?;
        let buffer = unsafe { core::create_buffer(&context, flags::MEM_ALLOC_HOST_PTR, 32, None::<&[f32]>)? };
        let fill1 = core::create_kernel(&program1, "fill")?;
        let fill2 = core::create_kernel(&program2, "fill")?;
        core::set_kernel_arg(&fill1, 0, ArgVal::mem(&buffer))?;
        unsafe {
            core::enqueue_kernel(&queue, &fill1, 1, None, &[32, 1, 1],
                                 None, None::<core::Event>, None::<&mut core::Event>)?;
        }

        let mut vec = vec![0.0f32; 32];
        unsafe {
            core::enqueue_read_buffer(&queue, &buffer, true, 0, vec.as_mut_slice(),
                                      None::<core::Event>, None::<&mut core::Event>)?;
        }
        assert!(vec.iter().all(|&x|x==1.0));

        core::set_kernel_arg(&fill2, 0, ArgVal::mem(&buffer))?;
        unsafe {
            core::enqueue_kernel(&queue, &fill2, 1, None, &[32, 1, 1],
                                 None, None::<core::Event>, None::<&mut core::Event>)?;
            core::enqueue_read_buffer(&queue, &buffer, true, 0, vec.as_mut_slice(),
                                      None::<core::Event>, None::<&mut core::Event>)?;
        }
        assert!(vec.iter().all(|&x|x==2.0));
        Ok(())
    }

    #[test]
    fn test_tch2() -> ocl::Result<()> {
        let src = r#"
        __kernel void sin32(__global float * buffer) {
            buffer[get_global_id(0)] = sin(buffer[get_global_id(0)]);
        }
        __kernel void cos32(__global float * buffer) {
            buffer[get_global_id(0)] = cos(buffer[get_global_id(0)]);
        }
        __kernel void tan32(__global float * buffer) {
            buffer[get_global_id(0)] = tan(buffer[get_global_id(0)]);
        }
        __kernel void tanh32(__global float * buffer) {
            buffer[get_global_id(0)] = tanh(buffer[get_global_id(0)]);
        }
        __kernel void relu32(__global float * buffer) {
            buffer[get_global_id(0)] = max(buffer[get_global_id(0)], 0.f);
        }
        __kernel void sigmoid32(__global float * buffer) {
            buffer[get_global_id(0)] = 1.0 / (1.0 + exp(-buffer[get_global_id(0)]));
        }
        __kernel void abs32(__global float * buffer) {
            buffer[get_global_id(0)] = fabs(buffer[get_global_id(0)]);
        }
        __kernel void square32(__global float * buffer) {
            float f = buffer[get_global_id(0)];
            buffer[get_global_id(0)] = f*f;
        }
        __kernel void const32(__global float * buffer, float val) {
            buffer[get_global_id(0)] = val;
        }
        __kernel void identity32(__global float * buffer) {
        }
        __kernel void add32(__global float * in_buffer,__global float * out_buffer) {
            out_buffer[get_global_id(0)] += in_buffer[get_global_id(0)];
        }

    "#;
        let dims = SpatialDims::new(Some(64 * 64 * 2), None, None)?;
        let platform = Platform::default();
        let device = Device::first(platform)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()?;
        let program = Program::builder()
            .devices(device)
            .src(src)
            .build(&context)?;
        let queue = Queue::new(&context, device, None)?;
        let buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(dims)
            .build()?;
        let d = SpatialDims::new(Some(64 * 64), None, None)?;
        let in_buffer = buffer.create_sub_buffer(None, SpatialDims::new(Some(0), None, None)?, d)?;
        let out_buffer = buffer.create_sub_buffer(None, d, d)?;
        println!("{}", buffer.len());
        println!("{:?}", buffer);
        println!("{}", in_buffer.len());
        println!("{}", out_buffer.len());
        let const32 = Kernel::builder()
            .program(&program)
            .name("const32")
            .queue(queue.clone())
            .global_work_size(dims)
            .arg(&buffer)
            .arg(3f32)
            .build()?;
        let add32 = Kernel::builder()
            .program(&program)
            .name("add32")
            .queue(queue.clone())
            .global_work_size(d)
            .arg(&in_buffer)
            .arg(&out_buffer)
            .build()?;
        let square32 = Kernel::builder()
            .program(&program)
            .name("square32")
            .queue(queue.clone())
            .global_work_size(d)
            .arg(&out_buffer)
            .build()?;

        unsafe {
            const32.cmd()
                .queue(&queue)
                .global_work_offset(const32.default_global_work_offset())
                .global_work_size(dims)
                .local_work_size(const32.default_local_work_size())
                .enq()?;
        }
        println!("const32 {:?} {:?}", add32.default_global_work_offset(), add32.default_global_work_size());
        unsafe {
            add32.cmd()
                .queue(&queue)
                .global_work_offset(add32.default_global_work_offset())
                .global_work_size(d)
                .local_work_size(add32.default_local_work_size())
                .enq()?;
        }
        unsafe {
            square32.cmd()
                .queue(&queue)
                .global_work_offset(square32.default_global_work_offset())
                .global_work_size(d)
                .local_work_size(square32.default_local_work_size())
                .enq()?;
        }

        let mut vec = vec![0.0f32; in_buffer.len()];
        in_buffer.cmd()
            .queue(&queue)
            .offset(0)
            .read(&mut vec)
            .enq()?;

        println!("The value at index [{}] is now '{}'!", 4, vec[4]);
        let mut vec = vec![0.0f32; out_buffer.len()];
        out_buffer.cmd()
            .queue(&queue)
            .offset(0)
            .read(&mut vec)
            .enq()?;

        println!("The value at index [{}] is now '{}'!", 4, vec[4]);
        Ok(())
    }

    #[test]
    fn test_tch_core() -> ocl::Result<()> {
        use std::ffi::CString;
        use ocl::{core, flags};
        use ocl::enums::ArgVal;
        use ocl::builders::ContextProperties;
        let src = r#"
        __kernel void sin32(__global float * buffer) {
            buffer[get_global_id(0)] = sin(buffer[get_global_id(0)]);
        }
        __kernel void cos32(__global float * buffer) {
            buffer[get_global_id(0)] = cos(buffer[get_global_id(0)]);
        }
        __kernel void tan32(__global float * buffer) {
            buffer[get_global_id(0)] = tan(buffer[get_global_id(0)]);
        }
        __kernel void tanh32(__global float * buffer) {
            buffer[get_global_id(0)] = tanh(buffer[get_global_id(0)]);
        }
        __kernel void relu32(__global float * buffer) {
            buffer[get_global_id(0)] = max(buffer[get_global_id(0)], 0.f);
        }
        __kernel void sigmoid32(__global float * buffer) {
            buffer[get_global_id(0)] = 1.0 / (1.0 + exp(-buffer[get_global_id(0)]));
        }
        __kernel void abs32(__global float * buffer) {
            buffer[get_global_id(0)] = fabs(buffer[get_global_id(0)]);
        }
        __kernel void square32(__global float * buffer) {
            float f = buffer[get_global_id(0)];
            buffer[get_global_id(0)] = f*f;
        }
        __kernel void const32(__global float * buffer, float val) {
            buffer[get_global_id(0)] = val;
        }
        __kernel void identity32(__global float * buffer) {
        }
        __kernel void add32(__global float * in_buffer,__global float * out_buffer) {
            out_buffer[get_global_id(0)] += in_buffer[get_global_id(0)];
        }

    "#;
        let platform_id = core::default_platform()?;
        let device_ids = core::get_device_ids(&platform_id, None, None)?;
        let device_id = device_ids[0];
        let context_properties = ContextProperties::new().platform(platform_id);
        let context = core::create_context(Some(&context_properties),
                                           &[device_id], None, None)?;
        let src_cstring = CString::new(src)?;
        let program = core::create_program_with_source(&context, &[src_cstring])?;
        core::build_program(&program, Some(&[device_id]), &CString::new("")?,
                            None, None)?;
        let queue = core::create_command_queue(&context, &device_id, None)?;
        let dims = [64 * 64, 1, 1];

        let buffer = unsafe { core::create_buffer(&context, flags::MEM_ALLOC_HOST_PTR, 64 * 64 * 2, None::<&[f32]>)? };
        let buff_reg = BufferRegion::<f32>::new(0, 64 * 64);
        let in_buffer = unsafe { core::create_sub_buffer(&buffer, flags::MEM_READ_WRITE, &buff_reg)? };
        let buff_reg = BufferRegion::<f32>::new(64 * 64, 64 * 64);
        let out_buffer = unsafe { core::create_sub_buffer(&buffer, flags::MEM_READ_WRITE, &buff_reg)? };
        let const32 = core::create_kernel(&program, "const32")?;
        core::set_kernel_arg(&const32, 0, ArgVal::mem(&in_buffer))?;
        core::set_kernel_arg(&const32, 1, ArgVal::scalar(&3f32))?;
        unsafe {
            core::enqueue_kernel(&queue, &const32, 1, None, &dims,
                                 None, None::<core::Event>, None::<&mut core::Event>)?;
        }
        let const32 = core::create_kernel(&program, "const32")?;
        core::set_kernel_arg(&const32, 0, ArgVal::mem(&out_buffer))?;
        core::set_kernel_arg(&const32, 1, ArgVal::scalar(&0f32))?;
        unsafe {
            core::enqueue_kernel(&queue, &const32, 1, None, &dims,
                                 None, None::<core::Event>, None::<&mut core::Event>)?;
        }
        let add32 = core::create_kernel(&program, "add32")?;
        core::set_kernel_arg(&add32, 0, ArgVal::mem(&in_buffer))?;
        core::set_kernel_arg(&add32, 1, ArgVal::mem(&out_buffer))?;
        unsafe {
            core::enqueue_kernel(&queue, &add32, 1, None, &dims,
                                 None, None::<core::Event>, None::<&mut core::Event>)?;
        }
        let square32 = core::create_kernel(&program, "square32")?;
        core::set_kernel_arg(&square32, 0, ArgVal::mem(&out_buffer))?;
        unsafe {
            core::enqueue_kernel(&queue, &square32, 1, None, &dims,
                                 None, None::<core::Event>, None::<&mut core::Event>)?;
        }

        let mut vec = vec![0.0f32; 64 * 64];
        unsafe {
            core::enqueue_read_buffer(&queue, &in_buffer, true, 0, &mut vec,
                                      None::<core::Event>, None::<&mut core::Event>)?;
        }
        println!("The value at index [{}] is now '{}'!", 4, vec[4]);
        let mut vec = vec![0.0f32; 64 * 64];
        unsafe {
            core::enqueue_read_buffer(&queue, &out_buffer, true, 0, &mut vec,
                                      None::<core::Event>, None::<&mut core::Event>)?;
        }
        println!("The value at index [{}] is now '{}'!", 4, vec[4]);
        Ok(())
    }

}
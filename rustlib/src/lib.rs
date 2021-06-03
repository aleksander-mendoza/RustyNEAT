#[macro_use] extern crate maplit;
#[macro_use] extern crate lazy_static;


pub mod neat;
pub mod activations;
pub mod num;
pub mod util;
pub mod cppn;

extern crate ocl;
use ocl::ProQue;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tch() -> ocl::Result<()>  {
        let src = r#"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }
    "#;

        let pro_que = ProQue::builder()
            .src(src)
            .dims(1 << 20)
            .build()?;

        let buffer = pro_que.create_buffer::<f32>()?;
        println!("{}",buffer.len());
        println!("{:?}",buffer);
        let kernel = pro_que.kernel_builder("add")
            .arg(&buffer)
            .arg(10.0f32)
            .build()?;

        unsafe { kernel.enq()?; }

        let mut vec = vec![0.0f32; buffer.len()];
        buffer.read(&mut vec).enq()?;

        println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);
        Ok(())
    }
}

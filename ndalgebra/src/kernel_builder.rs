use ocl::core::{Kernel, ArgVal, Mem, Error};
use ocl::{OclPrm, Queue, Program};
use crate::buffer::Buffer;
use crate::mat::AsShape;

pub struct KernelBuilder {
    kernel: Kernel,
    arg_idx: u32,
}

impl KernelBuilder {
    pub fn new<S: AsRef<str>>(program:&Program, name:S) -> Result<Self, Error> {
        ocl::core::create_kernel(program, name).map(Self::from_kernel)
    }
    pub fn from_kernel(kernel: Kernel) -> Self {
        Self { kernel, arg_idx: 0 }
    }
    pub fn add(mut self, arg: ArgVal) -> ocl::core::Result<Self> {
        let index = self.arg_idx;
        self.arg_idx += 1;
        ocl::core::set_kernel_arg(&self.kernel, index, arg)?;
        Ok(self)
    }

    pub fn add_mem(self, arg: &Mem) -> ocl::core::Result<Self> {
        self.add(ArgVal::mem(arg))
    }
    pub fn add_buff<T: OclPrm>(self, arg: &Buffer<T>) -> ocl::core::Result<Self> {
        self.add_mem(arg.as_core())
    }
    pub fn add_num<T: OclPrm>(self, arg: T) -> ocl::core::Result<Self> {
        self.add(ArgVal::scalar(&arg))
    }
    pub fn done(self) -> Kernel {
        self.kernel
    }
    pub fn enq(&self, queue:&Queue, global_work_dimensions:&[usize]) -> ocl::core::Result<()> {
        if global_work_dimensions.len()>3{
            return Err(ocl::core::Error::from(format!("OpenCL has at most 3 work dimensions but provided shape is {}",global_work_dimensions.as_shape())));
        }
        unsafe {
            let mut tmp:[usize;3] = [1;3];
            for (i,&d) in global_work_dimensions.iter().enumerate(){
                tmp[i] = d;
            }
            ocl::core::enqueue_kernel(queue.as_core(), &self.kernel, global_work_dimensions.len() as u32, None, &tmp, None, None::<ocl::core::Event>, None::<&mut ocl::core::Event>)
        }
    }
}

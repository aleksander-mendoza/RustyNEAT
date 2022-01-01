use ocl::{ProQue, Error, SpatialDims, Platform, Device, DeviceType, MemFlags, Queue, Program};
use std::fmt::{Formatter, Display};
use std::marker::PhantomData;
use ocl::core::{DeviceInfoResult, DeviceInfo, ArgVal};
use std::fs::File;
use std::io::Write;
use std::ops::Deref;
use ndalgebra::context::Context;
use ndalgebra::kernel_builder::KernelBuilder;

#[derive(Clone)]
pub struct EccProgram {
    pub ctx: Context,
    pub prog: Program,
}
impl Deref for EccProgram {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}
impl EccProgram {

    pub fn new(ctx:Context)->Result<Self,Error>{
        let src = include_str!("kernel.cl");
        let prog = Program::builder().source(src).build(ctx.context())?;
        Ok(Self{ctx:ctx.clone(),prog})
    }
    pub fn default()->Result<Self,Error>{
        Context::default().and_then(Self::new)
    }

    pub fn gpu()->Result<Self,Error>{
        Context::gpu().and_then(Self::new)
    }

    pub fn cpu()->Result<Self,Error>{
        Context::cpu().and_then(Self::new)
    }

    pub fn program(&self)->&Program{
        &self.prog
    }
    pub fn kernel_builder<S: AsRef<str>>(&self, name:S) -> Result<KernelBuilder, ocl::core::Error> {
        KernelBuilder::new(self.program(),name)
    }


}


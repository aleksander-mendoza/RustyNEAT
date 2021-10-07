use ocl::{ProQue, Error, SpatialDims, Platform, Device, DeviceType, MemFlags, Queue, Program};
use std::fmt::{Formatter, Display};
use std::marker::PhantomData;
use ocl::core::{DeviceInfoResult, DeviceInfo, ArgVal};
use std::fs::File;
use std::io::Write;
use std::ops::Deref;
use ndalgebra::context::Context;
use ndalgebra::kernel_builder::KernelBuilder;
use crate::htm_program::HtmProgram;

#[derive(Clone)]
pub struct HtmProgram2 {
    pub ctx: Context,
    pub prog: Program,
}
impl Deref for HtmProgram2{
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}
impl HtmProgram2 {

    pub fn from(t: &HtmProgram) -> Result<Self, Error> {
        Self::new(t.ctx.clone())
    }

    pub fn new(ctx:Context)->Result<Self,Error>{
        let src = include_str!("kernel2.cl");
        let prog = Program::builder().source(src).build(ctx.context())?;
        Ok(Self{ctx:ctx.clone(),prog})
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


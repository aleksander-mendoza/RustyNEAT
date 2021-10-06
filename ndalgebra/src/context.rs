use ocl::{Platform, Error, Device, DeviceType, SpatialDims, MemFlags, Queue, OclPrm};
use ocl::core::{DeviceInfoResult, DeviceInfo, ArgVal};
use crate::buffer::Buffer;
use crate::num::Num;
use crate::kernel_builder::KernelBuilder;

#[derive(Clone)]
pub struct Context {
    pub ctx: ocl::Context,
    pub q: Queue,
}

impl Context {
    pub fn device_list(platform: &Platform) -> Vec<Device> {
        Device::list_all(platform).unwrap_or_else(|_| vec![])
    }


    pub fn device_by_type(platform: Platform, dev_type: DeviceType) -> Option<Device> {
        Device::list_all(platform).ok().and_then(|dl| dl.into_iter().find(|d| match d.info(DeviceInfo::Type) {
            Ok(DeviceInfoResult::Type(d_type)) => d_type==dev_type,
            _ => false
        }))
    }
    pub fn gpu() -> Result<Self, Error> {
        let p = Platform::default();
        Self::device_by_type(p, DeviceType::GPU).ok_or_else(|| Error::from(format!("No GPU device"))).and_then(|d| Self::new(p, d))
    }
    pub fn cpu() -> Result<Self, Error> {
        let p = Platform::default();
        Self::device_by_type(p, DeviceType::CPU).ok_or_else(|| Error::from(format!("No CPU device"))).and_then(|d| Self::new(p, d))
    }
    pub fn new(platform: Platform, device: Device) -> Result<Self, Error> {
        let ctx = ocl::Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        let q = Queue::new(&ctx,device,None)?;
        Ok(Self{ctx,q})
    }

    pub fn buffer_from_slice<T:OclPrm>(&self, flags:MemFlags, slice:&[T]) -> Result<Buffer<T>, Error> {
        Buffer::from_slice(self.context(),flags,slice)
    }
    pub unsafe fn buffer_empty<T:OclPrm>(&self, flags:MemFlags, len:usize) -> Result<Buffer<T>, Error> {
        Buffer::empty(self.context(),flags,len)
    }
    pub fn queue(&self) -> &Queue{
        &self.q
    }
    pub fn context(&self) -> &ocl::Context{
        &self.ctx
    }
    pub fn buffer_filled<T:Num>(&self, flags:MemFlags, len:usize, fill_val:T) -> Result<Buffer<T>, Error> {
        let mut buff = unsafe{self.buffer_empty(flags,len)}?;
        buff.fill(self.queue(),fill_val)?;
        Ok(buff)
    }



}

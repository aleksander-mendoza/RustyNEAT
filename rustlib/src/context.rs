use ndalgebra::kernel::LinAlgProgram;
use ocl::{Platform, Device, Error};
use ocl::{DeviceType};
use std::any::Any;
use ocl::core::{DeviceInfo, DeviceInfoResult};


pub struct NeatContext {
    lin_alg: LinAlgProgram,
    platform:Platform,
    device:Device
}


impl NeatContext {
    pub fn lin_alg(&self) -> &LinAlgProgram {
        &self.lin_alg
    }
    pub fn platform(&self) -> &Platform {
        &self.platform
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn default() -> Result<Self, Error> {
        let p = Platform::default();
        Device::first(p).and_then(|d|Self::new(p, d))
    }
    pub fn gpu() -> Result<Self, Error> {
        let p = Platform::default();
        Self::device_by_type(&p, DeviceType::GPU).ok_or_else(|| Error::from(format!("No GPU device"))).and_then(|d| Self::new(p, d))
    }
    pub fn cpu() -> Result<Self, Error> {
        let p = Platform::default();
        Self::device_by_type(&p, DeviceType::CPU).ok_or_else(|| Error::from(format!("No CPU device"))).and_then(|d| Self::new(p, d))
    }
    pub fn new(platform: Platform, device: Device)->Result<Self, Error>{
        LinAlgProgram::new(platform.clone(),device.clone()).map(|lin_alg|Self{lin_alg,platform,device})
    }

    pub fn opencl_platforms() -> Vec<Platform> {
        Platform::list()
    }

    pub fn device_list(platform: &Platform) -> Vec<Device> {
        Device::list_all(platform).unwrap_or_else(|_| vec![])
    }

    pub fn opencl_default_platform() -> Platform {
        Platform::default()
    }

    pub fn device_by_type(platform: &Platform, dev_type: DeviceType) -> Option<Device> {
        LinAlgProgram::device_by_type(platform, dev_type)
    }

}
#![feature(option_result_contains)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(array_map)]
#![feature(maybe_uninit_extra)]
#[macro_use]
extern crate maplit;
#[macro_use]
extern crate lazy_static;


pub mod neat;
pub mod activations;
pub mod num;
pub mod util;
pub mod cppn;
pub mod envs;
pub mod gpu;

extern crate ocl;

use ocl::{DeviceType};
pub use ocl::{Platform, Device};
use std::any::Any;
use ocl::core::{DeviceInfo, DeviceInfoResult};


pub fn opencl_platforms() -> Vec<Platform> {
    Platform::list()
}

pub fn device_list(platform: &Platform) -> Vec<Device> {
    Device::list_all(platform).unwrap_or_else(|_| vec![])
}

pub fn opencl_default_platform() -> Platform {
    Platform::default()
}

pub fn default_device(platform: &Platform) -> Option<Device> {
    let d = Device::list_all(platform).ok().and_then(|dl|dl.into_iter().find(|d|match d.info(DeviceInfo::Type){
        Ok(DeviceInfoResult::Type(DeviceType::GPU)) => true,
        _ => false
    }));
    d.or_else(||Device::first(platform).ok())
}


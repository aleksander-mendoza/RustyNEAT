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
pub mod context;
pub use ocl::Device;
pub use ocl::Platform;
extern crate ocl;


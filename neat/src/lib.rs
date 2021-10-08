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
pub mod gpu;
extern crate ocl;


use std::collections::HashMap;
use std::iter::FromIterator;
use crate::util::RandRange;

pub fn sigmoid_f64(z: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-z))
}

pub fn sigmoid_f32(z: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-z))
}

pub fn relu_f64(z: f64) -> f64 {
    z.max(0.0)
}

pub fn relu_f32(z: f32) -> f32 {
    z.max(0.0)
}

pub fn square_f64(z: f64) -> f64 {
    z*z
}

pub fn square_f32(z: f32) -> f32 {
    z*z
}

pub fn inv_f64(z: f64) -> f64 {
    1.0/z
}

pub fn inv_f32(z: f32) -> f32 {
    1.0/z
}

pub fn step_f64(z: f64) -> f64 {
    if z > 0.0 {1.0}else{0.0}
}

pub fn step_f32(z: f32) -> f32 {
    if z > 0.0 {1.0}else{0.0}
}

pub fn identity<X>(z: X) -> X {
    z
}

pub struct ActFn{
    name:&'static str,
    opencl_name:&'static str,
    fn64:fn(f64)->f64,
    fn32:fn(f32)->f32,
}
impl ActFn{
    pub fn name(&self)->&'static str{
        self.name
    }
    pub fn opencl_name(&self)->&'static str{
        self.opencl_name
    }
    pub fn fn64(&self)->fn(f64)->f64{
        self.fn64
    }
    pub fn fn32(&self)->fn(f32)->f32{
        self.fn32
    }
    pub fn is_identity(&self)->bool{
        assert_eq!(self.fn64 == identity::<f64>, self.name == "identity");
        assert_eq!(self.fn64 == identity::<f64>, self.fn32 == identity::<f32>);
        self.fn64 == identity::<f64>
    }
}
pub const ALL_ACT_FN: [ActFn; 11] = [
    ActFn{name:"identity", fn64:identity, fn32:identity,opencl_name:""},
    ActFn{name:"sigmoid", fn64:sigmoid_f64, fn32:sigmoid_f32,opencl_name:"sigmoid32"},
    ActFn{name:"relu", fn64:relu_f64, fn32:relu_f32,opencl_name:"relu32"},
    ActFn{name:"sin", fn64:f64::sin, fn32:f32::sin,opencl_name:"sin"},
    ActFn{name:"cos", fn64:f64::cos, fn32:f32::cos,opencl_name:"cos"},
    ActFn{name:"tan", fn64:f64::tan, fn32:f32::tan,opencl_name:"tan"},
    ActFn{name:"tanh", fn64:f64::tanh, fn32:f32::tanh,opencl_name:"tanh"},
    ActFn{name:"abs", fn64:f64::abs, fn32:f32::abs,opencl_name:"fabs"},
    ActFn{name:"square", fn64:square_f64, fn32:square_f32,opencl_name:"square32"},
    ActFn{name:"inv", fn64:inv_f64, fn32:inv_f32,opencl_name:"1.0f/"},
    ActFn{name:"step", fn64:step_f64, fn32:step_f32,opencl_name:"step32"},
];
pub const IDENTITY:&'static ActFn = &ALL_ACT_FN[0];
lazy_static! {
    pub static ref STR_TO_IDX:HashMap<String,usize> = HashMap::<String,usize>::from_iter(ALL_ACT_FN.iter().enumerate().map(|(i,s)|(String::from(s.name),i)));
}

pub fn random_activation_fn() -> &'static ActFn{
    &ALL_ACT_FN[ALL_ACT_FN.len().random()]
}

use std::collections::HashMap;
use std::iter::FromIterator;

pub fn sigmoid_f64(z: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-z))
}

pub fn sigmoid_f32(z: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-z))
}

pub fn relu_f64(z: f64) -> f64 {
    if z > 0.0 { z } else { 0.0 }
}

pub fn relu_f32(z: f32) -> f32 {
    if z > 0.0 { z } else { 0.0 }
}

pub fn identity<X>(z: X) -> X {
    z
}


pub const ALL_STR: [&'static str; 8] = ["sigmoid", "relu", "sin", "cos", "tan", "tanh", "abs", "identity"];
pub const ALL_F64: [fn(f64) -> f64; 8] = [sigmoid_f64, relu_f64, f64::sin, f64::cos, f64::tan, f64::tanh, f64::abs, identity];
pub const ALL_F32: [fn(f32) -> f32; 8] = [sigmoid_f32, relu_f32, f32::sin, f32::cos, f32::tan, f32::tanh, f32::abs, identity];
lazy_static! {
    pub static ref STR_TO_IDX:HashMap<String,usize> = HashMap::<String,usize>::from_iter(ALL_STR.iter().enumerate().map(|(i,&s)|(String::from(s),i)));
}

pub fn random_activation_fn_name() -> &'static str{
    let r: f32 = rand::random();
    ALL_STR[ALL_STR.len() * r as usize]
}
//
//

//
//
// def softplus_activation(z):
// z = max(-60.0, min(60.0, 5.0 * z))
// return 0.2 * math.log(1 + math.exp(z))
//
//
// def identity_activation(z):
// return z
//
//
// def clamped_activation(z):
// return max(-1.0, min(1.0, z))
//
//
// def inv_activation(z):
// if z == 0:
// return 0.0
//
// return 1.0 / z
//
//
// def log_activation(z):
// z = max(1e-7, z)
// return math.log(z)
//
//
// def exp_activation(z):
// z = max(-60.0, min(60.0, z))
// return math.exp(z)
//
//
// def abs_activation(z):
// return abs(z)
//
//
// def hat_activation(z):
// return max(0.0, 1 - abs(z))
//
//
// def square_activation(z):
// return z ** 2
//
//
// def cube_activation(z):
// return z ** 3

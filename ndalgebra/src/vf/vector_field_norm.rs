use std::fmt::{Debug, Formatter};

use serde::{Serialize, Deserialize};
use crate::VectorField;
use num_traits::{Zero, Pow};
use std::ops::{Mul, Add};
pub trait Sqrt{
    fn sqrt(self)->Self;
}
impl Sqrt for f32{
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}
impl Sqrt for f64{
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

pub trait LNorm<Scalar> {
    fn pow(scalar: Scalar) -> Scalar;
    fn root(scalar: Scalar) -> Scalar;
}

pub struct L<const l: usize> {}

impl<Scalar> LNorm<Scalar> for L<1> {
    fn pow(scalar: Scalar) -> Scalar {
        scalar
    }

    fn root(scalar: Scalar) -> Scalar {
        scalar
    }
}

impl <Scalar:Copy+Sqrt+Mul<Output=Scalar>> LNorm<Scalar> for L<2> {
    fn pow(scalar: Scalar) -> Scalar {
        scalar * scalar
    }
    fn root(scalar: Scalar) -> Scalar {
        scalar.sqrt()
    }
}


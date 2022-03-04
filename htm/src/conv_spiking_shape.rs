use crate::{Idx, Shape3, VectorFieldOne, Shape, VectorFieldPartialOrd, range_contains, from_xyz, Shape2, w_idx, ConvShape, ConvShapeTrait, HasShape};
use serde::{Serialize, Deserialize};
use std::ops::{Range, Deref};

/**Time indices*/
pub type TIdx = u8;

pub trait ConvSpikingShape:HasShape {
    fn time(&self)->TIdx;
}
use crate::{Idx, Shape3, VectorFieldOne, Shape, VectorFieldPartialOrd, range_contains, from_xyz, Shape2, w_idx, ConvShape, ConvShapeTrait, HasConvShape};
use serde::{Serialize, Deserialize};
use std::ops::{Range, Deref};

/**Time indices*/
pub type TIdx = u8;

pub trait ConvSpikingShape: HasConvShape {
    fn time(&self)->TIdx;
}
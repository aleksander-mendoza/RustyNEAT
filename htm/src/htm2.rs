use ocl::OclPrm;
use serde::{Serialize, Deserialize};
use std::ops::Range;

#[derive(Copy, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct HtmFeedforwardConnection2 {
    pub permanence: f32,
    pub input_id: u32,
}

unsafe impl OclPrm for HtmFeedforwardConnection2{}


#[derive(Copy, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct HtmMinicolumn2 {
    pub connection_offset: u32,
    pub connection_len: u32,
    pub overlap: i32,
}
unsafe impl OclPrm for HtmMinicolumn2{}
impl HtmMinicolumn2{
    pub fn range(&self)->Range<usize>{
        self.connection_offset as usize..(self.connection_offset+self.connection_len)as usize
    }
}
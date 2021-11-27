use ocl::OclPrm;
use serde::{Serialize, Deserialize};

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

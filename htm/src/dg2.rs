use ocl::OclPrm;
use serde::{Serialize, Deserialize};


#[derive(Copy, Clone, Serialize, Deserialize, Debug, Default, PartialEq)]
#[repr(C)]
pub struct DgMinicolumn2 {
    pub connection_offset: u32,
    pub connection_len: u32,
    pub overlap: i32,
}
unsafe impl OclPrm for DgMinicolumn2{}

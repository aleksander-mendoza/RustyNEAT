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


#[derive(Copy, Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct DgCoord2d {
    pub y: u32,
    pub x: u32,
}
impl DgCoord2d{
    pub fn new_yx(y:u32, x:u32)->Self{
        Self{x,y}
    }
    pub fn as_yx(&self)->(u32,u32){
        (self.y,self.x)
    }
}
unsafe impl OclPrm for DgCoord2d{}

// use ocl::OclPrm;
// use serde::{Serialize, Deserialize};
// use std::ops::Range;
//
// #[derive(Copy, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
// #[repr(C)]
// pub struct HtmFeedforwardConnection {
//     pub permanence: f32,
//     pub input_id: u32,
// }
//
// unsafe impl OclPrm for HtmFeedforwardConnection {}
//
//
// #[derive(Copy, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
// #[repr(C)]
// pub struct HtmMinicolumn {
//     pub connection_offset: u32,
//     pub connection_len: u32,
//     pub overlap: i32,
// }
// unsafe impl OclPrm for HtmMinicolumn {}
// impl HtmMinicolumn {
//     pub fn range(&self)->Range<usize>{
//         self.connection_offset as usize..(self.connection_offset+self.connection_len)as usize
//     }
// }
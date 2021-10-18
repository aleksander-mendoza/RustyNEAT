// use ocl::OclPrm;
//
// #[derive(Copy, Clone, Debug, Default, PartialEq)]
// #[repr(C)]
// pub struct HomDistalConnection {
//     pub minicolumn_id: u32,
//     pub permanence: f32,
//     pub input_id: u32,
// }
//
// unsafe impl OclPrm for HomDistalConnection{}
//
// #[derive(Copy, Clone, Debug, Default, PartialEq)]
// #[repr(C)]
// pub struct HtmInput {
//     pub connection_offset: u32,
//     pub connection_len: u32,
//     pub is_active: bool,
// }
// unsafe impl OclPrm for HtmInput{}
//
// #[derive(Copy, Clone, Debug, Default, PartialEq)]
// #[repr(C)]
// pub struct HtmMinicolumn {
//     pub connection_index_offset: u32,
//     pub connection_index_len: u32,
//     pub overlap: i32,
// }
// unsafe impl OclPrm for HtmMinicolumn{}

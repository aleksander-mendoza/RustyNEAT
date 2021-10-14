use ocl::OclPrm;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct HtmFeedforwardConnection3 {
    pub minicolumn_id: u32,
    pub permanence: f32,
    pub input_id: u32,
    pub is_inhibitory:bool,
}

unsafe impl OclPrm for HtmFeedforwardConnection3{}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct HtmInput3 {
    pub connection_offset: u32,
    pub connection_len: u32,
    pub is_active: bool,
}
unsafe impl OclPrm for HtmInput3{}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct HtmMinicolumn3 {
    pub connection_index_offset: u32,
    pub connection_index_len: u32,
    pub overlap: i32,
}
unsafe impl OclPrm for HtmMinicolumn3{}

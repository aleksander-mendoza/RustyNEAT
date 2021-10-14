use ocl::OclPrm;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct HtmFeedforwardConnection4 {
    pub permanence: f32,
    pub input_id: u32,
    pub overlap_gain: i32, //inhibitory has -1, excitatory has 1
}

unsafe impl OclPrm for HtmFeedforwardConnection4{}


#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct HtmMinicolumn4 {
    pub connection_offset: u32,
    pub connection_len: u32,
    pub overlap: i32,
}
unsafe impl OclPrm for HtmMinicolumn4{}

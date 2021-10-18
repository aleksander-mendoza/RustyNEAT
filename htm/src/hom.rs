use ocl::OclPrm;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct HomDistalConnection {
    pub minicolumn_id: u32,
    pub permanence: f32,
}

unsafe impl OclPrm for HomDistalConnection{}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct HomDistalSegment {
    pub connection_offset: u32,
    pub connection_len: u32,
    pub is_active: bool,
}
unsafe impl OclPrm for HomDistalSegment{}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct HomCell {
    pub segment_offset: u32,
    pub segment_len: u32,
}
unsafe impl OclPrm for HomCell{}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct HomMinicolumn {
    pub cell_offset: u32,
    pub cell_len: u32,
    pub winner_cell: i32,
}
unsafe impl OclPrm for HomMinicolumn{}

use crate::{Shape3, Idx};
use std::ops::Range;

pub fn assert_valid_for(k: Idx, shape: &[Idx; 3]) {
    assert!(k <= shape.channels(), "k is larger than layer output!");
    assert_eq!(shape.channels() % k, 0, "k=={} does not divide out_channels=={}! Disable top1_per_region first!", k, shape.channels());
}

pub fn get_region_size(k: Idx, shape: &[Idx; 3]) -> Idx {
    shape.channels() / k
}

pub fn get_region_range(k: Idx, shape: &[Idx; 3], region_idx: Idx) -> Range<Idx> {
    get_region_range_(get_region_size(k, shape), region_idx)
}

pub fn get_region_range_(region_size: Idx, region_idx: Idx) -> Range<Idx> {
    let channel_region_offset = region_size * region_idx;
    channel_region_offset..channel_region_offset + region_size
}

pub fn get_region_count(k: Idx, shape: &[Idx; 3]) -> Idx {
    let a = shape.area();
    k * a
}
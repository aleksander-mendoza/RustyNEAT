use std::ops::Range;
use crate::init::{empty, init_fold_rev, InitEmptyWithCapacity};
use crate::{VectorField, VectorFieldOne};
use crate::layout::BorrowAsLayout;


pub type Shape = Vec<usize>;
pub type Strides = Vec<usize>;
pub type Layout = (Shape, Strides);

pub fn null() -> Layout {
    (Vec::new(), Vec::new())
}

pub fn from(shape: Vec<usize>) -> Layout {
    let strides = strides_for_shape(&shape);
    (shape, strides)
}

pub fn view(layout: &Layout, ranges: &[Range<usize>]) -> Layout {
    assert!(ranges.len() <= layout.ndim());
    assert!(ranges.iter().zip(shape(layout).iter()).all(|(r, &s)| r.start > s || r.end > s || r.start > r.end), "Ranges {:?} are out of bounds for shape {:?}", ranges, shape(layout));
    let mut new_shape = Vec::with_capacity(layout.ndim() - ranges.len());
    let mut new_strides = Vec::with_capacity(new_shape.capacity());
    for (i, &stride) in strides(layout).iter().enumerate() {
        let len = ranges.get(i).map(|r| r.len()).unwrap_or(shape(layout)[i]);
        if len == 0 {
            return null();
        }
        if len > 1 {
            new_shape.push(len);
            new_strides.push(stride);
        }
    }
    if new_shape.is_empty() {
        new_shape.push(1);
        new_strides.push(1);
    }
    (new_shape, new_strides)
}

pub fn unsqueeze(layout: &Layout, idx: usize) -> Layout {
    _unsqueeze(layout.clone(), idx)
}

pub fn _unsqueeze(mut layout: Layout, idx: usize) -> Layout {
    unsqueeze_(&mut layout, idx);
    layout
}

/**Inserts an additional dimension of length 1 at a specified index.  Works the same way as in numpy.*/
pub fn unsqueeze_(layout: &mut Layout, idx: usize) -> &mut Layout {
    shape_mut(layout).insert(idx, 1);
    strides_mut(layout).insert(idx, 1);
    layout
}

pub fn squeeze(layout: &Layout, idx: usize) -> Layout {
    _squeeze(layout.clone(), idx)
}

pub fn _squeeze(mut layout: Layout, idx: usize) -> Layout {
    squeeze_(&mut layout, idx);
    layout
}

/**Collapses a dimension at a specified index, provided that the length of that dimension is 1.  Works the same way as in numpy.*/
pub fn squeeze_(layout: &mut Layout, idx: usize) -> &mut Layout {
    if shape(layout)[idx] != 1 {
        panic!("Cannot squeeze {:?} as dimension {}", shape(layout), idx);
    } else {
        shape_mut(layout).remove(idx);
        strides_mut(layout).remove(idx);
        layout
    }
}


pub fn shape(layout: &Layout) -> &Strides {
    &layout.0
}

pub fn strides(layout: &Layout) -> &Shape {
    &layout.1
}

pub fn shape_mut(layout: &mut Layout) -> &mut Shape {
    &mut layout.0
}

pub fn strides_mut(layout: &mut Layout) -> &mut Strides {
    &mut layout.1
}

pub fn strides_for_shape(shape: &[usize]) -> Vec<usize> {
    shape.rfold_map(1, |prod, next_dim| (prod * next_dim, prod)).1
}

pub fn reshape_wildcard(layout: &Layout, reshape: &[isize]) -> Vec<usize> {
    let mut inferred = Vec::with_capacity(layout.ndim());
    let mut wildcard_pos = usize::MAX;
    let mut new_size = 1;
    for (i, &len) in reshape.iter().enumerate() {
        if len < 0 {
            inferred.push(0);
            if wildcard_pos < usize::MAX {
                panic!("Multiple wildcards {:?} lead to ambiguity", reshape);
            }
            wildcard_pos = i;
        } else {
            let len = len as usize;
            inferred.push(len);
            new_size *= len;
        }
    }
    if wildcard_pos < usize::MAX {
        let my_size = layout.size();
        if my_size % new_size == 0 {
            inferred[wildcard_pos] = my_size / new_size;
        } else {
            panic!("Shape {:?} cannot be reshaped into {:?} because their total volume is different!", shape(layout), reshape);
        }
    }
    inferred
}
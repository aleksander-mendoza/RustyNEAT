use std::ops::Range;
use crate::init::{empty, init_fold_rev};
use crate::{VectorField, VectorFieldOne};

pub type Shape<const DIM: usize> = [usize; DIM];
pub type Strides<const DIM: usize> = [usize; DIM];
pub type Layout<const DIM: usize> = (Shape<DIM>, Strides<DIM>);

pub fn cut_out<T: Copy, const DIM: usize>(z: &[T; DIM], idx: usize) -> [T; DIM - 1] {
    let mut s = empty();
    s[..idx].copy_from_slice(&z[..idx]);
    s[idx..].copy_from_slice(&z[idx + 1..]);
    s
}

pub fn insert<T: Copy, const DIM: usize>(z: &[T; DIM], idx: usize, elem: T) -> [T; DIM + 1] {
    let mut s = empty();
    s[..idx].copy_from_slice(&z[..idx]);
    s[idx] = elem;
    s[idx + 1..].copy_from_slice(&z[idx..]);
    s
}

pub fn null() -> Layout<0> {
    ([], [])
}

pub fn from<const DIM: usize>(shape: Shape<DIM>) -> Layout<DIM> {
    let strides = strides_for_shape(&shape);
    (shape, strides)
}

pub fn view<const DIM: usize>(layout: &Layout<DIM>, ranges: &[Range<usize>; DIM]) -> Layout<DIM> {
    if ranges.iter().zip(shape(layout).iter()).all(|(r, &s)| r.start > s || r.end > s || r.start > r.end) {
        panic!("Ranges {:?} are out of bounds for shape {:?}", ranges, shape(layout));
    } else {
        let mut new_shape = empty();
        let mut new_strides = empty();
        for (i, &stride) in strides(layout).iter().enumerate() {
            let len = ranges.get(i).map(|r| r.len()).unwrap_or(shape(layout)[i]);
            new_shape[i] = len;
        }
        (new_shape, new_strides)
    }
}

/**Inserts an additional dimension of length 1 at a specified index.  Works the same way as in numpy.*/
pub fn unsqueeze<const DIM: usize>(layout: &Layout<DIM>, idx: usize) -> Layout<{ DIM + 1 }> {
    (insert(shape(layout), idx, 1), insert(strides(layout), idx, /*whatever*/1))
}

/**Collapses a dimension at a specified index, provided that the length of that dimension is 1.  Works the same way as in numpy.*/
pub fn squeeze<const DIM: usize>(layout: &Layout<DIM>, idx: usize) -> Layout<{ DIM - 1 }> {
    (cut_out(shape(layout), idx), cut_out(strides(layout), idx))
}

pub fn shape<const DIM: usize>(layout: &Layout<DIM>) -> &Shape<DIM> {
    &layout.0
}

pub fn strides<const DIM: usize>(layout: &Layout<DIM>) -> &Strides<DIM> {
    &layout.1
}

pub fn shape_mut<const DIM: usize>(layout: &mut Layout<DIM>) -> &mut Shape<DIM> {
    &mut layout.0
}

pub fn strides_mut<const DIM: usize>(layout: &mut Layout<DIM>) -> &mut Strides<DIM> {
    &mut layout.1
}


pub fn strides_for_shape<const DIM: usize>(shape: &[usize; DIM]) -> [usize; DIM] {
    shape.rfold_map(1, |prod, next_dim| (prod * next_dim, prod)).1
}

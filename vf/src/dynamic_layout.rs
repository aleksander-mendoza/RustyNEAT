use std::ops::Range;
use crate::init::{empty, init_fold_rev};
use crate::VectorFieldOne;


pub type Shape = Vec<usize>;
pub type Strides = Vec<usize>;
pub type Layout = (Shape, Strides);
pub fn cut_out<T:Copy>(z:&[T], idx:usize)->Vec<T>{
    let mut s = Vec::with_capacity(z.len()-1);
    s.extend_from_slice(&z[0..idx]);
    s.extend_from_slice(&z[idx + 1..]);
    s
}
pub fn insert<T:Copy>(z:&[T], idx:usize, elem:T)->Vec<T>{
    let mut s = Vec::with_capacity(z.len()+1);
    s.extend_from_slice(&z[0..idx]);
    s.push(elem);
    s.extend_from_slice(&z[idx + 1..]);
    s
}
pub fn from(shape: Vec<usize>) -> Layout {
    let strides = strides_for_shape(&shape);
    (shape, strides)
}

pub fn end_offset(layout:&Layout) -> usize {
    shape(layout).iter().zip(strides(layout).iter()).map(|(&len, &stride)| (len - 1) * stride).sum()
}

pub fn view(layout:&Layout, ranges: &[Range<usize>]) -> Layout {
    if ranges.len() > ndim(layout) || ranges.iter().zip(shape(layout).iter()).all(|(r, &s)| r.start > s || r.end > s || r.start > r.end) {
        Err(MatError::InvalidView(ranges.to_vec(), shape(layout)))
    } else {
        let mut new_shape = Vec::with_capacity(self.ndim() - ranges.len());
        let mut new_strides = Vec::with_capacity(new_shape.capacity());
        for (i, &stride) in self.strides.iter().enumerate() {
            let len = ranges.get(i).map(|r| r.len()).unwrap_or(self.shape[i]);
            if len == 0 {
                return Ok(Self::null());
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
        let mut slf = Self {
            shape: new_shape.into_boxed_slice(),
            strides: new_strides.into_boxed_slice(),
            contiguous: false,
        };
        slf.contiguous = slf.test_contiguity();
        Ok(slf)
    }
}

/**Inserts an additional dimension of length 1 at a specified index.  Works the same way as in numpy.*/
pub fn unsqueeze(&mut self, idx: usize) -> Result<&mut Self, MatError> {
    if idx > self.ndim() {//notice that idx can be equal to dimensionality!
        Err(MatError::DimensionOutOfBounds(self.shape.to_vec(), idx))
    } else if self.ndim() >= MAX_MAT_DIMS {
        Err(MatError::DimensionalityLimitExceeded(self.ndim() + 1))
    } else {
        self.shape = insert(self.shape(), idx, 1).into_boxed_slice();
        self.strides = insert(self.stride(), idx, /*whatever*/1).into_boxed_slice();
    }
}

/**Collapses a dimension at a specified index, provided that the length of that dimension is 1.  Works the same way as in numpy.*/
pub fn squeeze(&mut self, idx: usize) -> Result<&mut Self, MatError> {
    if idx >= self.ndim() {
        Err(MatError::DimensionOutOfBounds(self.shape.to_vec(), idx))
    } else if self.shape[idx] != 1 {
        Err(MatError::NonsingularDimension(self.shape.to_vec(), idx))
    } else {
        self.shape = cut_out(self.shape(), idx).into_boxed_slice();
        self.strides = cut_out(self.stride(), idx).into_boxed_slice();
    }
}

pub fn transpose(&mut self, dim0: usize, dim1: usize) -> Result<&mut Self, MatError> {
    if dim0 >= self.ndim() {
        return Err(MatError::DimensionOutOfBounds(self.shape.to_vec(), dim0));
    }
    if dim1 >= self.ndim() {
        return Err(MatError::DimensionOutOfBounds(self.shape.to_vec(), dim1));
    }
    self.strides.swap(dim0, dim1);
    self.shape.swap(dim0, dim1);
    self.contiguous = self.test_contiguity();
    Ok(self)
}

pub fn contiguous(layout:&Layout) -> bool {
    let mut i = 1;
    for (dim, stride) in shape(layout).iter().cloned().zip(strides(layout).iter().cloned()).rev() {
        if stride != i {
            return false;
        }
        i *= dim;
    }
    true
}

pub fn offset(layout:&Layout, index: &[usize]) -> Result<usize, MatError> {
    if index.len() == ndim(layout) && index.iter().zip(shape(layout).iter()).all(|(&a, &b)| a < b) {
        Ok(index.iter().zip(strides(layout).iter()).map(|(a, b)| a * b).sum())
    } else {
        Err(MatError::InvalidIndex(index.to_vec(), shape(layout).to_vec()))
    }
}


pub fn shape(layout:&Layout) -> &[usize] {
    &layout.0
}

pub fn strides(layout:&Layout) -> &[usize] {
    &layout.1
}

pub fn ndim(layout:&Layout) -> usize {
    layout.0.len()
}

pub fn size(layout:&Layout) -> usize {
    shape(layout).product()
}

/**Length of first dimension*/
pub fn len(layout:&Layout) -> usize {
    shape(layout).first().cloned().unwrap_or(0usize)
}

pub fn strides_for_shape(shape: &[usize]) -> Box<[usize]> {
    if shape.is_empty() {
        vec![].into_boxed_slice()
    } else {
        let mut strides = empty();
        strides[shape.len() - 1] = 1;
        for i in (1..shape.len()).rev() {
            strides[i - 1] = strides[i] * shape[i];
        }
        Box::new(strides)
    }
}

pub fn infer_reshape_wildcard(layout:&Layout, reshape: &[isize]) -> Result<Box<[usize]>, MatError> {
    let mut inferred = Vec::with_capacity(self.ndim());
    let mut wildcard_pos = usize::MAX;
    let mut new_size = 1;
    for (i, &len) in reshape.iter().enumerate() {
        if len < 0 {
            inferred.push(0);
            if wildcard_pos < usize::MAX {
                return Err(MatError::MultipleWildcards(shape.to_vec()));
            }
            wildcard_pos = i;
        } else {
            let len = len as usize;
            inferred.push(len);
            new_size *= len;
        }
    }
    if wildcard_pos < usize::MAX {
        let my_size = self.size();
        if my_size % new_size == 0 {
            inferred[wildcard_pos] = my_size / new_size;
        } else {
            return Err(MatError::IncompatibleWildcardShapes(self.shape.to_vec(), reshape.to_vec()));
        }
    }
    Ok(inferred.into_boxed_slice())
}
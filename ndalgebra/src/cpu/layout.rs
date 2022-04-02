use crate::cpu::buffer::empty;
use crate::cpu::error::MatError;
use crate::lin_alg_program::MAX_MAT_DIMS;
use std::ops::Range;
use crate::cpu::shape::AsShape;

#[derive(Clone)]
pub struct Layout {
    shape: Box<[usize]>,
    strides: Box<[usize]>,
    contiguous: bool,
}
impl Eq for Layout{
}
impl PartialEq for Layout{
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
    }
}

impl From<&[usize]> for Layout {
    fn from(a: &[usize]) -> Self {
        Self::from_slice(a)
    }
}

impl From<Box<[usize]>> for Layout {
    fn from(a: Box<[usize]>) -> Self {
        Self::new(a)
    }
}

impl<const DIM: usize> From<[usize; DIM]> for Layout {
    fn from(a: [usize; DIM]) -> Self {
        Self::new(a.into())
    }
}
fn cut_out<T:Copy>(z:&[T], idx:usize)->Vec<T>{
    let mut s = Vec::with_capacity(z.len()-1);
    s.extend_from_slice(&z[0..idx]);
    s.extend_from_slice(&z[idx + 1..]);
    s
}
fn insert<T:Copy>(z:&[T], idx:usize, elem:T)->Vec<T>{
    let mut s = Vec::with_capacity(z.len()+1);
    s.extend_from_slice(&z[0..idx]);
    s.push(elem);
    s.extend_from_slice(&z[idx + 1..]);
    s
}
impl Layout {
    pub fn null()->Self{
        Self{
            shape: empty(0),
            strides: empty(0),
            contiguous: true
        }
    }
    pub fn end_offset(&self)->usize{
        self.shape.iter().zip(self.strides.iter()).map(|(&len, &stride)| (len - 1) * stride).sum()
    }
    pub fn view(&self, ranges: &[Range<usize>]) -> Result<Self, MatError> {
        if ranges.len() > self.ndim() || ranges.iter().zip(self.shape.iter()).all(|(r, &s)| r.start > s || r.end > s || r.start > r.end) {
            Err(MatError::InvalidView(ranges.to_vec(), self.shape.to_vec()))
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
            let mut slf = Self{
                shape: new_shape.into_boxed_slice(),
                strides: new_strides.into_boxed_slice(),
                contiguous: false
            };
            slf.contiguous = slf.test_contiguity();
            Ok(slf)
        }
    }
    /**Length of first dimension*/
    pub fn len(&self) -> usize {
        self.shape.first().cloned().unwrap_or(0usize)
    }
    /**Inserts an additional dimension of length 1 at a specified index.  Works the same way as in numpy.*/
    pub fn unsqueeze(&mut self, idx: usize) -> Result<&mut Self, MatError> {
        if idx > self.ndim() {//notice that idx can be equal to dimensionality!
            Err(MatError::DimensionOutOfBounds(self.shape.to_vec(), idx))
        } else if self.ndim() >= MAX_MAT_DIMS {
            Err(MatError::DimensionalityLimitExceeded(self.ndim() + 1))
        } else {
            self.shape = insert(self.shape(),idx,1).into_boxed_slice();
            self.strides = insert(self.stride(),idx, /*whatever*/1).into_boxed_slice();
        }
    }
    /**Collapses a dimension at a specified index, provided that the length of that dimension is 1.  Works the same way as in numpy.*/
    pub fn squeeze(&mut self, idx: usize) -> Result<&mut Self, MatError> {
        if idx >= self.ndim() {
            Err(MatError::DimensionOutOfBounds(self.shape.to_vec(), idx))
        } else if self.shape[idx] != 1 {
            Err(MatError::NonsingularDimension(self.shape.to_vec(), idx))
        } else {
            self.shape = cut_out(self.shape(),idx).into_boxed_slice();
            self.strides = cut_out(self.stride(),idx).into_boxed_slice();
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
    fn test_contiguity(&self)->bool{
        let mut i = 1;
        for (dim,stride) in self.shape.iter().cloned().zip(self.strides.iter().cloned()).rev(){
            if stride != i{
                return false;
            }
            i*=dim;
        }
        true
    }
    pub fn offset(&self, index: &[usize]) -> Result<usize, MatError> {
        if index.len() == self.ndim() && index.iter().zip(self.shape.iter()).all(|(&a, &b)| a < b) {
            Ok(index.iter().zip(self.strides.iter()).map(|(a, b)| a * b).sum())
        } else {
            Err(MatError::InvalidIndex(index.to_vec(), self.shape.to_vec()))
        }
    }
    pub fn contiguous(&self) -> bool {
        self.contiguous
    }
    pub fn shape(&self) -> &[usize] {
        self.shape.as_ref()
    }
    pub fn strides(&self) -> &[usize] {
        self.strides.as_ref()
    }
    pub fn into_vec(self) -> Vec<usize> {
        let Self { shape, .. } = self;
        shape.into_vec()
    }
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn from_slice(shape: &[usize]) -> Self {
        Self::new(shape.into())
    }
    pub fn new(shape: Box<[usize]>) -> Self {
        let strides = Self::strides_for_shape(&shape);
        Self { shape, strides, contiguous:true }
    }
    pub fn strides_for_shape(shape: &[usize]) -> Box<[usize]> {
        if shape.is_empty() {
            Box::<[usize]>::from([])
        } else {
            let mut strides = empty(shape.len());
            strides[shape.len() - 1] = 1;
            for i in (1..shape.len()).rev() {
                strides[i - 1] = strides[i] * shape[i];
            }
            strides
        }
    }
    pub fn infer_reshape_wildcard(&self, reshape: &[isize]) -> Result<Box<[usize]>, MatError> {
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
}
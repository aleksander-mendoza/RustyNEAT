use std::fmt::{Formatter, Display, Debug};
use crate::cpu::shape::AsShape;

pub enum MatError {
    BufferLengthMismatch(usize, usize),
    OutOfBufferBounds(usize, usize, usize),
    CannotReadNonContiguous(),
    NotSingletonMatrix(Vec<usize>),
    NonsingularDimension(Vec<usize>, usize),
    DimensionOutOfBounds(Vec<usize>, usize),
    MultipleWildcards(Vec<isize>),
    IncompatibleShapes(Vec<usize>, Vec<usize>),
    InvalidIndex(Vec<usize>, Vec<usize>),
    IncompatibleWildcardShapes(Vec<usize>, Vec<isize>),
    MatMulUndefined(),
    MatMulDimIncompatible(usize, usize),
    MatMulShapeIncompatible(Vec<usize>, Vec<usize>),
    DimensionalityLimitExceeded(usize),
    InvalidLiteral(),
    BufferLengthAndShapeMismatch(usize, Vec<usize>),
    InvalidView(Vec<Range<usize>>, Vec<usize>),
}

impl From<ocl::OclCoreError> for MatError {
    fn from(e: ocl::OclCoreError) -> Self {
        Self::OpenCLCoreError(e)
    }
}

impl From<MatError> for String {
    fn from(s: MatError) -> Self {
        format!("{}", s)
    }
}

impl Debug for MatError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for MatError {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MatError::OutOfBufferBounds(offset, dst_len, buff_len) => write!(fmt, "Offset {} and length {} of destination slice are out of bounds for buffer of length {}", offset, dst_len, buff_len),
            MatError::InvalidIndex(index, shape) => write!(fmt, "Indices {} are not valid for {}", index.as_shape(), shape.as_shape()),
            MatError::BufferLengthMismatch(tensor_buff, dst_buff) => write!(fmt, "Tensor has buffer of length {} but provided destination has length {}", tensor_buff, dst_buff),
            MatError::NotSingletonMatrix(shape) => write!(fmt, "Tensor of shape {} is not a singleton", shape.as_shape()),
            MatError::CannotReadNonContiguous() => write!(fmt, "Cannot read the tensor because its view is not contiguous"),
            MatError::InvalidView(view, shape) => write!(fmt, "View {:?} is not valid for shape {}", view, shape.as_shape()),
            MatError::BufferLengthAndShapeMismatch(buff_len, shape) => write!(fmt, "Provided buffer has length {} which does not equal total size of the shape {}", buff_len, shape.as_shape()),
            MatError::MultipleWildcards(shape) => write!(fmt, "More than one wildcard is not allowed: {}", shape.as_shape()),
            MatError::IncompatibleShapes(original, new) => write!(fmt, "Cannot reshape {} into {}", original.as_shape(), new.as_shape()),
            MatError::IncompatibleWildcardShapes(original, new) => write!(fmt, "Cannot reshape {} into {}", original.as_shape(), new.as_shape()),
            MatError::DimensionOutOfBounds(shape, dim) => write!(fmt, "Dimension {} is out of bounds for shape {}", dim, shape.as_shape()),
            MatError::MatMulUndefined() => write!(fmt, "Matrix multiplication cannot be performed on vectors"),
            MatError::MatMulDimIncompatible(i, j) => write!(fmt, "Matrix multiplication must work with shapes (...,i,j) * (...,k,i) but was provided with (...,{},j)*(...,k,{})", i, j),
            MatError::MatMulShapeIncompatible(lhs, rhs) => write!(fmt, "Matrix multiplication must work with shapes (s*,i,j) * (s*,k,i) but s* was not equal for both sides in {} {}", lhs.as_shape(), rhs.as_shape()),
            MatError::DimensionalityLimitExceeded(dim) => write!(fmt, "Matrix has {} dimensions but {} is the maximum", dim, MAX_MAT_DIMS),
            MatError::InvalidLiteral() => write!(fmt, "Provided literal matrix was invalid. All rows, columns, etc must be of the same size."),
            MatError::NonsingularDimension(shape, idx) => write!(fmt, "Shape {} has length {} at index {} but expected it to be of length 1", shape.as_shape(), shape[*idx], idx),
        }
    }
}


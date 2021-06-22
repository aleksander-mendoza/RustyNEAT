use ocl::{ProQue, SpatialDims, flags, Platform, Device, Buffer, Error, Queue};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive};
use std::fmt::{Display, Formatter, Debug};
use crate::kernel::{LinAlgProgram, MAX_MAT_DIMS};
use crate::num::Num;
use ocl::builders::KernelBuilder;

#[derive(Clone)]
pub struct Mat<T: Num> {
    lin_alg: LinAlgProgram,
    buff: Option<Buffer<T>>,
    shape: Box<[usize]>,
    strides: Box<[usize]>,
}

pub enum MatError {
    BufferLengthMismatch(usize, usize),
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
    OpenCLError(Error),
    BufferLengthAndShapeMismatch(usize, Vec<usize>),
    InvalidView(Vec<Range<usize>>, Vec<usize>),
}

impl From<ocl::Error> for MatError {
    fn from(e: Error) -> Self {
        Self::OpenCLError(e)
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
            MatError::OpenCLError(err) => write!(fmt, "OpenCL error: {}", err),
            MatError::InvalidLiteral() => write!(fmt, "Provided literal matrix was invalid. All rows, columns, etc must be of the same size."),
            MatError::NonsingularDimension(shape, idx) => write!(fmt, "Shape {} has length {} at index {} but expected it to be of length 1", shape.as_shape(), shape[*idx], idx),
        }
    }
}

pub struct Shape<'a, T: Num>(&'a [T]);

pub trait AsShape<T: Num> {
    fn as_shape(&self) -> Shape<T>;
}

impl<T: Num, const S: usize> AsShape<T> for [T; S] {
    fn as_shape(&self) -> Shape<T> {
        Shape(self)
    }
}


impl<T: Num> AsShape<T> for Box<[T]> {
    fn as_shape(&self) -> Shape<T> {
        Shape(self)
    }
}

impl<T: Num> AsShape<T> for &[T] {
    fn as_shape(&self) -> Shape<T> {
        Shape(self)
    }
}

impl<T: Num> AsShape<T> for Vec<T> {
    fn as_shape(&self) -> Shape<T> {
        Shape(self.as_slice())
    }
}

impl<'a, T: Num> Shape<'a, T> {
    fn size(&self) -> T {
        if self.0.is_empty() {
            T::zero()
        } else {
            self.0.iter().fold(T::one(), |a, &b| a * b)
        }
    }
}


impl<'a, T: Num> Display for Shape<'a, T> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "(")?;
        if !self.0.is_empty() {
            write!(fmt, "{}", self.0[0])?;
            for &i in &self.0[1..] {
                write!(fmt, ", {}", i)?;
            }
        }
        write!(fmt, ")")
    }
}

impl<T: Num> Mat<T> {
    pub fn null(lin_alg: &LinAlgProgram) -> Result<Self, MatError> {
        unsafe { Self::empty(lin_alg, &[]) }
    }

    /**Tensor with a single dimension of length 1 and a*/
    pub fn array0(lin_alg: &LinAlgProgram, arr: T) -> Result<Mat<T>, MatError> {
        Self::from_slice(lin_alg, &[arr], &[1])
    }

    pub fn array1<const X: usize>(lin_alg: &LinAlgProgram, arr: [T; X]) -> Result<Mat<T>, MatError> {
        Self::from_slice(lin_alg, &arr, &[X])
    }

    pub fn array2<const X: usize, const Y: usize>(lin_alg: &LinAlgProgram, arr: [[T; X]; Y]) -> Result<Mat<T>, MatError> where [T; { X * Y }]: Sized {
        let a: &[T] = unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const T, X * Y) };
        Self::from_slice(lin_alg, a, &[Y, X])
    }

    pub fn array3<const X: usize, const Y: usize, const Z: usize>(lin_alg: &LinAlgProgram, arr: [[[T; X]; Y]; Z]) -> Result<Mat<T>, MatError> where [T; { X * Y * Z }]: Sized {
        let a: &[T] = unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const T, X * Y * Z) };
        Self::from_slice(lin_alg, a, &[Z, Y, X])
    }

    pub fn lin_alg(&self) -> &LinAlgProgram {
        &self.lin_alg
    }

    fn new_with_strides(lin_alg: &LinAlgProgram, buff: Option<Buffer<T>>, contiguous: bool, strides: Box<[usize]>, shape: Box<[usize]>) -> Result<Self, MatError> {
        if shape.len() > 20 {
            Err(MatError::DimensionalityLimitExceeded(shape.len()))
        } else {
            if let Some(buff) = &buff {
                assert_eq!(strides.len(), shape.len(), "strides.len()!=shape.len()");
                if contiguous {
                    assert_eq!(buff.len(), if strides.is_empty() { 0 } else { strides[0] * shape[0] }, "buff.len()!=strides[0]*shape[0]");
                }
                assert_eq!(buff.len(), if strides.is_empty() || strides[0] == 0 { 0 } else { strides.iter().zip(shape.iter()).fold(0, |sum, (a, b)| sum + a * (b - 1)) + 1 }, "buff.len()!=strides.last_index+1");
            }
            let lin_alg = lin_alg.clone();
            Ok(Self { lin_alg, buff, strides, shape })
        }
    }
    pub fn from_buffer(lin_alg: &LinAlgProgram, buff: Buffer<T>, shape: &[usize]) -> Result<Self, MatError> {
        Self::from_buffer_boxed(lin_alg, buff, shape.into())
    }
    pub fn from_buffer_boxed(lin_alg: &LinAlgProgram, buff: Buffer<T>, shape: Box<[usize]>) -> Result<Self, MatError> {
        let strides = Self::strides_for_shape(&shape);
        let len = if strides.is_empty() { 0 } else { strides[0] * shape[0] };
        if buff.len() != len {
            Err(MatError::BufferLengthAndShapeMismatch(buff.len(), shape.to_vec()))
        } else {
            Self::new_with_strides(lin_alg, Some(buff), true, strides, shape)
        }
    }
    pub fn from_slice_infer_wildcard(lin_alg: &LinAlgProgram, arr: &[T], shape: &[isize]) -> Result<Mat<T>, MatError> {
        Self::from_slice_boxed(lin_alg, arr, Self::infer_wildcard(&[arr.len()], shape)?)
    }
    pub fn from_slice(lin_alg: &LinAlgProgram, arr: &[T], shape: &[usize]) -> Result<Mat<T>, MatError> {
        Self::from_slice_boxed(lin_alg, arr, shape.into())
    }
    pub fn from_slice_boxed(lin_alg: &LinAlgProgram, arr: &[T], shape: Box<[usize]>) -> Result<Mat<T>, MatError> {
        let strides = Self::strides_for_shape(&shape);
        let len = if strides.is_empty() { 0 } else { strides[0] * shape[0] };
        if arr.len() != len {
            Err(MatError::BufferLengthAndShapeMismatch(arr.len(), shape.to_vec()))
        } else {
            lin_alg.pro_que.buffer_builder::<T>()
                .flags(flags::MEM_READ_WRITE)
                .len(arr.len())
                .copy_host_slice(arr)
                .build().map_err(MatError::from)
                .and_then(|buff| Self::new_with_strides(lin_alg, Some(buff), true, strides, shape))
        }
    }
    pub fn buffer(&self) -> Option<&Buffer<T>> {
        self.buff.as_ref()
    }
    pub fn strides_for_shape(shape: &[usize]) -> Box<[usize]> {
        if shape.is_empty() {
            Box::<[usize]>::from([])
        } else {
            let mut strides = Box::new_uninit_slice(shape.len());
            strides[shape.len() - 1].write(1);
            unsafe {
                for i in (1..shape.len()).rev() {
                    strides[i - 1].write(strides[i].assume_init() * shape[i]);
                }
                strides.assume_init()
            }
        }
    }
    pub unsafe fn empty(lin_alg: &LinAlgProgram, shape: &[usize]) -> Result<Self, MatError> {
        Self::empty_boxed(lin_alg, shape.into())
    }
    unsafe fn empty_boxed_with_strides(lin_alg: &LinAlgProgram, strides: Box<[usize]>, shape: Box<[usize]>, len: usize) -> Result<Self, MatError> {
        assert_eq!(len, shape.as_shape().size());
        if len == 0 {
            Self::new_with_strides(lin_alg, None, true, strides, shape)
        } else {
            lin_alg.pro_que.buffer_builder::<T>()
                .flags(flags::MEM_READ_WRITE)
                .len(len)
                .build().map_err(|e| MatError::OpenCLError(e))
                .and_then(|buff| Self::new_with_strides(lin_alg, Some(buff), true, strides, shape))
        }
    }
    pub unsafe fn empty_boxed(lin_alg: &LinAlgProgram, shape: Box<[usize]>) -> Result<Self, MatError> {
        let strides = Self::strides_for_shape(&shape);
        let len = if strides.is_empty() { 0 } else { strides[0] * shape[0] };
        Self::empty_boxed_with_strides(lin_alg, strides, shape, len)
    }
    pub unsafe fn empty_like<D: Num>(other: &Mat<T>) -> Result<Mat<D>, MatError> {
        Mat::empty_boxed(&other.lin_alg, other.shape.clone())
    }
    pub fn full_like<D: Num>(other: &Mat<T>, fill_val: D) -> Result<Mat<D>, MatError> {
        Mat::full_boxed(&other.lin_alg, other.shape.clone(), fill_val)
    }
    pub fn full(lin_alg: &LinAlgProgram, shape: &[usize], fill_val: T) -> Result<Self, MatError> {
        Self::full_boxed(lin_alg, shape.into(), fill_val)
    }
    pub fn full_boxed(lin_alg: &LinAlgProgram, shape: Box<[usize]>, fill_val: T) -> Result<Self, MatError> {
        let strides = Self::strides_for_shape(&shape);
        let len = if strides.is_empty() { 0 } else { strides[0] * shape[0] };
        Self::full_boxed_with_strides(lin_alg, strides, shape, len, fill_val)
    }
    fn full_boxed_with_strides(lin_alg: &LinAlgProgram, strides: Box<[usize]>, shape: Box<[usize]>, len: usize, fill_val: T) -> Result<Self, MatError> {
        assert_eq!(len, shape.as_shape().size());
        lin_alg.pro_que.buffer_builder::<T>()
            .flags(flags::MEM_READ_WRITE)
            .len(len)
            .fill_val(fill_val)
            .build().map_err(MatError::from)
            .and_then(|buff| Self::new_with_strides(lin_alg, Some(buff), true, strides, shape))
    }

    pub fn ones(lin_alg: &LinAlgProgram, shape: &[usize]) -> Result<Self, MatError> {
        Self::full(lin_alg, shape, T::one())
    }

    pub fn zeros(lin_alg: &LinAlgProgram, shape: &[usize]) -> Result<Self, MatError> {
        Self::full(lin_alg, shape, T::zero())
    }

    pub fn offset_into_buffer(&self, index: &[usize]) -> Result<usize, MatError> {
        if index.len() == self.ndim() && index.iter().zip(self.shape.iter()).all(|(&a, &b)| a < b) {
            Ok(index.iter().zip(self.strides.iter()).fold(0, |sum, (&i, &s)| sum + i * s))
        } else {
            Err(MatError::InvalidIndex(index.to_vec(), self.shape.to_vec()))
        }
    }
    pub fn clone_transpose(&self, dim0: usize, dim1: usize) -> Result<Self, MatError> {
        let mut c = self.clone();
        c.transpose(dim0, dim1)?;
        Ok(c)
    }
    pub fn transpose(&mut self, dim0: usize, dim1: usize) -> Result<(), MatError> {
        if dim0 >= self.ndim() {
            return Err(MatError::DimensionOutOfBounds(self.shape.to_vec(), dim0));
        }
        if dim1 >= self.ndim() {
            return Err(MatError::DimensionOutOfBounds(self.shape.to_vec(), dim1));
        }
        self.strides.swap(dim0, dim1);
        self.shape.swap(dim0, dim1);
        Ok(())
    }
    pub fn view(&self, ranges: &[Range<usize>]) -> Result<Self, MatError> {
        if ranges.len() > self.ndim() || ranges.iter().zip(self.shape.iter()).all(|(r, &s)| r.start > s || r.end > s || r.start > r.end) {
            Err(MatError::InvalidView(ranges.to_vec(), self.shape.to_vec()))
        } else if let Some(buff) = &self.buff {
            let first_element_offset = ranges.iter().zip(self.strides.iter()).fold(0, |sum, (r, s)| sum + r.start * s);
            let mut new_shape = Vec::with_capacity(self.ndim());
            let mut last_element_offset = first_element_offset;
            for (i, stride) in self.strides.iter().enumerate() {
                let len = ranges.get(i).map(|r| r.end - r.start).unwrap_or(self.shape[i]);
                if len == 0 {
                    return Self::null(&self.lin_alg);
                }
                new_shape.push(len);
                last_element_offset += (len - 1) * stride;
            }
            let size = last_element_offset + 1 - first_element_offset;
            assert!(size > 0);
            let sub = buff.create_sub_buffer(None, first_element_offset, size)?;
            Self::new_with_strides(&self.lin_alg, Some(sub), false, self.strides.clone(), new_shape.into_boxed_slice())
        } else {
            assert_eq!(self.size(), 0);
            Ok(self.clone())
        }
    }
    /**Number of dimensions. Works the same way as in numpy.*/
    pub fn ndim(&self) -> usize {
        assert_eq!(self.shape.len(), self.strides.len());
        self.shape.len()
    }

    /**Changes dimensions of tensor. The total size (all dimension length multiplied together)
    must remain the same. If tensor is contiguous it will return a different view to the same buffer.
    If tensor is not contiguous a copy will be necessary. Works the same way as in numpy.*/
    pub fn reshape(&self, shape: &[usize]) -> Result<Mat<T>, MatError> {
        self.reshape_boxed(shape.into())
    }
    pub fn reshape_boxed(&self, shape: Box<[usize]>) -> Result<Mat<T>, MatError> {
        let strides = Self::strides_for_shape(&shape);
        if strides[0] * shape[0] != self.size() {
            return Err(MatError::IncompatibleShapes(self.shape.to_vec(), shape.to_vec()));
        }
        let buff = if self.contiguous() {
            self.buff.clone() //this is reference-counted
        } else {
            self.copy()?.buff
        };
        Self::new_with_strides(&self.lin_alg, buff, true, strides, shape)
    }
    pub fn reshape1(&self, dim0: usize) -> Result<Mat<T>, MatError> {
        self.reshape(&[dim0])
    }
    pub fn reshape2(&self, dim0: usize, dim1: usize) -> Result<Mat<T>, MatError> {
        self.reshape(&[dim0, dim1])
    }
    pub fn reshape3(&self, dim0: usize, dim1: usize, dim2: usize) -> Result<Mat<T>, MatError> {
        self.reshape(&[dim0, dim1, dim2])
    }
    pub fn reshape4(&self, dim0: usize, dim1: usize, dim2: usize, dim3: usize) -> Result<Mat<T>, MatError> {
        self.reshape(&[dim0, dim1, dim2, dim3])
    }
    pub fn reshape5(&self, dim0: usize, dim1: usize, dim2: usize, dim3: usize, dim4: usize) -> Result<Mat<T>, MatError> {
        self.reshape(&[dim0, dim1, dim2, dim3, dim4])
    }
    /**Inserts an additional dimension of length 1 at a specified index.  Works the same way as in numpy.*/
    pub fn unsqueeze(&self, idx: usize) -> Result<Mat<T>, MatError> {
        if idx > self.ndim() {//notice that idx can be equal to dimensionality!
            Err(MatError::DimensionOutOfBounds(self.shape.to_vec(), idx))
        } else if self.ndim() >= MAX_MAT_DIMS {
            Err(MatError::DimensionalityLimitExceeded(self.ndim() + 1))
        } else {
            let mut new_shape = Vec::with_capacity(self.ndim() + 1);
            new_shape.extend_from_slice(&self.shape[0..idx]);
            new_shape.push(1);
            new_shape.extend_from_slice(&self.shape[idx..]);
            assert_eq!(new_shape[idx], 1);
            self.reshape_boxed(new_shape.into_boxed_slice())
        }
    }
    /**Collapses a dimension at a specified index, provided that the length of that dimension is 1.  Works the same way as in numpy.*/
    pub fn squeeze(&self, idx: usize) -> Result<Mat<T>, MatError> {
        if idx >= self.ndim() {
            Err(MatError::DimensionOutOfBounds(self.shape.to_vec(), idx))
        } else if self.shape[idx] != 1 {
            Err(MatError::NonsingularDimension(self.shape.to_vec(), idx))
        } else {
            assert!(self.ndim() > 0);
            let mut new_shape = Vec::with_capacity(self.ndim() - 1);
            new_shape.extend_from_slice(&self.shape[0..idx]);
            new_shape.extend_from_slice(&self.shape[idx + 1..]);
            self.reshape_boxed(new_shape.into_boxed_slice())
        }
    }
    fn infer_wildcard(my_shape: &[usize], shape: &[isize]) -> Result<Box<[usize]>, MatError> {
        let mut inferred = Vec::with_capacity(shape.len());
        let mut wildcard_pos = usize::MAX;
        let mut new_size = 1;
        for (i, &len) in shape.iter().enumerate() {
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
            let my_size = my_shape.as_shape().size();
            if my_size % new_size == 0 {
                inferred[wildcard_pos] = my_size / new_size;
            } else {
                return Err(MatError::IncompatibleWildcardShapes(my_shape.to_vec(), shape.to_vec()));
            }
        }
        Ok(inferred.into_boxed_slice())
    }
    /**Same as reshape() but you can additionally use -1 as a wildcard. Works the same way as in numpy.*/
    pub fn reshape_infer_wildcard(&self, shape: &[isize]) -> Result<Mat<T>, MatError> {
        self.reshape_boxed(Self::infer_wildcard(self.shape(), shape)?)
    }
    /**Stride corresponding to each dimension of tensor*/
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    /**Shape of tensor is the array containing lengths of each dimension*/
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /**Length of first dimension*/
    pub fn len(&self) -> usize {
        self.shape.first().cloned().unwrap_or(0usize)
    }
    /**Length of the underlying buffer (or sub-buffer)*/
    pub fn len_buffer(&self) -> usize {
        self.buffer().map(|b| b.len()).unwrap_or(0)
    }
    /**Total size obtained by multiplying all dimensions together*/
    pub fn size(&self) -> usize {
        self.shape().as_shape().size()
    }
    /**Reads a single value from buffer. Notice that this memory could lie on GPU.
    This method might have large overhead. It's not recommended to use it in a loop to iterate over buffer.
    Use subviews instead.*/
    pub fn get(&self, index: &[usize]) -> Result<T, MatError> {
        if let Some(buff) = &self.buff {
            let mut tmp = [T::zero(); 1];
            buff.cmd()
                .offset(self.offset_into_buffer(index)?)
                .read(&mut tmp[..])
                .enq()?;
            Ok(tmp[0])
        } else {
            Err(MatError::InvalidIndex(index.to_vec(), self.shape.to_vec()))
        }
    }
    /**Reads entire tensor into a new vector*/
    pub fn to_vec(&self) -> Result<Vec<T>, MatError> {
        if !self.contiguous() {
            Err(MatError::CannotReadNonContiguous())
        } else {
            assert_eq!(self.len_buffer(), self.size());
            self.to_vec_non_contiguous().map_err(MatError::from)
        }
    }
    fn to_vec_non_contiguous(&self) -> ocl::Result<Vec<T>> {
        let mut v = Vec::with_capacity(self.len_buffer());
        unsafe { v.set_len(v.capacity()) };
        self.read_non_contiguous(v.as_mut_slice())?;
        Ok(v)
    }
    /**Reads entire tensor into the provided slice*/
    pub fn read(&self, dst: &mut [T]) -> Result<(), MatError> {
        if !self.contiguous() {
            Err(MatError::CannotReadNonContiguous())
        } else {
            let size = self.len_buffer();
            if dst.len() != size {
                Err(MatError::BufferLengthMismatch(size, dst.len()))
            } else {
                self.read_non_contiguous(dst).map_err(MatError::from)
            }
        }
    }
    fn read_non_contiguous(&self, dst: &mut [T]) -> ocl::Result<()> {
        assert_eq!(dst.len(), self.len_buffer());
        unsafe {
            if let Some(buff) = &self.buff {
                buff.cmd().read(dst).enq()
            } else {
                Ok(())
            }
        }
    }

    pub fn contiguous(&self) -> bool {
        self.size() == self.len_buffer()
    }

    /**This function performs a deep clone of the tensor together with its
    underlying buffer. If the tensor was not contiguous, its copy
     will become contiguous. Works the same way as in numpy. */
    pub fn copy(&self) -> Result<Self, MatError> {
        let mut out = unsafe { Self::empty(&self.lin_alg, self.shape()) }?;
        out.copy_from(self)?;
        Ok(out)
    }
    /**alias for mm()*/
    pub fn matmul(&self, rhs: &Self) -> Result<Self, MatError> {
        self.mm(rhs)
    }
    fn add_dim_args<'b>(kernel: &mut KernelBuilder<'b>, shape: &'b [usize]) -> usize {
        let mut total_s = 1;
        for dim_s in shape {
            kernel.arg(dim_s);
            total_s *= dim_s;
        }
        total_s
    }
    fn add_stride_args<'b>(&'b self, kernel: &mut KernelBuilder<'b>, range: Range<usize>) {
        for stride in &self.strides[range] {
            kernel.arg(stride);
        }
    }
    /**Matrix multiplication*/
    pub fn mm(&self, rhs: &Self) -> Result<Self, MatError> {
        if self.ndim() <= 1 {
            return Err(MatError::MatMulUndefined());
        }
        // lhs.shape==(s*,j,i)
        // rhs.shape==(s*,i,k)
        // out.shape==(s*,j,k)
        let lhs_i = self.shape[self.ndim() - 1];
        let rhs_i = rhs.shape[self.ndim() - 2];
        let j = self.shape[self.ndim() - 2];
        let k = rhs.shape[self.ndim() - 1];
        if lhs_i != rhs_i {
            return Err(MatError::MatMulDimIncompatible(lhs_i, rhs_i));
        }
        if !self.shape[..self.ndim() - 2].iter().zip(rhs.shape[..self.ndim() - 2].iter()).all(|(a, b)| a == b) {
            return Err(MatError::MatMulShapeIncompatible(self.shape.to_vec(), rhs.shape.to_vec()));
        }

        let mut out_shape = rhs.shape.clone();
        out_shape[self.ndim() - 1] = self.shape[self.ndim() - 1];
        assert_eq!(out_shape[self.ndim() - 2], j);
        assert_eq!(out_shape[self.ndim() - 1], k);
        let out = unsafe { Self::empty_boxed(&self.lin_alg, out_shape)? };

        let mut kernel = self.lin_alg.pro_que.kernel_builder(format!("{}_mm{}", T::OPENCL_TYPE_STR, self.ndim()));
        kernel.arg(self.buffer().unwrap())
            .arg(rhs.buffer().unwrap())
            .arg(out.buffer().unwrap())
            .arg(&lhs_i) // i_len
            .arg(&self.strides[self.ndim() - 2])//lhs_j_stride
            .arg(&self.strides[self.ndim() - 1])//lhs_i_stride
            .arg(&rhs.strides[self.ndim() - 2])//rhs_i_stride
            .arg(&rhs.strides[self.ndim() - 1])//rhs_k_stride
            .arg(&out.strides[self.ndim() - 2])//out_j_stride
            .arg(&out.strides[self.ndim() - 1]);//out_k_stride;
        let total_s = Self::add_dim_args(&mut kernel, &self.shape[0..self.ndim() - 2]);
        self.add_stride_args(&mut kernel, 0..self.ndim() - 2);
        rhs.add_stride_args(&mut kernel, 0..self.ndim() - 2);
        out.add_stride_args(&mut kernel, 0..self.ndim() - 2);
        kernel.global_work_size(SpatialDims::Three(j, k, total_s));
        let kernel = kernel.build()?;
        unsafe {
            kernel.cmd().enq()?
        }
        Ok(out)
    }
    fn mat_cmp_mat(&self, other: &Self, mode: &'static str) -> Result<Mat<u8>, MatError> {
        if self.shape != other.shape {
            Err(MatError::IncompatibleShapes(self.shape.to_vec(), other.shape.to_vec()))
        } else if let Some(buff) = &self.buff {
            let out = unsafe { Mat::<u8>::empty(&self.lin_alg, &self.shape)? };
            let kernel = self.lin_alg.pro_que.kernel_builder(format!("{}_mat_cmp_mat_{}", T::OPENCL_TYPE_STR, mode))
                .arg(buff)
                .arg(other.buffer().unwrap())
                .arg(out.buffer().unwrap())
                .global_work_size(self.size())
                .build()?;
            unsafe {
                kernel.cmd().enq()?;
            }
            Ok(out)
        } else {
            Mat::<u8>::null(&self.lin_alg)
        }
    }
    pub fn eq_mat(&self, other: &Self) -> Result<Mat<u8>, MatError> {
        self.mat_cmp_mat(other, "eq")
    }
    pub fn lt_mat(&self, other: &Self) -> Result<Mat<u8>, MatError> {
        self.mat_cmp_mat(other, "lt")
    }
    pub fn le_mat(&self, other: &Self) -> Result<Mat<u8>, MatError> {
        self.mat_cmp_mat(other, "le")
    }
    pub fn gt_mat(&self, other: &Self) -> Result<Mat<u8>, MatError> {
        self.mat_cmp_mat(other, "gt")
    }
    pub fn ge_mat(&self, other: &Self) -> Result<Mat<u8>, MatError> {
        self.mat_cmp_mat(other, "ge")
    }
    pub fn ne_mat(&self, other: &Self) -> Result<Mat<u8>, MatError> {
        self.mat_cmp_mat(other, "ne")
    }
    fn mat_cmp_scalar(&self, scalar: T, mode: &'static str) -> Result<Mat<u8>, MatError> {
        if let Some(buff) = self.buffer() {
            let out = unsafe { Mat::<u8>::empty(&self.lin_alg, self.shape())? };
            let kernel = self.lin_alg.pro_que.kernel_builder(format!("{}_scalar_cmp_{}", T::OPENCL_TYPE_STR, mode))
                .arg(buff)
                .arg(&scalar)
                .arg(out.buffer().unwrap())
                .global_work_size(self.size())
                .build()?;
            unsafe {
                kernel.cmd().enq()?;
            }
            Ok(out)
        } else {
            Mat::<u8>::null(&self.lin_alg)
        }
    }
    pub fn eq_scalar(&self, other: T) -> Result<Mat<u8>, MatError> {
        self.mat_cmp_scalar(other, "eq")
    }
    pub fn lt_scalar(&self, other: T) -> Result<Mat<u8>, MatError> {
        self.mat_cmp_scalar(other, "lt")
    }
    pub fn le_scalar(&self, other: T) -> Result<Mat<u8>, MatError> {
        self.mat_cmp_scalar(other, "le")
    }
    pub fn gt_scalar(&self, other: T) -> Result<Mat<u8>, MatError> {
        self.mat_cmp_scalar(other, "gt")
    }
    pub fn ge_scalar(&self, other: T) -> Result<Mat<u8>, MatError> {
        self.mat_cmp_scalar(other, "ge")
    }
    pub fn ne_scalar(&self, other: T) -> Result<Mat<u8>, MatError> {
        self.mat_cmp_scalar(other, "ne")
    }
    /**Creates a copy of matrix and converts all its elements to a different type.
    Is equivalent to copy() if both source and target types are the same*/
    pub fn cast<D: Num>(&self) -> Result<Mat<D>, MatError> {
        let mut out = unsafe { Self::empty_like::<D>(self)? };
        out.mat_to_lhs_mat(self, "cast")?;
        Ok(out)
    }
    pub fn abs(&mut self) -> Result<Mat<T>, MatError> {
        let mut out = self.copy()?;
        out.abs_in_place()?;
        Ok(out)
    }
    pub fn abs_in_place(&mut self) -> Result<(), MatError> {
        self.unary_to_lhs_mat(if T::IS_FLOAT { "fabs" } else { "abs" })
    }
    fn scalar_to_lhs_mat(&mut self, scalar: T, mode: &'static str) -> Result<(), MatError> {
        if let Some(buff) = self.buffer() {
            let kernel = self.lin_alg.pro_que.kernel_builder(format!("{}_scalar_to_lhs_mat_{}", T::OPENCL_TYPE_STR, mode))
                .arg(buff)
                .arg(&scalar)
                .global_work_size(self.size())
                .build()?;
            unsafe {
                kernel.cmd().enq().map_err(MatError::from)
            }
        } else {
            Ok(())
        }
    }
    pub fn fill(&mut self, scalar: T) -> Result<(), MatError> {
        self.scalar_to_lhs_mat(scalar, "fill")
    }
    pub fn add_scalar(&mut self, scalar: T) -> Result<(), MatError> {
        self.scalar_to_lhs_mat(scalar, "add")
    }
    pub fn sub_scalar(&mut self, scalar: T) -> Result<(), MatError> {
        self.scalar_to_lhs_mat(scalar, "sub")
    }
    pub fn mul_scalar(&mut self, scalar: T) -> Result<(), MatError> {
        self.scalar_to_lhs_mat(scalar, "mul")
    }
    pub fn div_scalar(&mut self, scalar: T) -> Result<(), MatError> {
        self.scalar_to_lhs_mat(scalar, "div")
    }
    pub fn min_scalar(&mut self, scalar: T) -> Result<(), MatError> {
        self.scalar_to_lhs_mat(scalar, "min")
    }
    pub fn max_scalar(&mut self, scalar: T) -> Result<(), MatError> {
        self.scalar_to_lhs_mat(scalar, "max")
    }
    fn mat_to_lhs_mat<D: Num>(&self, other: &Mat<D>, mode: &'static str) -> Result<(), MatError> {
        if self.shape != other.shape {
            Err(MatError::IncompatibleShapes(self.shape.to_vec(), other.shape.to_vec()))
        } else if let Some(buff) = self.buffer() {
            let fn_name = format!("mat_{input_type}_to_lhs_mat_{output_type}_{dims}_{mode}", input_type = D::OPENCL_TYPE_STR, output_type = T::OPENCL_TYPE_STR, dims = self.ndim(), mode = mode);
            println!("fn_name={}", fn_name);
            let mut kernel = self.lin_alg.pro_que.kernel_builder(fn_name);
            kernel.arg(buff)
                .arg(other.buffer().unwrap());
            let size = Self::add_dim_args(&mut kernel, &self.shape);
            self.add_stride_args(&mut kernel, 0..self.ndim());
            other.add_stride_args(&mut kernel, 0..self.ndim());
            kernel.global_work_size(size);

            let kernel = kernel.build()?;
            unsafe {
                kernel.cmd().enq().map_err(MatError::from)
            }
        } else {
            Ok(())
        }
    }
    fn unary_mat<D: Num>(&self, mode: &'static str) -> Result<Mat<D>, MatError> {
        let mut out = unsafe { Self::empty_like::<D>(self)? };
        out.mat_to_lhs_mat(self, mode)?;
        Ok(out)
    }
    fn unary_to_lhs_mat(&self, mode: &'static str) -> Result<(), MatError> {
        self.mat_to_lhs_mat(self, mode)
    }
    pub fn copy_from(&mut self, other: &Self) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "copy")
    }
    pub fn add_mat(&mut self, other: &Self) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "add")
    }
    pub fn div_mat(&mut self, other: &Self) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "div")
    }
    pub fn mul_mat(&mut self, other: &Self) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "hadamard")
    }
    pub fn sub_mat(&mut self, other: &Self) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "sub")
    }
    pub fn min_mat(&mut self, other: &Self) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "min")
    }
    pub fn max_mat(&mut self, other: &Self) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "max")
    }
    pub fn sum(&self) -> Result<T, MatError> {
        // let kernel = self.lin_alg.pro_que.kernel_builder("aggregate_sum")
        //     .arg(&self.buff)
        //     .arg(&other.buff)
        //     .arg(&out.buff)
        //     .global_work_size(self.len())
        //     .build()?;
        Ok(T::zero())
    }
    pub fn item(&self) -> Result<T, MatError> {
        if self.len_buffer() == 1 {
            let mut tmp = [T::zero()];
            self.read(&mut tmp)?;
            Ok(tmp[0])
        } else {
            Err(MatError::NotSingletonMatrix(self.shape.to_vec()))
        }
    }
    pub fn sin(&self) -> Result<Mat<f32>, MatError> {
        self.unary_mat("sin")
    }
    pub fn cos(&self) -> Result<Mat<f32>, MatError> {
        self.unary_mat("cos")
    }
    pub fn tan(&self) -> Result<Mat<f32>, MatError> {
        self.unary_mat("tan")
    }
    pub fn tanh(&self) -> Result<Mat<f32>, MatError> {
        self.unary_mat("tanh")
    }
    pub fn exp(&self) -> Result<Mat<f32>, MatError> {
        self.unary_mat("exp")
    }
    pub fn exp2(&self) -> Result<Mat<f32>, MatError> {
        self.unary_mat("exp2")
    }
    pub fn exp10(&self) -> Result<Mat<f32>, MatError> {
        self.unary_mat("exp10")
    }
    pub fn log(&self) -> Result<Mat<f32>, MatError> {
        self.unary_mat("log")
    }
    pub fn log2(&self) -> Result<Mat<f32>, MatError> {
        self.unary_mat("log2")
    }
    pub fn log10(&self) -> Result<Mat<f32>, MatError> {
        self.unary_mat("log10")
    }
}

impl Mat<f32> {
    pub fn sin_in_place(&mut self) -> Result<(), MatError> {
        self.unary_to_lhs_mat("sin")
    }
    pub fn cos_in_place(&mut self) -> Result<(), MatError> {
        self.unary_to_lhs_mat("cos")
    }
    pub fn tan_in_place(&mut self) -> Result<(), MatError> {
        self.unary_to_lhs_mat("tan")
    }
    pub fn tanh_in_place(&mut self) -> Result<(), MatError> {
        self.unary_to_lhs_mat("tanh")
    }
    pub fn exp_in_place(&mut self) -> Result<(), MatError> {
        self.unary_to_lhs_mat("exp")
    }
    pub fn exp2_in_place(&mut self) -> Result<(), MatError> {
        self.unary_to_lhs_mat("exp2")
    }
    pub fn exp10_in_place(&mut self) -> Result<(), MatError> {
        self.unary_to_lhs_mat("exp10")
    }
    pub fn log_in_place(&mut self) -> Result<(), MatError> {
        self.unary_to_lhs_mat("log")
    }
    pub fn log2_in_place(&mut self) -> Result<(), MatError> {
        self.unary_to_lhs_mat("log2")
    }
    pub fn log10_in_place(&mut self) -> Result<(), MatError> {
        self.unary_to_lhs_mat("log10")
    }
}

impl<T: Num> Display for Mat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.ndim() == 0 {
            write!(f, "[]")
        } else {
            let buff = self.to_vec_non_contiguous().map_err(|e| std::fmt::Error)?;
            fn recursive_print<T: Num>(me: &Mat<T>, idx: &mut [usize], level: usize, buff: &[T], f: &mut Formatter<'_>) -> std::fmt::Result {
                if level == me.ndim() {
                    write!(f, "{}", buff[me.offset_into_buffer(idx).unwrap()])
                } else {
                    write!(f, "[");
                    let dim = me.shape()[level];
                    if dim > 0 {
                        idx[level] = 0;
                        recursive_print(me, idx, level + 1, buff, f)?;
                        for coord in 1..dim {
                            write!(f, ", ");
                            idx[level] = coord;
                            recursive_print(me, idx, level + 1, buff, f)?;
                        }
                    }
                    write!(f, "]")
                }
            }
            let mut idx = vec![0; self.ndim()];
            recursive_print(self, idx.as_mut_slice(), 0, buff.as_slice(), f)
        }
    }
}

impl<T: Num> Debug for Mat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl<T: Num> PartialEq for Mat<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            false
        } else {
            let b = self.ne_mat(other).unwrap();
            b.sum().unwrap() == 0 // the number of different elements is zero, hence matrices are equal
        }
    }
}

impl<T: Num> Add<&Self> for Mat<T> {
    type Output = Mat<T>;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self.add_mat(rhs).unwrap();
        self
    }
}

impl<T: Num> Sub<&Self> for Mat<T> {
    type Output = Mat<T>;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.sub_mat(rhs).unwrap();
        self
    }
}

impl<T: Num> Div<&Self> for Mat<T> {
    type Output = Mat<T>;

    fn div(mut self, rhs: &Self) -> Self::Output {
        self.div_mat(rhs).unwrap();
        self
    }
}

impl<T: Num> Mul<&Self> for Mat<T> {
    type Output = Mat<T>;

    fn mul(mut self, rhs: &Self) -> Self::Output {
        self.mul_mat(rhs).unwrap();
        self
    }
}


impl<T: Num> Add<T> for Mat<T> {
    type Output = Mat<T>;

    fn add(mut self, rhs: T) -> Self::Output {
        self.add_scalar(rhs).unwrap();
        self
    }
}

impl<T: Num> Sub<T> for Mat<T> {
    type Output = Mat<T>;

    fn sub(mut self, rhs: T) -> Self::Output {
        self.sub_scalar(rhs).unwrap();
        self
    }
}

impl<T: Num> Div<T> for Mat<T> {
    type Output = Mat<T>;

    fn div(mut self, rhs: T) -> Self::Output {
        self.div_scalar(rhs).unwrap();
        self
    }
}

impl<T: Num> Mul<T> for Mat<T> {
    type Output = Mat<T>;

    fn mul(mut self, rhs: T) -> Self::Output {
        self.mul_scalar(rhs).unwrap();
        self
    }
}


impl<T: Num> AddAssign<&Self> for Mat<T> {
    fn add_assign(&mut self, rhs: &Self) {
        self.add_mat(rhs).unwrap()
    }
}

impl<T: Num> SubAssign<&Self> for Mat<T> {
    fn sub_assign(&mut self, rhs: &Self) {
        self.sub_mat(rhs).unwrap()
    }
}

impl<T: Num> DivAssign<&Self> for Mat<T> {
    fn div_assign(&mut self, rhs: &Self) {
        self.div_mat(rhs).unwrap()
    }
}

impl<T: Num> MulAssign<&Self> for Mat<T> {
    fn mul_assign(&mut self, rhs: &Self) {
        self.mul_mat(rhs).unwrap()
    }
}


impl<T: Num> AddAssign<T> for Mat<T> {
    fn add_assign(&mut self, rhs: T) {
        self.add_scalar(rhs).unwrap()
    }
}

impl<T: Num> SubAssign<T> for Mat<T> {
    fn sub_assign(&mut self, rhs: T) {
        self.sub_scalar(rhs).unwrap()
    }
}

impl<T: Num> DivAssign<T> for Mat<T> {
    fn div_assign(&mut self, rhs: T) {
        self.div_scalar(rhs).unwrap()
    }
}

impl<T: Num> MulAssign<T> for Mat<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.mul_scalar(rhs).unwrap()
    }
}

pub trait IntoRange {
    fn into(self, dim_len: usize) -> Range<usize>;
}

impl IntoRange for Range<usize> {
    fn into(self, _: usize) -> Range<usize> {
        self
    }
}

impl IntoRange for RangeFull {
    fn into(self, dim_len: usize) -> Range<usize> {
        0..dim_len
    }
}

impl IntoRange for RangeFrom<usize> {
    fn into(self, dim_len: usize) -> Range<usize> {
        self.start..dim_len
    }
}

impl IntoRange for RangeTo<usize> {
    fn into(self, dim_len: usize) -> Range<usize> {
        0..self.end
    }
}

impl IntoRange for RangeToInclusive<usize> {
    fn into(self, dim_len: usize) -> Range<usize> {
        0..self.end + 1
    }
}

impl IntoRange for usize {
    fn into(self, dim_len: usize) -> Range<usize> {
        self..self + 1
    }
}

impl<T: Num> Mat<T> {
    /**Shorthand for sub view with 1 range*/
    pub fn v1<R1: IntoRange>(&self, index: R1) -> Self {
        self.view(&[index.into(self.shape[0])]).unwrap()
    }
    /**Shorthand for sub view with 2 ranges*/
    pub fn v2<R1: IntoRange, R2: IntoRange>(&self, r1: R1, r2: R2) -> Self {
        self.view(&[r1.into(self.shape[0]), r2.into(self.shape[1])]).unwrap()
    }
    /**Shorthand for sub view with 3 ranges*/
    pub fn v3<R1: IntoRange, R2: IntoRange, R3: IntoRange>(&self, r1: R1, r2: R2, r3: R3) -> Self {
        self.view(&[r1.into(self.shape[0]), r2.into(self.shape[1]), r3.into(self.shape[2])]).unwrap()
    }
    /**Shorthand for sub view with 4 ranges*/
    pub fn v4<R1: IntoRange, R2: IntoRange, R3: IntoRange, R4: IntoRange>(&self, r1: R1, r2: R2, r3: R3, r4: R4) -> Self {
        self.view(&[r1.into(self.shape[0]), r2.into(self.shape[1]), r3.into(self.shape[2]), r4.into(self.shape[3])]).unwrap()
    }
}

use ocl::{ProQue, SpatialDims, flags, Platform, Device, Buffer, Error, Queue};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull,RangeFrom, RangeTo, RangeToInclusive, RangeInclusive};
use std::fmt::{Display, Formatter, Debug};
use crate::kernel::{LinAlgProgram, MAX_MAT_DIMS};
use crate::num::Num;
use ocl::builders::KernelBuilder;


pub struct Mat<T: Num, const S: usize> {
    pro_que: LinAlgProgram,
    buff: Buffer<T>,
    shape: [usize; S],
    strides: [usize; S],
}

pub enum MatError {
    BufferLengthMismatch(usize,usize),
    CannotReadNonContiguous(),
    NotSingletonMatrix(Vec<usize>),
    NonsingularDimension(Vec<usize>, usize),
    DimensionOutOfBounds(Vec<usize>, usize),
    MultipleWildcards(Vec<isize>),
    IncompatibleShapes(Vec<usize>, Vec<usize>),
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
            MatError::BufferLengthMismatch(tensor_buff, dst_buff) => write!(fmt, "Tensor has buffer of length {} but provided destination has length {}", tensor_buff, dst_buff),
            MatError::NotSingletonMatrix(shape) =>write!(fmt, "Tensor of shape {} is not a singleton",shape.as_shape()),
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

struct Shape<'a, T: Num>(&'a [T]);

trait AsShape<T: Num> {
    fn as_shape(&self) -> Shape<T>;
}

impl<T: Num, const S: usize> AsShape<T> for [T; S] {
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
        self.0.iter().fold(T::one(), |a, &b| a * b)
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
        write!(fmt, "(")
    }
}

impl<T: Num> Mat<T, 1> {


    /**Tensor with a single dimension of length 1 and a*/
    pub fn array0(lin_alg: &LinAlgProgram, arr: T) -> Result<Mat<T, 1>, MatError> {
        Self::from_slice(lin_alg, &[arr], [1])
    }

    pub fn array1<const X: usize>(lin_alg: &LinAlgProgram, arr: [T; X]) -> Result<Mat<T, 1>, MatError> {
        Self::from_slice(lin_alg, &arr, [X])
    }
}

impl<T: Num> Mat<T, 2> {
    pub fn array2<const X: usize, const Y: usize>(lin_alg: &LinAlgProgram, arr: [[T; X]; Y]) -> Result<Mat<T, 2>, MatError> where [T; { X * Y }]: Sized {
        let a: &[T] = unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const T, X * Y) };
        Self::from_slice(lin_alg, a, [Y, X])
    }
}

impl<T: Num> Mat<T, 3> {
    pub fn array3<const X: usize, const Y: usize, const Z: usize>(lin_alg: &LinAlgProgram, arr: [[[T; X]; Y]; Z]) -> Result<Mat<T, 3>, MatError> where [T; { X * Y * Z }]: Sized {
        let a: &[T] = unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const T, X * Y * Z) };
        Self::from_slice(lin_alg, a, [Z, Y, X])
    }
}

impl<T: Num, const S: usize> Mat<T, S> {
    pub fn null(lin_alg: &LinAlgProgram) -> Result<Self, MatError> {
        unsafe{Self::empty(lin_alg,[0;S])}
    }

    fn new_with_strides(lin_alg: &LinAlgProgram, buff: Buffer<T>, contiguous:bool, strides: [usize; S], shape: [usize; S]) -> Result<Self, MatError> {
        if shape.len() > 20 {
            Err(MatError::DimensionalityLimitExceeded(shape.len()))
        } else {
            assert_eq!(strides.len(), shape.len());
            if contiguous{
                assert_eq!(buff.len(), if strides.is_empty() { 0 } else { strides[0] * shape[0] });
            }
            assert_eq!(buff.len(), if strides.is_empty()||strides[0]==0{0}else{strides.iter().zip(shape.iter()).fold(0,|sum,(a,b)|sum+a*(b-1))+1});

            let pro_que = lin_alg.clone();
            Ok(Self { pro_que, buff, strides, shape })
        }
    }
    pub fn from_buffer(lin_alg: &LinAlgProgram, buff: Buffer<T>, shape: [usize; S]) -> Result<Self, MatError> {
        let strides = Self::strides_for_shape(&shape);
        let len = if strides.is_empty() { 1 } else { strides[0] * shape[0] };
        if buff.len() != len {
            Err(MatError::BufferLengthAndShapeMismatch(buff.len(), shape.to_vec()))
        } else {
            Self::new_with_strides(lin_alg, buff, true,strides, shape)
        }
    }
    pub fn from_slice(lin_alg: &LinAlgProgram, arr: &[T], shape: [usize; S]) -> Result<Mat<T, S>, MatError> {
        let strides = Self::strides_for_shape(&shape);
        let len = if strides.is_empty() { 1 } else { strides[0] * shape[0] };
        if arr.len() != len {
            Err(MatError::BufferLengthAndShapeMismatch(arr.len(), shape.to_vec()))
        } else {
            lin_alg.pro_que.buffer_builder::<T>()
                .flags(flags::MEM_READ_WRITE)
                .len(arr.len())
                .copy_host_slice(arr)
                .build().map_err(MatError::from)
                .and_then(|buff| Self::new_with_strides(lin_alg, buff, true,strides, shape))
        }
    }
    pub fn buffer(&self) -> &Buffer<T> {
        &self.buff
    }
    pub fn buffer_mut(&mut self) -> &mut Buffer<T> {
        &mut self.buff
    }
    pub fn strides_for_shape(shape: &[usize; S]) -> [usize; S] {
        if shape.is_empty() {
            assert_eq!(S,0);
            [0usize; S]
        } else {
            let mut strides: [MaybeUninit<usize>; S] = MaybeUninit::uninit_array();
            strides[S - 1] = MaybeUninit::new(1);
            unsafe {
                for i in (1..S).rev() {
                    strides[i - 1].write(strides[i].assume_init() * shape[i]);
                }
                MaybeUninit::array_assume_init(strides)
            }
        }
    }
    pub unsafe fn empty(lin_alg: &LinAlgProgram, shape: [usize; S]) -> Result<Self, MatError> {
        let strides = Self::strides_for_shape(&shape);
        let len = if strides.is_empty() { 0 } else { strides[0] * shape[0] };
        lin_alg.pro_que.buffer_builder::<T>()
            .flags(flags::MEM_READ_WRITE)
            .len(len)
            .build().map_err(|e| MatError::OpenCLError(e))
            .and_then(|buff| Self::new_with_strides(lin_alg, buff, true,strides, shape))
    }
    pub fn filled(lin_alg: &LinAlgProgram, shape: [usize; S], fill_val: T) -> Result<Self, MatError> {
        let strides = Self::strides_for_shape(&shape);
        let len = if strides.is_empty() { 1 } else { strides[0] * shape[0] };
        lin_alg.pro_que.buffer_builder::<T>()
            .flags(flags::MEM_READ_WRITE)
            .len(len)
            .fill_val(fill_val)
            .build().map_err(MatError::from)
            .and_then(|buff| Self::new_with_strides(lin_alg, buff, true,strides, shape))
    }

    pub fn ones(lin_alg: &LinAlgProgram, shape: [usize; S]) -> Result<Self, MatError> {
        Self::filled(lin_alg, shape, T::one())
    }

    pub fn zeros(lin_alg: &LinAlgProgram, shape: [usize; S]) -> Result<Self, MatError> {
        Self::filled(lin_alg, shape, T::zero())
    }

    pub fn offset_into_buffer(&self, index: &[usize; S]) -> Option<usize> {
        if index.iter().zip(self.shape.iter()).all(|(&a, &b)| a < b) {
            Some(index.iter().zip(self.strides.iter()).fold(0, |sum, (&i, &s)| sum + i * s))
        } else {
            None
        }
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
    pub fn view(&self, ranges: [Range<usize>; S]) -> Result<Self, MatError> {
        if ranges.iter().zip(self.shape.iter()).all(|(r, &s)| r.start > s || r.end > s || r.start > r.end) {
            Err(MatError::InvalidView(ranges.to_vec(), self.shape.to_vec()))
        } else {
            let first_element_offset = ranges.iter().zip(self.strides.iter()).fold(0, |sum, (r, s)| sum + r.start * s);
            let mut new_shape: [MaybeUninit<usize>; S] = MaybeUninit::uninit_array();
            let mut last_element_offset = first_element_offset;
            let new_shape = unsafe {
                for (i, r) in ranges.into_iter().enumerate() {
                    let len = r.end - r.start;
                    new_shape[i].write(len);
                    if len==0{
                        return Self::null(&self.pro_que)
                    }
                    last_element_offset += (len-1)*self.strides[i];
                }
                MaybeUninit::array_assume_init(new_shape)
            };
            let size = last_element_offset + 1 - first_element_offset;
            let sub = self.buff.create_sub_buffer(None, first_element_offset, size)?;
            Self::new_with_strides(&self.pro_que, sub, false,self.strides, new_shape)
        }
    }
    /**Number of dimensions. Works the same way as in numpy.*/
    pub fn ndim(&self) -> usize { S }

    /**Changes dimensions of tensor. The total size (all dimension length multiplied together)
    must remain the same. If tensor is contiguous it will return a different view to the same buffer.
    If tensor is not contiguous a copy will be necessary. Works the same way as in numpy.*/
    pub fn reshape<const S2: usize>(&self, shape: [usize; S2]) -> Result<Mat<T, S2>, MatError> {
        let strides = Mat::<T, S2>::strides_for_shape(&shape);
        if strides[0] * shape[0] != self.size() {
            return Err(MatError::IncompatibleShapes(self.shape.to_vec(), shape.to_vec()));
        }
        let buff = if self.contiguous() {
            self.buff.clone() //this is reference-counted
        } else {
            self.copy()?.buff
        };
        Mat::<T, S2>::new_with_strides(&self.pro_que, buff, true,strides, shape)
    }
    pub fn reshape1(&self, dim0: usize) -> Result<Mat<T, 1>, MatError> {
        self.reshape([dim0])
    }
    pub fn reshape2(&self, dim0: usize,dim1: usize) -> Result<Mat<T, 2>, MatError> {
        self.reshape([dim0,dim1])
    }
    pub fn reshape3(&self, dim0: usize,dim1: usize,dim2: usize) -> Result<Mat<T, 3>, MatError> {
        self.reshape([dim0,dim1,dim2])
    }
    pub fn reshape4(&self, dim0: usize,dim1: usize,dim2: usize,dim3: usize) -> Result<Mat<T, 4>, MatError> {
        self.reshape([dim0,dim1,dim2,dim3])
    }
    pub fn reshape5(&self, dim0: usize,dim1: usize,dim2: usize,dim3: usize,dim4: usize) -> Result<Mat<T, 5>, MatError> {
        self.reshape([dim0,dim1,dim2,dim3,dim4])
    }
    /**Inserts an additional dimension of length 1 at a specified index.  Works the same way as in numpy.*/
    pub fn unsqueeze(&self, idx: usize) -> Result<Mat<T, { S + 1 }>, MatError> where [usize; S + 1]: Sized {
        if idx > self.ndim() {//notice that idx can be equal to dimensionality!
            Err(MatError::DimensionOutOfBounds(self.shape.to_vec(), idx))
        } else {
            let mut new_shape = [0; { S + 1 }];
            new_shape[0..idx].copy_from_slice(&self.shape[0..idx]);
            new_shape[idx] = 1;
            new_shape[idx + 1..].copy_from_slice(&self.shape[idx..]);
            self.reshape(new_shape)
        }
    }
    /**Collapses a dimension at a specified index, provided that the length of that dimension is 1.  Works the same way as in numpy.*/
    pub fn squeeze(&self, idx: usize) -> Result<Mat<T, { S - 1 }>, MatError> where [usize; S - 1]: Sized {
        if idx >= self.ndim() {
            Err(MatError::DimensionOutOfBounds(self.shape.to_vec(), idx))
        } else if self.shape[idx] != 1 {
            Err(MatError::NonsingularDimension(self.shape.to_vec(), idx))
        } else {
            let mut new_shape = [0; { S - 1 }];
            new_shape[0..idx].copy_from_slice(&self.shape[0..idx]);
            new_shape[idx..].copy_from_slice(&self.shape[idx + 1..]);
            self.reshape(new_shape)
        }
    }

    /**Same as reshape() but you can additionally use -1 as a wildcard. Works the same way as in numpy.*/
    pub fn reshape_infer_wildcard<const S2: usize>(&self, mut shape: [isize; S2]) -> Result<Mat<T, S2>, MatError> {
        if let Some(wildcard_pos) = shape.iter().position(|&a| a < 0) {
            if shape[wildcard_pos + 1..].iter().find(|&&a| a < 0).is_some() {
                return Err(MatError::MultipleWildcards(shape.to_vec()));
            }
            let new_size = shape.iter().filter(|&&a| a >= 0).fold(1, |a, &b| a * b) as usize;
            let my_size = self.size();
            if my_size % new_size == 0 {
                shape[wildcard_pos] = (my_size / new_size) as isize;
            } else {
                return Err(MatError::IncompatibleWildcardShapes(self.shape.to_vec(), shape.to_vec()));
            }
        }
        let shape = shape.map(|a| a as usize);
        self.reshape(shape)
    }
    /**Stride corresponding to each dimension of tensor*/
    pub fn strides(&self) -> &[usize; S] {
        &self.strides
    }
    /**Shape of tensor is the array containing lengths of each dimension*/
    pub fn shape(&self) -> &[usize; S] {
        &self.shape
    }
    /**Length of first dimension*/
    pub fn len(&self) -> usize {
        self.shape.first().cloned().unwrap_or(1usize)
    }
    /**Length of the underlying buffer (or sub-buffer)*/
    pub fn len_buffer(&self) -> usize {
        self.buff.len()
    }
    /**Total size obtained by multiplying all dimensions together*/
    pub fn size(&self) -> usize {
        self.shape.as_shape().size()
    }
    /**Reads a single value from buffer. Notice that this memory could lie on GPU.
    This method might have large overhead. It's not recommended to use it in a loop to iterate over buffer.
    Use subviews instead.*/
    pub fn get(&self, index: &[usize; S]) -> Result<T, Error> {
        let mut tmp = [T::zero(); 1];
        self.buff.cmd()
            .offset(self.offset_into_buffer(index).ok_or_else(|| Error::from(format!("Indices {:?} are out of bounds {:?}", index, self.shape)))?)
            .read(&mut tmp[..])
            .enq()?;
        Ok(tmp[0])
    }
    /**Reads entire tensor into a new vector*/
    pub fn to_vec(&self) -> Result<Vec<T>, MatError> {
        if !self.contiguous() {
            Err(MatError::CannotReadNonContiguous())
        } else {
            assert_eq!(self.len_buffer(),self.size());
            let mut v = Vec::with_capacity(self.len_buffer());
            unsafe { v.set_len(v.capacity()) };
            self.read(v.as_mut_slice())?;
            Ok(v)
        }
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
                unsafe {
                    self.buff.cmd().read(dst).enq().map_err(MatError::from)
                }
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
        let mut out = unsafe { Self::empty(&self.pro_que, self.shape) }?;
        out.copy_from(self)?;
        Ok(out)
    }
    /**alias for mm()*/
    pub fn matmul(&mut self, rhs: &Self) -> Result<Self, MatError> {
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
        if S <= 1 {
            return Err(MatError::MatMulUndefined());
        }
        // lhs.shape==(s*,j,i)
        // rhs.shape==(s*,i,k)
        // out.shape==(s*,j,k)
        let lhs_i = self.shape[S - 1];
        let rhs_i = rhs.shape[S - 2];
        let j = self.shape[S - 2];
        let k = rhs.shape[S - 1];
        if lhs_i != rhs_i {
            return Err(MatError::MatMulDimIncompatible(lhs_i, rhs_i));
        }
        if !self.shape[..S - 2].iter().zip(rhs.shape[..S - 2].iter()).all(|(a, b)| a == b) {
            return Err(MatError::MatMulShapeIncompatible(self.shape.to_vec(), rhs.shape.to_vec()));
        }

        let mut out_shape = rhs.shape.clone();
        out_shape[S - 1] = self.shape[S - 1];
        assert_eq!(out_shape[S - 2], j);
        assert_eq!(out_shape[S - 1], k);
        let out = unsafe { Self::empty(&self.pro_que, out_shape)? };

        let mut kernel = self.pro_que.pro_que.kernel_builder(format!("{}_mm{}", T::opencl_type_str(), S));
        kernel.arg(&self.buff)
            .arg(&rhs.buff)
            .arg(&out.buff)
            .arg(&lhs_i) // i_len
            .arg(&self.strides[S - 2])//lhs_j_stride
            .arg(&self.strides[S - 1])//lhs_i_stride
            .arg(&rhs.strides[S - 2])//rhs_i_stride
            .arg(&rhs.strides[S - 1])//rhs_k_stride
            .arg(&out.strides[S - 2])//out_j_stride
            .arg(&out.strides[S - 1]);//out_k_stride;
        let total_s = Self::add_dim_args(&mut kernel, &self.shape[0..S - 2]);
        self.add_stride_args(&mut kernel, 0..S - 2);
        rhs.add_stride_args(&mut kernel, 0..S - 2);
        out.add_stride_args(&mut kernel, 0..S - 2);
        kernel.global_work_size(SpatialDims::Three(j, k, total_s));
        let kernel = kernel.build()?;
        unsafe {
            kernel.cmd().enq()?
        }
        Ok(out)
    }
    fn mat_cmp_mat(&self, other: &Self, mode: &'static str) -> Result<Mat<u8, S>, MatError> {
        if self.shape != other.shape {
            Err(MatError::IncompatibleShapes(self.shape.to_vec(), other.shape.to_vec()))
        } else {
            let out = unsafe { Mat::<u8, S>::empty(&self.pro_que, self.shape)? };
            let kernel = self.pro_que.pro_que.kernel_builder(format!("{}_mat_cmp_mat_{}", T::opencl_type_str(), mode))
                .arg(&self.buff)
                .arg(&other.buff)
                .arg(&out.buff)
                .global_work_size(self.size())
                .build()?;
            unsafe {
                kernel.cmd().enq()?;
            }
            Ok(out)
        }
    }
    pub fn eq_mat(&self, other: &Self) -> Result<Mat<u8, S>, MatError> {
        self.mat_cmp_mat(other, "eq")
    }
    pub fn lt_mat(&self, other: &Self) -> Result<Mat<u8, S>, MatError> {
        self.mat_cmp_mat(other, "lt")
    }
    pub fn le_mat(&self, other: &Self) -> Result<Mat<u8, S>, MatError> {
        self.mat_cmp_mat(other, "le")
    }
    pub fn gt_mat(&self, other: &Self) -> Result<Mat<u8, S>, MatError> {
        self.mat_cmp_mat(other, "gt")
    }
    pub fn ge_mat(&self, other: &Self) -> Result<Mat<u8, S>, MatError> {
        self.mat_cmp_mat(other, "ge")
    }
    pub fn ne_mat(&self, other: &Self) -> Result<Mat<u8, S>, MatError> {
        self.mat_cmp_mat(other, "ne")
    }
    fn mat_cmp_scalar(&self, scalar: T, mode: &'static str) -> Result<Mat<u8, S>, MatError> {
        let out = unsafe { Mat::<u8, S>::empty(&self.pro_que, self.shape)? };
        let kernel = self.pro_que.pro_que.kernel_builder(format!("{}_scalar_cmp_{}", T::opencl_type_str(), mode))
            .arg(&self.buff)
            .arg(&scalar)
            .arg(&out.buff)
            .global_work_size(self.size())
            .build()?;
        unsafe {
            kernel.cmd().enq()?;
        }
        Ok(out)
    }
    pub fn eq_scalar(&self, other: T) -> Result<Mat<u8, S>, MatError> {
        self.mat_cmp_scalar(other, "eq")
    }
    pub fn lt_scalar(&self, other: T) -> Result<Mat<u8, S>, MatError> {
        self.mat_cmp_scalar(other, "lt")
    }
    pub fn le_scalar(&self, other: T) -> Result<Mat<u8, S>, MatError> {
        self.mat_cmp_scalar(other, "le")
    }
    pub fn gt_scalar(&self, other: T) -> Result<Mat<u8, S>, MatError> {
        self.mat_cmp_scalar(other, "gt")
    }
    pub fn ge_scalar(&self, other: T) -> Result<Mat<u8, S>, MatError> {
        self.mat_cmp_scalar(other, "ge")
    }
    pub fn ne_scalar(&self, other: T) -> Result<Mat<u8, S>, MatError> {
        self.mat_cmp_scalar(other, "ne")
    }
    fn unary_mat(&mut self, mode: &'static str) -> Result<(), MatError> {
        let kernel = self.pro_que.pro_que.kernel_builder(format!("{}_unary_mat_{}", T::opencl_type_str(), mode))
            .arg(&self.buff)
            .global_work_size(self.size())
            .build()?;
        unsafe {
            kernel.cmd().enq()?;
        }
        Ok(())
    }
    pub fn abs(&mut self) -> Result<(), MatError> {
        self.unary_mat(if T::IS_FLOAT { "fabs" } else { "abs" })
    }
    fn scalar_to_lhs_mat(&mut self, scalar: T, mode: &'static str) -> Result<(), MatError> {
        let kernel = self.pro_que.pro_que.kernel_builder(format!("{}_scalar_to_lhs_mat_{}", T::opencl_type_str(), mode))
            .arg(&self.buff)
            .arg(&scalar)
            .global_work_size(self.size())
            .build()?;
        unsafe {
            kernel.cmd().enq()?;
        }
        Ok(())
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
    fn mat_to_lhs_mat(&mut self, other: &Self, mode: &'static str) -> Result<(), MatError> {
        if self.shape != other.shape {
            Err(MatError::IncompatibleShapes(self.shape.to_vec(), other.shape.to_vec()))
        } else {
            let mut kernel = self.pro_que.pro_que.kernel_builder(format!("{}_mat_to_lhs_mat{}_{}", T::opencl_type_str(), S, mode));
            kernel.arg(&self.buff)
                .arg(&other.buff);
            let size = Self::add_dim_args(&mut kernel, &self.shape);
            self.add_stride_args(&mut kernel, 0..S);
            other.add_stride_args(&mut kernel, 0..S);
            kernel.global_work_size(size);

            let kernel = kernel.build()?;
            unsafe {
                kernel.cmd().enq()?;
            }
            Ok(())
        }
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
        // let kernel = self.pro_que.pro_que.kernel_builder("aggregate_sum")
        //     .arg(&self.buff)
        //     .arg(&other.buff)
        //     .arg(&out.buff)
        //     .global_work_size(self.len())
        //     .build()?;
        Ok(T::zero())
    }
    pub fn item(&self) -> Result<T,MatError> {
        if self.len_buffer()==1{
            let mut tmp = [T::zero()];
            self.read(&mut tmp)?;
            Ok(tmp[0])
        }else{
            Err(MatError::NotSingletonMatrix(self.shape.to_vec()))
        }
    }
}

impl<const S: usize> Mat<f32, S> {
    pub fn sin(&mut self) -> Result<(), MatError> {
        self.unary_mat("sin")
    }
    pub fn cos(&mut self) -> Result<(), MatError> {
        self.unary_mat("cos")
    }
    pub fn tan(&mut self) -> Result<(), MatError> {
        self.unary_mat("tan")
    }
    pub fn tanh(&mut self) -> Result<(), MatError> {
        self.unary_mat("tanh")
    }
    pub fn exp(&mut self) -> Result<(), MatError> {
        self.unary_mat("exp")
    }
    pub fn exp2(&mut self) -> Result<(), MatError> {
        self.unary_mat("exp2")
    }
    pub fn exp10(&mut self) -> Result<(), MatError> {
        self.unary_mat("exp10")
    }
    pub fn log(&mut self) -> Result<(), MatError> {
        self.unary_mat("log")
    }
    pub fn log2(&mut self) -> Result<(), MatError> {
        self.unary_mat("log2")
    }
    pub fn log10(&mut self) -> Result<(), MatError> {
        self.unary_mat("log10")
    }
}

impl<T: Num, const S: usize> Display for Mat<T, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl<T: Num, const S: usize> Debug for Mat<T, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl<T: Num, const S: usize> PartialEq for Mat<T, S> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            false
        } else {
            let b = self.ne_mat(other).unwrap();
            b.sum().unwrap() == 0 // the number of different elements is zero, hence matrices are equal
        }
    }
}

impl<T: Num, const S: usize> Add<&Self> for Mat<T, S> {
    type Output = Mat<T, S>;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self.add_mat(rhs).unwrap();
        self
    }
}

impl<T: Num, const S: usize> Sub<&Self> for Mat<T, S> {
    type Output = Mat<T, S>;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.sub_mat(rhs).unwrap();
        self
    }
}

impl<T: Num, const S: usize> Div<&Self> for Mat<T, S> {
    type Output = Mat<T, S>;

    fn div(mut self, rhs: &Self) -> Self::Output {
        self.div_mat(rhs).unwrap();
        self
    }
}

impl<T: Num, const S: usize> Mul<&Self> for Mat<T, S> {
    type Output = Mat<T, S>;

    fn mul(mut self, rhs: &Self) -> Self::Output {
        self.mul_mat(rhs).unwrap();
        self
    }
}


impl<T: Num, const S: usize> Add<T> for Mat<T, S> {
    type Output = Mat<T, S>;

    fn add(mut self, rhs: T) -> Self::Output {
        self.add_scalar(rhs).unwrap();
        self
    }
}

impl<T: Num, const S: usize> Sub<T> for Mat<T, S> {
    type Output = Mat<T, S>;

    fn sub(mut self, rhs: T) -> Self::Output {
        self.sub_scalar(rhs).unwrap();
        self
    }
}

impl<T: Num, const S: usize> Div<T> for Mat<T, S> {
    type Output = Mat<T, S>;

    fn div(mut self, rhs: T) -> Self::Output {
        self.div_scalar(rhs).unwrap();
        self
    }
}

impl<T: Num, const S: usize> Mul<T> for Mat<T, S> {
    type Output = Mat<T, S>;

    fn mul(mut self, rhs: T) -> Self::Output {
        self.mul_scalar(rhs).unwrap();
        self
    }
}


impl<T: Num, const S: usize> AddAssign<&Self> for Mat<T, S> {
    fn add_assign(&mut self, rhs: &Self) {
        self.add_mat(rhs).unwrap()
    }
}

impl<T: Num, const S: usize> SubAssign<&Self> for Mat<T, S> {
    fn sub_assign(&mut self, rhs: &Self) {
        self.sub_mat(rhs).unwrap()
    }
}

impl<T: Num, const S: usize> DivAssign<&Self> for Mat<T, S> {
    fn div_assign(&mut self, rhs: &Self) {
        self.div_mat(rhs).unwrap()
    }
}

impl<T: Num, const S: usize> MulAssign<&Self> for Mat<T, S> {
    fn mul_assign(&mut self, rhs: &Self) {
        self.mul_mat(rhs).unwrap()
    }
}


impl<T: Num, const S: usize> AddAssign<T> for Mat<T, S> {
    fn add_assign(&mut self, rhs: T) {
        self.add_scalar(rhs).unwrap()
    }
}

impl<T: Num, const S: usize> SubAssign<T> for Mat<T, S> {
    fn sub_assign(&mut self, rhs: T) {
        self.sub_scalar(rhs).unwrap()
    }
}

impl<T: Num, const S: usize> DivAssign<T> for Mat<T, S> {
    fn div_assign(&mut self, rhs: T) {
        self.div_scalar(rhs).unwrap()
    }
}

impl<T: Num, const S: usize> MulAssign<T> for Mat<T, S> {
    fn mul_assign(&mut self, rhs: T) {
        self.mul_scalar(rhs).unwrap()
    }
}

pub trait IntoRange{
    fn into(self, dim_len:usize)->Range<usize>;
}
impl IntoRange for Range<usize>{
    fn into(self, _:usize) -> Range<usize> {
        self
    }
}
impl IntoRange for RangeFull{
    fn into(self, dim_len:usize) -> Range<usize> {
        0..dim_len
    }
}
impl IntoRange for RangeFrom<usize>{
    fn into(self, dim_len:usize) -> Range<usize> {
        self.start..dim_len
    }
}
impl IntoRange for RangeTo<usize>{
    fn into(self, dim_len:usize) -> Range<usize> {
        0..self.end
    }
}
impl IntoRange for RangeToInclusive<usize>{
    fn into(self, dim_len:usize) -> Range<usize> {
        0..self.end+1
    }
}
impl IntoRange for usize{
    fn into(self, dim_len:usize) -> Range<usize> {
        self..self+1
    }
}

impl<T: Num> Mat<T, 1> {
    /**Shorthand for sub view with 1 range*/
    pub fn v1<R1: IntoRange>(&self, index: R1) -> Self {
        self.view([index.into(self.shape[0])]).unwrap()
    }
}

impl<T: Num> Mat<T, 2> {
    /**Shorthand for sub view with 1 range*/
    pub fn v1<R1: IntoRange>(&self, index: R1) -> Self {
        self.v2(index, ..)
    }
    /**Shorthand for sub view with 2 ranges*/
    pub fn v2<R1: IntoRange, R2: IntoRange>(&self, r1: R1, r2: R2) -> Self {
        self.view([r1.into(self.shape[0]), r2.into(self.shape[1])]).unwrap()
    }
}

impl<T: Num> Mat<T, 3> {
    /**Shorthand for sub view with 1 range*/
    pub fn v1<R1: IntoRange>(&self, index: R1) -> Self {
        self.v2(index, ..)
    }
    /**Shorthand for sub view with 2 ranges*/
    pub fn v2<R1: IntoRange, R2: IntoRange>(&self, r1: R1, r2: R2) -> Self {
        self.v3(r1,r2,..)
    }
    /**Shorthand for sub view with 3 ranges*/
    pub fn v3<R1: IntoRange, R2: IntoRange, R3: IntoRange>(&self, r1: R1, r2: R2, r3: R3) -> Self {
        self.view([r1.into(self.shape[0]), r2.into(self.shape[1]), r3.into(self.shape[2])]).unwrap()
    }
}

impl<T: Num> Mat<T, 4> {
    /**Shorthand for sub view with 1 range*/
    pub fn v1<R1: IntoRange>(&self, index: R1) -> Self {
        self.v2(index, ..)
    }
    /**Shorthand for sub view with 2 ranges*/
    pub fn v2<R1: IntoRange, R2: IntoRange>(&self, r1: R1, r2: R2) -> Self {
        self.v3(r1,r2,..)
    }
    /**Shorthand for sub view with 3 ranges*/
    pub fn v3<R1: IntoRange, R2: IntoRange, R3: IntoRange>(&self, r1: R1, r2: R2, r3: R3) -> Self {
        self.v4(r1,r2,r3,..)
    }
    /**Shorthand for sub view with 4 ranges*/
    pub fn v4<R1: IntoRange, R2: IntoRange, R3: IntoRange, R4: IntoRange>(&self, r1: R1, r2: R2, r3: R3, r4:R4) -> Self {
        self.view([r1.into(self.shape[0]), r2.into(self.shape[1]), r3.into(self.shape[2]), r4.into(self.shape[3])]).unwrap()
    }
}

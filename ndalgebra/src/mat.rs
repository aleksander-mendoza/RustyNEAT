use ocl::{ProQue, SpatialDims, flags, Platform, Device, Buffer, Error, Queue};
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul};
use std::fmt::{Display, Formatter, Debug};
use crate::kernel::{LinAlgProgram, MAX_MAT_DIMS};
use crate::num::Num;



pub struct Mat<T: Num, const S: usize> {
    pro_que: LinAlgProgram,
    buff: Buffer<T>,
    shape: [usize; S],
    strides: [usize; S],
}

pub enum MatError {
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
        Display::fmt(self,f)
    }
}
impl Display for MatError {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
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
            MatError::NonsingularDimension(shape, idx) => write!(fmt, "Shape {} has length {} at index {} but expected it to be of length 1",shape.as_shape(),shape[*idx], idx),
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
impl <'a,T:Num> Shape<'a, T>{
    fn size(&self)->T{
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

impl<T: Num> Mat<T, 0> {
    pub fn array0(lin_alg: &LinAlgProgram, arr: T) -> Result<Mat<T, 0>, MatError> {
        Self::from_slice(lin_alg, &[arr], [])
    }
}

impl<T: Num> Mat<T, 1> {
    pub fn array1<const X: usize>(lin_alg: &LinAlgProgram, arr: [T; X]) -> Result<Mat<T, 1>, MatError> {
        Self::from_slice(lin_alg, &arr, [X])
    }
}

impl<T: Num> Mat<T, 2> {
    pub fn array2<const X: usize, const Y: usize>(lin_alg: &LinAlgProgram, arr: [[T; X]; Y]) -> Result<Mat<T, 2>, MatError> where [T; { X * Y }]: Sized {
        let a: &[T] = unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const T,X*Y) };
        Self::from_slice(lin_alg, a, [Y, X])
    }
}

impl<T: Num> Mat<T, 3> {
    pub fn array3<const X: usize, const Y: usize, const Z: usize>(lin_alg: &LinAlgProgram, arr: [[[T; X]; Y]; Z]) -> Result<Mat<T, 3>, MatError> where [T; { X * Y * Z }]: Sized {
        let a: &[T] = unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const T,X*Y*Z) };
        Self::from_slice(lin_alg, a, [Z, Y, X])
    }
}

impl<T: Num, const S: usize> Mat<T, S> {
    fn new_with_strides(lin_alg: &LinAlgProgram, buff: Buffer<T>, strides: [usize; S], shape: [usize; S]) -> Result<Self, MatError> {
        if shape.len() > 20 {
            Err(MatError::DimensionalityLimitExceeded(shape.len()))
        } else {
            assert_eq!(strides.len(), shape.len());
            assert_eq!(buff.len(), if strides.is_empty() { 1 } else { strides[0] * shape[0] });
            let pro_que = lin_alg.clone();
            Ok(Self { pro_que, buff, strides, shape})
        }
    }
    pub fn from_buffer(lin_alg: &LinAlgProgram, buff: Buffer<T>, shape: [usize; S]) -> Result<Self, MatError> {
        let strides = Self::strides_for_shape(&shape);
        let len = if strides.is_empty() { 1 } else { strides[0] * shape[0] };
        if buff.len() != len {
            Err(MatError::BufferLengthAndShapeMismatch(buff.len(), shape.to_vec()))
        } else {
            Self::new_with_strides(lin_alg, buff, strides, shape)
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
                .and_then(|buff| Self::new_with_strides(lin_alg, buff, strides, shape))
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
        let len = if strides.is_empty() { 1 } else { strides[0] * shape[0] };
        lin_alg.pro_que.buffer_builder::<T>()
            .flags(flags::MEM_READ_WRITE)
            .len(len)
            .build().map_err(|e| MatError::OpenCLError(e))
            .and_then(|buff| Self::new_with_strides(lin_alg, buff, strides, shape))
    }
    pub fn filled(lin_alg: &LinAlgProgram, shape: [usize; S], fill_val: T) -> Result<Self, MatError> {
        let strides = Self::strides_for_shape(&shape);
        let len = if strides.is_empty() { 1 } else { strides[0] * shape[0] };
        lin_alg.pro_que.buffer_builder::<T>()
            .flags(flags::MEM_READ_WRITE)
            .len(len)
            .fill_val(fill_val)
            .build().map_err(MatError::from)
            .and_then(|buff| Self::new_with_strides(lin_alg, buff, strides, shape))
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
    pub fn ndim(&self) -> usize {
        S
    }
    pub fn reshape<const S2: usize>(&self, shape: [usize; S2]) -> Result<Mat<T, S2>, MatError> {
        let strides = Mat::<T, S2>::strides_for_shape(&shape);
        if strides[0] * shape[0] != self.size() {
            return Err(MatError::IncompatibleShapes(self.shape.to_vec(), shape.to_vec()));
        }
        let buff = self.buff.clone(); //this is reference-counted
        Mat::<T, S2>::new_with_strides(&self.pro_que, buff, strides, shape)
    }
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
    pub fn squeeze(&self, idx: usize) -> Result<Mat<T, { S - 1 }>, MatError> where [usize; S - 1]: Sized {
        if idx >= self.ndim() {
            Err(MatError::DimensionOutOfBounds(self.shape.to_vec(), idx))
        } else if self.shape[idx] != 1 {
            Err(MatError::NonsingularDimension(self.shape.to_vec(), idx))
        }else{
            let mut new_shape = [0; { S - 1 }];
            new_shape[0..idx].copy_from_slice(&self.shape[0..idx]);
            new_shape[idx..].copy_from_slice(&self.shape[idx+1..]);
            self.reshape(new_shape)
        }
    }

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
    /**Length of the underlying buffer (if this tensor uses only a certain sub-buffer,
    this method will return the total length of the parent buffer.)*/
    pub fn len_buffer(&self) -> usize {
        self.buff.len()
    }
    /**Length of the underlying sub-buffer. Every tensor has some underlying buffer.
     Some tensors, however, only use a fragment of that large buffer. */
    pub fn len_sub_buffer(&self) -> usize {
        self.buff.len()-self.buff.offset().unwrap_or(0)
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
    pub fn to_vec(&self) -> Result<Vec<T>, Error> {
        let mut v = Vec::with_capacity(self.size());
        unsafe { v.set_len(v.capacity()) };
        self.read(v.as_mut_slice())?;
        Ok(v)
    }
    /**Reads entire tensor into the provided slice*/
    pub fn read(&self, dst: &mut [T]) -> Result<(), Error> {
        let size = self.size();
        if dst.len() != size{
            Err(Error::from(format!("Expected buffer length {} but got {}", size, dst.len())))
        } else {
            unsafe {
                self.buff.cmd().read(dst).enq()
            }
        }
    }
    pub fn contiguous(&self)->bool{
        self.size() == self.len_sub_buffer()
    }
    /**alias for mm()*/
    pub fn matmul(&mut self, rhs: &Self) -> Result<Self, MatError> {
        self.mm(rhs)
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
        assert_eq!(out_shape[S-2],j);
        assert_eq!(out_shape[S-1],k);
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
        let mut total_s = 1;
        for (s, dim_s) in self.shape[0..S - 2].iter().enumerate() {
            kernel.arg(dim_s).arg(&self.strides[s]).arg(&rhs.strides[s]).arg(&out.strides[s]);
            total_s *= dim_s;
        }
        kernel.global_work_size(SpatialDims::Three(j, k, total_s));
        let kernel = kernel.build()?;
        unsafe {
            kernel.cmd().enq()?
        }
        Ok(out)
    }
    fn cmp_scalar(&self, scalar: T, mode: &'static str) -> Result<Mat<u8, S>, MatError> {
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
    fn cmp_mat(&self, other: &Self, mode: &'static str) -> Result<Mat<u8, S>, MatError> {
        if self.shape != other.shape {
            Err(MatError::IncompatibleShapes(self.shape.to_vec(), other.shape.to_vec()))
        } else {
            let out = unsafe { Mat::<u8, S>::empty(&self.pro_que, self.shape)? };
            let kernel = self.pro_que.pro_que.kernel_builder(format!("{}_cmp_{}",T::opencl_type_str(), mode))
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
        self.cmp_mat(other, "eq")
    }
    pub fn lt_mat(&self, other: &Self) -> Result<Mat<u8, S>, MatError> {
        self.cmp_mat(other, "lt")
    }
    pub fn le_mat(&self, other: &Self) -> Result<Mat<u8, S>, MatError> {
        self.cmp_mat(other, "le")
    }
    pub fn gt_mat(&self, other: &Self) -> Result<Mat<u8, S>, MatError> {
        self.cmp_mat(other, "gt")
    }
    pub fn ge_mat(&self, other: &Self) -> Result<Mat<u8, S>, MatError> {
        self.cmp_mat(other, "ge")
    }
    pub fn ne_mat(&self, other: &Self) -> Result<Mat<u8, S>, MatError> {
        self.cmp_mat(other, "ne")
    }
    pub fn eq_scalar(&self, other: T) -> Result<Mat<u8, S>, MatError> {
        self.cmp_scalar(other, "eq")
    }
    pub fn lt_scalar(&self, other: T) -> Result<Mat<u8, S>, MatError> {
        self.cmp_scalar(other, "lt")
    }
    pub fn le_scalar(&self, other: T) -> Result<Mat<u8, S>, MatError> {
        self.cmp_scalar(other, "le")
    }
    pub fn gt_scalar(&self, other: T) -> Result<Mat<u8, S>, MatError> {
        self.cmp_scalar(other, "gt")
    }
    pub fn ge_scalar(&self, other: T) -> Result<Mat<u8, S>, MatError> {
        self.cmp_scalar(other, "ge")
    }
    pub fn ne_scalar(&self, other: T) -> Result<Mat<u8, S>, MatError> {
        self.cmp_scalar(other, "ne")
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
    /**scalar multiplication*/
    pub fn mul_in_place(&mut self, scalar: T) {}
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
            b.sum().unwrap()==0 // the number of different elements is zero, hence matrices are equal
        }
    }
}


// impl<T: Num, const S: usize> Index<[usize; S]> for Mat<T, S> {
//     type Output = T;
//
//     fn index(&self, index: [usize; S]) -> &Self::Output {
//         self.get(&index).unwrap()
//     }
// }



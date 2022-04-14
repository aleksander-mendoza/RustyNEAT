use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut, Mul, Add, Range, Sub, Div, AddAssign, DivAssign, SubAssign, MulAssign, RangeFull, RangeFrom, RangeTo, RangeToInclusive, RangeInclusive, Neg};
use std::fmt::{Display, Formatter, Debug};
use crate::num::Num;
use crate::cpu::shape::AsShape;
use crate::cpu::error::MatError;
use crate::cpu::buffer::{Buffer, empty};
use crate::cpu::layout::Layout;



#[derive(Clone)]
pub struct Mat<T: Num, B: Buffer<T>> {
    buff: B,
    layout: Layout
}

impl<T: Num> Mat<T, &[T]> {
    /**Tensor with a single dimension of length 1 and a*/
    pub fn array0(arr: T) -> Result<Self, MatError> {
        Self::new(&[arr], &[1])
    }

    pub fn array1<const X: usize>(arr: [T; X]) -> Result<Self, MatError> {
        Self::new(&arr, &[X])
    }

    pub fn array2<const X: usize, const Y: usize>(arr: [[T; X]; Y]) -> Result<Self, MatError> where [T; { X * Y }]: Sized {
        let a: &[T] = unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const T, X * Y) };
        Self::new(a, &[Y, X])
    }

    pub fn array3<const X: usize, const Y: usize, const Z: usize>(arr: [[[T; X]; Y]; Z]) -> Result<Self, MatError> where [T; { X * Y * Z }]: Sized {
        let a: &[T] = unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const T, X * Y * Z) };
        Self::new(a, &[Z, Y, X])
    }
}
impl<T: Num, B: Buffer<T>> Mat<T, B> {
    pub fn null() -> Result<Self, MatError> {
        unsafe { Self::empty(&[]) }
    }
    pub fn empty(shape: impl Into<Layout>) -> Result<Self, MatError> {
        Self::new(empty(len),shape)
    }
    pub fn new(buff: B, layout:impl Into<Layout>) -> Result<Self, MatError> {
        let layout = layout.into();
        if layout.ndim() > 20 {
            Err(MatError::DimensionalityLimitExceeded(layout.ndim()))
        } else if buff.len() != layout.size() {
            Err(MatError::BufferLengthAndShapeMismatch(arr.len(), shape.into_vec()))
        } else {
            Ok(Self { buff, layout })
        }
    }
    pub fn from_slice_infer_wildcard(arr: &[T], shape: &[isize]) -> Result<Self, MatError> {
        Self::from_slice_boxed(arr, Self::infer_wildcard(&[arr.len()], shape)?)
    }
    pub fn buffer(&self) -> &B {
        &self.buff
    }
    pub fn buffer_mut(&mut self) -> &mut B {
        &mut self.buff
    }
    pub unsafe fn empty_like<D: Num,B:Buffer<T>>(other: &Self) -> Result<Mat<D,B>, MatError> {
        Self::empty(other.layout.clone())
    }
    pub fn full_like<D: Num,B:Buffer<T>>(other: &Self, fill_val: D) -> Result<Mat<D,B>, MatError> {
        Self::full(other.layout.clone(), fill_val)
    }
    pub fn full(shape: impl Into<Layout>, fill_val: T) -> Result<Self, MatError> {
        Self::empty(shape.into()).map(|mut slf|{
            slf.buff.fill(fill_val);
            slf
        })
    }
    pub fn ones(shape: &[usize]) -> Result<Self, MatError> {
        Self::full(shape, T::one())
    }
    pub fn zeros(shape: &[usize]) -> Result<Self, MatError> {
        Self::full(shape, T::zero())
    }
    pub fn offset(&self, index: &[usize]) -> Result<usize, MatError> {
        self.layout.offset(index)
    }
    pub fn transpose(&mut self, dim0: usize, dim1: usize) -> Result<&mut Self, MatError> {
        self.layout.transpose(dim0,dim1);
        Ok(self)
    }
    pub fn view(&self, ranges: &[Range<usize>]) -> Result<Mat<T,&[T]>, MatError> {
        let layout = self.layout.view(ranges)?;
        let offset = ranges.iter().zip(layout.strides().iter()).map(|(r, s)| r.start * s).sum::<usize>();
        let range = layout.end_offset();
        let sub = &self.buff.as_slice()[offset..offset+range];
        Self::new(sub,layout)
    }
    /**Number of dimensions. Works the same way as in numpy.*/
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }
    /**Changes dimensions of tensor. The total size (all dimension length multiplied together)
    must remain the same. If tensor is contiguous it will return a different view to the same buffer.
    If tensor is not contiguous a copy will be necessary. Works the same way as in numpy.*/
    pub fn reshape(&self, shape: impl Into<Layout>) -> Result<Mat<T,Box<[T]>>, MatError> {
        let shape = shape.into();
        if shape.size() != self.size() {
            return Err(MatError::IncompatibleShapes(self.shape().to_vec(), shape.to_vec()));
        }
        let buff = if self.contiguous() {
            self.buff.clone() //this is reference-counted
        } else {
            self.copy()?.buff
        };
        Self::new(buff, true, strides, shape)
    }
    /**Inserts an additional dimension of length 1 at a specified index.  Works the same way as in numpy.*/
    pub fn unsqueeze(&mut self, idx: usize) -> Result<&mut Self, MatError> {
        self.layout.unsqueeze(idx)?;
        Ok(self)
    }
    /**Collapses a dimension at a specified index, provided that the length of that dimension is 1.  Works the same way as in numpy.*/
    pub fn squeeze(&mut self, idx: usize) -> Result<&mut Self, MatError> {
        self.layout.squeeze(idx)?;
        Ok(self)
    }
    /**Same as reshape() but you can additionally use -1 as a wildcard. Works the same way as in numpy.*/
    pub fn reshape_infer_wildcard(&self, shape: &[isize]) -> Result<Self, MatError> {
        self.reshape_boxed(Self::infer_wildcard(self.shape(), shape)?)
    }
    /**Stride corresponding to each dimension of tensor*/
    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }
    /**Shape of tensor is the array containing lengths of each dimension*/
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }
    /**Length of first dimension*/
    pub fn len(&self) -> usize {
        self.layout.len()
    }
    /**Total size obtained by multiplying all dimensions together*/
    pub fn size(&self) -> usize {
        self.layout.size()
    }
    /**Reads a single value from buffer. Notice that this memory could lie on GPU.
    This method might have large overhead. It's not recommended to use it in a loop to iterate over buffer.
    Use subviews instead.*/
    pub fn get(&self, index: &[usize]) -> Result<T, MatError> {
        self.offset_into_buffer(index).map(|i|self.buff.get(i))
    }
    pub fn contiguous(&self) -> bool {
        self.layout.contiguous()
    }
    /**This function performs a deep clone of the tensor together with its
    underlying buffer. If the tensor was not contiguous, its copy
     will become contiguous. Works the same way as in numpy. */
    pub fn copy(&self) -> Result<Self, MatError> {
        let mut out = unsafe { Self::empty(&self.buff, self.shape()) }?;
        out.copy_from(self)?;
        Ok(out)
    }
    /**alias for mm()*/
    pub fn matmul(&self, rhs: &Self) -> Result<Self, MatError> {
        self.mm(rhs)
    }
    pub fn fill(&mut self, scalar: T) -> Result<(), MatError> {
        self.scalar_to_lhs_mat(scalar, "fill")
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
        let out = unsafe { Self::empty_boxed(&self.out_shape)? };

        let kernel = self.lin_alg.kernel_builder(format!("{}_mm{}", T::OPENCL_TYPE_STR, self.ndim()))?
            .add_buff(self.buffer().unwrap())?
            .add_buff(rhs.buffer().unwrap())?
            .add_buff(out.buffer().unwrap())?
            .add_num(lhs_i)? // i_len
            .add_num(self.strides[self.ndim() - 2])?//lhs_j_stride
            .add_num(self.strides[self.ndim() - 1])?//lhs_i_stride
            .add_num(rhs.strides[self.ndim() - 2])?//rhs_i_stride
            .add_num(rhs.strides[self.ndim() - 1])?//rhs_k_stride
            .add_num(out.strides[self.ndim() - 2])?//out_j_stride
            .add_num(out.strides[self.ndim() - 1])?;//out_k_stride;
        let (kernel, total_s) = Self::add_dim_args(kernel, &self.shape[0..self.ndim() - 2])?;
        let kernel = self.add_stride_args(kernel, 0..self.ndim() - 2)?;
        let kernel = rhs.add_stride_args(kernel, 0..self.ndim() - 2)?;
        let kernel = out.add_stride_args(kernel, 0..self.ndim() - 2)?;
        kernel.enq(self.queue(), &[j, k, total_s]);
        Ok(out)
    }
    fn zip<D:Num,V:Buffer<D>,E:Num,W:Buffer<D>>(&self, other: &Mat<D,V>, output:&mut Mat<E,W>, f: fn(T, D, &mut E)) -> Result<(), MatError> {
        if self.layout != other.layout {
            Err(MatError::IncompatibleShapes(self.layout.to_vec(), other.layout.to_vec()))
        } else if let Some(buff) = &self.buff {
            let out = unsafe { Mat::<u8>::empty(&self. & self.shape)? };
            self.lin_alg.kernel_builder(format!("{}_mat_cmp_mat_{}", T::OPENCL_TYPE_STR, mode))?
                .add_buff(buff)?
                .add_buff(other.buffer().unwrap())?
                .add_buff(out.buffer().unwrap())?
                .enq(self.queue(), &[self.size()])?;
            Ok(out)
        } else {
            Mat::<u8>::null(&self.lin_alg)
        }
    }
    pub fn eq_mat(&self, other: &Self) -> Result<Mat<bool>, MatError> {
        self.mat_cmp_mat(other, "eq")
    }
    pub fn lt_mat(&self, other: &Self) -> Result<Mat<bool>, MatError> {
        self.mat_cmp_mat(other, "lt")
    }
    pub fn le_mat(&self, other: &Self) -> Result<Mat<bool>, MatError> {
        self.mat_cmp_mat(other, "le")
    }
    pub fn gt_mat(&self, other: &Self) -> Result<Mat<bool>, MatError> {
        self.mat_cmp_mat(other, "gt")
    }
    pub fn ge_mat(&self, other: &Self) -> Result<Mat<bool>, MatError> {
        self.mat_cmp_mat(other, "ge")
    }
    pub fn ne_mat(&self, other: &Self) -> Result<Mat<bool>, MatError> {
        self.mat_cmp_mat(other, "ne")
    }
    fn mat_cmp_scalar(&self, scalar: T, mode: &'static str) -> Result<Mat<bool>, MatError> {
        if let Some(buff) = self.buffer() {
            let out = unsafe { Mat::<u8>::empty(&self. self.shape())? };
            self.lin_alg.kernel_builder(format!("{}_mat_cmp_scalar_{}", T::OPENCL_TYPE_STR, mode))?
                .add_buff(buff)?
                .add_num(scalar)?
                .add_buff(out.buffer().unwrap())?
                .enq(self.queue(), &[self.size()])?;
            Ok(out)
        } else {
            Mat::<u8>::null(&self.lin_alg)
        }
    }
    pub fn eq_scalar(&self, other: T) -> Result<Mat<bool>, MatError> {
        self.mat_cmp_scalar(other, "eq")
    }
    pub fn lt_scalar(&self, other: T) -> Result<Mat<bool>, MatError> {
        self.mat_cmp_scalar(other, "lt")
    }
    pub fn le_scalar(&self, other: T) -> Result<Mat<bool>, MatError> {
        self.mat_cmp_scalar(other, "le")
    }
    pub fn gt_scalar(&self, other: T) -> Result<Mat<bool>, MatError> {
        self.mat_cmp_scalar(other, "gt")
    }
    pub fn ge_scalar(&self, other: T) -> Result<Mat<bool>, MatError> {
        self.mat_cmp_scalar(other, "ge")
    }
    pub fn ne_scalar(&self, other: T) -> Result<Mat<bool>, MatError> {
        self.mat_cmp_scalar(other, "ne")
    }
    /**Creates a copy of matrix and converts all its elements to a different type.
    Is equivalent to copy() if both source and target types are the same*/
    pub fn cast<D: Num>(&self) -> Result<Mat<D>, MatError> {
        let mut out = unsafe { Self::empty_like::<D>(self)? };
        out.copy_from(self)?;
        Ok(out)
    }
    pub fn abs(&mut self) -> Result<Self, MatError> {
        let mut out = self.copy()?;
        out.abs_in_place()?;
        Ok(out)
    }
    pub fn abs_in_place(&mut self) -> Result<(), MatError> {
        self.unary_to_lhs_mat(if T::IS_FLOAT { "fabs" } else { "abs" })
    }
    fn scalar_to_lhs_mat(&mut self, scalar: T, mode: &'static str) -> Result<(), MatError> {
        if let Some(buff) = self.buffer() {
            let fn_name = format!("scalar_to_lhs_mat_{dtype}_{dims}_{name}", dtype = T::OPENCL_TYPE_STR, dims = self.ndim(), name = mode);
            let mut kernel = self.lin_alg.kernel_builder(fn_name)?
                .add_buff(buff)?
                .add_num(scalar)?;
            let (kernel, size) = Self::add_dim_args(kernel, &self.shape)?;
            let kernel = self.add_stride_args(kernel, 0..self.ndim())?;
            kernel.enq(self.queue(), &[size]).map_err(MatError::from)
        } else {
            Ok(())
        }
    }

    pub fn add_scalar(&mut self, scalar: T) -> Result<(), MatError> {
        self.scalar_to_lhs_mat(scalar, "add")
    }
    pub fn sub_scalar(&mut self, scalar: T) -> Result<(), MatError> {
        self.scalar_to_lhs_mat(scalar, "sub")
    }
    /**Instead of performing (self - scalar), it performs (scalar - self)*/
    pub fn swapped_sub_scalar(&mut self, scalar: T) -> Result<(), MatError> {
        self.scalar_to_lhs_mat(scalar, "swapped_sub")
    }
    /**Instead of performing (self / scalar), it performs (scalar / self)*/
    pub fn swapped_div_scalar(&mut self, scalar: T) -> Result<(), MatError> {
        self.scalar_to_lhs_mat(scalar, "swapped_div")
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
            let mut kernel = self.lin_alg.kernel_builder(fn_name)?
                .add_buff(buff)?
                .add_buff(other.buffer().unwrap())?;
            let (kernel, size) = Self::add_dim_args(kernel, &self.shape)?;
            let kernel = self.add_stride_args(kernel, 0..self.ndim())?;
            let kernel = other.add_stride_args(kernel, 0..self.ndim())?;
            kernel.enq(self.queue(), &[size]).map_err(MatError::from)
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
    pub fn copy_from<D: Num>(&mut self, other: &Mat<D>) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "cast")
    }
    pub fn add_mat(&mut self, other: &Self) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "add")
    }
    pub fn div_mat(&mut self, other: &Self) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "div")
    }
    /**performs in-place division but instead of saving output in lhs matrix, it does so in rhs*/
    pub fn swapped_div_mat(&self, other: &mut Self) -> Result<(), MatError> {
        other.mat_to_lhs_mat(self, "swapped_div")
    }
    pub fn mul_mat(&mut self, other: &Self) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "hadamard")
    }
    pub fn sub_mat(&mut self, other: &Self) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "sub")
    }
    /**performs in-place subtraction but instead of saving output in lhs matrix, it does so in rhs*/
    pub fn swapped_sub_mat(&self, other: &mut Self) -> Result<(), MatError> {
        other.mat_to_lhs_mat(self, "swapped_sub")
    }
    pub fn min_mat(&mut self, other: &Self) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "min")
    }
    pub fn max_mat(&mut self, other: &Self) -> Result<(), MatError> {
        self.mat_to_lhs_mat(other, "max")
    }
    pub fn sum(&self) -> Result<T, MatError> {
    }
    pub fn item(&self) -> Result<T, MatError> {
        if self.size() == 1 {
            self.buffer().get(0)
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

impl<T: Num> Display for Self {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.ndim() == 0 {
            write!(f, "[]")
        } else {
            let buff = self.to_vec_non_contiguous().map_err(|e| std::fmt::Error)?;
            fn recursive_print<T: Num>(me: &Self, idx: &mut [usize], level: usize, buff: &[T], f: &mut Formatter<'_>) -> std::fmt::Result {
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

impl<T: Num> Debug for Self {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl<T: Num> PartialEq for Self {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            false
        } else {
            let b = self.ne_mat(other).unwrap();
            b.sum().unwrap() == 0 // the number of different elements is zero, hence matrices are equal
        }
    }
}

impl<T: Num> Add<&Self> for Self {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self.add_mat(rhs).unwrap();
        self
    }
}

impl<T: Num> Sub<&Self> for Self {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.sub_mat(rhs).unwrap();
        self
    }
}

impl<T: Num> Div<&Self> for Self {
    type Output = Self;

    fn div(mut self, rhs: &Self) -> Self::Output {
        self.div_mat(rhs).unwrap();
        self
    }
}

impl<T: Num> Mul<&Self> for Self {
    type Output = Self;

    fn mul(mut self, rhs: &Self) -> Self::Output {
        self.mul_mat(rhs).unwrap();
        self
    }
}


impl<T: Num> Add<T> for Self {
    type Output = Self;

    fn add(mut self, rhs: T) -> Self::Output {
        self.add_scalar(rhs).unwrap();
        self
    }
}

impl<T: Num> Sub<T> for Self {
    type Output = Self;

    fn sub(mut self, rhs: T) -> Self::Output {
        self.sub_scalar(rhs).unwrap();
        self
    }
}

impl<T: Num> Div<T> for Self {
    type Output = Self;

    fn div(mut self, rhs: T) -> Self::Output {
        self.div_scalar(rhs).unwrap();
        self
    }
}

impl<T: Num> Mul<T> for Self {
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self::Output {
        self.mul_scalar(rhs).unwrap();
        self
    }
}

impl<T: Num> Neg for Self {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.swapped_sub_scalar(T::zero()).unwrap();
        self
    }
}

impl<T: Num> AddAssign<&Self> for Self {
    fn add_assign(&mut self, rhs: &Self) {
        self.add_mat(rhs).unwrap()
    }
}

impl<T: Num> SubAssign<&Self> for Self {
    fn sub_assign(&mut self, rhs: &Self) {
        self.sub_mat(rhs).unwrap()
    }
}

impl<T: Num> DivAssign<&Self> for Self {
    fn div_assign(&mut self, rhs: &Self) {
        self.div_mat(rhs).unwrap()
    }
}

impl<T: Num> MulAssign<&Self> for Self {
    fn mul_assign(&mut self, rhs: &Self) {
        self.mul_mat(rhs).unwrap()
    }
}


impl<T: Num> AddAssign<T> for Self {
    fn add_assign(&mut self, rhs: T) {
        self.add_scalar(rhs).unwrap()
    }
}

impl<T: Num> SubAssign<T> for Self {
    fn sub_assign(&mut self, rhs: T) {
        self.sub_scalar(rhs).unwrap()
    }
}

impl<T: Num> DivAssign<T> for Self {
    fn div_assign(&mut self, rhs: T) {
        self.div_scalar(rhs).unwrap()
    }
}

impl<T: Num> MulAssign<T> for Self {
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

impl<T: Num> Self {
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

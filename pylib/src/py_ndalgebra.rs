use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol, PyTypeInfo, PyClass, PyMappingProtocol, FromPyPointer, PySequenceProtocol, PyDowncastError, AsPyPointer, PyIterProtocol, ffi, PyNativeType, PyNumberProtocol};
use pyo3::PyResult;
use ndalgebra::mat::{Mat, MatError, AsShape};
use ndalgebra::num::Num;
use crate::py_ocl::NeatContext;
use crate::{ocl_err_to_py_ex, slice_box};
use std::fmt::{Display, Formatter};
use pyo3::types::{PyType, PyList, PyTuple, PyInt, PyFloat, PyBool, PySlice, PyLong};
use pyo3::exceptions::PyValueError;
use pyo3::basic::CompareOp;
use std::ops::Range;
use numpy::{PyArray, PY_ARRAY_API, npyffi, Element, PyReadonlyArrayDyn, DataType, PyArrayDyn, ToNpyDims};
use numpy::npyffi::{PyArray_Dims, NPY_TYPES, NPY_ARRAY_WRITEABLE};
use ndalgebra::kernel::LinAlgProgram;

#[pyclass]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct DType {
    e: DTypeEnum,
}


pub const U8: DType = DType { e: DTypeEnum::u8 };
pub const U16: DType = DType { e: DTypeEnum::u16 };
pub const U32: DType = DType { e: DTypeEnum::u32 };
pub const U64: DType = DType { e: DTypeEnum::u64 };
pub const I8: DType = DType { e: DTypeEnum::i8 };
pub const I16: DType = DType { e: DTypeEnum::i16 };
pub const I32: DType = DType { e: DTypeEnum::i32 };
pub const I64: DType = DType { e: DTypeEnum::i64 };
pub const F32: DType = DType { e: DTypeEnum::f32 };

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum DTypeEnum {
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    f32,
}

impl DTypeEnum {
    pub fn into_ctype(self) -> NPY_TYPES {
        match self {
            Self::i8 => NPY_TYPES::NPY_BYTE,
            Self::i16 => NPY_TYPES::NPY_SHORT,
            Self::i32 => NPY_TYPES::NPY_INT,
            Self::i64 => NPY_TYPES::NPY_LONGLONG,
            Self::u8 => NPY_TYPES::NPY_UBYTE,
            Self::u16 => NPY_TYPES::NPY_USHORT,
            Self::u32 => NPY_TYPES::NPY_UINT,
            Self::u64 => NPY_TYPES::NPY_ULONGLONG,
            Self::f32 => NPY_TYPES::NPY_FLOAT,
        }
    }
}

impl Display for DTypeEnum {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

pub trait PyNum: Num + ToDtype + ToPyObject {
    fn extract_from(val: &PyAny) -> PyResult<Self>;
}

impl PyNum for f32 {
    fn extract_from(val: &PyAny) -> PyResult<Self> {
        let v = unsafe { ffi::PyFloat_AsDouble(val.as_ptr()) };

        if v == -1.0 && PyErr::occurred(val.py()) {
            Err(PyErr::fetch(val.py()))
        } else {
            Ok(v as f32)
        }
    }
}

impl PyNum for i64 {
    fn extract_from(val: &PyAny) -> PyResult<Self> {
        let v = unsafe { ffi::PyLong_AsLongLong(val.as_ptr()) };

        if v == -1 && PyErr::occurred(val.py()) {
            Err(PyErr::fetch(val.py()))
        } else {
            Ok(v)
        }
    }
}

impl PyNum for i32 {
    fn extract_from(val: &PyAny) -> PyResult<Self> {
        i64::extract_from(val).map(|v| v as i32)
    }
}

impl PyNum for i16 {
    fn extract_from(val: &PyAny) -> PyResult<Self> {
        i64::extract_from(val).map(|v| v as i16)
    }
}

impl PyNum for i8 {
    fn extract_from(val: &PyAny) -> PyResult<Self> {
        i64::extract_from(val).map(|v| v as i8)
    }
}

impl PyNum for u64 {
    fn extract_from(val: &PyAny) -> PyResult<Self> {
        let v = unsafe { ffi::PyLong_AsUnsignedLongLong(val.as_ptr()) };

        if v == u64::MAX && PyErr::occurred(val.py()) {
            Err(PyErr::fetch(val.py()))
        } else {
            Ok(v)
        }
    }
}

impl PyNum for usize {
    fn extract_from(val: &PyAny) -> PyResult<Self> {
        u64::extract_from(val).map(|v| v as usize)
    }
}

impl PyNum for u32 {
    fn extract_from(val: &PyAny) -> PyResult<Self> {
        u64::extract_from(val).map(|v| v as u32)
    }
}

impl PyNum for u16 {
    fn extract_from(val: &PyAny) -> PyResult<Self> {
        u64::extract_from(val).map(|v| v as u16)
    }
}

impl PyNum for u8 {
    fn extract_from(val: &PyAny) -> PyResult<Self> {
        u64::extract_from(val).map(|v| v as u8)
    }
}

pub trait DynMatTrait: Display {
    fn len(&self) -> usize;
    fn dtype(&self) -> DTypeEnum;
    fn lin_alg(&self) -> &LinAlgProgram;
    fn dtype_size(&self) -> usize;
    fn untyped_mat(&self) -> &Mat<u8>;
    fn untyped_mat_mut(&mut self) -> &mut Mat<u8>;
    fn ndim(&self) -> usize {
        self.untyped_mat().ndim()
    }
    fn shape(&self) -> &[usize] {
        self.untyped_mat().shape()
    }
    fn strides(&self) -> &[usize] {
        self.untyped_mat().strides()
    }
    fn astype(&self, dtype:DTypeEnum) -> Result<DynMat, MatError>;
    fn unsqueeze(&self, idx: usize) -> Result<DynMat, MatError>;
    fn squeeze(&self, idx: usize) -> Result<DynMat, MatError>;
    fn reshape(&self, shape: Box<[usize]>) -> Result<DynMat, MatError>;
    fn reshape_infer_wildcard(&self, shape: &[isize]) -> Result<DynMat, MatError>;
    fn view(&self, view: &[Range<usize>]) -> Result<DynMat, MatError>;
    fn matmul(&self, tensor: &PyAny) -> PyResult<DynMat>;
    fn add(&self, scalar_or_tensor: &PyAny) -> PyResult<DynMat>;
    fn sub(&self, scalar_or_tensor: &PyAny) -> PyResult<DynMat>;
    fn swapped_sub(&self, scalar_or_tensor: &PyAny) -> PyResult<DynMat>;
    fn mul(&self, scalar_or_tensor: &PyAny) -> PyResult<DynMat>;
    fn div(&self, scalar_or_tensor: &PyAny) -> PyResult<DynMat>;
    fn swapped_div(&self, scalar_or_tensor: &PyAny) -> PyResult<DynMat>;
    fn min(&self, scalar_or_tensor: &PyAny) -> PyResult<DynMat>;
    fn max(&self, scalar_or_tensor: &PyAny) -> PyResult<DynMat>;
    fn add_(&mut self, scalar_or_tensor: &PyAny) -> PyResult<()>;
    fn sub_(&mut self, scalar_or_tensor: &PyAny) -> PyResult<()>;
    fn swapped_sub_(&mut self, scalar_or_tensor: &PyAny) -> PyResult<()>;
    fn mul_(&mut self, scalar_or_tensor: &PyAny) -> PyResult<()>;
    fn div_(&mut self, scalar_or_tensor: &PyAny) -> PyResult<()>;
    fn swapped_div_(&mut self, scalar_or_tensor: &PyAny) -> PyResult<()>;
    fn min_(&mut self, scalar_or_tensor: &PyAny) -> PyResult<()>;
    fn max_(&mut self, scalar_or_tensor: &PyAny) -> PyResult<()>;
    fn neg_(&mut self) -> Result<(), MatError>;
    fn neg(&self) -> Result<DynMat, MatError>;
    fn exp(&self) -> Result<DynMat, MatError>;
    fn exp2(&self) -> Result<DynMat, MatError>;
    fn exp10(&self) -> Result<DynMat, MatError>;
    fn sin(&self) -> Result<DynMat, MatError>;
    fn cos(&self) -> Result<DynMat, MatError>;
    fn tan(&self) -> Result<DynMat, MatError>;
    fn tanh(&self) -> Result<DynMat, MatError>;
    fn log(&self) -> Result<DynMat, MatError>;
    fn log2(&self) -> Result<DynMat, MatError>;
    fn log10(&self) -> Result<DynMat, MatError>;
    fn fill(&mut self, scalar: &PyAny) -> PyResult<()>;
    fn copy_from(&mut self, other: &DynMat) -> Result<(), MatError>;
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny>;
    fn list<'py>(&self, py: Python<'py>) -> PyResult<&'py PyList>;
    fn item<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny>;
    fn read_item<'py>(&self, py: Python<'py>, idx: &[usize]) -> PyResult<PyObject>;
}

pub trait DynMatCasts {
    fn as_dtype<D: Num + ToDtype>(&self) -> Result<&Mat<D>, String>;
    fn as_dtype_mut<D: Num + ToDtype>(&mut self) -> Result<&mut Mat<D>, String>;
}

#[pyclass(name = "ndarray")]
pub struct DynMat {
    pub(crate) m: Box<dyn DynMatTrait + Send>,
}

impl DynMat {
    pub fn try_as_dtype<D: Num + ToDtype>(&self) -> PyResult<&Mat<D>> {
        if self.m.dtype() == D::to_dtype() {
            Ok(unsafe { std::mem::transmute::<&Mat<u8>, &Mat<D>>(self.m.untyped_mat()) })
        } else {
            Err(PyValueError::new_err(format!("Tensor is of type {} but expected {}", self.m.dtype(), D::to_dtype())))
        }
    }

    pub fn try_as_dtype_mut<D: Num + ToDtype>(&mut self) -> PyResult<&mut Mat<D>> {
        if self.m.dtype() == D::to_dtype() {
            Ok(unsafe { std::mem::transmute::<&mut Mat<u8>, &mut Mat<D>>(self.m.untyped_mat_mut()) })
        } else {
            Err(PyValueError::new_err(format!("Tensor is of type {} but expected {}", self.m.dtype(), D::to_dtype())))
        }
    }

    pub fn match_dtype<X, Y,
        U8: FnOnce(X, &Mat<u8>) -> Y,
        U16: FnOnce(X, &Mat<u16>) -> Y,
        U32: FnOnce(X, &Mat<u32>) -> Y,
        U64: FnOnce(X, &Mat<u64>) -> Y,
        I8: FnOnce(X, &Mat<i8>) -> Y,
        I16: FnOnce(X, &Mat<i16>) -> Y,
        I32: FnOnce(X, &Mat<i32>) -> Y,
        I64: FnOnce(X, &Mat<i64>) -> Y,
        F32: FnOnce(X, &Mat<f32>) -> Y,
    >(&self, x: X, u8: U8, u16: U16, u32: U32, u64: U64, i8: I8, i16: I16, i32: I32, i64: I64, f32: F32) -> Y {
        match self.m.dtype() {
            DTypeEnum::u8 => u8(x, self.try_as_dtype::<u8>().unwrap()),
            DTypeEnum::u16 => u16(x, self.try_as_dtype::<u16>().unwrap()),
            DTypeEnum::u32 => u32(x, self.try_as_dtype::<u32>().unwrap()),
            DTypeEnum::u64 => u64(x, self.try_as_dtype::<u64>().unwrap()),
            DTypeEnum::i8 => i8(x, self.try_as_dtype::<i8>().unwrap()),
            DTypeEnum::i16 => i16(x, self.try_as_dtype::<i16>().unwrap()),
            DTypeEnum::i32 => i32(x, self.try_as_dtype::<i32>().unwrap()),
            DTypeEnum::i64 => i64(x, self.try_as_dtype::<i64>().unwrap()),
            DTypeEnum::f32 => f32(x, self.try_as_dtype::<f32>().unwrap()),
        }
    }
}


#[pymethods]
impl DynMat {
    #[getter]
    fn ndim(&self) -> usize {
        self.m.ndim()
    }
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> &'py PyTuple {
        PyTuple::new(py, self.m.shape())
    }
    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> &'py PyTuple {
        PyTuple::new(py, self.m.strides())
    }
    #[getter]
    fn dtype(&self) -> DType {
        DType { e: self.m.dtype() }
    }
    fn astype(&self, dtype:DType) -> PyResult<DynMat> {
        self.m.astype(dtype.e).map_err(ocl_err_to_py_ex)
    }
    fn unsqueeze(&self, position: Option<usize>) -> PyResult<DynMat> {
        self.m.unsqueeze(position.unwrap_or(0)).map_err(ocl_err_to_py_ex)
    }
    fn squeeze(&self, position: Option<usize>) -> PyResult<DynMat> {
        self.m.squeeze(position.unwrap_or(0)).map_err(ocl_err_to_py_ex)
    }
    #[args(py_args = "*")]
    fn reshape(&self, py_args: &PyTuple) -> PyResult<DynMat> {
        let mut shape = Vec::with_capacity(py_args.len());
        for arg in py_args {
            let dim = arg.extract::<isize>()?;
            shape.push(dim);
        }
        self.m.reshape_infer_wildcard(shape.as_slice()).map_err(ocl_err_to_py_ex)
    }
    fn exp(&self) -> PyResult<DynMat> {
        self.m.exp().map_err(ocl_err_to_py_ex)
    }
    fn exp2(&self) -> PyResult<DynMat> {
        self.m.exp2().map_err(ocl_err_to_py_ex)
    }
    fn exp10(&self) -> PyResult<DynMat> {
        self.m.exp10().map_err(ocl_err_to_py_ex)
    }
    fn sin(&self) -> PyResult<DynMat> {
        self.m.sin().map_err(ocl_err_to_py_ex)
    }
    fn cos(&self) -> PyResult<DynMat> {
        self.m.cos().map_err(ocl_err_to_py_ex)
    }
    fn tan(&self) -> PyResult<DynMat> {
        self.m.tan().map_err(ocl_err_to_py_ex)
    }
    fn tanh(&self) -> PyResult<DynMat> {
        self.m.tanh().map_err(ocl_err_to_py_ex)
    }
    fn log(&self) -> PyResult<DynMat> {
        self.m.log().map_err(ocl_err_to_py_ex)
    }
    fn log2(&self) -> PyResult<DynMat> {
        self.m.log2().map_err(ocl_err_to_py_ex)
    }
    fn log10(&self) -> PyResult<DynMat> {
        self.m.log10().map_err(ocl_err_to_py_ex)
    }
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        self.m.numpy(py)
    }
    fn item<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        self.m.item(py)
    }
}

pub trait ToDtype {
    fn to_dtype() -> DTypeEnum;
}

macro_rules! dtype_for_prim {
    ($t:ident) =>{
        impl ToDtype for $t{
            fn to_dtype() -> DTypeEnum {
                DTypeEnum::$t
            }
        }
    };
}
dtype_for_prim!(u8);
dtype_for_prim!(u16);
dtype_for_prim!(u32);
dtype_for_prim!(u64);
dtype_for_prim!(i8);
dtype_for_prim!(i16);
dtype_for_prim!(i32);
dtype_for_prim!(i64);
dtype_for_prim!(f32);
impl ToDtype for usize {
    fn to_dtype() -> DTypeEnum {
        if std::mem::size_of::<usize>() == std::mem::size_of::<u64>() { DTypeEnum::u64 } else { DTypeEnum::u32 }
    }
}

impl ToDtype for isize {
    fn to_dtype() -> DTypeEnum {
        if std::mem::size_of::<isize>() == std::mem::size_of::<i64>() { DTypeEnum::i64 } else { DTypeEnum::i32 }
    }
}

fn dyn_mat_op<T: PyNum>(slf: &Mat<T>, scalar_or_tensor: &PyAny, op: fn(&mut Mat<T>, T) -> Result<(), MatError>, mat: fn(&mut Mat<T>, &Mat<T>) -> Result<(), MatError>) -> PyResult<DynMat> {
    if PyFloat::is_type_of(scalar_or_tensor) || PyInt::is_type_of(scalar_or_tensor) || PyBool::is_type_of(scalar_or_tensor) {
        let scalar = T::extract_from(scalar_or_tensor).map_err(ocl_err_to_py_ex)?;
        let mut out = slf.copy().map_err(ocl_err_to_py_ex)?;
        op(&mut out, scalar).map_err(ocl_err_to_py_ex)?;
        Ok(DynMat::from(out))
    } else if let Ok(tensor) = scalar_or_tensor.cast_as::<PyCell<DynMat>>() {
        let tensor = tensor.try_borrow()?;
        let mut out = slf.copy().map_err(ocl_err_to_py_ex)?;
        mat(&mut out, tensor.try_as_dtype()?).map_err(ocl_err_to_py_ex)?;
        Ok(DynMat::from(out))
    } else {
        Err(PyValueError::new_err(format!("Could not perform this operation on {}", scalar_or_tensor.get_type())))
    }
}

fn dyn_mat_op_in_place<T: PyNum>(slf: &mut Mat<T>, scalar_or_tensor: &PyAny, op: fn(&mut Mat<T>, T) -> Result<(), MatError>, mat: fn(&mut Mat<T>, &Mat<T>) -> Result<(), MatError>) -> PyResult<()> {
    if PyFloat::is_type_of(scalar_or_tensor) || PyInt::is_type_of(scalar_or_tensor) || PyBool::is_type_of(scalar_or_tensor) {
        let scalar = T::extract_from(scalar_or_tensor).map_err(ocl_err_to_py_ex)?;
        op(slf, scalar).map_err(ocl_err_to_py_ex)?;
        Ok(())
    } else if let Ok(tensor) = scalar_or_tensor.cast_as::<PyCell<DynMat>>() {
        let tensor = tensor.try_borrow()?;
        mat(slf, tensor.try_as_dtype()?).map_err(ocl_err_to_py_ex)?;
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("Could not perform this operation on {}", scalar_or_tensor.get_type())))
    }
}

impl<T: PyNum> DynMatTrait for Mat<T> {
    fn len(&self) -> usize {
        Mat::len(self)
    }

    fn dtype(&self) -> DTypeEnum {
        T::to_dtype()
    }

    fn lin_alg(&self) -> &LinAlgProgram {
        Mat::lin_alg(self)
    }

    fn dtype_size(&self) -> usize {
        std::mem::size_of::<T>()
    }

    fn untyped_mat(&self) -> &Mat<u8> {
        unsafe { std::mem::transmute::<&Mat<T>, &Mat<u8>>(self) }
    }
    fn untyped_mat_mut(&mut self) -> &mut Mat<u8> {
        unsafe { std::mem::transmute::<&mut Mat<T>, &mut Mat<u8>>(self) }
    }
    fn astype(&self, dtype:DTypeEnum) -> Result<DynMat, MatError>{
        match dtype{
            DTypeEnum::u8 => self.cast::<u8>()  .map(DynMat::from),
            DTypeEnum::u16 => self.cast::<u16>().map(DynMat::from),
            DTypeEnum::u32 => self.cast::<u32>().map(DynMat::from),
            DTypeEnum::u64 => self.cast::<u64>().map(DynMat::from),
            DTypeEnum::i8 =>  self.cast::<i8>() .map(DynMat::from),
            DTypeEnum::i16 => self.cast::<i16>().map(DynMat::from),
            DTypeEnum::i32 => self.cast::<i32>().map(DynMat::from),
            DTypeEnum::i64 => self.cast::<i64>().map(DynMat::from),
            DTypeEnum::f32 => self.cast::<f32>().map(DynMat::from),
        }
    }
    fn unsqueeze(&self, idx: usize) -> Result<DynMat, MatError> {
        self.unsqueeze(idx).map(DynMat::from)
    }

    fn squeeze(&self, idx: usize) -> Result<DynMat, MatError> {
        self.squeeze(idx).map(DynMat::from)
    }

    fn reshape(&self, shape: Box<[usize]>) -> Result<DynMat, MatError> {
        self.reshape_boxed(shape).map(DynMat::from)
    }

    fn reshape_infer_wildcard(&self, shape: &[isize]) -> Result<DynMat, MatError> {
        Mat::reshape_infer_wildcard(self, shape).map(DynMat::from)
    }

    fn view(&self, view: &[Range<usize>]) -> Result<DynMat, MatError> {
        self.view(view).map(DynMat::from)
    }
    fn matmul(&self, tensor: &PyAny) -> PyResult<DynMat> {
        let tensor = tensor.cast_as::<PyCell<DynMat>>()?;
        let tensor = tensor.try_borrow()?;
        let out = Mat::matmul(self,tensor.try_as_dtype::<T>()?).map_err(ocl_err_to_py_ex)?;
        Ok(DynMat::from(out))
    }
    fn add(&self, scalar: &PyAny) -> PyResult<DynMat> {
        dyn_mat_op(self, scalar, Mat::add_scalar, Mat::add_mat)
    }

    fn sub(&self, scalar: &PyAny) -> PyResult<DynMat> {
        dyn_mat_op(self, scalar, Mat::sub_scalar, Mat::sub_mat)
    }

    fn swapped_sub(&self, scalar: &PyAny) -> PyResult<DynMat> {
        dyn_mat_op(self, scalar, Mat::swapped_sub_scalar, |a,b|unreachable!("Should be handled by __sub__ instead of __rsub__"))
    }

    fn mul(&self, scalar: &PyAny) -> PyResult<DynMat> {
        dyn_mat_op(self, scalar, Mat::mul_scalar, Mat::mul_mat)
    }

    fn div(&self, scalar: &PyAny) -> PyResult<DynMat> {
        dyn_mat_op(self, scalar, Mat::div_scalar, Mat::div_mat)
    }

    fn swapped_div(&self, scalar: &PyAny) -> PyResult<DynMat> {
        dyn_mat_op(self, scalar, Mat::swapped_div_scalar, |a,b|unreachable!("Should be handled by __truediv__ instead of __rtruediv__"))
    }

    fn min(&self, scalar: &PyAny) -> PyResult<DynMat> {
        dyn_mat_op(self, scalar, Mat::min_scalar, Mat::min_mat)
    }

    fn max(&self, scalar: &PyAny) -> PyResult<DynMat> {
        dyn_mat_op(self, scalar, Mat::max_scalar, Mat::max_mat)
    }

    fn add_(&mut self, scalar: &PyAny) -> PyResult<()> {
        dyn_mat_op_in_place(self, scalar, Mat::add_scalar, Mat::add_mat)
    }

    fn sub_(&mut self, scalar: &PyAny) -> PyResult<()> {
        dyn_mat_op_in_place(self, scalar, Mat::sub_scalar, Mat::sub_mat)
    }

    fn swapped_sub_(&mut self, scalar: &PyAny) -> PyResult<()> {
        dyn_mat_op_in_place(self, scalar, Mat::swapped_sub_scalar, |a,b|unreachable!("Should be handled by __sub__ instead of __rsub__"))
    }

    fn mul_(&mut self, scalar: &PyAny) -> PyResult<()> {
        dyn_mat_op_in_place(self, scalar, Mat::mul_scalar, Mat::mul_mat)
    }

    fn div_(&mut self, scalar: &PyAny) -> PyResult<()> {
        dyn_mat_op_in_place(self, scalar, Mat::div_scalar, Mat::div_mat)
    }

    fn swapped_div_(&mut self, scalar: &PyAny) -> PyResult<()> {
        dyn_mat_op_in_place(self, scalar, Mat::swapped_div_scalar, |a,b|unreachable!("Should be handled by __truediv__ instead of __rtruediv__"))
    }

    fn min_(&mut self, scalar: &PyAny) -> PyResult<()> {
        dyn_mat_op_in_place(self, scalar, Mat::min_scalar, Mat::min_mat)
    }

    fn max_(&mut self, scalar: &PyAny) -> PyResult<()> {
        dyn_mat_op_in_place(self, scalar, Mat::max_scalar, Mat::max_mat)
    }

    fn neg_(&mut self) -> Result<(), MatError> {
        Mat::swapped_sub_scalar(self, T::zero())
    }

    fn neg(&self) -> Result<DynMat, MatError> {
        let mut out = self.copy()?;
        Mat::swapped_sub_scalar(&mut out, T::zero())?;
        Ok(DynMat::from(out))
    }

    fn exp(&self) -> Result<DynMat, MatError> {
        Mat::exp(self).map(DynMat::from)
    }
    fn exp2(&self) -> Result<DynMat, MatError> {
        Mat::exp2(self).map(DynMat::from)
    }
    fn exp10(&self) -> Result<DynMat, MatError> {
        Mat::exp10(self).map(DynMat::from)
    }
    fn sin(&self) -> Result<DynMat, MatError> { Mat::sin(self).map(DynMat::from) }
    fn cos(&self) -> Result<DynMat, MatError> {
        Mat::cos(self).map(DynMat::from)
    }
    fn tan(&self) -> Result<DynMat, MatError> {
        Mat::tan(self).map(DynMat::from)
    }
    fn tanh(&self) -> Result<DynMat, MatError> {
        Mat::tanh(self).map(DynMat::from)
    }
    fn log(&self) -> Result<DynMat, MatError> {
        Mat::log(self).map(DynMat::from)
    }
    fn log2(&self) -> Result<DynMat, MatError> {
        Mat::log2(self).map(DynMat::from)
    }
    fn log10(&self) -> Result<DynMat, MatError> {
        Mat::log10(self).map(DynMat::from)
    }
    fn fill(&mut self, scalar: &PyAny) -> PyResult<()> {
        Mat::fill(self, T::extract_from(scalar)?).map_err(ocl_err_to_py_ex)
    }
    fn copy_from(&mut self, other: &DynMat) -> Result<(), MatError> {
        match other.m.dtype() {
            DTypeEnum::u8 => Mat::copy_from(self, other.try_as_dtype::<u8>().unwrap()),
            DTypeEnum::u16 => Mat::copy_from(self, other.try_as_dtype::<u16>().unwrap()),
            DTypeEnum::u32 => Mat::copy_from(self, other.try_as_dtype::<u32>().unwrap()),
            DTypeEnum::u64 => Mat::copy_from(self, other.try_as_dtype::<u64>().unwrap()),
            DTypeEnum::i8 => Mat::copy_from(self, other.try_as_dtype::<i8>().unwrap()),
            DTypeEnum::i16 => Mat::copy_from(self, other.try_as_dtype::<i16>().unwrap()),
            DTypeEnum::i32 => Mat::copy_from(self, other.try_as_dtype::<i32>().unwrap()),
            DTypeEnum::i64 => Mat::copy_from(self, other.try_as_dtype::<i64>().unwrap()),
            DTypeEnum::f32 => Mat::copy_from(self, other.try_as_dtype::<f32>().unwrap()),
        }
    }
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let vec = self.to_vec().map_err(ocl_err_to_py_ex)?;
        let slice = vec.into_boxed_slice();
        let container = slice_box::SliceBox::new(slice);
        let data_ptr = container.data;
        let cell = pyo3::PyClassInitializer::from(container)
            .create_cell(py)
            .expect("Object creation failed.");
        let elem_size = std::mem::size_of::<T>();
        let strides_in_bytes = self.strides().iter().map(|s| s * elem_size).collect::<Vec<usize>>();
        unsafe {
            let ptr = PY_ARRAY_API.PyArray_New(
                PY_ARRAY_API.get_type_object(npyffi::NpyTypes::PyArray_Type),
                self.ndim() as i32,
                self.shape().as_ptr() as *mut npyffi::npy_intp,
                self.dtype().into_ctype() as i32,
                strides_in_bytes.as_ptr() as *mut npyffi::npy_intp,
                data_ptr as _,
                elem_size as i32,
                NPY_ARRAY_WRITEABLE,
                std::ptr::null_mut(),
            );
            PY_ARRAY_API.PyArray_SetBaseObject(ptr as *mut npyffi::PyArrayObject, cell as _);
            Ok(PyAny::from_owned_ptr(py, ptr))
        }
    }

    fn list<'py>(&self, py: Python<'py>) -> PyResult<&'py PyList> {
        let vec = self.to_vec().map_err(ocl_err_to_py_ex)?;
        Ok(PyList::new(py, vec))
    }

    fn item<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        self.item().map_err(ocl_err_to_py_ex).map(|i| i.to_object(py).into_ref(py))
    }

    fn read_item<'py>(&self, py: Python<'py>, idx: &[usize]) -> PyResult<PyObject> {
        Mat::read_item(self, idx).map(|s| s.to_object(py)).map_err(ocl_err_to_py_ex)
    }
}

pub fn try_as_dtype<D: Num + ToDtype, T: DynMatTrait>(tensor: &T) -> Result<&Mat<D>, String> {
    if tensor.dtype() == D::to_dtype() {
        Ok(unsafe { std::mem::transmute::<&Mat<u8>, &Mat<D>>(tensor.untyped_mat()) })
    } else {
        Err(format!("Tensor is of type {} but expected {}", tensor.dtype(), D::to_dtype()))
    }
}

pub fn try_as_dtype_mut<D: Num + ToDtype, T: DynMatTrait>(tensor: &mut T) -> Result<&mut Mat<D>, String> {
    if tensor.dtype() == D::to_dtype() {
        Ok(unsafe { std::mem::transmute::<&mut Mat<u8>, &mut Mat<D>>(tensor.untyped_mat_mut()) })
    } else {
        Err(format!("Tensor is of type {} but expected {}", tensor.dtype(), D::to_dtype()))
    }
}


impl<T: PyNum> From<Mat<T>> for DynMat {
    fn from(m: Mat<T>) -> Self {
        DynMat { m: Box::new(m) as Box<dyn DynMatTrait + Send> }
    }
}

#[pyfunction]
#[text_signature = "(shape_tuple, context, dtype/)"]
pub fn empty(shape: Vec<usize>, context: &NeatContext, dtype: Option<DType>) -> PyResult<DynMat> {
    unsafe {
        match dtype.map(|d| d.e).unwrap_or(DTypeEnum::f32) {
            DTypeEnum::u8 => Mat::<u8>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
            DTypeEnum::u16 => Mat::<u16>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
            DTypeEnum::u32 => Mat::<u32>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
            DTypeEnum::u64 => Mat::<u64>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
            DTypeEnum::i8 => Mat::<i8>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
            DTypeEnum::i16 => Mat::<i16>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
            DTypeEnum::i32 => Mat::<i32>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
            DTypeEnum::i64 => Mat::<i64>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
            DTypeEnum::f32 => Mat::<f32>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
        }.map_err(ocl_err_to_py_ex)
    }
}

#[pyfunction]
#[text_signature = "(shape_tuple, fill_value, ontext, dtype/)"]
pub fn full(shape: Vec<usize>, fill_value: &PyAny, context: &NeatContext, dtype: Option<DType>) -> PyResult<DynMat> {
    unsafe {
        match dtype.map(|d| d.e).unwrap_or_else(|| if PyInt::is_type_of(fill_value) { DTypeEnum::i64 } else if PyBool::is_type_of(fill_value) { DTypeEnum::u8 } else { DTypeEnum::f32 }) {
            DTypeEnum::u8 => Mat::<u8>::full(context.c.lin_alg(), shape.as_slice(), fill_value.extract::<u8>()?).map(DynMat::from),
            DTypeEnum::u16 => Mat::<u16>::full(context.c.lin_alg(), shape.as_slice(), fill_value.extract::<u16>()?).map(DynMat::from),
            DTypeEnum::u32 => Mat::<u32>::full(context.c.lin_alg(), shape.as_slice(), fill_value.extract::<u32>()?).map(DynMat::from),
            DTypeEnum::u64 => Mat::<u64>::full(context.c.lin_alg(), shape.as_slice(), fill_value.extract::<u64>()?).map(DynMat::from),
            DTypeEnum::i8 => Mat::<i8>::full(context.c.lin_alg(), shape.as_slice(), fill_value.extract::<i8>()?).map(DynMat::from),
            DTypeEnum::i16 => Mat::<i16>::full(context.c.lin_alg(), shape.as_slice(), fill_value.extract::<i16>()?).map(DynMat::from),
            DTypeEnum::i32 => Mat::<i32>::full(context.c.lin_alg(), shape.as_slice(), fill_value.extract::<i32>()?).map(DynMat::from),
            DTypeEnum::i64 => Mat::<i64>::full(context.c.lin_alg(), shape.as_slice(), fill_value.extract::<i64>()?).map(DynMat::from),
            DTypeEnum::f32 => Mat::<f32>::full(context.c.lin_alg(), shape.as_slice(), fill_value.extract::<f32>()?).map(DynMat::from),
        }.map_err(ocl_err_to_py_ex)
    }
}

#[pyfunction]
#[text_signature = "(shape_tuple, context, dtype/)"]
pub fn zeros(py: Python<'_>, shape: Vec<usize>, context: &NeatContext, dtype: Option<DType>) -> PyResult<DynMat> {
    full(shape, PyFloat::new(py, 0.), context, dtype)
}

#[pyfunction]
#[text_signature = "(shape_tuple, context, dtype/)"]
pub fn ones(py: Python<'_>, shape: Vec<usize>, context: &NeatContext, dtype: Option<DType>) -> PyResult<DynMat> {
    full(shape, PyFloat::new(py, 1.), context, dtype)
}

#[pyfunction]
#[text_signature = "(tensor/)"]
pub fn exp<'py>(py: Python<'py>, tensor_or_scalar: &'py PyAny) -> PyResult<&'py PyAny> {
    unary_op(py,tensor_or_scalar,f64::exp,|x|x.exp())
}
#[pyfunction]
#[text_signature = "(tensor/)"]
pub fn exp2<'py>(py: Python<'py>, tensor_or_scalar: &'py PyAny) -> PyResult<&'py PyAny> {
    unary_op(py,tensor_or_scalar,f64::exp2,|x|x.exp2())
}
#[pyfunction]
#[text_signature = "(tensor/)"]
pub fn exp10<'py>(py: Python<'py>, tensor_or_scalar: &'py PyAny) -> PyResult<&'py PyAny> {
    unary_op(py,tensor_or_scalar,|x|10f64.powf(x),|x|x.exp10())
}
#[pyfunction]
#[text_signature = "(tensor/)"]
pub fn log<'py>(py: Python<'py>, tensor_or_scalar: &'py PyAny) -> PyResult<&'py PyAny> {
    unary_op(py,tensor_or_scalar,f64::ln,|x|x.log())
}
#[pyfunction]
#[text_signature = "(tensor/)"]
pub fn log2<'py>(py: Python<'py>, tensor_or_scalar: &'py PyAny) -> PyResult<&'py PyAny> {
    unary_op(py,tensor_or_scalar,f64::log2,|x|x.log2())
}
#[pyfunction]
#[text_signature = "(tensor/)"]
pub fn log10<'py>(py: Python<'py>, tensor_or_scalar: &'py PyAny) -> PyResult<&'py PyAny> {
    unary_op(py,tensor_or_scalar,f64::log10,|x|x.log10())
}
#[pyfunction]
#[text_signature = "(tensor/)"]
pub fn sin<'py>(py: Python<'py>, tensor_or_scalar: &'py PyAny) -> PyResult<&'py PyAny> {
    unary_op(py,tensor_or_scalar,f64::sin,|x|x.sin())
}
#[pyfunction]
#[text_signature = "(tensor/)"]
pub fn cos<'py>(py: Python<'py>, tensor_or_scalar: &'py PyAny) -> PyResult<&'py PyAny> {
    unary_op(py,tensor_or_scalar,f64::cos,|x|x.cos())
}
#[pyfunction]
#[text_signature = "(tensor/)"]
pub fn tan<'py>(py: Python<'py>, tensor_or_scalar: &'py PyAny) -> PyResult<&'py PyAny> {
    unary_op(py,tensor_or_scalar,f64::tan,|x|x.tan())
}
#[pyfunction]
#[text_signature = "(tensor/)"]
pub fn tanh<'py>(py: Python<'py>, tensor_or_scalar: &'py PyAny) -> PyResult<&'py PyAny> {
    unary_op(py,tensor_or_scalar,f64::tanh,|x|x.tanh())
}


pub fn unary_op<'py>(py: Python<'py>, tensor_or_scalar: &'py PyAny, op:fn(f64)->f64, mat:fn(PyRef<DynMat>)->PyResult<DynMat>) -> PyResult<&'py PyAny> {
    if let Ok(scalar) = tensor_or_scalar.cast_as::<PyFloat>() {
        let scalar: &PyAny = PyFloat::new(py, op(scalar.extract::<f64>()?));
        Ok(scalar)
    } else if let Ok(scalar) = tensor_or_scalar.cast_as::<PyInt>() {
        let scalar: &PyAny = PyFloat::new(py, op(scalar.extract::<i64>()? as f64));
        Ok(scalar)
    } else if let Ok(tensor) = tensor_or_scalar.cast_as::<PyCell<DynMat>>() {
        let mut tensor: PyRef<DynMat> = tensor.try_borrow()?;
        let tensor = mat(tensor)?;
        let tensor_as_py = Py::new(py, tensor)?.into_ref(py);
        return Ok(tensor_as_py);
    } else {
        Err(PyValueError::new_err(format!("Could not perform this operation on {}", tensor_or_scalar.get_type())))
    }
}

#[pyfunction]
#[text_signature = "(numpy_array, context/)"]
pub fn from_numpy(input: &PyAny, context: &NeatContext) -> Result<DynMat, PyErr> {
    let array = unsafe {
        if npyffi::PyArray_Check(input.as_ptr()) == 0 {
            return Err(PyDowncastError::new(input, "PyArray<T, D>").into());
        }
        &*(input as *const PyAny as *const PyArrayDyn<u8>)
    };
    let dtype = array.dtype().get_datatype().ok_or_else(|| PyValueError::new_err("No numpy array has no dtype"))?;
    let shape = array.shape();

    macro_rules! e {
        ($dtype:ident) =>{
            {
                let array = unsafe{ &*(input as *const PyAny as *const PyArrayDyn<$dtype>) };
                let arr = unsafe { array.as_slice()? };
                Mat::<$dtype>::from_slice(context.c.lin_alg(),arr,shape).map(DynMat::from).map_err(ocl_err_to_py_ex)
            }
        };
    }
    match dtype {
        DataType::Bool => e!(u8),
        DataType::Int8 => e!(i8),
        DataType::Int16 => e!(i16),
        DataType::Int32 => e!(i32),
        DataType::Int64 => e!(i64),
        DataType::Uint8 => e!(u8),
        DataType::Uint16 => e!(u16),
        DataType::Uint32 => e!(u32),
        DataType::Uint64 => e!(u64),
        DataType::Float32 => e!(f32),
        DataType::Float64 => Err(PyValueError::new_err("Float64 dtype is not yet available in RustyNEAT")),
        DataType::Complex32 => Err(PyValueError::new_err("Complex32 dtype is not yet available in RustyNEAT")),
        DataType::Complex64 => Err(PyValueError::new_err("Complex64 dtype is not yet available in RustyNEAT")),
        DataType::Object => Err(PyValueError::new_err("Object dtype is not yet available in RustyNEAT")),
    }
}

#[pyfunction]
#[text_signature = "(list, context, dtype/)"]
pub fn array(array: &PyAny, context: &NeatContext, dtype: Option<DType>) -> PyResult<DynMat> {
    list_to_tensor(array, context.c.lin_alg(), dtype)
}

pub fn list_to_tensor(array: &PyAny, context: &LinAlgProgram, dtype: Option<DType>) -> PyResult<DynMat> {
    let mut dims = vec![];
    fn find_shape<'a>(elem: &'a PyAny, dims: &mut Vec<usize>) -> Option<&'a PyAny> {
        if let Ok(array) = elem.cast_as::<PyList>() {
            dims.push(array.len());
            array.iter().next().and_then(|first| find_shape(first, dims))
        } else if let Ok(array) = elem.cast_as::<PyTuple>() {
            dims.push(array.len());
            array.iter().next().and_then(|first| find_shape(first, dims))
        } else {
            Some(elem)
        }
    }

    let first_elem = find_shape(array, &mut dims);
    let dtype = dtype.map(|e| e.e).unwrap_or_else(|| if let Some(first_elem) = first_elem {
        if PyInt::is_type_of(first_elem) {
            DTypeEnum::i64
        } else if PyFloat::is_type_of(first_elem) {
            DTypeEnum::f32
        } else if PyBool::is_type_of(first_elem) {
            DTypeEnum::u8
        } else {
            DTypeEnum::f32
        }
    } else {
        DTypeEnum::f32
    });
    trait From_u8: Sized {
        fn from_u8(u: u8) -> PyResult<Self> {
            Err(PyValueError::new_err(format!("Cannot convert to {:?} from bool", std::any::type_name::<Self>())))
        }
    }
    trait From_f64: Sized {
        fn from_f64(u: f64) -> PyResult<Self> {
            Err(PyValueError::new_err(format!("Cannot convert to {:?} from float", std::any::type_name::<Self>())))
        }
    }
    trait From_i64: Sized {
        fn from_i64(u: i64) -> PyResult<Self> {
            Err(PyValueError::new_err(format!("Cannot convert to {:?} from int", std::any::type_name::<Self>())))
        }
    }
    macro_rules! can_be_converted {
        ($from_type:ident,$from_trait:ident,$from_fn:ident,$to:ident)=>{
            impl $from_trait for $to{
                fn $from_fn(u:$from_type)->PyResult<Self>{
                    Ok(Self::from(u))
                }
            }
        };
    }
    macro_rules! can_be_converted_with_overflow {
        ($from_type:ident,$from_trait:ident,$from_fn:ident,$to:ident)=>{
            impl $from_trait for $to{
                fn $from_fn(u:$from_type)->PyResult<Self>{
                    Self::try_from(u).map_err(ocl_err_to_py_ex)
                }
            }
        };
    }
    macro_rules! can_be_converted_with_overflow {
        ($from_type:ident,$from_trait:ident,$from_fn:ident,$to:ident)=>{
            impl $from_trait for $to{
                fn $from_fn(u:$from_type)->PyResult<Self>{
                    Ok(u as $to)
                }
            }
        };
    }
    macro_rules! cant_be_converted {
        ($from_type:ident,$from_trait:ident,$from_fn:ident,$to:ident)=>{
            impl $from_trait for $to{
            }
        };
    }
    can_be_converted!(u8,From_u8,from_u8,u8);
    can_be_converted!(u8,From_u8,from_u8,u16);
    can_be_converted!(u8,From_u8,from_u8,u32);
    can_be_converted!(u8,From_u8,from_u8,u64);
    cant_be_converted!(u8,From_u8,from_u8,i8);
    can_be_converted!(u8,From_u8,from_u8,i16);
    can_be_converted!(u8,From_u8,from_u8,i32);
    can_be_converted!(u8,From_u8,from_u8,i64);
    can_be_converted!(u8,From_u8,from_u8,f32);
    can_be_converted_with_overflow!(i64,From_i64,from_i64,u8);
    can_be_converted_with_overflow!(i64,From_i64,from_i64,u16);
    can_be_converted_with_overflow!(i64,From_i64,from_i64,u32);
    can_be_converted_with_overflow!(i64,From_i64,from_i64,u64);
    can_be_converted_with_overflow!(i64,From_i64,from_i64,i8);
    can_be_converted_with_overflow!(i64,From_i64,from_i64,i16);
    can_be_converted_with_overflow!(i64,From_i64,from_i64,i32);
    can_be_converted!(i64,From_i64,from_i64,i64);
    can_be_converted_with_overflow!(i64,From_i64,from_i64,f32);
    can_be_converted_with_overflow!(f64,From_f64,from_f64,u8);
    can_be_converted_with_overflow!(f64,From_f64,from_f64,u16);
    can_be_converted_with_overflow!(f64,From_f64,from_f64,u32);
    can_be_converted_with_overflow!(f64,From_f64,from_f64,u64);
    can_be_converted_with_overflow!(f64,From_f64,from_f64,i8);
    can_be_converted_with_overflow!(f64,From_f64,from_f64,i16);
    can_be_converted_with_overflow!(f64,From_f64,from_f64,i32);
    can_be_converted_with_overflow!(f64,From_f64,from_f64,i64);
    can_be_converted_with_overflow!(f64,From_f64,from_f64,f32);
    fn recursive_cast<'a, T: From_f64 + From_i64 + From_u8>(elem: &'a PyAny, dtype: DTypeEnum, dims: &[usize], elements: &mut Vec<T>) -> PyResult<()> {
        if let Ok(array) = elem.cast_as::<PyList>() {
            if dims.len() == 0 {
                return Err(PyValueError::new_err(format!("The nesting level of sub-lists is irregular")));
            }
            if array.len() != dims[0] {
                return Err(PyValueError::new_err(format!("The lengths of sub-lists are irregular")));
            }
            for item in array {
                recursive_cast(item, dtype, &dims[1..], elements)?;
            }
            Ok(())
        } else if let Ok(array) = elem.cast_as::<PyTuple>() {
            if dims.len() == 0 {
                return Err(PyValueError::new_err(format!("The nesting level of sub-lists is irregular")));
            }
            if array.len() != dims[0] {
                return Err(PyValueError::new_err(format!("The lengths of sub-lists are irregular")));
            }
            for item in array.iter() {
                recursive_cast(item, dtype, &dims[1..], elements)?;
            }
            Ok(())
        } else if let Ok(f) = elem.cast_as::<PyFloat>() {
            f.extract::<f64>().and_then(T::from_f64).map(|f| elements.push(f))
        } else if let Ok(f) = elem.cast_as::<PyInt>() {
            f.extract::<i64>().and_then(T::from_i64).map(|f| elements.push(f))
        } else if let Ok(f) = elem.cast_as::<PyBool>() {
            f.extract::<bool>().and_then(|f| T::from_u8(if f { 1u8 } else { 0 })).map(|f| elements.push(f))
        } else {
            Err(PyValueError::new_err(format!("Cannot cast {} to {}", elem.get_type(), dtype)))
        }
    }

    macro_rules! e {
        ($dtype:ident)=>{
            {
                let mut elements = vec![];
                recursive_cast::<$dtype>(array,dtype, &dims,&mut elements)?;
                Mat::<$dtype>::from_slice_boxed(context,elements.as_slice(), dims.into_boxed_slice()).map(DynMat::from)
            }
        };
    }
    match dtype {
        DTypeEnum::u8 => e!(u8),
        DTypeEnum::u16 => e!(u16),
        DTypeEnum::u32 => e!(u32),
        DTypeEnum::u64 => e!(u64),
        DTypeEnum::i8 => e!(i8),
        DTypeEnum::i16 => e!(i16),
        DTypeEnum::i32 => e!(i32),
        DTypeEnum::i64 => e!(i64),
        DTypeEnum::f32 => e!(f32),
    }.map_err(ocl_err_to_py_ex)
}


#[pyproto]
impl PyObjectProtocol for DynMat {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.m))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}


#[pyproto]
impl PySequenceProtocol for DynMat {
    fn __len__(&self) -> usize {
        self.m.len()
    }
}

fn extract_py_slice(key: &PyAny, dim: usize) -> PyResult<Range<usize>> {
    if let Ok(slice) = key.cast_as::<PySlice>() {
        let from = slice.getattr("start")?;
        let to = slice.getattr("stop")?;
        let from = if from.is_none() {
            0
        } else {
            from.extract::<usize>()?
        };
        let to = if to.is_none() {
            dim
        } else {
            to.extract::<usize>()?
        };
        Ok(from..to)
    } else {
        let index = key.extract::<usize>()?;
        Ok(index..index + 1)
    }
}

fn get_subview(me: &DynMat, key: &PyAny) -> PyResult<DynMat> {
    let slices = if let Ok(tuple) = key.cast_as::<PyTuple>() {
        tuple.as_slice().to_vec()
    } else if let Ok(list) = key.cast_as::<PyList>() {
        list.iter().collect::<Vec<&PyAny>>()
    } else {
        vec![key]
    };
    if slices.len() > me.m.ndim() {
        return Err(PyValueError::new_err(format!("Dimensionality of shape {} is less than the number {} of provided indices", me.m.shape().as_shape(), slices.len())));
    }
    let mut ranges = Vec::with_capacity(slices.len());
    for (i, key) in slices.iter().enumerate() {
        let slice = extract_py_slice(key, me.m.shape()[i])?;
        ranges.push(slice)
    }
    me.m.view(ranges.as_slice()).map_err(ocl_err_to_py_ex)
}

#[pyproto]
impl PyMappingProtocol for DynMat {
    fn __getitem__(&self, key: &PyAny) -> PyResult<PyObject> {
        if self.m.ndim() == 1 {
            if let Ok(slice) = extract_py_slice(key, self.m.len()) {
                if slice.start + 1 == slice.end {
                    return self.m.read_item(key.py(), &[slice.start]);
                }
            }
        }
        get_subview(self, key).map(|v| v.into_py(key.py()))
    }
    fn __setitem__(&mut self, key: &PyAny, value: &PyAny) -> PyResult<()> {
        let mut sub_view = get_subview(self, key)?;
        if PyFloat::is_type_of(value) || PyLong::is_type_of(value) || PyBool::is_type_of(value) {
            sub_view.m.fill(value)
        } else if PyList::is_type_of(value) {
            let tensor = list_to_tensor(value, self.m.lin_alg(), Some(DType { e: self.m.dtype() }))?;
            sub_view.m.copy_from(&tensor).map_err(ocl_err_to_py_ex)
        } else {
            let tensor = value.cast_as::<PyCell<DynMat>>()?;
            let tensor = tensor.try_borrow()?;
            sub_view.m.copy_from(&tensor).map_err(ocl_err_to_py_ex)
        }
    }
}


#[pyproto]
impl PyObjectProtocol for DType {
    fn __richcmp__(&self, other: PyRef<DType>, op: CompareOp) -> PyResult<bool> {
        let eq = self.e == other.e;
        match op {
            CompareOp::Eq => Ok(eq),
            CompareOp::Ne => Ok(!eq),
            op => Err(ocl_err_to_py_ex("Cannot compare platforms"))
        }
    }
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.e))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyclass]
pub struct ViewIter {
    inner: Py<DynMat>,
    idx: usize,
}

#[pyproto]
impl PyIterProtocol for ViewIter {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<DynMat>> {
        let i = slf.idx;
        slf.idx += 1;
        let r = Py::try_borrow(&slf.inner, slf.py())?;
        Ok(if i >= r.m.len() {
            None
        } else {
            let v = r.m.view(&[i..i + 1]).map_err(ocl_err_to_py_ex)?;
            Some(v)
        })
    }
}

#[pyproto]
impl PyIterProtocol for DynMat {
    fn __iter__(slf: Py<Self>) -> PyResult<PyObject> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let n = Py::try_borrow(&slf, py)?.m.ndim();
        if n == 1 {
            Py::try_borrow(&slf, py)?.m.list(py).and_then(|p| PyAny::iter(p)).map(|l| l.to_object(py))
        } else {
            let i = ViewIter {
                inner: slf,
                idx: 0,
            };
            Ok(i.into_py(py))
        }
    }
}

#[pyproto]
impl PyNumberProtocol for DynMat {
    fn __add__(lhs: PyRef<DynMat>, rhs: &PyAny) -> PyResult<DynMat> {
        lhs.m.add(rhs)
    }
    fn __sub__(lhs: PyRef<DynMat>, rhs: &PyAny) -> PyResult<DynMat> {
        lhs.m.sub(rhs)
    }
    fn __truediv__(lhs: PyRef<DynMat>, rhs: &PyAny) -> PyResult<DynMat> {
        lhs.m.div(rhs)
    }
    fn __mul__(lhs: PyRef<DynMat>, rhs: &PyAny) -> PyResult<DynMat> {
        lhs.m.mul(rhs)
    }
    fn __matmul__(lhs: PyRef<DynMat>, rhs: &PyAny) -> PyResult<DynMat> {
        lhs.m.matmul(rhs)
    }
    fn __iadd__(&mut self, rhs: &PyAny) -> PyResult<()> {
        self.m.add_(rhs)
    }
    fn __isub__(&mut self, rhs: &PyAny) -> PyResult<()> {
        self.m.sub_(rhs)
    }
    fn __itruediv__(&mut self, rhs: &PyAny) -> PyResult<()> {
        self.m.div_(rhs)
    }
    fn __imul__(&mut self, rhs: &PyAny) -> PyResult<()> {
        self.m.mul_(rhs)
    }
    fn __radd__(&self, lhs: &PyAny) -> PyResult<DynMat> {
        self.m.add(lhs)
    }
    fn __rsub__(&self, lhs: &PyAny) -> PyResult<DynMat> {
        self.m.swapped_sub(lhs)
    }
    fn __rtruediv__(&self, lhs: &PyAny) -> PyResult<DynMat> {
        self.m.swapped_div(lhs)
    }
    fn __rmul__(&self, lhs: &PyAny) -> PyResult<DynMat> {
        self.m.mul(lhs)
    }
    fn __neg__(&self) -> PyResult<DynMat>{
        self.m.neg().map_err(ocl_err_to_py_ex)
    }
}
use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol, PyTypeInfo, PyClass, PyMappingProtocol, FromPyPointer};
use pyo3::PyResult;
use ndalgebra::mat::{Mat, MatError, AsShape};
use ndalgebra::num::Num;
use crate::py_ocl::NeatContext;
use crate::{ocl_err_to_py_ex, slice_box};
use std::fmt::{Display, Formatter};
use pyo3::types::{PyType, PyList, PyTuple, PyInt, PyFloat, PyBool, PySlice};
use pyo3::exceptions::PyValueError;
use pyo3::basic::CompareOp;
use std::ops::Range;
use numpy::{PyArray, PY_ARRAY_API, npyffi, Element};
use numpy::npyffi::{PyArray_Dims, NPY_TYPES, NPY_ARRAY_WRITEABLE};

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

pub trait DynMatTrait: Display {
    fn dtype(&self) -> DTypeEnum;
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
    fn unsqueeze(&self, idx: usize) -> Result<DynMat, MatError>;
    fn squeeze(&self, idx: usize) -> Result<DynMat, MatError>;
    fn reshape(&self, shape: Box<[usize]>) -> Result<DynMat, MatError>;
    fn view(&self, view: &[Range<usize>]) -> Result<DynMat, MatError>;
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
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny>;
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
}


#[pymethods]
impl DynMat {
    #[getter]
    fn ndim(&self) -> usize {
        self.m.ndim()
    }
    #[getter]
    fn shape<'py>(&self, py:Python<'py>) -> &'py PyTuple {
        PyTuple::new(py,self.m.shape())
    }
    #[getter]
    fn strides<'py>(&self, py:Python<'py>) -> &'py PyTuple {
        PyTuple::new(py,self.m.strides())
    }
    #[getter]
    fn dtype(&self) -> DType {
        DType { e: self.m.dtype() }
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
        for arg in py_args{
            let dim = arg.extract::<usize>()?;
            shape.push(dim);
        }
        self.m.reshape(shape.into_boxed_slice()).map_err(ocl_err_to_py_ex)
    }
    fn exp(&self) -> PyResult<DynMat> {
        self.m.exp().map_err(ocl_err_to_py_ex)
    }
    fn exp2(&mut self) -> PyResult<DynMat> {
        self.m.exp2().map_err(ocl_err_to_py_ex)
    }
    fn exp10(&mut self) -> PyResult<DynMat> {
        self.m.exp10().map_err(ocl_err_to_py_ex)
    }
    fn sin(&mut self) -> PyResult<DynMat> {
        self.m.sin().map_err(ocl_err_to_py_ex)
    }
    fn cos(&mut self) -> PyResult<DynMat> {
        self.m.cos().map_err(ocl_err_to_py_ex)
    }
    fn tan(&mut self) -> PyResult<DynMat> {
        self.m.tan().map_err(ocl_err_to_py_ex)
    }
    fn tanh(&mut self) -> PyResult<DynMat> {
        self.m.tanh().map_err(ocl_err_to_py_ex)
    }
    fn log(&mut self) -> PyResult<DynMat> {
        self.m.log().map_err(ocl_err_to_py_ex)
    }
    fn log2(&mut self) -> PyResult<DynMat> {
        self.m.log2().map_err(ocl_err_to_py_ex)
    }
    fn log10(&mut self) -> PyResult<DynMat> {
        self.m.log10().map_err(ocl_err_to_py_ex)
    }
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        self.m.numpy(py)
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


impl<T: Num + ToDtype> DynMatTrait for Mat<T> {
    fn dtype(&self) -> DTypeEnum {
        T::to_dtype()
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

    fn unsqueeze(&self, idx: usize) -> Result<DynMat, MatError> {
        self.unsqueeze(idx).map(DynMat::from)
    }

    fn squeeze(&self, idx: usize) -> Result<DynMat, MatError> {
        self.squeeze(idx).map(DynMat::from)
    }

    fn reshape(&self, shape: Box<[usize]>) -> Result<DynMat, MatError> {
        self.reshape_boxed(shape).map(DynMat::from)
    }

    fn view(&self, view: &[Range<usize>]) -> Result<DynMat, MatError> {
        self.view(view).map(DynMat::from)
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


    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let vec = self.to_vec().map_err(ocl_err_to_py_ex)?;
        let slice = vec.into_boxed_slice();
        let container = slice_box::SliceBox::new(slice);
        let data_ptr = container.data;
        let cell = pyo3::PyClassInitializer::from(container)
            .create_cell(py)
            .expect("Object creation failed.");
        let elem_size = std::mem::size_of::<T>();
        let strides_in_bytes = self.strides().iter().map(|s|s*elem_size).collect::<Vec<usize>>();
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
// unsafe fn from_boxed_slice<T: Element, D: Dimension, ID>(
//     py: Python,
//     dims: ID,
//     flags: c_int,
//     strides: *const npy_intp,
//     slice: Box<[T]>,
// ) -> &PyArray<T, D>
//     where
//         ID: IntoDimension<Dim=D>,
// {
//     let dims = dims.into_dimension();
//     let container = slice_box::SliceBox::new(slice);
//     let data_ptr = container.data;
//     let cell = pyo3::PyClassInitializer::from(container)
//         .create_cell(py)
//         .expect("Object creation failed.");
//     let ptr = PY_ARRAY_API.PyArray_New(
//         PY_ARRAY_API.get_type_object(npyffi::NpyTypes::PyArray_Type),
//         dims.ndim_cint(),
//         dims.as_dims_ptr(),
//         T::npy_type() as i32,
//         strides as *mut _,          // strides
//         data_ptr as _,              // data
//         std::mem::size_of::<T>() as i32, // itemsize
//         flags,                          // flag
//         std::ptr::null_mut(),            //obj
//     );
//     PY_ARRAY_API.PyArray_SetBaseObject(ptr as *mut npyffi::PyArrayObject, cell as _);
//     PyArray::from_owned_ptr(py, ptr)
// }
//
// pub fn new_ndarray<T: Element, D: Dimension, ID>(py: Python, dims: ID, vec: Vec<T>) -> PyResult<&PyArray<T, D>>
//     where ID: IntoDimension<Dim=D> {
//     let vec = vec.into_boxed_slice();
//     let len = vec.len();
//     let strides = [std::mem::size_of::<T>() as npy_intp];
//     let vec = unsafe { from_boxed_slice(py, [len], NPY_ARRAY_WRITEABLE, strides.as_ptr(), vec) };
//     vec.reshape(dims)
// }
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


impl<T: Num + ToDtype> From<Mat<T>> for DynMat {
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
            DTypeEnum::i8 => Mat::<u8>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
            DTypeEnum::i16 => Mat::<u16>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
            DTypeEnum::i32 => Mat::<u32>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
            DTypeEnum::i64 => Mat::<u64>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
            DTypeEnum::f32 => Mat::<f32>::empty(context.c.lin_alg(), shape.as_slice()).map(DynMat::from),
        }.map_err(ocl_err_to_py_ex)
    }
}


#[pyfunction]
#[text_signature = "(tensor/)"]
pub fn exp<'py>(py: Python<'py>, tensor_or_scalar: &'py PyAny) -> PyResult<&'py PyAny> {
    if let Ok(scalar) = tensor_or_scalar.cast_as::<PyFloat>() {
        let scalar: &PyAny = PyFloat::new(py, scalar.extract::<f64>()?.exp());
        Ok(scalar)
    } else if let Ok(scalar) = tensor_or_scalar.cast_as::<PyInt>() {
        let scalar: &PyAny = PyFloat::new(py, (scalar.extract::<i64>()? as f64).exp());
        Ok(scalar)
    } else if let Ok(tensor) = tensor_or_scalar.cast_as::<PyCell<DynMat>>() {
        let mut tensor: PyRef<DynMat> = tensor.try_borrow()?;
        let tensor = tensor.exp()?;
        let tensor_as_py = Py::new(py, tensor)?.into_ref(py);
        return Ok(tensor_as_py);
    } else {
        Err(PyValueError::new_err(format!("Could not perform this operation on {}", tensor_or_scalar.get_type())))
    }
}

#[pyfunction]
#[text_signature = "(list, context, dtype/)"]
pub fn array(array: &PyAny, context: &NeatContext, dtype: Option<DType>) -> PyResult<DynMat> {
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
                Mat::<$dtype>::from_slice_boxed(context.c.lin_alg(),elements.as_slice(), dims.into_boxed_slice()).map(DynMat::from)
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
    // fn __richcmp__(&self, other: PyRef<NeatContext>, op: CompareOp) -> PyResult<bool> {
    //     let eq = self.c.device() == other.c.device() && self.c.platform().as_core()==other.c.platform().as_core();
    //     match op {
    //         CompareOp::Eq => Ok(eq),
    //         CompareOp::Ne => Ok(!eq),
    //         op => Err(ocl_err_to_py_ex("Cannot compare platforms"))
    //     }
    // }
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.m))
    }
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

#[pyproto]
impl PyMappingProtocol for DynMat {
    fn __getitem__(&self, key: &PyAny) -> PyResult<DynMat> {
        let slices = if let Ok(tuple) = key.cast_as::<PyTuple>() {
            tuple.as_slice().to_vec()
        } else if let Ok(list) = key.cast_as::<PyList>() {
            list.iter().collect::<Vec<&PyAny>>()
        } else {
            vec![key]
        };
        if slices.len() > self.m.ndim() {
            return Err(PyValueError::new_err(format!("Dimensionality of shape {} is less than the number {} of provided indices", self.m.shape().as_shape(), slices.len())));
        }
        let mut ranges = Vec::with_capacity(slices.len());
        for (i, key) in slices.iter().enumerate() {
            let (from,to) = if let Ok(slice) = key.cast_as::<PySlice>() {
                let from = slice.getattr("start")?;
                let to = slice.getattr("stop")?;
                let from = if from.is_none() {
                    0
                } else {
                    from.extract::<usize>()?
                };
                let to = if to.is_none() {
                    self.m.shape()[i]
                } else {
                    to.extract::<usize>()?
                };
                (from,to)
            } else {
                let index = key.extract::<usize>()?;
                (index, index+1)
            };


            ranges.push(from..to)
        }
        self.m.view(ranges.as_slice()).map_err(ocl_err_to_py_ex)
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
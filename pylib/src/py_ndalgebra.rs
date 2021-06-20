use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyObjectProtocol, PyTypeInfo, PyClass};
use pyo3::PyResult;
use ndalgebra::mat::Mat;
use ndalgebra::num::Num;
use crate::py_ocl::NeatContext;
use crate::ocl_err_to_py_ex;
use std::fmt::{Display, Formatter};
use pyo3::types::{PyType, PyList, PyTuple, PyInt, PyFloat, PyBool};
use pyo3::exceptions::PyValueError;

#[pyclass]
#[derive(Copy, Clone, Eq, PartialEq)]
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

impl Display for DTypeEnum {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

pub trait DynMatTrait {
    fn dtype(&self) -> DTypeEnum;
    fn dtype_size(&self) -> usize;
    fn untyped_mat(&self) -> &Mat<u8>;
    fn as_f32(&self) -> Result<&Mat<f32>, String>;
    fn ndim(&self) -> usize {
        self.untyped_mat().ndim()
    }
    fn shape(&self) -> &[usize] {
        self.untyped_mat().shape()
    }
    fn strides(&self) -> &[usize] {
        self.untyped_mat().strides()
    }
}

#[pyclass]
pub struct DynMat {
    pub(crate) m: Box<dyn DynMatTrait + Send>,
}


#[pymethods]
impl DynMat {
    #[getter]
    fn ndim(&self) -> usize {
        self.m.ndim()
    }
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.m.shape().to_vec()
    }
    #[getter]
    fn strides(&self) -> Vec<usize> {
        self.m.strides().to_vec()
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

    fn as_f32(&self) -> Result<&Mat<f32>, String> {
        if self.dtype() == DTypeEnum::f32 {
            Ok(unsafe { std::mem::transmute::<&Mat<T>, &Mat<f32>>(self) })
        } else {
            Err(format!("Tensor is of type {} but expectde f32", self.dtype()))
        }
    }
}

impl<T: Num + ToDtype> From<Mat<T>> for DynMat {
    fn from(m: Mat<T>) -> Self {
        DynMat { m: Box::new(m) as Box<dyn DynMatTrait + Send> }
    }
}

#[pyfunction]
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
    let dtype = if let Some(first_elem) = first_elem {
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
    };

    trait From_u8:Sized{
        fn from_u8(u:u8)->PyResult<Self>{
            Err(PyValueError::new_err(format!("Cannot convert to {:?} from bool",std::any::type_name::<Self>())))
        }
    }
    trait From_f64:Sized{
        fn from_f64(u:f64)->PyResult<Self>{
            Err(PyValueError::new_err(format!("Cannot convert to {:?} from float",std::any::type_name::<Self>())))
        }
    }
    trait From_i64:Sized{
        fn from_i64(u:i64)->PyResult<Self>{
            Err(PyValueError::new_err(format!("Cannot convert to {:?} from int",std::any::type_name::<Self>())))
        }
    }
    macro_rules! can_be_converted{
        ($from_type:ident,$from_trait:ident,$from_fn:ident,$to:ident)=>{
            impl $from_trait for $to{
                fn $from_fn(u:$from_type)->PyResult<Self>{
                    Ok(Self::from(u))
                }
            }
        };
    }
    macro_rules! can_be_converted_with_overflow{
        ($from_type:ident,$from_trait:ident,$from_fn:ident,$to:ident)=>{
            impl $from_trait for $to{
                fn $from_fn(u:$from_type)->PyResult<Self>{
                    Self::try_from(u).map_err(ocl_err_to_py_ex)
                }
            }
        };
    }
    macro_rules! can_be_converted_with_overflow{
        ($from_type:ident,$from_trait:ident,$from_fn:ident,$to:ident)=>{
            impl $from_trait for $to{
                fn $from_fn(u:$from_type)->PyResult<Self>{
                    Ok(u as $to)
                }
            }
        };
    }
    macro_rules! cant_be_converted{
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
    fn recursive_cast<'a, T:From_f64+From_i64+From_u8>(elem: &'a PyAny, dtype: DTypeEnum, dims: &[usize], elements: &mut Vec<T>) -> PyResult<()> {

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
            f.extract::<f64>().and_then(T::from_f64).map(|f|elements.push(f))
        } else if let Ok(f) = elem.cast_as::<PyInt>() {
            f.extract::<i64>().and_then(T::from_i64).map(|f|elements.push(f))
        } else if let Ok(f) = elem.cast_as::<PyBool>() {
            f.extract::<bool>().and_then(|f|T::from_u8(if f{1u8}else{0})).map(|f|elements.push(f))
        } else {
            Err(PyValueError::new_err(format!("Cannot cast {} to {}", elem.get_type(), dtype)))
        }
    }

    macro_rules! e {
        ($dtype:ident)=>{
            {
                let mut elements = vec![];
                recursive_cast::<u8>(array,dtype, &dims,&mut elements)?;
                Mat::<u8>::from_slice_boxed(context.c.lin_alg(),elements.as_slice(), dims.into_boxed_slice()).map(DynMat::from)
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

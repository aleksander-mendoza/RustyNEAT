use std::ops::{Deref, DerefMut, SubAssign, AddAssign};
use serde::{Serialize, Deserialize};
use std::fmt::{Debug, Formatter};
use crate::as_usize::AsUsize;
use std::iter::Sum;
use crate::tensor_trait::TensorTrait;
use crate::{Idx, HasShape, Shape2, VectorFieldOne, Shape3, Weight};

#[derive(Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct Tensor<D> {
    arr: Vec<D>,
    shape: [Idx; 3],
}

impl<D> HasShape for Tensor<D> {
    fn shape(&self) -> &[Idx; 3] {
        &self.shape
    }
}
impl<D: Copy> TensorTrait<D> for Tensor<D> {
    fn as_slice(&self) -> &[D] {
        self.arr.as_slice()
    }

    fn repeat_column(&self, column_grid: [Idx; 2], column_pos: [Idx; 2]) -> Self {
        let shape = column_grid.add_channels(self.shape().channels());
        let mut slf = unsafe{Self::empty(shape)};
        slf.copy_repeated_column(self,column_pos);
        slf
    }

    fn as_mut_slice(&mut self) -> &mut [D] {
        self.arr.as_mut_slice()
    }
}

impl<D: Debug> Debug for Tensor<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut i0 = 0;
        let mut i1 = 0;
        let mut i2 = 0;
        write!(f, "[[[")?;
        for x in self.arr.as_slice() {
            write!(f, "{:?}", x)?;
            i2 += 1;
            if i2 == self.shape[2] {
                write!(f, "]")?;
                i2 = 0;
                i1 += 1;
                if i1 == self.shape[1] {
                    write!(f, "]")?;
                    i1 = 0;
                    i0 += 1;
                    if i0 == self.shape[0] {
                        write!(f, "]")?;
                    } else {
                        write!(f, ",\n  [[")?;
                    }
                } else {
                    write!(f, ", [")?;
                }
            } else {
                write!(f, ", ")?;
            }
        }
        Ok(())
    }
}

impl<D> Tensor<D> {
    pub fn null() -> Self {
        Self{
            arr: vec![],
            shape: [0,0,0]
        }
    }
    pub fn new(shape: [Idx; 3], initial_value: D) -> Self where D: Clone {
        Self {
            arr: vec![initial_value; shape.product().as_usize()],
            shape,
        }
    }
    pub unsafe fn empty(shape: [Idx; 3]) -> Self {
        let l = shape.product().as_usize();
        let mut arr = Vec::with_capacity(l);
        arr.set_len(l);
        Self { arr, shape, }
    }
    pub fn from(shape: [Idx; 3], arr: Vec<D>) -> Self {
        assert_eq!(shape.product().as_usize(), arr.len());
        Self {
            arr,
            shape,
        }
    }
    pub fn from_slice(shape: [Idx; 3], arr: &[D]) -> Self where D: Clone {
        assert_eq!(shape.product().as_usize(), arr.len());
        Self::from(shape, Vec::from(arr))
    }
    pub fn into_vec(self)->Vec<D>{
        let Self{arr,..} = self;
        arr
    }
}

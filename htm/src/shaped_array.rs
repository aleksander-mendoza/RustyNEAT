use std::ops::{Deref, DerefMut, SubAssign, AddAssign};
use crate::{Idx, as_usize, VectorFieldOne, CpuEccPopulation, DenseWeight, Metric, HasConvShape, ConvShape, HasShape};
use serde::{Serialize, Deserialize};
use std::fmt::{Debug, Formatter};
use crate::as_usize::AsUsize;

#[derive(Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct ShapedArray<D> {
    arr: Vec<D>,
    shape: [Idx; 3],
}

pub trait ForwardTarget<D> {
    fn target(&self) -> &[D];
    fn len(&self) -> usize {
        self.target().len()
    }
    fn state(&self, idx: Idx) -> &D {
        &self.target()[idx.as_usize()]
    }
    fn excite(&mut self, idx: Idx, w: D) where D:AddAssign{
        self.target_mut()[idx.as_usize()] += w;
    }
    fn inhibit(&mut self, idx: Idx, w: D) where D:SubAssign{
        self.target_mut()[idx.as_usize()] -= w;
    }
    fn target_mut(&mut self) -> &mut [D];
    fn target_vec(&self) -> Vec<D> where D: Clone{
        self.target().to_vec()
    }
}

pub trait ConvShapedArrayTrait<D>: HasConvShape + ShapedArrayTrait<D> {}

pub trait ShapedArrayTrait<D>: HasShape + ForwardTarget<D> {}

impl<D> HasShape for ShapedArray<D> {
    fn shape(&self) -> &[Idx; 3] {
        &self.shape
    }
}

impl<D> ShapedArrayTrait<D> for ShapedArray<D> {}

impl<D> ForwardTarget<D> for ShapedArray<D> {
    fn target(&self) -> &[D] {
        self.arr.as_slice()
    }

    fn target_mut(&mut self) -> &mut [D] {
        self.arr.as_mut_slice()
    }
}

impl<D: Debug> Debug for ShapedArray<D> {
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

impl ShapedArray<f32> {
    pub fn from_pop<M: Metric<f32>>(pop: &CpuEccPopulation<M>) -> Self {
        Self {
            shape: *pop.shape(),
            arr: pop.sums.clone(),
        }
    }
}

impl<D> ShapedArray<D> {
    pub fn new(shape: [Idx; 3], initial_value: D) -> Self where D: Clone {
        Self {
            arr: vec![initial_value; shape.product().as_usize()],
            shape,
        }
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

impl<D> Deref for ShapedArray<D> {
    type Target = Vec<D>;

    fn deref(&self) -> &Self::Target {
        &self.arr
    }
}

impl<D> DerefMut for ShapedArray<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.arr
    }
}
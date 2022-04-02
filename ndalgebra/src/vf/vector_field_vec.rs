use crate::vf::*;
use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, Neg, AddAssign, SubAssign, DivAssign, MulAssign, RemAssign};
use std::mem::MaybeUninit;
use num_traits::{Zero, One, Num, AsPrimitive, NumAssign};
use rand::Rng;
use rand::distributions::{Standard, Distribution};

impl<T: Copy> VectorField<T> for Vec<T> {
    #[inline]
    fn fold<I>(&self, zero: I, mut f: impl FnMut(I, T) -> I) -> I {
        self.iter().fold(zero, |a, b| f(a, *b))
    }
    #[inline]
    fn map(&self, mut f: impl FnMut(T) -> T) -> Self {
        self.iter().cloned().map(f).collect()
    }

    fn map_assign(&mut self, f: impl FnMut(&mut T)) -> &mut Self {
        self.iter_mut().for_each(f);
        self
    }

    #[inline]
    fn new_const(s: T) -> Self {
        vec![s]
    }
    #[inline]
    fn all(&self, mut f: impl FnMut(T) -> bool) -> bool {
        self.iter().cloned().all(f)
    }
    #[inline]
    fn any(&self, mut f: impl FnMut(T) -> bool) -> bool {
        self.iter().cloned().any(f)
    }
    #[inline]
    fn all_zip(&self, other: &Self, mut f: impl FnMut(T, T) -> bool) -> bool {
        self.iter().zip(other.iter()).all(|(&a, &b)| f(a, b))
    }
    #[inline]
    fn any_zip(&self, other: &Self, mut f: impl FnMut(T, T) -> bool) -> bool {
        self.iter().zip(other.iter()).any(|(&a, &b)| f(a, b))
    }
    #[inline]
    fn zip(&self, other: &Self, mut f: impl FnMut(T, T) -> T) -> Self {
        self.iter().cloned().zip(other.iter().cloned()).map(|(a,b)|f(a,b)).collect()
    }

    fn zip_assign(&mut self, other: &Self, mut f: impl FnMut(&mut T, T)) -> &mut Self {
        self.iter_mut().zip(other.iter().cloned()).for_each(|(a,b)|f(a,b));
        self
    }
}

impl<T: Copy + Add<Output=T>> VectorFieldAdd<T> for Vec<T> {}

impl<T: Copy + Add<Output=T> + Zero> VectorFieldZero<T> for Vec<T> {}

impl<T: Copy + Sub<Output=T>> VectorFieldSub<T> for Vec<T> {}

impl<T: Copy + Div<Output=T>> VectorFieldDiv<T> for Vec<T> {}

impl<T: Copy + Div<Output=T> + Zero> VectorFieldDivDefaultZero<T> for Vec<T> {}

impl<T: Copy + Rem<Output=T> + Zero> VectorFieldRemDefaultZero<T> for Vec<T> {}

impl<T: Copy + Mul<Output=T>> VectorFieldMul<T> for Vec<T> {}

impl<T: Copy + Mul<Output=T> + One> VectorFieldOne<T> for Vec<T> {}

impl<T: Copy + Rem<Output=T>> VectorFieldRem<T> for Vec<T> {}

impl<T: Copy + PartialOrd> VectorFieldPartialOrd<T> for Vec<T> {}

impl<T: Neg<Output=T> + Zero + PartialOrd + Copy> VectorFieldAbs<T> for Vec<T> {}

impl<T: Neg<Output=T> + Copy> VectorFieldNeg<T> for Vec<T> {}


impl<T: Copy + AddAssign> VectorFieldAddAssign<T> for Vec<T> {}

impl<T: Copy + SubAssign> VectorFieldSubAssign<T> for Vec<T> {}

impl<T: Copy + DivAssign> VectorFieldDivAssign<T> for Vec<T> {}

impl<T: Copy + DivAssign + Zero> VectorFieldDivAssignDefaultZero<T> for Vec<T> {}

impl<T: Copy + RemAssign + Zero> VectorFieldRemAssignDefaultZero<T> for Vec<T> {}

impl<T: Copy + MulAssign> VectorFieldMulAssign<T> for Vec<T> {}

impl<T: Copy + RemAssign> VectorFieldRemAssign<T> for Vec<T> {}


impl<T: Num + Copy + PartialOrd> VectorFieldNum<T> for Vec<T> {}

impl<T: NumAssign + Copy + PartialOrd> VectorFieldNumAssign<T> for Vec<T> {}

impl<T: Copy> VecCast<T> for Vec<T> {
    fn as_scalar<IntoScalar: 'static + Copy>(&self) -> Vec<IntoScalar> where T: AsPrimitive<IntoScalar> {
        self.iter().map(|i|i.as_()).collect()
    }
}

impl<T: Copy + Rem<Output=T>> VectorFieldRng<T> for Vec<T> where Standard: Distribution<T>{
    fn rand_vec(&self, rng: &mut impl rand::Rng)->Self{
        self.iter().map(|i|rng.gen::<T>() % *i).collect()
    }
}

impl<T: Copy + Rem<Output=T>> VectorFieldRngAssign<T> for Vec<T> where Standard: Distribution<T>{
    fn rand_vec_assign(&mut self, rng: &mut impl rand::Rng)->&mut Self{
        self.iter_mut().for_each(|i|*i = rng.gen::<T>() % *i);
        self
    }
}
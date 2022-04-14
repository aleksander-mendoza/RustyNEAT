use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, Neg, AddAssign, SubAssign, DivAssign, MulAssign, RemAssign};
use std::mem::MaybeUninit;
use num_traits::{Zero, One, Num, AsPrimitive, NumAssign};
use rand::Rng;
use rand::distributions::{Standard, Distribution};
use crate::*;

impl<T: Copy> VectorFieldOwned<T> for Vec<T> {
    fn _map(mut self, f: impl FnMut(T) -> T) -> Self {
        self.iter_mut().for_each(|e| *e = f(*e));
        self
    }
    fn _zip(mut self, other: &Self, mut f: impl FnMut(T, T) -> T) -> Self {
        self.iter_mut().zip(other.iter().cloned()).for_each(|(a, b)| *a = f(*a, b));
        self
    }
}

impl<T: Copy + Add<Output=T>> VectorFieldAddOwned<T> for Vec<T> {}

impl<T: Copy + Sub<Output=T>> VectorFieldSubOwned<T> for Vec<T> {}

impl<T: Copy + Div<Output=T>> VectorFieldDivOwned<T> for Vec<T> {}

impl<T: Copy + Div<Output=T> + Zero> VectorFieldDivDefaultZeroOwned<T> for Vec<T> {}

impl<T: Copy + Rem<Output=T> + Zero> VectorFieldRemDefaultZeroOwned<T> for Vec<T> {}

impl<T: Copy + Mul<Output=T>> VectorFieldMulOwned<T> for Vec<T> {}

impl<T: Copy + Rem<Output=T>> VectorFieldRemOwned<T> for Vec<T> {}

impl<T: Copy + PartialOrd> VectorFieldPartialOrdOwned<T> for Vec<T> {}

impl<T: Neg<Output=T> + Zero + PartialOrd + Copy> VectorFieldAbsOwned<T> for Vec<T> {}

impl<T: Neg<Output=T> + Copy> VectorFieldNegOwned<T> for Vec<T> {}

impl<T: Num + Copy + PartialOrd> VectorFieldNumOwned<T> for Vec<T> {}

impl<T: Copy> VectorFieldRngOwned<T> for Vec<T> where Standard: Distribution<T> {
    fn _rand(mut self, rng: &mut impl Rng) -> Self {
        self.iter_mut().map(|i| *i = rng.gen::<T>());
        self
    }
}

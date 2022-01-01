use std::fmt::{Debug, Formatter};

use serde::{Serialize, Deserialize};
use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, Neg};
use std::mem::MaybeUninit;
use num_traits::{Zero, One, Num, AsPrimitive};
use rand::Rng;
use rand::distributions::{Standard, Distribution};

pub trait VectorField<Scalar: Copy>: Sized {
    fn fold(&self, zero: Scalar, f: impl FnMut(Scalar, Scalar) -> Scalar) -> Scalar;
    fn map(&self, f: impl FnMut(Scalar) -> Scalar) -> Self;
    fn new_const(s:Scalar) -> Self;
    fn all(&self, f: impl FnMut(Scalar) -> bool) -> bool;
    fn any(&self, f: impl FnMut(Scalar) -> bool) -> bool;
    fn all_zip(&self, other: &Self, f: impl FnMut(Scalar, Scalar) -> bool) -> bool;
    fn any_zip(&self, other: &Self, f: impl FnMut(Scalar, Scalar) -> bool) -> bool;
    fn zip(&self, other: &Self, f: impl FnMut(Scalar, Scalar) -> Scalar) -> Self;
}

pub trait VectorFieldAdd<Scalar: Add<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn add(&self, rhs: &Self) -> Self {
        self.zip(&rhs, |a, b| a + b)
    }
    fn add_scalar(&self, rhs: Scalar) -> Self {
        self.map(|a| a + rhs)
    }
}

pub trait VectorFieldZero<Scalar: Zero + Add<Output=Scalar> + Copy>: VectorFieldAdd<Scalar> {
    fn sum(&self) -> Scalar {
        self.fold(Scalar::zero(), |a, b| a + b)
    }
}

pub trait VectorFieldPartialOrd<Scalar: PartialOrd + Copy>: VectorField<Scalar> {
    fn all_le(&self, rhs: &Self) -> bool {
        self.all_zip(rhs, |l, r| l <= r)
    }
    fn all_le_scalar(&self, rhs: Scalar) -> bool {
        self.all( |l| l <= rhs)
    }
    fn all_lt(&self, rhs: &Self) -> bool {
        self.all_zip(rhs, |l, r| l < r)
    }
    fn all_lt_scalar(&self, rhs: Scalar) -> bool {
        self.all( |l| l < rhs)
    }
    fn all_gt(&self, rhs: &Self) -> bool {
        self.all_zip(rhs, |l, r| l > r)
    }
    fn all_gt_scalar(&self, rhs: Scalar) -> bool {
        self.all( |l| l > rhs)
    }
    fn all_ge(&self, rhs: &Self) -> bool {
        self.all_zip(rhs, |l, r| l >= r)
    }
    fn all_ge_scalar(&self, rhs: Scalar) -> bool {
        self.all( |l| l >= rhs)
    }
    fn all_eq(&self, rhs: &Self) -> bool {
        self.all_zip(rhs, |l, r| l == r)
    }
    fn all_eq_scalar(&self, rhs: Scalar) -> bool {
        self.all(|l| l == rhs)
    }
    fn all_neq(&self, rhs: &Self) -> bool {
        self.all_zip(rhs, |l, r| l != r)
    }
    fn all_neq_scalar(&self, rhs: Scalar) -> bool {
        self.all( |l| l != rhs)
    }
    fn max(&self, rhs: &Self) -> Self {
        self.zip(rhs, |l, r| if l > r { l } else { r })
    }
    fn min(&self, rhs: &Self) -> Self {
        self.zip(rhs, |l, r| if l < r { l } else { r })
    }
    fn max_scalar(&self, rhs: Scalar) -> Self {
        self.map(|l| if l > rhs { l } else { rhs })
    }
    fn min_scalar(&self, rhs: Scalar) -> Self {
        self.map(|l| if l < rhs { l } else { rhs })
    }
    fn any_le(&self, rhs: &Self) -> bool {
        self.any_zip(rhs, |l, r| l <= r)
    }
    fn any_le_scalar(&self, rhs: Scalar) -> bool { self.any( |l| l <= rhs) }
    fn any_lt(&self, rhs: &Self) -> bool { self.any_zip(rhs, |l, r| l < r) }
    fn any_lt_scalar(&self, rhs: Scalar) -> bool { self.any( |l| l < rhs) }
    fn any_gt(&self, rhs: &Self) -> bool { self.any_zip(rhs, |l, r| l > r) }
    fn any_gt_scalar(&self, rhs: Scalar) -> bool { self.any( |l| l > rhs) }
    fn any_ge(&self, rhs: &Self) -> bool { self.any_zip(rhs, |l, r| l >= r) }
    fn any_ge_scalar(&self, rhs: Scalar) -> bool { self.any( |l| l >= rhs) }
    fn any_eq(&self, rhs: &Self) -> bool { self.any_zip(rhs, |l, r| l == r) }
    fn any_eq_scalar(&self, rhs: Scalar) -> bool { self.any(|l| l == rhs) }
    fn any_neq(&self, rhs: &Self) -> bool { self.any_zip(rhs, |l, r| l != r) }
    fn any_neq_scalar(&self, rhs: Scalar) -> bool { self.any( |l| l != rhs) }
}

pub trait VectorFieldAbs<Scalar: Neg<Output=Scalar> + Zero + PartialOrd + Copy>: VectorField<Scalar> {
    fn abs(&self) -> Self {
        self.map(|b| if b < Scalar::zero() { -b } else { b })
    }
}

pub trait VectorFieldNeg<Scalar: Neg<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn neg(&self) -> Self {
        self.map( |b| -b)
    }
}

pub trait VectorFieldSub<Scalar: Sub<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn sub(&self, rhs: &Self) -> Self {
        self.zip(&rhs, |a, b| a - b)
    }
    fn sub_scalar(&self, scalar: Scalar) -> Self {
        self.map(|a| a - scalar)
    }
}

pub trait VectorFieldDiv<Scalar: Div<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn div(&self, rhs: &Self) -> Self {
        self.zip(&rhs, |a, b| a / b)
    }
    fn div_scalar(&self, scalar: Scalar) -> Self {
        self.map(|a| a / scalar)
    }
}

pub trait VectorFieldDivDefaultZero<Scalar: Div<Output=Scalar> + Copy + Zero>: VectorFieldDiv<Scalar> {
    fn div_default_zero(&self, rhs: &Self, defult_value_for_division_by_zero:Scalar) -> Self {
        self.zip(&rhs, |a, b| if b.is_zero() {defult_value_for_division_by_zero}else{a / b})
    }
}
pub trait VectorFieldMul<Scalar: Mul<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn mul(&self, rhs: &Self) -> Self {
        self.zip(&rhs, |a, b| a * b)
    }
    fn mul_scalar(&self, scalar: Scalar) -> Self {
        self.map(|a| a * scalar)
    }
}

pub trait VectorFieldOne<Scalar: One + Mul<Output=Scalar> + Copy>: VectorFieldMul<Scalar> {
    fn product(&self) -> Scalar {
        self.fold(Scalar::one(), |a, b| a * b)
    }
}

pub trait VectorFieldRem<Scalar: Rem<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn rem(&self, rhs: &Self) -> Self {
        self.zip(&rhs, |a, b| a % b)
    }
    fn rem_scalar(&self, scalar: Scalar) -> Self {
        self.map(|a| a % scalar)
    }
}
pub trait VectorFieldRemDefaultZero<Scalar: Rem<Output=Scalar> + Copy + Zero>: VectorFieldRem<Scalar> {
    fn rem_default_zero(&self, rhs: &Self, default_value_for_division_by_zero:Scalar) -> Self {
        self.zip(&rhs, |a, b| if b.is_zero(){default_value_for_division_by_zero}else{a % b})
    }
}
pub trait VectorFieldRng<Scalar: rand::Fill + Copy>: VectorField<Scalar> {
    fn rand(&mut self, rng: &mut impl rand::Rng);
}


pub trait VectorFieldNum<S: Num + Copy + PartialOrd>: VectorField<S> +
VectorFieldAdd<S> + VectorFieldSub<S> +
VectorFieldMul<S> + VectorFieldDiv<S> +
VectorFieldDivDefaultZero<S> + VectorFieldRemDefaultZero<S> +
VectorFieldPartialOrd<S> + VectorFieldRem<S> +
VectorFieldOne<S> + VectorFieldZero<S> {}

impl<T: Copy, const DIM: usize> VectorField<T> for [T; DIM] {
    fn fold(&self, zero: T, mut f: impl FnMut(T, T) -> T) -> T {
        self.iter().fold(zero, |a, b| f(a, *b))
    }

    fn map(&self, mut f: impl FnMut(T) -> T) -> Self {
        let mut arr: [T; DIM] = unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..DIM {
            arr[i] = f(self[i]);
        }
        arr
    }

    fn new_const(s: T) -> Self {
        [s;DIM]
    }

    fn all(&self, mut f: impl FnMut(T) -> bool) -> bool {
        self.iter().cloned().all(f)
    }
    fn any(&self, mut f: impl FnMut(T) -> bool) -> bool {
        self.iter().cloned().any(f)
    }

    fn all_zip(&self, other: &Self, mut f: impl FnMut(T, T) -> bool) -> bool {
        self.iter().zip(other.iter()).all(|(&a, &b)| f(a, b))
    }

    fn any_zip(&self, other: &Self, mut f: impl FnMut(T, T) -> bool) -> bool {
        self.iter().zip(other.iter()).any(|(&a, &b)| f(a, b))
    }

    fn zip(&self, other: &Self, mut f: impl FnMut(T, T) -> T) -> Self {
        let mut arr: [T; DIM] = unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..DIM {
            arr[i] = f(self[i], other[i]);
        }
        arr
    }
}

impl<T: Copy + Add<Output=T>, const DIM: usize> VectorFieldAdd<T> for [T; DIM] {}

impl<T: Copy + Add<Output=T> + Zero, const DIM: usize> VectorFieldZero<T> for [T; DIM] {}

impl<T: Copy + Sub<Output=T>, const DIM: usize> VectorFieldSub<T> for [T; DIM] {}

impl<T: Copy + Div<Output=T>, const DIM: usize> VectorFieldDiv<T> for [T; DIM] {}

impl<T: Copy + Div<Output=T> + Zero, const DIM: usize> VectorFieldDivDefaultZero<T> for [T; DIM] {}

impl<T: Copy + Rem<Output=T> + Zero, const DIM: usize> VectorFieldRemDefaultZero<T> for [T; DIM] {}

impl<T: Copy + Mul<Output=T>, const DIM: usize> VectorFieldMul<T> for [T; DIM] {}

impl<T: Copy + Mul<Output=T> + One, const DIM: usize> VectorFieldOne<T> for [T; DIM] {}

impl<T: Copy + Rem<Output=T>, const DIM: usize> VectorFieldRem<T> for [T; DIM] {}

impl<T: Copy + PartialOrd, const DIM: usize> VectorFieldPartialOrd<T> for [T; DIM] {}

impl<T: Neg<Output=T> + Zero + PartialOrd + Copy, const DIM: usize> VectorFieldAbs<T> for [T; DIM] {}

impl<T: Neg<Output=T> + Copy, const DIM: usize> VectorFieldNeg<T> for [T; DIM] {}

impl<T: Num + Copy + PartialOrd, const DIM: usize> VectorFieldNum<T> for [T; DIM] {}

pub trait ArrayCast<Scalar: Copy, const DIM: usize> {
    fn as_scalar<IntoScalar: 'static + Copy>(&self) -> [IntoScalar; DIM] where Scalar: AsPrimitive<IntoScalar>;
}

impl<T: Copy, const DIM: usize> ArrayCast<T, DIM> for [T; DIM] {
    fn as_scalar<IntoScalar: 'static + Copy>(&self) -> [IntoScalar; DIM] where T: AsPrimitive<IntoScalar> {
        let mut arr: [IntoScalar; DIM] = unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..DIM {
            arr[i] = self[i].as_();
        }
        arr
    }
}

impl<T: rand::Fill + Copy, const DIM: usize> VectorFieldRng<T> for [T; DIM] where Standard: Distribution<T>{
    fn rand(&mut self, rng: &mut impl rand::Rng){
        for i in 0..DIM{
            self[i] = rng.gen();
        }
    }
}
use std::fmt::{Debug, Formatter};

use serde::{Serialize, Deserialize};
use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, Neg, AddAssign, SubAssign, DivAssign, MulAssign, RemAssign};
use std::mem::MaybeUninit;
use num_traits::{Zero, One, Num, AsPrimitive, NumAssign};
use rand::Rng;
use rand::distributions::{Standard, Distribution};

pub trait VectorField<Scalar: Copy>: Sized {
    fn fold<T>(&self, zero: T, f: impl FnMut(T, Scalar) -> T) -> T;
    fn for_each(&self, mut f: impl FnMut(Scalar)) {
        self.fold((), |(), s| f(s))
    }
    fn for_each_enumerated(&self, mut f: impl FnMut(usize, Scalar)) {
        self.fold(0usize, |i, s| {
            f(i, s);
            i + 1
        });
    }
    fn map(&self, f: impl FnMut(Scalar) -> Scalar) -> Self;
    fn map_assign(&mut self, f: impl FnMut(&mut Scalar)) -> &mut Self;
    fn new_const(s: Scalar) -> Self;
    fn all(&self, f: impl FnMut(Scalar) -> bool) -> bool;
    fn any(&self, f: impl FnMut(Scalar) -> bool) -> bool;
    fn all_zip(&self, other: &Self, f: impl FnMut(Scalar, Scalar) -> bool) -> bool;
    fn any_zip(&self, other: &Self, f: impl FnMut(Scalar, Scalar) -> bool) -> bool;
    fn zip(&self, other: &Self, f: impl FnMut(Scalar, Scalar) -> Scalar) -> Self;
    fn zip_assign(&mut self, other: &Self, f: impl FnMut(&mut Scalar, Scalar)) -> &mut Self;
}

pub trait VectorFieldAdd<Scalar: Add<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn add(&self, rhs: &Self) -> Self {
        self.zip(rhs, |a, b| a + b)
    }
    fn add_scalar(&self, rhs: Scalar) -> Self {
        self.map(|a| a + rhs)
    }
}

pub trait VectorFieldAddAssign<Scalar: AddAssign + Copy>: VectorField<Scalar> {
    fn add_assign(&mut self, rhs: &Self) -> &mut Self {
        self.zip_assign(rhs, |a, b| *a += b)
    }
    fn add_assign_scalar(&mut self, rhs: Scalar) -> &mut Self {
        self.map_assign(|a| *a += rhs)
    }
}

pub trait VectorFieldZero<Scalar: Zero + Add<Output=Scalar> + Copy>: VectorFieldAdd<Scalar> {
    fn sum(&self) -> Scalar {
        self.fold(Scalar::zero(), |a, b| a + b)
    }
}

pub trait VectorFieldPartialOrd<Scalar: PartialOrd + Copy>: VectorField<Scalar> {
    #[inline]
    fn all_le(&self, rhs: &Self) -> bool {
        self.all_zip(rhs, |l, r| l <= r)
    }
    #[inline]
    fn all_le_scalar(&self, rhs: Scalar) -> bool {
        self.all(|l| l <= rhs)
    }
    #[inline]
    fn all_lt(&self, rhs: &Self) -> bool {
        self.all_zip(rhs, |l, r| l < r)
    }
    #[inline]
    fn all_lt_scalar(&self, rhs: Scalar) -> bool {
        self.all(|l| l < rhs)
    }
    #[inline]
    fn all_gt(&self, rhs: &Self) -> bool {
        self.all_zip(rhs, |l, r| l > r)
    }
    #[inline]
    fn all_gt_scalar(&self, rhs: Scalar) -> bool {
        self.all(|l| l > rhs)
    }
    #[inline]
    fn all_ge(&self, rhs: &Self) -> bool {
        self.all_zip(rhs, |l, r| l >= r)
    }
    #[inline]
    fn all_ge_scalar(&self, rhs: Scalar) -> bool {
        self.all(|l| l >= rhs)
    }
    #[inline]
    fn all_eq(&self, rhs: &Self) -> bool {
        self.all_zip(rhs, |l, r| l == r)
    }
    #[inline]
    fn all_eq_scalar(&self, rhs: Scalar) -> bool {
        self.all(|l| l == rhs)
    }
    #[inline]
    fn all_neq(&self, rhs: &Self) -> bool {
        self.all_zip(rhs, |l, r| l != r)
    }
    #[inline]
    fn all_neq_scalar(&self, rhs: Scalar) -> bool {
        self.all(|l| l != rhs)
    }
    #[inline]
    fn max(&self, rhs: &Self) -> Self {
        self.zip(rhs, |l, r| if l > r { l } else { r })
    }
    #[inline]
    fn min(&self, rhs: &Self) -> Self {
        self.zip(rhs, |l, r| if l < r { l } else { r })
    }
    #[inline]
    fn max_assign(&mut self, rhs: &Self) -> &mut Self {
        self.zip_assign(rhs, |l, r| if *l > r { *l = r })
    }
    #[inline]
    fn min_assign(&mut self, rhs: &Self) -> &mut Self {
        self.zip_assign(rhs, |l, r| if r < *l { *l = r })
    }
    #[inline]
    fn max_scalar(&self, rhs: Scalar) -> Self {
        self.map(|l| if l > rhs { l } else { rhs })
    }
    #[inline]
    fn min_scalar(&self, rhs: Scalar) -> Self {
        self.map(|l| if l < rhs { l } else { rhs })
    }
    #[inline]
    fn any_le(&self, rhs: &Self) -> bool {
        self.any_zip(rhs, |l, r| l <= r)
    }
    #[inline]
    fn any_le_scalar(&self, rhs: Scalar) -> bool { self.any(|l| l <= rhs) }
    #[inline]
    fn any_lt(&self, rhs: &Self) -> bool { self.any_zip(rhs, |l, r| l < r) }
    #[inline]
    fn any_lt_scalar(&self, rhs: Scalar) -> bool { self.any(|l| l < rhs) }
    #[inline]
    fn any_gt(&self, rhs: &Self) -> bool { self.any_zip(rhs, |l, r| l > r) }
    #[inline]
    fn any_gt_scalar(&self, rhs: Scalar) -> bool { self.any(|l| l > rhs) }
    #[inline]
    fn any_ge(&self, rhs: &Self) -> bool { self.any_zip(rhs, |l, r| l >= r) }
    #[inline]
    fn any_ge_scalar(&self, rhs: Scalar) -> bool { self.any(|l| l >= rhs) }
    #[inline]
    fn any_eq(&self, rhs: &Self) -> bool { self.any_zip(rhs, |l, r| l == r) }
    #[inline]
    fn any_eq_scalar(&self, rhs: Scalar) -> bool { self.any(|l| l == rhs) }
    #[inline]
    fn any_neq(&self, rhs: &Self) -> bool { self.any_zip(rhs, |l, r| l != r) }
    #[inline]
    fn any_neq_scalar(&self, rhs: Scalar) -> bool { self.any(|l| l != rhs) }
    fn find_gt(&self, scalar: Scalar, destination: &mut CpuSDR) {
        self.for_each_enumerated(|i,s|if s>scalar{destination.push(as_idx(i))})
    }
    fn find_lt(&self, scalar: Scalar, destination: &mut CpuSDR) {
        self.for_each_enumerated(|i,s|if s<scalar{destination.push(as_idx(i))})
    }
    fn find_ge(&self, scalar: Scalar, destination: &mut CpuSDR) {
        self.for_each_enumerated(|i,s|if s>=scalar{destination.push(as_idx(i))})
    }
    fn find_le(&self, scalar: Scalar, destination: &mut CpuSDR) {
        self.for_each_enumerated(|i,s|if s<=scalar{destination.push(as_idx(i))})
    }
    fn find_eq(&self, scalar: Scalar, destination: &mut CpuSDR) {
        self.for_each_enumerated(|i,s|if s==scalar{destination.push(as_idx(i))})
    }
    fn find_neq(&self, scalar: Scalar, destination: &mut CpuSDR) {
        self.for_each_enumerated(|i,s|if s!=scalar{destination.push(as_idx(i))})
    }
}

pub trait VectorFieldAbs<Scalar: Neg<Output=Scalar> + Zero + PartialOrd + Copy>: VectorField<Scalar> {
    fn abs(&self) -> Self {
        self.map(|b| if b < Scalar::zero() { -b } else { b })
    }
    fn abs_assign(&mut self) -> &mut Self {
        self.map_assign(|b| if *b < Scalar::zero() { *b = -*b })
    }
}

pub trait VectorFieldNeg<Scalar: Neg<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn neg(&self) -> Self {
        self.map(|b| -b)
    }
    fn neg_assign(&mut self) -> &mut Self {
        self.map_assign(|b| *b = -*b)
    }
}

pub trait VectorFieldSub<Scalar: Sub<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn sub(&self, rhs: &Self) -> Self {
        self.zip(rhs, |a, b| a - b)
    }
    fn sub_scalar(&self, scalar: Scalar) -> Self {
        self.map(|a| a - scalar)
    }
}

pub trait VectorFieldSubAssign<Scalar: SubAssign + Copy>: VectorField<Scalar> {
    fn sub_assign(&mut self, rhs: &Self) -> &mut Self {
        self.zip_assign(rhs, |a, b| *a -= b)
    }
    fn sub_assign_scalar(&mut self, scalar: Scalar) -> &mut Self {
        self.map_assign(|a| *a -= scalar)
    }
}

pub trait VectorFieldDiv<Scalar: Div<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn div(&self, rhs: &Self) -> Self {
        self.zip(rhs, |a, b| a / b)
    }
    fn div_scalar(&self, scalar: Scalar) -> Self {
        self.map(|a| a / scalar)
    }
}

pub trait VectorFieldDivDefaultZero<Scalar: Div<Output=Scalar> + Copy + Zero>: VectorFieldDiv<Scalar> {
    fn div_default_zero(&self, rhs: &Self, default_value_for_division_by_zero: Scalar) -> Self {
        self.zip(&rhs, |a, b| if b.is_zero() { default_value_for_division_by_zero } else { a / b })
    }
}

pub trait VectorFieldDivAssign<Scalar: DivAssign + Copy>: VectorField<Scalar> {
    fn div_assign(&mut self, rhs: &Self) -> &mut Self {
        self.zip_assign(rhs, |a, b| *a /= b)
    }
    fn div_assign_scalar(&mut self, scalar: Scalar) -> &mut Self {
        self.map_assign(|a| *a /= scalar)
    }
}

pub trait VectorFieldDivAssignDefaultZero<Scalar: DivAssign + Copy + Zero>: VectorFieldDivAssign<Scalar> {
    fn div_assign_default_zero(&mut self, rhs: &Self, default_value_for_division_by_zero: Scalar) -> &mut Self {
        self.zip_assign(&rhs, |a, b| if b.is_zero() { *a = default_value_for_division_by_zero } else { *a /= b })
    }
}

pub trait VectorFieldMul<Scalar: Mul<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn mul(&self, rhs: &Self) -> Self {
        self.zip(rhs, |a, b| a * b)
    }
    fn mul_scalar(&self, scalar: Scalar) -> Self {
        self.map(|a| a * scalar)
    }
}

pub trait VectorFieldMulAssign<Scalar: MulAssign + Copy>: VectorField<Scalar> {
    fn mul_assign(&mut self, rhs: &Self) -> &mut Self {
        self.zip_assign(rhs, |a, b| *a *= b)
    }
    fn mul_scalar_assign(&mut self, scalar: Scalar) -> &mut Self {
        self.map_assign(|a| *a *= scalar)
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

pub trait VectorFieldRemAssign<Scalar: RemAssign + Copy>: VectorField<Scalar> {
    fn rem_assign(&mut self, rhs: &Self) -> &mut Self {
        self.zip_assign(rhs, |a, b| *a %= b)
    }
    fn rem_assign_scalar(&mut self, scalar: Scalar) -> &mut Self {
        self.map_assign(|a| *a %= scalar)
    }
}

pub trait VectorFieldRemDefaultZero<Scalar: Rem<Output=Scalar> + Copy + Zero>: VectorFieldRem<Scalar> {
    fn rem_default_zero(&self, rhs: &Self, default_value_for_division_by_zero: Scalar) -> Self {
        self.zip(&rhs, |a, b| if b.is_zero() { default_value_for_division_by_zero } else { a % b })
    }
}

pub trait VectorFieldRemAssignDefaultZero<Scalar: RemAssign + Copy + Zero>: VectorFieldRemAssign<Scalar> {
    fn rem_assign_default_zero(&mut self, rhs: &Self, default_value_for_division_by_zero: Scalar) -> &mut Self {
        self.zip_assign(rhs, |a, b| if b.is_zero() { *a = default_value_for_division_by_zero } else { *a %= b })
    }
}

pub trait VectorFieldRng<Scalar: Copy>: VectorField<Scalar> {
    fn rand_vec(&self, rng: &mut impl rand::Rng) -> Self;
}

pub trait VectorFieldRngAssign<Scalar: Copy>: VectorField<Scalar> {
    fn rand_vec_assign(&mut self, rng: &mut impl rand::Rng)->&mut Self;
}

pub trait VectorFieldNum<S: Num + Copy + PartialOrd>: VectorField<S> +
VectorFieldAdd<S> + VectorFieldSub<S> +
VectorFieldMul<S> + VectorFieldDiv<S> +
VectorFieldDivDefaultZero<S> + VectorFieldRemDefaultZero<S> +
VectorFieldPartialOrd<S> + VectorFieldRem<S> +
VectorFieldOne<S> + VectorFieldZero<S> {}

pub trait VectorFieldNumAssign<S: NumAssign + Copy + PartialOrd>: VectorFieldNum<S> +
VectorFieldAddAssign<S> + VectorFieldSubAssign<S> +
VectorFieldMulAssign<S> + VectorFieldDivAssign<S> +
VectorFieldDivAssignDefaultZero<S> + VectorFieldRemAssignDefaultZero<S> +
VectorFieldRemAssign<S> {}


pub trait ArrayCast<Scalar: Copy, const DIM: usize> {
    fn as_scalar<IntoScalar: 'static + Copy>(&self) -> [IntoScalar; DIM] where Scalar: AsPrimitive<IntoScalar>;
}

pub trait VecCast<Scalar: Copy> {
    fn as_scalar<IntoScalar: 'static + Copy>(&self) -> Vec<IntoScalar> where Scalar: AsPrimitive<IntoScalar>;
}

use std::fmt::{Debug, Formatter};

use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, Neg, AddAssign, SubAssign, DivAssign, MulAssign, RemAssign};
use std::mem::MaybeUninit;
use num_traits::{Zero, One, Num, AsPrimitive, NumAssign};
use rand::Rng;
use rand::distributions::{Standard, Distribution};

pub trait VectorField<Scalar: Copy> {
    fn fold<T>(&self, zero: T, f: impl FnMut(T, Scalar) -> T) -> T;
    fn rfold<T>(&self, zero: T, f: impl FnMut(T, Scalar) -> T) -> T;
    fn fold_<T>(&mut self, zero: T, f: impl FnMut(T, &mut Scalar) -> T) -> T;
    fn rfold_<T>(&mut self, zero: T, f: impl FnMut(T, &mut Scalar) -> T) -> T;

    fn for_each(&self, mut f: impl FnMut(Scalar)) {
        self.fold((), |(), s| f(s))
    }
    fn for_each_enumerated(&self, mut f: impl FnMut(usize, Scalar)) {
        self.fold(0usize, |i, s| {
            f(i, s);
            i + 1
        });
    }
    fn map_(&mut self, f: impl FnMut(&mut Scalar)) -> &mut Self;
    fn all(&self, f: impl FnMut(Scalar) -> bool) -> bool;
    fn any(&self, f: impl FnMut(Scalar) -> bool) -> bool;
    fn all_zip(&self, other: &Self, f: impl FnMut(Scalar, Scalar) -> bool) -> bool;
    fn any_zip(&self, other: &Self, f: impl FnMut(Scalar, Scalar) -> bool) -> bool;
    fn zip_(&mut self, other: &Self, f: impl FnMut(&mut Scalar, Scalar)) -> &mut Self;
    type O;
    fn map(&self, f: impl FnMut(Scalar) -> Scalar) -> Self::O;
    fn zip(&self, other: &Self, f: impl FnMut(Scalar, Scalar) -> Scalar) -> Self::O;
    fn fold_map<T>(&self, zero: T, f: impl FnMut(T, Scalar) -> (T, Scalar)) -> (T, Self::O);
    fn rfold_map<T>(&self, zero: T, f: impl FnMut(T, Scalar) -> (T, Scalar)) -> (T, Self::O);
}

pub trait VectorFieldOwned<Scalar: Copy>: Sized {
    fn _map(self, f: impl FnMut(Scalar) -> Scalar) -> Self;
    fn _zip(self, other: &Self, f: impl FnMut(Scalar, Scalar) -> Scalar) -> Self;
}

pub trait VectorFieldAdd<Scalar: Add<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn add(&self, rhs: &Self) -> Self::O {
        self.zip(rhs, |a, b| a + b)
    }
    fn add_scalar(&self, rhs: Scalar) -> Self::O {
        self.map(|a| a + rhs)
    }
}

pub trait VectorFieldAddOwned<Scalar: Add<Output=Scalar> + Copy>: VectorFieldOwned<Scalar> {
    fn _add(self, rhs: &Self) -> Self {
        self._zip(rhs, |a, b| a + b)
    }
    fn _add_scalar(self, rhs: Scalar) -> Self {
        self._map(|a| a + rhs)
    }
}

pub trait VectorFieldAddAssign<Scalar: AddAssign + Copy>: VectorField<Scalar> {
    fn add_(&mut self, rhs: &Self) -> &mut Self {
        self.zip_(rhs, |a, b| *a += b)
    }
    fn add_scalar_(&mut self, rhs: Scalar) -> &mut Self {
        self.map_(|a| *a += rhs)
    }
}

pub trait VectorFieldZero<Scalar: Zero + Add<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn sum(&self) -> Scalar {
        self.fold(Scalar::zero(), |a, b| a + b)
    }
}

pub trait VectorFieldPartialOrdOwned<Scalar: PartialOrd + Copy>: VectorFieldOwned<Scalar> {
    #[inline]
    fn _max(self, rhs: &Self) -> Self {
        self._zip(rhs, |l, r| if l > r { l } else { r })
    }
    #[inline]
    fn _min(self, rhs: &Self) -> Self {
        self._zip(rhs, |l, r| if l < r { l } else { r })
    }
    #[inline]
    fn _max_scalar(self, rhs: Scalar) -> Self {
        self._map(|l| if l > rhs { l } else { rhs })
    }
    #[inline]
    fn _min_scalar(self, rhs: Scalar) -> Self {
        self._map(|l| if l < rhs { l } else { rhs })
    }
}

pub trait VectorFieldPartialOrd<Scalar: PartialOrd + Copy>: VectorField<Scalar> {
    #[inline]
    fn max(&self, rhs: &Self) -> Self::O {
        self.zip(rhs, |l, r| if l > r { l } else { r })
    }
    #[inline]
    fn min(&self, rhs: &Self) -> Self::O {
        self.zip(rhs, |l, r| if l < r { l } else { r })
    }
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
    fn max_(&mut self, rhs: &Self) -> &mut Self {
        self.zip_(rhs, |l, r| if *l > r { *l = r })
    }
    #[inline]
    fn min_(&mut self, rhs: &Self) -> &mut Self {
        self.zip_(rhs, |l, r| if r < *l { *l = r })
    }
    #[inline]
    fn max_scalar(&self, rhs: Scalar) -> Self::O {
        self.map(|l| if l > rhs { l } else { rhs })
    }
    #[inline]
    fn min_scalar(&self, rhs: Scalar) -> Self::O {
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
    fn find_gt(&self, scalar: Scalar, destination: &mut Vec<usize>) {
        self.for_each_enumerated(|i, s| if s > scalar { destination.push(i) })
    }
    fn find_lt(&self, scalar: Scalar, destination: &mut Vec<usize>) {
        self.for_each_enumerated(|i, s| if s < scalar { destination.push(i) })
    }
    fn find_ge(&self, scalar: Scalar, destination: &mut Vec<usize>) {
        self.for_each_enumerated(|i, s| if s >= scalar { destination.push(i) })
    }
    fn find_le(&self, scalar: Scalar, destination: &mut Vec<usize>) {
        self.for_each_enumerated(|i, s| if s <= scalar { destination.push(i) })
    }
    fn find_eq(&self, scalar: Scalar, destination: &mut Vec<usize>) {
        self.for_each_enumerated(|i, s| if s == scalar { destination.push(i) })
    }
    fn find_neq(&self, scalar: Scalar, destination: &mut Vec<usize>) {
        self.for_each_enumerated(|i, s| if s != scalar { destination.push(i) })
    }
}

pub trait VectorFieldAbsOwned<Scalar: Neg<Output=Scalar> + Zero + PartialOrd + Copy>: VectorFieldOwned<Scalar> {
    fn _abs(self) -> Self {
        self._map(|b| if b < Scalar::zero() { -b } else { b })
    }
}

pub trait VectorFieldAbs<Scalar: Neg<Output=Scalar> + Zero + PartialOrd + Copy>: VectorField<Scalar> {
    fn abs(&self) -> Self::O {
        self.map(|b| if b < Scalar::zero() { -b } else { b })
    }
    fn abs_(&mut self) -> &mut Self {
        self.map_(|b| if *b < Scalar::zero() { *b = -*b })
    }
}

pub trait VectorFieldNegOwned<Scalar: Neg<Output=Scalar> + Copy>: VectorFieldOwned<Scalar> {
    fn _neg(self) -> Self {
        self._map(|b| -b)
    }
}

pub trait VectorFieldNeg<Scalar: Neg<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn neg(&self) -> Self::O {
        self.map(|b| -b)
    }
    fn neg_(&mut self) -> &mut Self {
        self.map_(|b| *b = -*b)
    }
}

pub trait VectorFieldSubOwned<Scalar: Sub<Output=Scalar> + Copy>: VectorFieldOwned<Scalar> {
    fn _sub(self, rhs: &Self) -> Self {
        self._zip(rhs, |a, b| a - b)
    }
    fn _sub_scalar(self, scalar: Scalar) -> Self {
        self._map(|a| a - scalar)
    }
}

pub trait VectorFieldSub<Scalar: Sub<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn sub(&self, rhs: &Self) -> Self::O {
        self.zip(rhs, |a, b| a - b)
    }
    fn sub_scalar(&self, scalar: Scalar) -> Self::O {
        self.map(|a| a - scalar)
    }
}

pub trait VectorFieldSubAssign<Scalar: SubAssign + Copy>: VectorField<Scalar> {
    fn sub_(&mut self, rhs: &Self) -> &mut Self {
        self.zip_(rhs, |a, b| *a -= b)
    }
    fn sub_scalar_(&mut self, scalar: Scalar) -> &mut Self {
        self.map_(|a| *a -= scalar)
    }
}

pub trait VectorFieldDivOwned<Scalar: Div<Output=Scalar> + Copy>: VectorFieldOwned<Scalar> {
    fn _div(self, rhs: &Self) -> Self {
        self._zip(rhs, |a, b| a / b)
    }
    fn _div_scalar(self, scalar: Scalar) -> Self {
        self._map(|a| a / scalar)
    }
}

pub trait VectorFieldDiv<Scalar: Div<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn div(&self, rhs: &Self) -> Self::O {
        self.zip(rhs, |a, b| a / b)
    }
    fn div_scalar(&self, scalar: Scalar) -> Self::O {
        self.map(|a| a / scalar)
    }
}

pub trait VectorFieldDivDefaultZeroOwned<Scalar: Div<Output=Scalar> + Copy + Zero>: VectorFieldDivOwned<Scalar> {
    fn _div_default_zero(self, rhs: &Self, default_value_for_division_by_zero: Scalar) -> Self {
        self._zip(&rhs, |a, b| if b.is_zero() { default_value_for_division_by_zero } else { a / b })
    }
}

pub trait VectorFieldDivDefaultZero<Scalar: Div<Output=Scalar> + Copy + Zero>: VectorFieldDiv<Scalar> {
    fn div_default_zero(&self, rhs: &Self, default_value_for_division_by_zero: Scalar) -> Self::O {
        self.zip(&rhs, |a, b| if b.is_zero() { default_value_for_division_by_zero } else { a / b })
    }
}

pub trait VectorFieldDivAssign<Scalar: DivAssign + Copy>: VectorField<Scalar> {
    fn div_(&mut self, rhs: &Self) -> &mut Self {
        self.zip_(rhs, |a, b| *a /= b)
    }
    fn div_scalar_(&mut self, scalar: Scalar) -> &mut Self {
        self.map_(|a| *a /= scalar)
    }
}

pub trait VectorFieldDivAssignDefaultZero<Scalar: DivAssign + Copy + Zero>: VectorFieldDivAssign<Scalar> {
    fn div_default_zero_(&mut self, rhs: &Self, default_value_for_division_by_zero: Scalar) -> &mut Self {
        self.zip_(&rhs, |a, b| if b.is_zero() { *a = default_value_for_division_by_zero } else { *a /= b })
    }
}

pub trait VectorFieldMulOwned<Scalar: Mul<Output=Scalar> + Copy>: VectorFieldOwned<Scalar> {
    fn _mul(self, rhs: &Self) -> Self {
        self._zip(rhs, |a, b| a * b)
    }
    fn _mul_scalar(self, scalar: Scalar) -> Self {
        self._map(|a| a * scalar)
    }
}

pub trait VectorFieldMul<Scalar: Mul<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn mul(&self, rhs: &Self) -> Self::O {
        self.zip(rhs, |a, b| a * b)
    }
    fn mul_scalar(&self, scalar: Scalar) -> Self::O {
        self.map(|a| a * scalar)
    }
}

pub trait VectorFieldMulAssign<Scalar: MulAssign + Copy>: VectorField<Scalar> {
    fn mul_(&mut self, rhs: &Self) -> &mut Self {
        self.zip_(rhs, |a, b| *a *= b)
    }
    fn mul_scalar_(&mut self, scalar: Scalar) -> &mut Self {
        self.map_(|a| *a *= scalar)
    }
}

pub trait VectorFieldOne<Scalar: One + Mul<Output=Scalar> + Copy>: VectorFieldMul<Scalar> {
    fn product(&self) -> Scalar {
        self.fold(Scalar::one(), |a, b| a * b)
    }
}

pub trait VectorFieldRemOwned<Scalar: Rem<Output=Scalar> + Copy>: VectorFieldOwned<Scalar> {
    fn _rem(self, rhs: &Self) -> Self {
        self._zip(&rhs, |a, b| a % b)
    }
    fn _rem_scalar(self, scalar: Scalar) -> Self {
        self._map(|a| a % scalar)
    }
}

pub trait VectorFieldRem<Scalar: Rem<Output=Scalar> + Copy>: VectorField<Scalar> {
    fn rem(&self, rhs: &Self) -> Self::O {
        self.zip(&rhs, |a, b| a % b)
    }
    fn rem_scalar(&self, scalar: Scalar) -> Self::O {
        self.map(|a| a % scalar)
    }
}

pub trait VectorFieldRemAssign<Scalar: RemAssign + Copy>: VectorField<Scalar> {
    fn rem_(&mut self, rhs: &Self) -> &mut Self {
        self.zip_(rhs, |a, b| *a %= b)
    }
    fn rem_scalar_(&mut self, scalar: Scalar) -> &mut Self {
        self.map_(|a| *a %= scalar)
    }
}

pub trait VectorFieldRemDefaultZeroOwned<Scalar: Rem<Output=Scalar> + Copy + Zero>: VectorFieldRemOwned<Scalar> {
    fn _rem_default_zero(self, rhs: &Self, default_value_for_division_by_zero: Scalar) -> Self {
        self._zip(rhs, |a, b| if b.is_zero() { default_value_for_division_by_zero } else { a % b })
    }
}

pub trait VectorFieldRemDefaultZero<Scalar: Rem<Output=Scalar> + Copy + Zero>: VectorFieldRem<Scalar> {
    fn rem_default_zero(&self, rhs: &Self, default_value_for_division_by_zero: Scalar) -> Self::O {
        self.zip(rhs, |a, b| if b.is_zero() { default_value_for_division_by_zero } else { a % b })
    }
}

pub trait VectorFieldRemAssignDefaultZero<Scalar: RemAssign + Copy + Zero>: VectorFieldRemAssign<Scalar> {
    fn rem_default_zero_(&mut self, rhs: &Self, default_value_for_division_by_zero: Scalar) -> &mut Self {
        self.zip_(rhs, |a, b| if b.is_zero() { *a = default_value_for_division_by_zero } else { *a %= b })
    }
}


pub trait VectorFieldRngOwned<Scalar: Copy>: VectorFieldOwned<Scalar> {
    fn _rand(self, rng: &mut impl rand::Rng) -> Self;
}

pub trait VectorFieldRngAssign<Scalar: Copy>: VectorField<Scalar> {
    fn rand_(&mut self, rng: &mut impl rand::Rng) -> &mut Self;
}

pub trait VectorFieldNum<S: Num + Copy + PartialOrd>: VectorField<S> +
VectorFieldAdd<S> + VectorFieldSub<S> +
VectorFieldMul<S> + VectorFieldDiv<S> +
VectorFieldDivDefaultZero<S> + VectorFieldRemDefaultZero<S> +
VectorFieldPartialOrd<S> + VectorFieldRem<S> +
VectorFieldOne<S> + VectorFieldZero<S> {}


pub trait VectorFieldNumOwned<S: Num + Copy + PartialOrd>: VectorFieldOwned<S> +
VectorFieldAddOwned<S> + VectorFieldSubOwned<S> +
VectorFieldMulOwned<S> + VectorFieldDivOwned<S> +
VectorFieldDivDefaultZeroOwned<S> + VectorFieldRemDefaultZeroOwned<S> +
VectorFieldPartialOrdOwned<S> + VectorFieldRemOwned<S> {}

pub trait VectorFieldNumAssign<S: NumAssign + Copy + PartialOrd>: VectorFieldNum<S> +
VectorFieldAddAssign<S> + VectorFieldSubAssign<S> +
VectorFieldMulAssign<S> + VectorFieldDivAssign<S> +
VectorFieldDivAssignDefaultZero<S> + VectorFieldRemAssignDefaultZero<S> +
VectorFieldRemAssign<S> {}



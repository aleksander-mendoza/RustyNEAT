use crate::vector_field::*;
use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, Neg, AddAssign, SubAssign, DivAssign, MulAssign, RemAssign};
use std::mem::MaybeUninit;
use num_traits::{Zero, One, Num, AsPrimitive, NumAssign};
use rand::Rng;
use rand::distributions::{Standard, Distribution};

impl<T: Copy, const DIM: usize> VectorField<T> for [T; DIM] {
    #[inline]
    fn fold<I>(&self, zero: I, mut f: impl FnMut(I, T) -> I) -> I {
        self.iter().fold(zero, |a, b| f(a, *b))
    }
    #[inline]
    fn map(&self, mut f: impl FnMut(T) -> T) -> Self {
        let mut arr: [T; DIM] = unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..DIM {
            arr[i] = f(self[i]);
        }
        arr
    }

    fn map_assign(&mut self, f: impl FnMut(&mut T)) -> &mut Self {
        self.iter_mut().for_each(f);
        self
    }

    #[inline]
    fn new_const(s: T) -> Self {
        [s;DIM]
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
        let mut arr: [T; DIM] = unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..DIM {
            arr[i] = f(self[i], other[i]);
        }
        arr
    }

    fn zip_assign(&mut self, other: &Self, mut f: impl FnMut(&mut T, T)) -> &mut Self {
        self.iter_mut().zip(other.iter().cloned()).for_each(|(a,b)|f(a,b));
        self
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

impl<T: Copy + AddAssign, const DIM: usize> VectorFieldAddAssign<T> for [T; DIM] {}

impl<T: Copy + SubAssign, const DIM: usize> VectorFieldSubAssign<T> for [T; DIM] {}

impl<T: Copy + DivAssign, const DIM: usize> VectorFieldDivAssign<T> for [T; DIM] {}

impl<T: Copy + DivAssign + Zero, const DIM: usize> VectorFieldDivAssignDefaultZero<T> for [T; DIM] {}

impl<T: Copy + RemAssign + Zero, const DIM: usize> VectorFieldRemAssignDefaultZero<T> for [T; DIM] {}

impl<T: Copy + MulAssign, const DIM: usize> VectorFieldMulAssign<T> for [T; DIM] {}

impl<T: Copy + RemAssign, const DIM: usize> VectorFieldRemAssign<T> for [T; DIM] {}

impl<T: Num + Copy + PartialOrd, const DIM: usize> VectorFieldNum<T> for [T; DIM] {}

impl<T: NumAssign + Copy + PartialOrd, const DIM: usize> VectorFieldNumAssign<T> for [T; DIM] {}

impl<T: Copy, const DIM: usize> ArrayCast<T, DIM> for [T; DIM] {
    fn as_scalar<IntoScalar: 'static + Copy>(&self) -> [IntoScalar; DIM] where T: AsPrimitive<IntoScalar> {
        let mut arr: [IntoScalar; DIM] = unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..DIM {
            arr[i] = self[i].as_();
        }
        arr
    }
}

impl<T: Copy + Rem<Output=T>, const DIM: usize> VectorFieldRng<T> for [T; DIM] where Standard: Distribution<T>{
    fn rand_vec(&self, rng: &mut impl rand::Rng)->Self{
        let mut s:[T;DIM] = unsafe{MaybeUninit::uninit().assume_init()};
        for i in 0..DIM{
            s[i] = rng.gen::<T>()%self[i];
        }
        s
    }
}

impl<T: Copy + Rem<Output=T>, const DIM: usize> VectorFieldRngAssign<T> for [T; DIM] where Standard: Distribution<T>{
    fn rand_vec_assign(&mut self, rng: &mut impl rand::Rng)->&mut Self{
        self.iter_mut().map(|i|*i = rng.gen::<T>() % *i);
        self
    }
}
#[cfg(test)]
mod tests {
    use crate::VectorFieldRng;
    use rand::thread_rng;

    #[test]
    fn test1() {
        let a = [3,3];
        let mut c = [[0;2];3];
        let mut rng = thread_rng();
        for i in 0..64{
            let b = a.rand_vec(&mut rng);
            c[b[0]][0] += 1;
            c[b[1]][1] += 1;
        }
        for c in c.iter() {
            for &c in c.iter() {
                assert!(c>0);
            }
        }
    }

}
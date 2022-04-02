use crate::num::Num;
use std::fmt::{Display, Formatter};

pub struct Shape<'a, T: Num>(&'a [T]);

pub trait AsShape<T: Num> {
    fn as_shape(&self) -> Shape<T>;
}

impl<T: Num, const S: usize> AsShape<T> for [T; S] {
    fn as_shape(&self) -> Shape<T> {
        Shape(self)
    }
}


impl<T: Num> AsShape<T> for Box<[T]> {
    fn as_shape(&self) -> Shape<T> {
        Shape(self)
    }
}

impl<T: Num> AsShape<T> for &[T] {
    fn as_shape(&self) -> Shape<T> {
        Shape(self)
    }
}

impl<T: Num> AsShape<T> for Vec<T> {
    fn as_shape(&self) -> Shape<T> {
        Shape(self.as_slice())
    }
}

impl<'a, T: Num> Shape<'a, T> {
    pub(crate) fn size(&self) -> T {
        if self.0.is_empty() {
            T::zero()
        } else {
            self.0.iter().cloned().product()
        }
    }
}


impl<'a, T: Num> Display for Shape<'a, T> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "(")?;
        if !self.0.is_empty() {
            write!(fmt, "{}", self.0[0])?;
            for &i in &self.0[1..] {
                write!(fmt, ", {}", i)?;
            }
        }
        write!(fmt, ")")
    }
}
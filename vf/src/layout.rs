use std::ops::Range;
use crate::init::{empty, init_fold_rev};
use crate::{VectorField, VectorFieldOne};
use crate::shape::Shape;


pub trait BorrowAsLayout {
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn contiguous(&self) -> bool {
        let mut i = 1;
        for (dim, stride) in self.shape().iter().cloned().zip(self.strides().iter().cloned()).rev() {
            if stride != i {
                return false;
            }
            i *= dim;
        }
        true
    }

    fn offset<const DIM: usize>(&self, index: &[usize]) -> usize {
        if index.len() == self.ndim() && index.iter().zip(self.shape().iter()).all(|(&a, &b)| a < b) {
            index.iter().zip(self.strides().iter()).map(|(a, b)| a * b).sum()
        } else {
            panic!("Invalid index {:?} into shape {:?}", index, self.shape())
        }
    }

    fn end_offset(&self) -> usize {
        self.shape().iter().zip(self.strides().iter()).map(|(&len, &stride)| (len - 1) * stride).sum()
    }

    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn size(&self) -> usize {
        self.shape().size()
    }

    /**Length of first dimension*/
    fn len(&self) -> usize {
        self.shape().first().cloned().unwrap_or(0usize)
    }
}


pub trait BorrowAsMutLayout: BorrowAsLayout {
    fn shape_mut(&mut self) -> &mut [usize];
    fn strides_mut(&mut self) -> &mut [usize];
    fn transpose_(&mut self, dim0: usize, dim1: usize) -> &mut Self {
        self.shape_mut().swap(dim0, dim1);
        self.shape_mut().swap(dim0, dim1);
        self
    }
    fn _transpose(mut self, dim0: usize, dim1: usize) -> Self where Self: Sized {
        self.transpose_(dim0, dim1);
        self
    }
}

pub trait CloneAsLayout {
    type C: BorrowAsMutLayout;
    fn clone_layout(&self) -> Self::C;

    fn transpose(&self, dim0: usize, dim1: usize) -> Self::C {
        self.clone_layout()._transpose(dim0, dim1)
    }
}


impl<const DIM: usize> BorrowAsLayout for ([usize; DIM], [usize; DIM]) {
    fn shape(&self) -> &[usize] {
        &self.0
    }

    fn strides(&self) -> &[usize] {
        &self.1
    }
}

impl<const DIM: usize> BorrowAsMutLayout for ([usize; DIM], [usize; DIM]) {
    fn shape_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }

    fn strides_mut(&mut self) -> &mut [usize] {
        &mut self.1
    }
}


impl BorrowAsLayout for (Vec<usize>, Vec<usize>) {
    fn shape(&self) -> &[usize] {
        &self.0
    }

    fn strides(&self) -> &[usize] {
        &self.1
    }
}

impl BorrowAsMutLayout for (Vec<usize>, Vec<usize>) {
    fn shape_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }

    fn strides_mut(&mut self) -> &mut [usize] {
        &mut self.1
    }
}


impl BorrowAsLayout for (&[usize], &[usize]) {
    fn shape(&self) -> &[usize] {
        self.0
    }

    fn strides(&self) -> &[usize] {
        self.1
    }
}

impl BorrowAsLayout for (&mut [usize], &mut [usize]) {
    fn shape(&self) -> &[usize] {
        self.0
    }

    fn strides(&self) -> &[usize] {
        self.1
    }
}

impl BorrowAsMutLayout for (&mut [usize], &mut [usize]) {
    fn shape_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }

    fn strides_mut(&mut self) -> &mut [usize] {
        &mut self.1
    }
}


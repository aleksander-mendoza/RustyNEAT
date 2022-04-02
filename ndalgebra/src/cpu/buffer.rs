//! Interfaces with a buffer.

use std;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Range};
use std::any::TypeId;
use std::mem::MaybeUninit;
use std::rc::Rc;
use std::cell::RefCell;

pub trait Buffer<T:Copy> {

    fn as_mut_slice(&mut self) -> &mut [T];
    fn as_slice(&self) -> &[T];
    fn len(&self) -> usize{
        self.as_slice().len()
    }
    fn fill(&mut self, fill_val: T) {
        self.as_mut_slice().fill(fill_val)
    }

    fn get(&self, index: usize) -> T {
        self.as_slice()[index]
    }
    fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }


}

pub fn empty<T:Copy>(len: usize) -> Box<[T]> {
    unsafe { Box::new_uninit_slice(len).assume_init() }
}
impl <T:Copy> Buffer<T> for Box<[T]>{
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut()
    }

    fn as_slice(&self) -> &[T] {
        self.as_ref()
    }
}
impl <T:Copy> Buffer<T> for &[T]{
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    fn as_slice(&self) -> &[T] {
        self
    }
}


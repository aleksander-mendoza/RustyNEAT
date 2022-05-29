use std::mem::MaybeUninit;

pub trait InitEmpty{
    fn empty()->Self;
}
pub trait InitEmptyWithCapacity{
    fn empty(capacity:usize)->Self;
}
pub trait InitWith<T>{
    fn init_with(f:impl FnMut(usize)->T)->Self;
}
pub trait InitWithCapacity<T>{
    fn init_with(capacity:usize, f:impl FnMut(usize)->T)->Self;
}
pub trait InitFold<T:Copy>{
    fn init_fold(start:T, f:impl FnMut(T,usize)->T)->Self;
}
pub trait InitFoldWithCapacity<T:Copy>{
    fn init_fold(capacity:usize, start:T, f:impl FnMut(T, usize)->T)->Self;
}
pub trait InitFoldRev<T:Copy>{
    fn init_fold_rev(end:T, f:impl FnMut(T,usize)->T)->Self;
}
pub trait InitFoldRevWithCapacity<T:Copy>{
    fn init_fold_rev(capacity:usize, end:T, f:impl FnMut(T, usize)->T)->Self;
}
impl <T:Copy,const DIM:usize> InitEmpty for [T;DIM]{
    fn empty() -> Self {
        empty()
    }
}
impl <T:Copy> InitEmptyWithCapacity for Vec<T>{
    fn empty(capacity: usize) -> Self {
        let mut v = Vec::with_capacity(capacity);
        unsafe{v.set_len(capacity)}
        v
    }
}
impl <T:Copy,const DIM:usize> InitWith<T> for [T;DIM]{
    fn init_with(f:impl FnMut(usize)->T)->Self{
        let mut e = Self::empty();
        e.iter_mut().enumerate().for_each(|(i,e)|*e=f(i));
        e
    }
}
impl <T> InitWithCapacity<T> for Vec<T>{
    fn init_with(capacity: usize, f: impl FnMut(usize) -> T) -> Self {
        (0..capacity).map(f).collect()
    }
}
impl <T:Copy,const DIM:usize> InitFold<T> for [T;DIM]{
    fn init_fold(start: T, f: impl FnMut(T, usize) -> T) -> Self {
        init_fold(start,f)
    }
}
impl <T:Copy> InitFoldWithCapacity<T> for Vec<T>{
    fn init_fold(capacity: usize, mut start: T, f: impl FnMut(T, usize) -> T) -> Self {
        let mut arr = Self::empty(capacity);
        arr.fill_fold(start,f);
        arr
    }
}
impl <T:Copy,const DIM:usize> InitFoldRev<T> for [T;DIM]{
    fn init_fold_rev(mut end: T, f: impl FnMut(T, usize) -> T) -> Self {
        init_fold_rev(end,f)
    }
}
impl <T:Copy> InitFoldRevWithCapacity<T> for Vec<T>{
    fn init_fold_rev(capacity: usize, mut end: T, f: impl FnMut(T, usize) -> T) -> Self {
        let mut arr = Self::empty(capacity);
        arr.fill_fold_rev(end,f);
        arr
    }
}

pub fn empty<T:Copy, const DIM:usize>()->[T;DIM]{
    unsafe{MaybeUninit::array_assume_init(MaybeUninit::uninit_array())}
}
pub fn init_fold<T:Copy, const DIM:usize>(start: T, f: impl FnMut(T, usize) -> T)->[T;DIM]{
    let mut arr = empty();
    arr.fill_fold(start,f);
    arr
}
pub fn init_fold_rev<T:Copy, const DIM:usize>(end: T, f: impl FnMut(T, usize) -> T)->[T;DIM]{
    let mut arr = empty();
    arr.fill_fold_rev(end,f);
    arr
}
pub trait FillFold<T:Copy>{
    fn fill_fold(&mut self, start:T,f: impl FnMut(T, usize) -> T);
}
impl <T:Copy> FillFold<T> for [T]{
    fn fill_fold(&mut self, mut start: T, f: impl FnMut(T, usize) -> T) -> T{
        for i in 0..self.len(){
            self[i]=start;
            start=f(start,i);
        }
        start
    }
}
pub trait FillFoldRev<T:Copy>{
    fn fill_fold_rev(&mut self, end:T,f: impl FnMut(T, usize) -> T);
}
impl <T:Copy> FillFoldRev<T> for [T]{
    fn fill_fold_rev(&mut self, mut end: T, f: impl FnMut(T, usize) -> T) -> T{
        for i in (0..self.len()).rev(){
            self[i]=end;
            end=f(end,i);
        }
        end
    }
}
use crate::dynamic_layout::{Layout, shape};
use std::ops::Add;

pub fn zip<A,B>(lhs_tensor: &[A], lhs_layout: &Layout, rhs_tensor: &[B], rhs_layout: &Layout, f:impl FnMut(A,B)){
    assert_eq!(shape(lhs_layout),shape(rhs_layout));
    
}
/**The resulting */
pub fn add<T: Add>(lhs_tensor: &[T], lhs_layout: &Layout, rhs_tensor: &[T], rhs_layout: &Layout) -> Vec<T>{

}
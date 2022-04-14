// use std::ops::Range;
//
// pub fn for_each<T: Copy,const DIM:usize>(range: &Range<[T; DIM]>, mut f: impl FnMut([T; DIM])) where Range<T>: Iterator<Item=T> {
//     for p0 in range.start[0]..range.end[0] {
//         for p1 in range.start[1]..range.end[1] {
//             f([p0, p1])
//         }
//     }
// }
//
// pub fn range_contains<T: Copy + PartialOrd + Debug, const DIM: usize>(range: &Range<[T; DIM]>, element: &[T; DIM]) -> bool {
//     range.start.all_le(element) && element.all_lt(&range.end)
// }
//
// pub fn range_translate<T: Copy + Num + std::cmp::PartialOrd + std::cmp::Eq + Debug, const DIM: usize>(range: &Range<[T; DIM]>, element: &[T; DIM]) -> Option<T> {
//     if range_contains(range, element) {
//         let element_within_range = element.sub(&range.start);
//         let range_size = range.end.sub(&range.start);
//         Some(range_size.idx(element_within_range))
//     } else {
//         None
//     }
// }
//
// pub fn resolve_range<T: Add<Output=T> + Copy + One + Zero + PartialOrd + Debug>(input_size: T, input_range: impl RangeBounds<T>) -> Range<T> {
//     let b = match input_range.start_bound() {
//         Bound::Included(&x) => x,
//         Bound::Excluded(&x) => x + T::one(),
//         Bound::Unbounded => T::zero()
//     };
//     let e = match input_range.end_bound() {
//         Bound::Included(&x) => x + T::one(),
//         Bound::Excluded(&x) => x,
//         Bound::Unbounded => input_size
//     };
//     assert!(b <= e, "Input range {:?}..{:?} starts later than it ends", b, e);
//     assert!(e <= input_size, "Input range {:?}..{:?} exceeds input size {:?}", b, e, input_size);
//     b..e
// }
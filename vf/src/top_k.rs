//
//
// pub fn top_small_k_indices<V: Copy + PartialOrd>(mut k: usize, n: usize, f: impl Fn(usize) -> V) -> Vec<(usize, V)> {
//     debug_assert!(k <= n);
//     let mut heap: Vec<(usize, V)> = (0..k).map(&f).enumerate().collect();
//     heap.sort_by(|v1, v2| if v1.1 > v2.1 { Greater } else { Less });
//     for (idx, v) in (k..n).map(f).enumerate() {
//         let idx = idx + k;
//         if v > heap[0].1 {
//             let mut i = 1;
//             while i < k && v > heap[i].1 {
//                 heap[i - 1] = heap[i];
//                 i += 1
//             }
//             heap[i - 1] = (idx, v);
//         }
//     }
//     heap
// }
//
// pub fn top_large_k_indices<T>(mut k: usize, values: &[T], candidates_per_value: &mut [usize], f: fn(&T) -> usize, mut output: impl FnMut(usize)) {
//     debug_assert!(candidates_per_value.iter().all(|&e| e == 0));
//     values.iter().for_each(|v| candidates_per_value[f(v)] += 1);
//     let mut min_candidate_value = 0;
//     for (value, candidates) in candidates_per_value.iter_mut().enumerate().rev() {
//         if k <= *candidates {
//             *candidates = k;
//             min_candidate_value = value;
//             break;
//         }
//         k -= *candidates;
//     }
//     candidates_per_value[0..min_candidate_value].fill(0);
//     for (i, v) in values.iter().enumerate() {
//         let v = f(v);
//         if candidates_per_value[v] > 0 {
//             output(i);
//             candidates_per_value[v] -= 1;
//         }
//     }
// }
//
//
// impl TopK for [Idx; 3] {
//     fn topk_per_column<D: PartialOrd + Copy>(&self, k: usize, value: impl Fn(usize, usize) -> D, mut target: impl FnMut(D, usize, usize)) {
//         let a = self.volume().as_usize();
//         let c = self.channels().as_usize();
//         for column_idx in 0..a {
//             let r = c * column_idx;
//             for (i, v) in top_small_k_indices(k, c, |i| {
//                 debug_assert!(i < c);
//                 value(column_idx, i + r)
//             }) {
//                 let e = r + i;
//                 debug_assert!(r <= e);
//                 debug_assert!(e < r + c);
//                 target(v, column_idx, e);
//             }
//         }
//     }
//     fn top1_in_range<D: PartialOrd>(&self, mut range: Range<Idx>, value: impl Fn(Idx) -> D) -> (D, Idx) {
//         let mut top1_idx = range.start;
//         let mut top1_val = value(top1_idx);
//         range.start += 1;
//         for i in range {
//             let val = value(i);
//             if val > top1_val {
//                 top1_val = val;
//                 top1_idx = i;
//             }
//         }
//         (top1_val, top1_idx)
//     }
//     fn top1_per_region<D: PartialOrd>(&self, k: Idx, value: impl Fn(Idx) -> D, mut target: impl FnMut(D, Idx)) {
//         assert_eq!(self.channels() % k, 0);
//         for region_idx in 0..k_reg::get_region_count(k, self) {
//             //There are k regions per column, each has region_size neurons.
//             //We need to pick the top 1 winner within each region.
//             //Giving us the total of k winners per output column.
//             //The channels of each column are arranged contiguously. Regions are also contiguous.
//             let mut range = k_reg::get_region_range(k, self, region_idx);
//             let (top1_val, top1_idx) = self.top1_in_range(range, &value);
//             target(top1_val, top1_idx)
//         }
//     }
//     fn top1_per_region_per_column<D: PartialOrd, V>(&self, k: Idx, column_value: impl Fn(Idx) -> V, value: impl Fn(&V, Idx) -> D, mut target: impl FnMut(D, Idx)) {
//         assert_eq!(self.channels() % k, 0);
//         for col_idx in 0..self.area() {
//             for region_idx_within_col in 0..k {
//                 //There are k regions per column, each has region_size neurons.
//                 //We need to pick the top 1 winner within each region.
//                 //Giving us the total of k winners per output column.
//                 //The channels of each column are arranged contiguously. Regions are also contiguous.
//                 let col_val = column_value(col_idx);
//                 let region_idx = k * col_idx + region_idx_within_col;
//                 let mut range = k_reg::get_region_range(k, self, region_idx);
//                 let (top1_val, top1_idx) = self.top1_in_range(range, |i| value(&col_val, i));
//                 target(top1_val, top1_idx)
//             }
//         }
//     }
// }
//
// #[cfg(test)]
// mod tests{
//
//     #[test]
//     fn test8() {
//         let mut rng = rand::thread_rng();
//         let max = 128usize;
//         for _ in 0..54 {
//             let k = rng.gen_range(2usize..8);
//             let arr: Vec<usize> = (0..64).map(|_| rng.gen_range(0..max)).collect();
//             let mut candidates = vec![0; max];
//             let mut o = Vec::new();
//             top_large_k_indices(k, &arr, &mut candidates, |&a| a, |t| o.push(t));
//             let mut top_values1: Vec<usize> = o.iter().map(|&i| arr[i]).collect();
//             let mut arr_ind: Vec<(usize, usize)> = arr.into_iter().enumerate().collect();
//             arr_ind.sort_by_key(|&(_, v)| v);
//             let top_values2: Vec<usize> = arr_ind[64 - k..].iter().map(|&(_, v)| v).collect();
//             top_values1.sort();
//             assert_eq!(top_values1, top_values2)
//         }
//     }
//
//     #[test]
//     fn test9() {
//         let mut rng = rand::thread_rng();
//         let max = 128usize;
//         for _ in 0..54 {
//             let k = rng.gen_range(2usize..8);
//             let arr: Vec<usize> = (0..64).map(|_| rng.gen_range(0..max)).collect();
//             let o = top_small_k_indices(k, arr.len(), |i| arr[i]);
//             let mut top_values1: Vec<usize> = o.into_iter().map(|(i, v)| v).collect();
//             let mut arr_ind: Vec<(usize, usize)> = arr.into_iter().enumerate().collect();
//             arr_ind.sort_by_key(|&(_, v)| v);
//             let top_values2: Vec<usize> = arr_ind[64 - k..].iter().map(|&(_, v)| v).collect();
//             top_values1.sort();
//             assert_eq!(top_values1, top_values2)
//         }
//     }
//
//     #[test]
//     fn test10() {
//         let mut rng = rand::thread_rng();
//         let max = 128usize;
//         for _ in 0..54 {
//             let arr: Vec<usize> = (0..64).map(|_| rng.gen_range(0..max)).collect();
//             let o = top_small_k_indices(1, arr.len(), |i| arr[i]);
//             let (top_idx, top_val) = o[0];
//             assert_eq!(top_val, *arr.iter().max().unwrap());
//             assert_eq!(top_idx, arr.len() - 1 - arr.iter().rev().position_max().unwrap());
//         }
//     }
// }
use num_traits::AsPrimitive;
use std::ops::Range;

pub trait SetCardinality {
    /**Set cardinality is the number of its member elements*/
    fn card(&self) -> usize;
}

pub trait SetIntersection {
    type O;
    fn intersection(&self, other: &Self) -> Self::O;
}

pub trait SetUnion {
    type O;
    fn union(&self, other: &Self) -> Self::O;
}

pub trait SetSubtract {
    fn subtract(&mut self, other: &Self);
}

pub trait SetOverlap {
    fn overlap(&self, other: &Self) -> usize;
}

pub trait SetContains<N: Copy> {
    fn contains(&self, elem: N) -> bool;
}

pub trait SetSparseIndexArray {
    fn normalize(&mut self) -> &mut Self;
    fn is_normalized(&self) -> bool;
}

pub trait SetSparseMask {
    fn mask<T>(&self, destination:&mut [T], f:impl Fn(&mut T));
}

pub trait SetSparseParallelMask {
    fn mask_par<T>(&self, destination:&mut [T], f:impl Fn(&mut T)+Send+Sync);
}

// pub trait SetSparseRand<T> {
//     fn add_unique_random(&mut self, n: T, range: Range<T>);
// }
impl<N: Ord + Copy> SetCardinality for [N] {
    fn card(&self) -> usize {
        self.len()
    }
}

impl<N: Ord + Copy> SetSparseIndexArray for [N] {
    fn normalize(&mut self) -> &mut Self {
        self.sort();
        self.dedup();
        self
    }
    fn is_normalized(&self) -> bool {
        if self.is_empty() { return true; }
        let mut prev = self[0];
        for &i in &self[1..] {
            if i <= prev { return false; }
            prev = i;
        }
        true
    }
}

impl<N: num_traits::PrimInt> SetIntersection for [N] {
    type O = Vec<N>;

    /**Requires that both SDRs are normalized. The resulting SDR is already in normalized form*/
    fn intersection(&self, other: &Self) -> Vec<N> {
        let mut intersection = Vec::with_capacity(self.len() + other.len());
        let mut i = 0;
        if other.is_empty() { return intersection; }
        for &idx in self {
            while other[i] < idx {
                i += 1;
                if i >= other.len() { return intersection; }
            }
            if other[i] == idx {
                intersection.push(idx);
            }
        }
        intersection
    }
}

impl<N: Ord + Copy> SetOverlap for Vec<N> {
    /**This method requires that both sets are normalized*/
    fn overlap(&self, other: &Self) -> usize {
        if self.is_empty() || other.is_empty() { return 0; }
        let mut i1 = 0;
        let mut i2 = 0;
        let mut overlap = 0;
        let (s1, s2) = if self[0] < other[0] { (self, other) } else { (other, self) };
        loop {
            while s1[i1] < s2[i2] {
                i1 += 1;
                if i1 >= s1.len() { return overlap; }
            }
            if s1[i1] == s2[i2] {
                overlap += 1;
                i1 += 1;
                if i1 >= s1.len() { return overlap; }
            }
            while s1[i1] > s2[i2] {
                i2 += 1;
                if i2 >= s2.len() { return overlap; }
            }
            if s1[i1] == s2[i2] {
                overlap += 1;
                i2 += 1;
                if i2 >= s2.len() { return overlap; }
            }
        }
    }
}

impl<N: Ord + Copy> SetUnion for [N] {
    type O = Vec<N>;

    fn union(&self, other: &Self) -> Self::O {
        let mut union = Vec::with_capacity(self.len() + other.len());
        let mut i1 = 0;
        let mut i2 = 0;
        if self.len() > 0 && other.len() > 0 {
            'outer: loop {
                while self[i1] < other[i2] {
                    union.push(self[i1]);
                    i1 += 1;
                    if i1 >= self.len() { break 'outer; }
                }
                if self[i1] == other[i2] {
                    union.push(self[i1]);
                    i1 += 1;
                    i2 += 1;
                    if i1 >= self.len() || i2 >= other.len() { break 'outer; }
                }
                while self[i1] > other[i2] {
                    union.push(other[i2]);
                    i2 += 1;
                    if i2 >= other.len() { break 'outer; }
                }
                if self[i1] == other[i2] {
                    union.push(other[i2]);
                    i1 += 1;
                    i2 += 1;
                    if i1 >= self.len() || i2 >= other.len() { break 'outer; }
                }
            }
        }
        if i1 < self.len() {
            union.extend_from_slice(&self[i1..])
        } else {
            union.extend_from_slice(&other[i2..])
        }
        union
    }
}

impl<N: Ord + Copy> SetSubtract for Vec<N> {
    fn subtract(&mut self, other: &Self) {
        let mut i1 = 0;
        let mut i2 = 0;
        let mut j = 0;
        if self.len() > 0 && other.len() > 0 {
            'outer: loop {
                while self[i1] < other[i2] {
                    self[j] = self[i1];
                    j += 1;
                    i1 += 1;
                    if i1 >= self.len() { break 'outer; }
                }
                if self[i1] == other[i2] {
                    i1 += 1;
                    i2 += 1;
                    if i1 >= self.len() || i2 >= other.len() { break 'outer; }
                }
                while self[i1] > other[i2] {
                    i2 += 1;
                    if i2 >= other.len() { break 'outer; }
                }
                if self[i1] == other[i2] {
                    i1 += 1;
                    i2 += 1;
                    if i1 >= self.len() || i2 >= other.len() { break 'outer; }
                }
            }
        }
        while i1 < self.len() {
            self[j] = self[i1];
            j += 1;
            i1 += 1;
        }
        self.truncate(j);
    }
}

impl<N: AsPrimitive<usize>+Copy> SetSparseMask for [N] {
    fn mask<T>(&self, destination: &mut [T], f: impl Fn(&mut T)) {
        for i in self {
            f(&mut destination[i.as_()])
        }
    }
}
impl<N: AsPrimitive<usize>+Copy> SetSparseParallelMask for [N] {
    fn mask_par<T>(&self, destination: &mut [T], f: impl Fn(&mut T)+Send+Sync) {
        let len = destination.len();
        let ptr = destination.as_mut_ptr() as usize;
        self.par_iter().for_each(|i:&N| {
            let s = unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, len) };
            f(&mut s[i.as_()])
        })
    }
}

// impl SetSparseRand<T> for Vec<N>{
//     pub fn add_unique_random(&mut self, n: u32, range: Range<u32>) {
//         let len = range.end - range.start;
//         assert!(len >= n, "The range of values {}..{} has {} elements. Can't get unique {} elements out of it!", range.start, range.end, len, n);
//         let mut set = HashSet::new();
//         for _ in 0..n {
//             let mut r = range.start + rand::random::<u32>() % len;
//             while !set.insert(r) {
//                 r += 1;
//                 if r >= range.end {
//                     r = range.start;
//                 }
//             }
//             self.push(r);
//         }
//     }
//
//     pub fn rand(cardinality: Idx, size: Idx) -> Self {
//         assert!(cardinality <= size);
//         let mut s = Self::with_capacity(cardinality.as_usize());
//         s.add_unique_random(cardinality, 0..size);
//         s
//     }
//     /**Randomly picks some neurons that a present in other SDR but not in self SDR.
//     Requires that both SDRs are already normalized.
//     It will only add so many elements so that self.len() <= n*/
//     pub fn randomly_extend_from(&mut self, other: &Self, n: usize) {
//         debug_assert!(self.is_normalized());
//         debug_assert!(other.is_normalized());
//         assert!(other.len() <= n, "The limit {} is less than the size of SDR {}", n, other.len());
//         self.subtract(other);
//         while self.len() + other.len() > n {
//             let idx = rand::random::<usize>() % self.0.len();
//             self.0.swap_remove(idx);
//         }
//         self.0.extend_from_slice(other.as_slice());
//         self.0.sort()
//     }
// }
// {
//     pub fn subregion(&self, total_shape: &[Idx; 3], subregion_range: &Range<[Idx; 3]>) -> Vec {
//         Vec(self.iter().cloned().filter_map(|i| range_translate(subregion_range, &total_shape.pos(i))).collect())
//     }
//
//     pub fn subregion2d(&self, total_shape: &[Idx; 3], subregion_range: &Range<[Idx; 2]>) -> Vec {
//         Vec(self.iter().cloned().filter_map(|i| range_translate(subregion_range, total_shape.pos(i).grid())).collect())
//     }
//
//     pub fn conv_rand_subregion(&self, shape: &ConvShape, rng: &mut impl Rng) -> Vec {
//         self.conv_subregion(shape, &shape.out_grid().rand_vec(rng))
//     }
//
//     pub fn conv_subregion(&self, shape: &ConvShape, output_column_position: &[Idx; 2]) -> Vec {
//         let mut s = Vec::new();
//         let r = shape.in_range(output_column_position);
//         let kc = shape.kernel_column();
//         for &i in self.iter() {
//             let pos = shape.in_shape().pos(i);
//             if range_contains(&r, pos.grid()) {
//                 let pos_within_subregion = pos.grid().sub(&r.start).add_channels(pos.channels());
//                 s.push(kc.idx(pos_within_subregion))
//             }
//         }
//         s
//     }
//
// }


#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;



    #[test]
    fn test6() -> Result<(), String> {
        fn overlap(a: &[u32], b: &[u32]) -> usize {
            let mut sdr1 = a.to_vec();
            let mut sdr2 = b.to_vec();
            sdr1.normalize();
            sdr2.normalize();
            sdr1.overlap(&sdr2)
        }
        assert_eq!(overlap(&[1, 5, 6, 76], &[1]), 1);
        assert_eq!(overlap(&[1, 5, 6, 76], &[]), 0);
        assert_eq!(overlap(&[], &[]), 0);
        assert_eq!(overlap(&[], &[1]), 0);
        assert_eq!(overlap(&[1, 5, 6, 76], &[1, 5, 6, 76]), 4);
        assert_eq!(overlap(&[1, 5, 6, 76], &[5, 76, 6, 1]), 4);
        assert_eq!(overlap(&[1, 5, 6, 76], &[53, 746, 6, 1]), 2);
        assert_eq!(overlap(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]), 3);
        Ok(())
    }

    #[test]
    fn test7() -> Result<(), String> {
        fn intersect(a: &[u32], b: &[u32]) -> Vec<u32> {
            let mut sdr1 = a.to_vec();
            let mut sdr2 = b.to_vec();
            sdr2.set_from_slice(b);
            sdr1.normalize();
            sdr2.normalize();
            sdr1.intersection(&sdr2)
        }
        assert_eq!(intersect(&[1, 5, 6, 76], &[1]).as_slice(), &[1]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[]).as_slice(), &[]);
        assert_eq!(intersect(&[], &[]).as_slice(), &[]);
        assert_eq!(intersect(&[], &[1]).as_slice(), &[]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[1, 5, 6, 76]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[5, 76, 6, 1]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[53, 746, 6, 1]).as_slice(), &[1, 6]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]).as_slice(), &[1, 5, 6]);
        Ok(())
    }

    #[test]
    fn test7_union() -> Result<(), String> {
        fn union(a: &[u32], b: &[u32]) -> Vec<u32> {
            let mut sdr1 = a.to_vec();
            let mut sdr2 = b.to_vec();
            sdr1.normalize();
            sdr2.normalize();
            sdr1.union(&sdr2)
        }
        assert_eq!(union(&[1, 5, 6, 76], &[1]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[1, 5, 6, 76], &[]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[], &[]).as_slice(), &[]);
        assert_eq!(union(&[1], &[]).as_slice(), &[1]);
        assert_eq!(union(&[], &[1]).as_slice(), &[1]);
        assert_eq!(union(&[1, 5, 6, 76], &[1, 5, 6, 76]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[1, 5, 6, 76], &[5, 76, 6, 1]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[1, 5, 6, 76], &[53, 746, 6, 1]).as_slice(), &[1, 5, 6, 53, 76, 746]);
        assert_eq!(union(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]).as_slice(), &[1, 3, 5, 6, 7, 53, 76, 78, 746]);
        Ok(())
    }

    #[test]
    fn test7_subtract() -> Result<(), String> {
        fn subtract(a: &[u32], b: &[u32]) -> Vec<u32> {
            let mut sdr1 = a.to_vec();
            let mut sdr2 = b.to_vec();
            sdr1.normalize();
            sdr2.normalize();
            sdr1.subtract(&sdr2);
            sdr1
        }
        assert_eq!(subtract(&[1, 5, 6, 76], &[1]).as_slice(), &[5, 6, 76]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(subtract(&[], &[]).as_slice(), &[]);
        assert_eq!(subtract(&[1], &[]).as_slice(), &[1]);
        assert_eq!(subtract(&[], &[1]).as_slice(), &[]);
        assert_eq!(subtract(&[1], &[1]).as_slice(), &[]);
        assert_eq!(subtract(&[1], &[2]).as_slice(), &[1]);
        assert_eq!(subtract(&[1, 2], &[2]).as_slice(), &[1]);
        assert_eq!(subtract(&[2, 3], &[2]).as_slice(), &[3]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[1, 5, 6, 76]).as_slice(), &[]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[5, 76, 6, 1]).as_slice(), &[]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[53, 746, 6, 1]).as_slice(), &[5, 76]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]).as_slice(), &[76]);
        Ok(())
    }

}
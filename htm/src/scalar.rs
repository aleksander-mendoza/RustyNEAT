use std::cmp::Ordering;

pub type D = f32;
pub type Idx = u32;
pub fn as_idx(u:usize)->Idx{
    u as Idx
}
pub trait NaiveCmp: PartialOrd + Copy {
    fn cmp_naive(&self, b: &Self) -> Ordering {
        if self > b { Ordering::Greater } else { Ordering::Less }
    }

    fn min_naive(&self, b: &Self) -> Self {
        *if self < b { self } else { b }
    }

    fn max_naive(&self, b: &Self) -> Self {
        *if self > b { self } else { b }
    }
}
impl <D:PartialOrd+Copy> NaiveCmp for D{

}
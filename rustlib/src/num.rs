use rand::prelude::Distribution;

pub trait Num: num_traits::Num + Copy{}

impl Num for f64 {}

impl Num for f32 {}

impl Num for i64 {}

impl Num for i32 {}

impl Num for i16 {}

impl Num for u64 {}

impl Num for u32 {}

impl Num for u16 {}
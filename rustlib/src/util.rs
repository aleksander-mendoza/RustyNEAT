

pub trait Initializer<T> {
    fn initialize(count: usize, f: fn(usize) -> T) -> Vec<T> {
        (0..count).map(f).collect()
    }
}

impl<T> Initializer<T> for Vec<T> {}
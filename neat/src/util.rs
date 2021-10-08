

pub trait Initializer<T> {
    fn initialize(count: usize, f: fn(usize) -> T) -> Vec<T> {
        (0..count).map(f).collect()
    }
}

impl<T> Initializer<T> for Vec<T> {}


pub trait RandRange{
    fn random(&self)->Self;
}
impl RandRange for usize{
    fn random(&self) -> usize{
        (rand::random::<f32>() * (*self as f32)) as usize
    }
}
pub trait AsUsize{
    fn as_usize(self)->usize;
    fn from_usize(_:usize)->Self;
}
impl AsUsize for u8{
    fn as_usize(self) -> usize {
        self as usize
    }

    fn from_usize(u: usize) -> Self {
        u as u8
    }
}
impl AsUsize for u32{
    fn as_usize(self) -> usize {
        self as usize
    }

    fn from_usize(u: usize) -> Self {
        u as u32
    }
}
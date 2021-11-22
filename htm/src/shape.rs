use std::fmt::{Debug, Formatter};

pub struct Shape<const dim:usize>{
    stride: [u32;dim],
    shape: [u32;dim],
}
impl <const dim:usize> Debug for Shape<dim>{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f,"{:?}",self.shape)
    }
}
impl <const dim:usize> PartialEq for Shape<dim>{
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
    }
}
impl <const dim:usize> Eq for Shape<dim>{}
impl Shape<2>{
    pub fn height(&self)->u32{
        self.shape[0]
    }
    pub fn width(&self)->u32{
        self.shape[1]
    }
    pub fn new(height:u32,width:u32)->Self{
        Self{ stride: [width, 1], shape: [height, width] }
    }
    pub fn index(&self, y:u32,x:u32)->u32{
        assert!(y<self.height());
        assert!(x<self.width());
        y*self.stride[0]+x*self.stride[1]
    }
    pub fn idx(&self, pos:[u32;2])->u32{
        let [y,x] = pos;
        self.index(y,x)
    }
    pub fn pos(&self, index:u32)->[u32;2]{
        assert!(index<self.width()*self.height());
        let y = (index/self.stride[0])%self.shape[0];
        let x = (index/self.stride[1])%self.shape[1];
        [y, x]
    }
    pub fn transpose(&mut self){
        self.shape.swap(0,1);
        self.stride.swap(0,1);
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn test1(){
        let s = Shape::new(4,3);
        assert_eq!(s.idx([0,0]),0);
        assert_eq!(s.idx([0,1]),1);
        assert_eq!(s.idx([0,2]),2);
        assert_eq!(s.pos(0),[0,0]);
        assert_eq!(s.pos(1),[0,1]);
        assert_eq!(s.pos(2),[0,2]);
        assert_eq!(s.pos(3),[1,0]);
        assert_eq!(s.pos(4),[1,1]);
        assert_eq!(s.pos(5),[1,2]);
        for i in 0..(3*4) {
            let p = s.pos(i);
            assert_eq!(s.idx(p), i, "{}=={:?}",i, p);
        }
    }
    #[test]
    fn test2(){
        let s = Shape::new(3,4);
        for x in 0..3{
            for y in 0..4 {
                assert_eq!(s.pos(s.idx([x,y])), [x,y]);
            }
        }
    }
    #[test]
    fn test3(){
        let mut s = Shape::new(3, 4);
        s.transpose();
        for x in 0..4{
            for y in 0..3 {
                assert_eq!(s.pos(s.idx([x,y])), [x,y]);
            }
        }
    }
}

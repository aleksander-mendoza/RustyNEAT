use std::fmt::{Debug, Formatter};

use serde::{Serialize, Deserialize};

#[derive(Eq, PartialEq, Clone, Serialize, Deserialize)]
pub struct Shape{
    shape: [u32;2],
}
impl Debug for Shape{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f,"{:?}",self.shape)
    }
}
impl Shape{
    pub fn height(&self)->u32{
        self.shape[0]
    }
    pub fn width(&self)->u32{
        self.shape[1]
    }
    pub fn new(height:u32,width:u32)->Self{
        Self{ shape: [height, width] }
    }
    pub fn index(&self, y:u32,x:u32)->u32{
        assert!(y<self.height());
        assert!(x<self.width());
        y*self.width()+x
    }
    pub fn idx(&self, pos:[u32;2])->u32{
        let [y,x] = pos;
        self.index(y,x)
    }
    pub fn pos(&self, index:u32)->[u32;2]{
        assert!(index<self.width()*self.height());
        let y = index/self.width();
        let x = index%self.width();
        [y, x]
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
}

use crate::Idx;

pub trait SDR{
    fn clear(&mut self);
    fn item(&self)->Idx;
    fn cardinality(&self)->Idx;

    fn set_from_slice(&mut self, other:&[Idx]);

    fn set_from_sdr(&mut self, other:&Self);

    fn to_vec(&self)->Vec<Idx>;
    fn into_vec(self) -> Vec<Idx>;
    fn is_empty(&self)->bool{
        self.cardinality()==0
    }
}
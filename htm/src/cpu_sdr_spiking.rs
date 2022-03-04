use crate::CpuSDR;
use crate::conv_spiking_shape::TIdx;
use crate::as_usize::AsUsize;
use serde::{Serialize, Deserialize};

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Default)]
pub struct CpuSdrSpiking{
    sdrs:Vec<CpuSDR>
}

impl CpuSdrSpiking{
    pub fn new(time:TIdx)->Self{
        assert!(time>0,"Time must be greater than 0");
        Self{sdrs:vec![CpuSDR::new();time.as_usize()]}
    }
    pub fn next(&mut self, sdr:CpuSDR){
        self.sdrs.pop();
        self.sdrs.insert(0,sdr);
    }
    pub fn get(&self, i:TIdx)->&CpuSDR{
        &self.sdrs[i.as_usize()]
    }
    pub fn get_mut(&mut self, i:TIdx)->&mut CpuSDR{
        &mut self.sdrs[i.as_usize()]
    }
}
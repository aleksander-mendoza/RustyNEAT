use serde::{Serialize, Deserialize};
use crate::D;

#[derive(Serialize, Deserialize, Clone, Debug, Copy, PartialEq, Eq)]
pub enum Activity {
    /**r = s+a*/
    Additive,
    /**r = s*(a-min(a)+1)*/
    Multiplicative,
    /**W=Q>a*/
    Thresholded,
}
impl Activity{
    pub fn cache_min_column_activity(&self)->bool{
        match self{
            Activity::Additive => false,
            Activity::Multiplicative => true,
            Activity::Thresholded => true,
        }
    }
}

impl From<&str> for Activity {
    fn from(v: &str) -> Self {
        match v {
            "Additive" => Self::Additive,
            "Multiplicative" => Self::Multiplicative,
            "Thresholded" => Self::Thresholded,
            _ => panic!("Invalid literal")
        }
    }
}

impl Default for Activity {
    fn default() -> Self {
        Self::Additive
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Copy, PartialEq, Eq)]
pub enum WNorm {
    /** W become a moving average*/
    None,
    /** W sum up to 1*/
    L1,
    /** inner product <W,W> is 1 */
    L2,
}

impl From<&str> for WNorm {
    fn from(v: &str) -> Self {
        match v {
            "None" => Self::None,
            "L1" => Self::L1,
            "L2" => Self::L2,
            _ => panic!("Invalid literal")
        }
    }
}

impl Default for WNorm {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct EccConfig<D: Copy> {
    pub biased: bool,
    pub activity: Activity,
    pub entropy_maximisation: D,
    pub w_norm: WNorm,
}

impl<D: Copy> EccConfig<D> {
    pub fn compatible(&self, other: &Self) -> bool {
        self.biased == other.biased && self.activity == other.activity && self.w_norm == other.w_norm
    }

}
impl EccConfig<D>{
    pub fn zero_order()->Self{
        Self{
            biased: false,
            activity: Activity::Additive,
            entropy_maximisation: 0.1,
            w_norm: WNorm::L1
        }
    }
    pub fn l1()->Self{
        Self{
            biased: false,
            activity: Activity::Thresholded,
            entropy_maximisation: 0.1,
            w_norm: WNorm::None
        }
    }
    pub fn l2()->Self{
        Self{
            biased: false,
            activity: Activity::Multiplicative,
            entropy_maximisation: 0.1,
            w_norm: WNorm::L2
        }
    }
}

pub trait HasEccConfig<D: Copy> {
    fn cfg(&self) -> &EccConfig<D>;
    fn cfg_entropy_maximisation(&self) -> D {
        self.cfg().entropy_maximisation
    }
    fn cfg_biased(&self) -> bool {
        self.cfg().biased
    }
    fn cfg_w_norm(&self) -> WNorm {
        self.cfg().w_norm
    }
    fn cfg_activity(&self) -> Activity {
        self.cfg().activity
    }
    fn cfg_compatible(&self, other: &impl HasEccConfig<D>) -> bool {
        self.cfg().compatible(other.cfg())
    }
}

pub trait HasEccConfigMut<D: Copy + 'static>: HasEccConfig<D> {
    fn cfg_mut(&mut self) -> &mut EccConfig<D>;
    fn cfg_entropy_maximisation_mut(&mut self) -> &mut D {
        &mut self.cfg_mut().entropy_maximisation
    }
    fn cfg_biased_mut(&mut self) -> &mut bool {
        &mut self.cfg_mut().biased
    }
    fn cfg_w_norm_mut(&mut self) -> &mut WNorm {
        &mut self.cfg_mut().w_norm
    }
    fn cfg_activity_mut(&mut self) -> &mut Activity {
        &mut self.cfg_mut().activity
    }
}

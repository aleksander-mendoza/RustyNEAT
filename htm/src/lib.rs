pub mod ocl_sdr;
pub mod ocl_htm;
pub mod htm_program;
pub mod cpu_htm;
pub mod htm;
pub mod cpu_sdr;


#[cfg(test)]
mod tests {
    use super::*;
    use ocl::{SpatialDims, Platform, Device, Context, Program, Queue, Buffer, flags, Kernel, ProQue};
    use ocl::core::{BufferRegion, Error};
    use crate::htm_program::HtmProgram;
    use crate::ocl_sdr::OclSDR;
    use crate::ocl_htm::OclHTM;
    use crate::cpu_sdr::CpuSDR;
    use crate::cpu_htm::CpuHTM;
    #[test]
    fn test1() -> Result<(), String> {
        let p = HtmProgram::gpu()?;
        let mut sdr = CpuSDR::new(16);
        sdr.set(&[4,6,14,3]);
        let mut htm = CpuHTM::new_globally_uniform_prob(16,16,4,0.76,-0.01,0.02,12);
        let output_sdr = htm.infer(&sdr,false);
        let output_sdr2 = htm.infer2(&sdr,false);

        let mut ocl_sdr = OclSDR::new(p.clone(), 16)?;
        ocl_sdr.set(&[4,6,14,3]);
        let mut ocl_htm = OclHTM::new(htm, p.clone())?;
        let output_sdr3 = ocl_htm.infer(&ocl_sdr,false)?;
        let output_sdr4 = ocl_htm.infer2(&ocl_sdr,false)?;
        let output_sdr = output_sdr.to_vec();
        let output_sdr2 = output_sdr2.to_vec();
        let output_sdr3 = output_sdr3.buffer().to_vec(p.queue())?;
        let output_sdr4 = output_sdr4.buffer().to_vec(p.queue())?;
        assert_eq!(output_sdr,output_sdr2,"{:?}=={:?}=={:?}=={:?}",output_sdr,output_sdr2,output_sdr3,output_sdr4);
        assert_eq!(output_sdr2,output_sdr3,"{:?}=={:?}=={:?}=={:?}",output_sdr,output_sdr2,output_sdr3,output_sdr4);
        assert_eq!(output_sdr3,output_sdr4,"{:?}=={:?}=={:?}=={:?}",output_sdr,output_sdr2,output_sdr3,output_sdr4);
        Ok(())
    }
}
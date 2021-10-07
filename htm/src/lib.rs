pub mod ocl_sdr;
pub mod ocl_htm;
pub mod htm_program;
pub mod cpu_htm;
pub mod htm;
pub mod cpu_sdr;
pub mod cpu_htm2;
pub mod htm2;
pub mod ocl_htm2;
pub mod htm_program2;


#[cfg(test)]
mod tests {
    use super::*;
    use ocl::{SpatialDims, Platform, Device, Program, Queue, Buffer, flags, Kernel, ProQue};
    use ocl::core::{BufferRegion, Error};
    use crate::htm_program::HtmProgram;
    use crate::ocl_sdr::OclSDR;
    use crate::ocl_htm::OclHTM;
    use crate::cpu_sdr::CpuSDR;
    use crate::cpu_htm::CpuHTM;
    use crate::cpu_htm2::CpuHTM2;
    use crate::ocl_htm2::OclHTM2;
    use crate::htm_program2::HtmProgram2;
    use ndalgebra::context::Context;

    #[test]
    fn test1() -> Result<(), String> {
        let c = Context::gpu()?;
        let p = HtmProgram::new(c.clone())?;
        let p2 = HtmProgram2::new(c)?;
        let mut sdr = CpuSDR::new(16);
        sdr.set(&[4,6,14,3]);
        let mut htm = CpuHTM::new_globally_uniform_prob(16,16,4,0.76,-0.01,0.02,12);
        let output_sdr = htm.infer(&sdr,false);
        let mut htm2 = CpuHTM2::from(&htm);
        let output_sdr2 = htm2.infer2(&sdr,false);

        let mut ocl_sdr = OclSDR::new(p.ctx.clone(), 16)?;
        ocl_sdr.set(&[4,6,14,3]);
        let mut ocl_htm2 = OclHTM2::new(htm2.clone(), p2.clone())?;
        let output_sdr4 = ocl_htm2.infer2(&ocl_sdr,false)?;
        let mut ocl_htm = OclHTM::new(htm.clone(), p.clone())?;
        let output_sdr3 = ocl_htm.infer(&ocl_sdr,false)?;

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
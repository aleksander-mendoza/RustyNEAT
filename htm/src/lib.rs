mod cpu_htm;
mod htm;
mod cpu_sdr;
mod cpu_htm2;
mod htm2;
mod cpu_hom;
mod hom;
mod encoder;
mod ocl_sdr;
mod ocl_htm;
mod htm_program;
mod ocl_htm2;
mod ocl_hom;
mod cpu_htm4;
mod htm4;
mod ocl_bitset;
mod cpu_bitset;
mod cpu_input;
mod ocl_input;
pub use ocl_htm2::OclHTM2;
pub use ocl_bitset::OclBitset;
pub use ocl_input::OclInput;
pub use ocl_sdr::OclSDR;
pub use ocl_htm::OclHTM;
pub use htm_program::HtmProgram;
pub use htm::*;
pub use htm4::*;
pub use htm2::*;
pub use encoder::*;
pub use cpu_htm::CpuHTM;
pub use cpu_hom::*;
pub use cpu_bitset::CpuBitset;
pub use cpu_input::CpuInput;
pub use cpu_htm4::CpuHTM4;
pub use cpu_sdr::CpuSDR;
pub use cpu_htm2::CpuHTM2;

// pub use cpu_higher_order_memory::CpuHOM;

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
    use ndalgebra::context::Context;
    use crate::encoder::{EncoderBuilder, Encoder};

    #[test]
    fn test1() -> Result<(), String> {
        let c = Context::gpu()?;
        let p = HtmProgram::new(c.clone())?;
        let mut sdr = CpuInput::from_sparse_slice(&[4, 6, 14, 3], 16);
        let mut htm = CpuHTM::new_globally_uniform_prob(16, 16, 4, 12);
        let output_sdr = htm.infer(&sdr, false);
        let mut htm2 = CpuHTM2::from(&htm);
        let output_sdr2 = htm2.infer2(sdr.get_dense(), false);

        let mut ocl_sdr = OclInput::new(p.clone(), 16,16)?;
        ocl_sdr.set_sparse_from_slice(&[4, 6, 14, 3]);
        let mut ocl_htm2 = OclHTM2::new(&htm2, p.clone())?;
        let output_sdr4 = ocl_htm2.infer2(ocl_sdr.get_dense(), false)?;
        let mut ocl_htm = OclHTM::new(&htm, p.clone())?;
        let output_sdr3 = ocl_htm.infer(&ocl_sdr, false)?;

        let output_sdr = output_sdr.to_vec();
        let output_sdr2 = output_sdr2.to_vec();
        let output_sdr3 = output_sdr3.buffer().to_vec(p.queue())?;
        let output_sdr4 = output_sdr4.buffer().to_vec(p.queue())?;
        assert_eq!(output_sdr, output_sdr2, "{:?}=={:?}=={:?}=={:?}", output_sdr, output_sdr2, output_sdr3, output_sdr4);
        assert_eq!(output_sdr2, output_sdr3, "{:?}=={:?}=={:?}=={:?}", output_sdr, output_sdr2, output_sdr3, output_sdr4);
        assert_eq!(output_sdr3, output_sdr4, "{:?}=={:?}=={:?}=={:?}", output_sdr, output_sdr2, output_sdr3, output_sdr4);
        Ok(())
    }

    #[test]
    fn test2() -> Result<(), String> {
        let mut encoder = EncoderBuilder::new();
        let scalar = encoder.add_integer(50..100, 100, 5);

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 50);
        assert_eq!(vec![0, 1, 2, 3, 4], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 51);
        assert_eq!(vec![1, 2, 3, 4, 5], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 52);
        assert_eq!(vec![3, 4, 5, 6, 7], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 100);
        assert_eq!(vec![95, 96, 97, 98, 99], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 99);
        assert_eq!(vec![95, 96, 97, 98, 99], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 98);
        assert_eq!(vec![93, 94, 95, 96, 97], sdr.to_vec());
        Ok(())
    }

    #[test]
    fn test3() -> Result<(), String> {
        let mut encoder = EncoderBuilder::new();
        let scalar = encoder.add_float(50f32..100f32, 100, 5);

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 50.);
        assert_eq!(vec![0, 1, 2, 3, 4], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 51.);
        assert_eq!(vec![1, 2, 3, 4, 5], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 52.);
        assert_eq!(vec![3, 4, 5, 6, 7], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 100.);
        assert_eq!(vec![95, 96, 97, 98, 99], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 99.);
        assert_eq!(vec![93, 94, 95, 96, 97], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 98.);
        assert_eq!(vec![91, 92, 93, 94, 95], sdr.to_vec());
        Ok(())
    }

    #[test]
    fn test4() -> Result<(), String> {
        let mut encoder = EncoderBuilder::new();
        let scalar = encoder.add_circular_integer(50..100, 100, 5);

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 50);
        assert_eq!(vec![0, 1, 2, 3, 4], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 51);
        assert_eq!(vec![2, 3, 4, 5, 6], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 52);
        assert_eq!(vec![4, 5, 6, 7, 8], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 101);
        assert_eq!(vec![2, 3, 4, 5, 6], sdr.to_vec());


        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 100);
        assert_eq!(vec![0, 1, 2, 3, 4], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 99);
        assert_eq!(vec![98, 99, 0, 1, 2], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 98);
        assert_eq!(vec![96, 97, 98, 99, 0], sdr.to_vec());

        let mut sdr = CpuSDR::new();
        scalar.encode(&mut sdr, 97);
        assert_eq!(vec![94, 95, 96, 97, 98], sdr.to_vec());
        Ok(())
    }

    #[test]
    fn test5() -> Result<(), String> {
        let mut encoder = EncoderBuilder::new();
        let scalar1 = encoder.add_circular_integer(50..100, 100, 5);
        let scalar2 = encoder.add_float(50f32..100f32, 100, 5);

        let mut sdr = CpuSDR::new();
        scalar1.encode(&mut sdr, 99);
        scalar2.encode(&mut sdr, 50.);
        assert_eq!(vec![98, 99, 0, 1, 2, 100, 101, 102, 103, 104], sdr.to_vec());
        Ok(())
    }

    #[test]
    fn test6() -> Result<(), String> {
        fn overlap(a: &[u32], b: &[u32]) -> u32 {
            let mut sdr1 = CpuSDR::new();
            sdr1.set(a);
            let mut sdr2 = CpuSDR::new();
            sdr2.set(b);
            sdr1.normalize();
            sdr2.normalize();
            sdr1.overlap(&sdr2)
        }
        assert_eq!(overlap(&[1, 5, 6, 76], &[1]), 1);
        assert_eq!(overlap(&[1, 5, 6, 76], &[]), 0);
        assert_eq!(overlap(&[], &[]), 0);
        assert_eq!(overlap(&[], &[1]), 0);
        assert_eq!(overlap(&[1, 5, 6, 76], &[1, 5, 6, 76]), 4);
        assert_eq!(overlap(&[1, 5, 6, 76], &[5, 76, 6, 1]), 4);
        assert_eq!(overlap(&[1, 5, 6, 76], &[53, 746, 6, 1]), 2);
        assert_eq!(overlap(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]), 3);
        Ok(())
    }

    #[test]
    fn test7() -> Result<(), String> {
        fn intersect(a: &[u32], b: &[u32]) -> CpuSDR {
            let mut sdr1 = CpuSDR::new();
            sdr1.set(a);
            let mut sdr2 = CpuSDR::new();
            sdr2.set(b);
            sdr1.normalize();
            sdr2.normalize();
            sdr1.intersection(&sdr2)
        }
        assert_eq!(intersect(&[1, 5, 6, 76], &[1]).as_slice(), &[1]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[]).as_slice(), &[]);
        assert_eq!(intersect(&[], &[]).as_slice(), &[]);
        assert_eq!(intersect(&[], &[1]).as_slice(), &[]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[1, 5, 6, 76]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[5, 76, 6, 1]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[53, 746, 6, 1]).as_slice(), &[1, 6]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]).as_slice(), &[1, 5, 6]);
        Ok(())
    }

    #[test]
    fn test8() -> Result<(), String> {
        let mut encoder = EncoderBuilder::new();
        let cat_enc = encoder.add_categorical(4, 8);
        let number_of_minicolumns = 100;
        let mut htm = CpuHTM2::new_globally_uniform_prob(
            encoder.input_size(),
            number_of_minicolumns,
            16,
            (encoder.input_size() as f32 *0.8) as u32
        );
        let mut hom = CpuHOM::new(8, number_of_minicolumns);
        hom.hyp.predicted_decrement = -0.05;
        let mut sdr = CpuInput::new(encoder.input_size());
        const NOTE_A:u32 = 0;
        const NOTE_B:u32 = 1;
        const NOTE_C_SHARP:u32 = 2;
        const NOTE_D:u32 = 3;
        let mut predicted_after_a = CpuSDR::new();
        let mut activated_by_b = CpuSDR::new();
        let mut predicted_after_b = CpuSDR::new();
        let mut activated_by_c = CpuSDR::new();
        let mut predicted_after_c = CpuSDR::new();
        let mut activated_by_d = CpuSDR::new();
        let mut predicted_after_d = CpuSDR::new();
        let mut activated_by_a = CpuSDR::new();

        for _ in 0..4{
            sdr.clear();
            cat_enc.encode(&mut sdr, NOTE_A);
            activated_by_a = htm.infer2(sdr.get_dense(),true);
            predicted_after_a = hom.infer(&activated_by_a,true);

            sdr.clear();
            cat_enc.encode(&mut sdr, NOTE_B);
            activated_by_b = htm.infer2( sdr.get_dense(),true);
            predicted_after_b = hom.infer(& activated_by_b,true);


            sdr.clear();
            cat_enc.encode(&mut sdr, NOTE_C_SHARP);
            activated_by_c = htm.infer2( sdr.get_dense(),true);
            predicted_after_c = hom.infer(& activated_by_c,true);


            sdr.clear();
            cat_enc.encode(&mut sdr, NOTE_D);
            activated_by_d = htm.infer2( sdr.get_dense(),true);
            predicted_after_d = hom.infer(& activated_by_d,true);
        }

        assert_eq!(predicted_after_a, activated_by_b);
        assert_eq!(predicted_after_b, activated_by_c);
        assert_eq!(predicted_after_c, activated_by_d);
        assert_eq!(predicted_after_d, activated_by_a);
        Ok(())
    }

    #[test]
    fn test9() -> Result<(), String> {
        let p = HtmProgram::default()?;
        let input = CpuInput::from_sparse_slice(&[1,2,4,7,15], 16);
        let ocl_input = OclInput::from_cpu(&input,p.clone(),16)?;
        let input2 = ocl_input.to_cpu()?;
        assert_eq!(input.get_sparse(),input2.get_sparse(),"sparse");
        assert_eq!(input.get_dense(),input2.get_dense(),"dense");
        Ok(())
    }

    #[test]
    fn test10() -> Result<(), String> {
        let p = HtmProgram::default()?;
        let input = CpuInput::from_dense_bools(&[true,false,false,true,true,false,false,true]);
        let ocl_input = OclInput::from_cpu(&input,p.clone(),16)?;
        let input2 = ocl_input.to_cpu()?;
        assert_eq!(input.get_sparse(),input2.get_sparse(),"sparse");
        assert_eq!(input.get_dense(),input2.get_dense(),"dense");
        Ok(())
    }

    #[test]
    fn test11() -> Result<(), String> {
        let p = HtmProgram::default()?;
        let input = CpuInput::from_sparse_slice(&[1,2,4,7,15], 16);
        let mut ocl_input = OclInput::from_cpu(&input,p.clone(),16)?;
        ocl_input.set_sparse_from_slice(&[1,5,13])?;
        let input = CpuInput::from_sparse_slice(&[1,5,13], 16);
        let input2 = ocl_input.to_cpu()?;
        assert_eq!(input.get_sparse(),input2.get_sparse(),"sparse");
        assert_eq!(input.get_dense(),input2.get_dense(),"dense");
        Ok(())
    }
}
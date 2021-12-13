
mod cpu_htm;
mod htm;
mod cpu_sdr;
mod cpu_htm2;
mod htm2;
mod cpu_hom;
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
mod rnd;
mod htm3;
mod cpu_htm3;
mod cpu_htm5;
mod htm5;
mod map;
mod dg2;
mod cpu_dg2;
mod shape;
mod cpu_big_htm;
mod ocl_dg2;
mod vector_field;
mod htm_builder;

pub use htm_builder::*;
pub use vector_field::*;
pub use cpu_big_htm::*;
pub use crate::rnd::auto_gen_seed;
pub use ocl_htm2::OclHTM2;
pub use ocl_bitset::OclBitset;
pub use ocl_input::OclInput;
pub use ocl_sdr::OclSDR;
pub use ocl_htm::OclHTM;
pub use ocl_dg2::OclDG2;
pub use htm_program::HtmProgram;
pub use dg2::*;
pub use shape::*;
pub use cpu_dg2::*;
pub use htm5::*;
pub use htm4::*;
pub use htm3::*;
pub use htm2::*;
pub use htm::*;
pub use map::*;
pub use encoder::*;
pub use cpu_hom::*;
pub use cpu_bitset::CpuBitset;
pub use cpu_input::CpuInput;
pub use cpu_htm5::CpuHTM5;
pub use cpu_htm4::CpuHTM4;
pub use cpu_htm3::CpuHTM3;
pub use cpu_htm2::CpuHTM2;
pub use cpu_htm::CpuHTM;
pub use cpu_sdr::CpuSDR;


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
    use crate::rnd::xorshift32;
    use crate::cpu_htm3::CpuHTM3;
    use crate::htm_builder::Population;

    #[test]
    fn test1() -> Result<(), String> {
        let c = Context::default()?;
        let p = HtmProgram::new(c.clone())?;
        let mut sdr = CpuInput::from_sparse_slice(&[4, 6, 14, 3], 16);
        let mut htm = CpuHTM::new_globally_uniform_prob(16, 16, 4, 12, 4);
        let output_sdr = htm.infer(&sdr, false);
        let mut htm2 = CpuHTM2::from(&htm);
        let output_sdr2 = htm2.infer(sdr.get_dense(), false);

        let mut ocl_sdr = OclInput::new(p.clone(), 16,16)?;
        ocl_sdr.set_sparse_from_slice(&[4, 6, 14, 3])?;
        let mut ocl_htm2 = OclHTM2::new(&htm2, p.clone())?;
        let output_sdr4 = ocl_htm2.infer(ocl_sdr.get_dense(), false)?;
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
    fn test7_union() -> Result<(), String> {
        fn union(a: &[u32], b: &[u32]) -> CpuSDR {
            let mut sdr1 = CpuSDR::new();
            sdr1.set(a);
            let mut sdr2 = CpuSDR::new();
            sdr2.set(b);
            sdr1.normalize();
            sdr2.normalize();
            sdr1.union(&sdr2)
        }
        assert_eq!(union(&[1, 5, 6, 76], &[1]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[1, 5, 6, 76], &[]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[], &[]).as_slice(), &[]);
        assert_eq!(union(&[1], &[]).as_slice(), &[1]);
        assert_eq!(union(&[], &[1]).as_slice(), &[1]);
        assert_eq!(union(&[1, 5, 6, 76], &[1, 5, 6, 76]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[1, 5, 6, 76], &[5, 76, 6, 1]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[1, 5, 6, 76], &[53, 746, 6, 1]).as_slice(), &[1, 5, 6, 53, 76, 746]);
        assert_eq!(union(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]).as_slice(), &[1, 3, 5, 6, 7, 53, 76, 78, 746]);
        Ok(())
    }
    #[test]
    fn test7_subtract() -> Result<(), String> {
        fn subtract(a: &[u32], b: &[u32]) -> CpuSDR {
            let mut sdr1 = CpuSDR::new();
            sdr1.set(a);
            let mut sdr2 = CpuSDR::new();
            sdr2.set(b);
            sdr1.normalize();
            sdr2.normalize();
            sdr1.subtract(&sdr2);
            sdr1
        }
        assert_eq!(subtract(&[1, 5, 6, 76], &[1]).as_slice(), &[5, 6, 76]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(subtract(&[], &[]).as_slice(), &[]);
        assert_eq!(subtract(&[1], &[]).as_slice(), &[1]);
        assert_eq!(subtract(&[], &[1]).as_slice(), &[]);
        assert_eq!(subtract(&[1], &[1]).as_slice(), &[]);
        assert_eq!(subtract(&[1], &[2]).as_slice(), &[1]);
        assert_eq!(subtract(&[1,2], &[2]).as_slice(), &[1]);
        assert_eq!(subtract(&[2,3], &[2]).as_slice(), &[3]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[1, 5, 6, 76]).as_slice(), &[]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[5, 76, 6, 1]).as_slice(), &[]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[53, 746, 6, 1]).as_slice(), &[5, 76]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]).as_slice(), &[76]);
        Ok(())
    }
    #[test]
    fn test8() -> Result<(), String> {
        let mut encoder = EncoderBuilder::new();
        let cat_enc = encoder.add_categorical(5, 10);
        let number_of_minicolumns = 50;
        let mut htm = CpuHTM2::new(
            encoder.input_size(),
            16,

        );
        let mut pop = Population::new(number_of_minicolumns,1);
        pop.add_uniform_rand_inputs_from_range(0..encoder.input_size(),25,544768);
        htm.add_population(&pop,64574);
        let mut hom = CpuHOM::new(1, number_of_minicolumns as u32);
        hom.hyp.activation_threshold = 8;
        hom.hyp.learning_threshold = 8;
        hom.hyp.predicted_decrement = -0.0;
        hom.hyp.permanence_decrement_increment = [0.,0.1];


        const NOTE_A:u32 = 0;
        const NOTE_B:u32 = 1;
        const NOTE_C_SHARP:u32 = 2;
        const NOTE_D:u32 = 3;
        const NOTE_E:u32 = 4;
        let sdrs:Vec<CpuInput> = (0..5).map(|i|{
            let mut sdr = CpuInput::new(encoder.input_size());
            cat_enc.encode(&mut sdr, i);
            sdr
        }).collect();
        for sdr in &sdrs{
            let activated = htm.infer(sdr.get_dense(),true);
            hom.infer(&activated,true);
        }
        hom.reset();
        let activated = htm.infer(sdrs[0].get_dense(),false);
        let mut predicted = hom.infer(&activated,false);
        for sdr in &sdrs[1..]{
            let activated = htm.infer(sdr.get_dense(),true);
            assert_eq!(predicted, activated);
            predicted = hom.infer(&activated,true);
        }

        Ok(())
    }

    #[test]
    fn test9() -> Result<(), String> {
        let p = HtmProgram::default()?;
        let input = CpuInput::from_sparse_slice(&[1,2,4,7,15], 16);
        let ocl_input = OclInput::from_cpu(&input,p.clone(),16)?;
        assert_eq!(input.cardinality(),ocl_input.cardinality(),"cardinality");
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
        assert_eq!(input.cardinality(),ocl_input.cardinality(),"cardinality");
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


    #[test]
    fn test12() -> Result<(), String> {
        let mut sdr = CpuInput::from_sparse_slice(&[4, 6, 14, 3], 16);
        let mut htm = CpuHTM::new_globally_uniform_prob(16, 16, 4, 12,36564);
        let output_sdr = htm.infer(&sdr, false);
        let mut htm2 = CpuHTM2::from(&htm);
        let output_sdr2 = htm2.infer(sdr.get_dense(), false);

        let output_sdr = output_sdr.to_vec();
        let output_sdr2 = output_sdr2.to_vec();
        assert_eq!(output_sdr, output_sdr2, "{:?}=={:?}", output_sdr, output_sdr2);
        Ok(())
    }

    #[test]
    fn test13() -> Result<(), String> {
        let mut encoder = EncoderBuilder::new();
        let cat_enc = encoder.add_categorical(5, 10);
        let number_of_minicolumns = encoder.input_size();
        let mut hom = CpuHOM::new(1, number_of_minicolumns);
        hom.hyp.activation_threshold = 8;
        hom.hyp.learning_threshold = 8;
        hom.hyp.predicted_decrement = -0.0;
        hom.hyp.permanence_decrement_increment = [0.,0.1];

        const NOTE_A:u32 = 0;
        const NOTE_B:u32 = 1;
        const NOTE_C_SHARP:u32 = 2;
        const NOTE_D:u32 = 3;
        const NOTE_E:u32 = 4;
        let sdrs:Vec<CpuInput> = (0..5).map(|i|{
            let mut sdr = CpuInput::new(encoder.input_size());
            cat_enc.encode(&mut sdr, i);
            sdr
        }).collect();
        for sdr in &sdrs{
            hom.infer(sdr.get_sparse(),true);
        }
        hom.reset();
        let mut predicted = hom.infer(sdrs[0].get_sparse(),false);
        for sdr in &sdrs[1..]{
            assert_eq!(&predicted, sdr.get_sparse());
            predicted = hom.infer(sdr.get_sparse(),true);
        }

        Ok(())
    }

    #[test]
    fn test14() -> Result<(), String> {
        const CELLS_PER_COL:u32=1;
        const CELLS_PER_IN:u32=4;
        const IN:u32=CELLS_PER_IN*5;
        const COLUMNS:u32=IN;
        const N:u32 = 4;
        const ACTIVATION_THRESHOLD:u32=4;
        let mut encoder = EncoderBuilder::new();
        let cat_enc = encoder.add_categorical(5, CELLS_PER_IN);
        let mut htm = CpuHTM2::new(encoder.input_size(), N );
        htm.add_globally_uniform_prob(COLUMNS as usize,COLUMNS/2,967786374);
        let mut hom = CpuHOM::new(1, COLUMNS);
        hom.hyp.activation_threshold = ACTIVATION_THRESHOLD;
        hom.hyp.learning_threshold = ACTIVATION_THRESHOLD;
        hom.hyp.predicted_decrement = -0.0;
        hom.hyp.permanence_decrement_increment = [0.,0.1];


        const NOTE_A:u32 = 0;
        const NOTE_B:u32 = 1;
        const NOTE_C_SHARP:u32 = 2;
        const NOTE_D:u32 = 3;
        const NOTE_E:u32 = 4;
        let sdrs:Vec<CpuInput> = (0..5).map(|i|{
            let mut sdr = CpuInput::new(encoder.input_size());
            cat_enc.encode(&mut sdr, i);
            sdr
        }).collect();
        for sdr in &sdrs{
            let activated = htm.infer(sdr.get_dense(),true);
            hom.infer(&activated,true);
        }
        hom.reset();
        let activated = htm.infer(sdrs[0].get_dense(),false);
        let mut predicted = hom.infer(&activated,false);
        for sdr in &sdrs[1..]{
            let activated = htm.infer(sdr.get_dense(),true);
            assert_eq!(predicted, activated);
            predicted = hom.infer(&activated,true);
        }

        Ok(())
    }

    #[test]
    fn test15() -> Result<(), String> {
        let mut htm = CpuHTM4::new(8,8);
        htm.add_globally_uniform_prob(64,4,0.8,464567);
        let in1 = CpuBitset::from_bools(&[false,false,false,true,false,false,true,false]);
        let in2 = CpuBitset::from_bools(&[false,false,false,true,false,true,true,false]);
        for _ in 0..128{
            let out1 = htm.infer(&in1,true);
            let out2 = htm.infer(&in2,true);
            println!("1{:?}",out1);
            println!("2{:?}",out2);
        }
        Ok(())
    }

    #[test]
    fn test16() {
        fn test(from:u32,to:u32) {
            let mut bits = CpuBitset::from_bools(&[true; 64]);
            bits.clear_range(from,to);
            for i in from..to {
                assert!(!bits.is_bit_on(i), "{},{}->{}", from,to,i);
            }
            for i in 0..from {
                assert!(bits.is_bit_on(i), "{},{}->{}", from,to,i);
            }
            for i in to..64 {
                assert!(bits.is_bit_on(i), "{},{}->{}", from,to,i);
            }
        }
        test(0,3);
        test(1,3);
        test(0,32);
        test(0,33);
        test(32,33);
        test(0,64);
        test(32,64);
        test(50,64);
        test(50,55);
    }

    // #[test]
    // fn test17() {
    //     let h = CpuHTM2::new_local_2d([4,4],[2,2],4,4,2.1, 64747);
    //
    // }

    #[test]
    fn test18() {
        let out_columns = 28 * (28 + 10 * 4);
        let mut htm_enc = EncoderBuilder::new();
        const S:u32 = 28 * 28;
        let img_enc = htm_enc.add_bits(S);
        let mut hom_enc = EncoderBuilder::new();
        let out_enc = hom_enc.add_bits(out_columns);
        let lbl_num = 10;
        let lbl_enc = hom_enc.add_categorical(lbl_num, 28 * 4);
        let mut sdr = CpuSDR::new();
        let data = (0..100).map(|i|CpuBitset::rand(htm_enc.input_size(),i*6546)).collect::<Vec<CpuBitset>>();
        let mut rand_seed = 53676;
        let labels = (0..data.len()).map(|_|{
            rand_seed = xorshift32(rand_seed);
            rand_seed%lbl_num
        }).collect::<Vec<u32>>();
        let label_sdrs = (0..lbl_num).map(|lbl|{
            let mut sdr = CpuSDR::new();
            lbl_enc.encode(&mut sdr,lbl);
            sdr
        }).collect::<Vec<CpuSDR>>();
        let mut htm1 = CpuHTM2::new(htm_enc.input_size(), 30);
        htm1.add_globally_uniform_prob(out_columns as usize,28 * 4,648679);
        let mut hom = CpuHOM::new(1, hom_enc.input_size());


        for (img, &lbl) in data.iter().zip(labels.iter()){
            let active_columns = htm1.infer(img,true);
            let predicted_columns = hom.infer(&active_columns,true);
            hom.infer(&label_sdrs[lbl as usize],true);
            hom.reset();
        }


    }


    #[test]
    fn test19() {
        let mut bits = CpuBitset::new(64);
        bits.set_bits_on(&[0,1,2,3,4,5,6,7,8]);
        assert_eq!(bits.cardinality_in_range(0,9),9);
        assert_eq!(bits.cardinality_in_range(0,8),8);
        assert_eq!(bits.cardinality_in_range(1,9),8);
        assert_eq!(bits.cardinality_in_range(2,2),0);
        assert_eq!(bits.cardinality_in_range(1,8),7);

        bits.set_bits_on(&[0,1,2,3,4,5,6,7,8,32,33,34]);
        assert_eq!(bits.cardinality_in_range(0,35),12);
        assert_eq!(bits.cardinality_in_range(0,34),11);
        assert_eq!(bits.cardinality_in_range(0,32),9);
        assert_eq!(bits.cardinality_in_range(1,32),8);
        assert_eq!(bits.cardinality_in_range(9,32),0);
        assert_eq!(bits.cardinality_in_range(9,35),3);
        assert_eq!(bits.cardinality_in_range(32,35),3);
    }

    #[test]
    fn test20() {
        let enc = EncoderBuilder::new().add_categorical(5,10);
        let mut i = CpuInput::new(64);
        i.set_sparse_from_slice(&[1,4,10,21,33,32,34,40]);
        assert_eq!(enc.find_category_with_highest_overlap_bitset(i.get_dense()),3);
        assert_eq!(enc.find_category_with_highest_overlap(i.get_sparse()),3);
        i.set_sparse_from_slice(&[1,4,10,21,34,40]);
        assert_eq!(enc.find_category_with_highest_overlap_bitset(i.get_dense()),0);
        assert_eq!(enc.find_category_with_highest_overlap(i.get_sparse()),0);
        i.set_sparse_from_slice(&[1,4,5,6,10,21,33,32,34,40]);
        assert_eq!(enc.find_category_with_highest_overlap_bitset(i.get_dense()),0);
        assert_eq!(enc.find_category_with_highest_overlap(i.get_sparse()),0);
        i.set_sparse_from_slice(&[1]);
        assert_eq!(enc.find_category_with_highest_overlap_bitset(i.get_dense()),0);
        assert_eq!(enc.find_category_with_highest_overlap(i.get_sparse()),0);
        i.set_sparse_from_slice(&[34]);
        assert_eq!(enc.find_category_with_highest_overlap_bitset(i.get_dense()),3);
        assert_eq!(enc.find_category_with_highest_overlap(i.get_sparse()),3);
    }

    // #[test]
    // fn test21() {//2={permanence:0.438, input_id:1}, 3={permanence:0.2970, input_id:1}
    //     let mut htm = CpuHTM3::new(3,2);
    //     let mut pop = Population::new(2,1);
    //     pop.add_uniform_rand_inputs_from_range(0..htm.input_size(),3,456457);
    //     htm.();
    //     let data = [
    //         CpuBitset::from_sdr(&[0,1],3),
    //         CpuBitset::from_sdr(&[1,2],3),
    //     ];
    //     let active_columns = [
    //         CpuBitset::from_sdr(&[0],2),
    //         CpuBitset::from_sdr(&[1],2),
    //     ];
    //     for _ in 0..20 {
    //         for (d, a) in data.iter().zip(active_columns.iter()) {
    //             htm.update_permanence_and_penalize(a, d)
    //         }
    //     }
    //     println!("{}",htm.permanence_threshold);
    // }
    #[test]
    fn test22() {

        let a = [1,4,6,7];
        assert_eq!(CpuSDR::from(&CpuBitset::from_sdr(&a,32)),CpuSDR::from(&a as &[u32]));
        let a = [6,7];
        assert_eq!(CpuSDR::from(&CpuBitset::from_sdr(&a,32)),CpuSDR::from(&a as &[u32]));
        let a = [31];
        assert_eq!(CpuSDR::from(&CpuBitset::from_sdr(&a,32)),CpuSDR::from(&a as &[u32]));
        let a = [63];
        assert_eq!(CpuSDR::from(&CpuBitset::from_sdr(&a,64)),CpuSDR::from(&a as &[u32]));
        let a = [4,63];
        assert_eq!(CpuSDR::from(&CpuBitset::from_sdr(&a,64)),CpuSDR::from(&a as &[u32]));
    }

    #[test]
    fn test23() {
        let sdr_grid = [[CpuSDR::from_slice(&[1,2,3])]];
        let o = CpuSDR::vote_conv2d_arr(4,0,[1,1],[1,1],[1,1],&sdr_grid);
        assert_eq!(o[0][0],sdr_grid[0][0])
    }
    #[test]
    fn test24() {
        let sdr_grid = [
            [
                CpuSDR::from_slice(&[1,2,3]),
                CpuSDR::from_slice(&[1,2,3])
            ],
            [
                CpuSDR::from_slice(&[1,2,3]),
                CpuSDR::from_slice(&[1,2,3])
            ]
        ];
        let o = CpuSDR::vote_conv2d_arr(4,0,[1,1],[2,2],[2,2],&sdr_grid);
        assert_eq!(o.len(),1);
        assert_eq!(o[0].len(),1);
        assert_eq!(o[0][0],sdr_grid[0][0])
    }
    #[test]
    fn test25() {
        let sdr_grid = [
            [
                CpuSDR::from_slice(&[1,2,3]),
                CpuSDR::from_slice(&[0,2,3])
            ],
            [
                CpuSDR::from_slice(&[0,2,3]),
                CpuSDR::from_slice(&[1,4,3])
            ]
        ];
        let o = CpuSDR::vote_conv2d_arr(2,0,[1,1],[2,2],[2,2],&sdr_grid);
        assert_eq!(o.len(),1);
        assert_eq!(o[0].len(),1);
        assert_eq!(o[0][0],CpuSDR::from_slice(&[2,3]))
    }
    #[test]
    fn test26() {
        let sdr_grid = [[
                CpuSDR::from_slice(&[1,2,3]), CpuSDR::from_slice(&[0,2,3]), CpuSDR::from_slice(&[0,2,4]), ], [
                CpuSDR::from_slice(&[0,2,3]), CpuSDR::from_slice(&[1,4,3]), CpuSDR::from_slice(&[1,4,3]), ]
        ];
        let o = CpuSDR::vote_conv2d_arr(2,0,[1,1],[2,2],[2,3],&sdr_grid);
        assert_eq!(o.len(),1);
        assert_eq!(o[0].len(),2);
        assert_eq!(o[0][0],CpuSDR::from_slice(&[2,3]));
        assert_eq!(o[0][1],CpuSDR::from_slice(&[3,4]));
    }

    #[test]
    fn test27() {
        let mut htm = CpuBigHTM::new(16, 16, 2,55348);
        let inp = CpuInput::from_sparse_slice(&[0,8],16);
        let out = CpuSDR::from_slice(&[4,7]);
        htm.update_permanence(&inp, &out);
        let out2 = htm.infer(&inp, false);
        assert_eq!(out2,out);
    }

    #[test]
    fn test28() {
        let mut htm1 = CpuBigHTM::new(16, 16, 2,55348);
        let mut htm2 = CpuBigHTM::new(16, 16, 2,55348);
        let inp = CpuInput::from_sparse_slice(&[0,8],16);
        let out1 = htm1.infer(&inp, false);
        htm1.update_permanence(&inp, &out1);
        let out2 = htm2.infer(&inp, true);
        assert_eq!(out2,out1);
        assert_eq!(htm1,htm2);
    }

    #[test]
    fn test29() {
        let sdr_grid = [[
            CpuSDR::from_slice(&[0]), CpuSDR::from_slice(&[1]), CpuSDR::from_slice(&[2]), ], [
            CpuSDR::from_slice(&[3]), CpuSDR::from_slice(&[4]), CpuSDR::from_slice(&[5]), ]
        ];
        let o = CpuSDR::vote_conv2d_transpose_arr([1,1],[2,2],[3,4],&|out0,out1|&sdr_grid[out0 as usize][out1 as usize]);
        assert_eq!(o.len(),3);
        assert_eq!(o[0].len(),4);
        assert_eq!(o[0][0],CpuSDR::from_slice(&[0]));
        assert_eq!(o[0][1],CpuSDR::from_slice(&[0,1]));
        assert_eq!(o[0][2],CpuSDR::from_slice(&[1,2]));
        assert_eq!(o[0][3],CpuSDR::from_slice(&[2]));
        assert_eq!(o[1][0],CpuSDR::from_slice(&[0,3]));
        assert_eq!(o[1][1],CpuSDR::from_slice(&[0,1,3,4]));
        assert_eq!(o[1][2],CpuSDR::from_slice(&[1,2,4,5]));
        assert_eq!(o[1][3],CpuSDR::from_slice(&[2,5]));
        assert_eq!(o[2][0],CpuSDR::from_slice(&[3]));
        assert_eq!(o[2][1],CpuSDR::from_slice(&[3,4]));
        assert_eq!(o[2][2],CpuSDR::from_slice(&[4,5]));
        assert_eq!(o[2][3],CpuSDR::from_slice(&[5]));
    }

    // #[test]
    // fn test30() {
    //     let sdr_grid = [
    //         [CpuSDR::from_slice(&[4, 13, 50, 64, 74, 94, 115, 162, 167, 202, 203, 253]), CpuSDR::from_slice(&[4, 5, 105, 112, 117, 188, 193, 202, 212, 217, 252, 255]), CpuSDR::from_slice(&[6, 7, 11, 48, 54, 75, 85, 86, 120, 178, 248, 251]), ],
    //         [CpuSDR::from_slice(&[11, 17, 20, 26, 77, 78, 88, 105, 128, 156, 173, 187]), CpuSDR::from_slice(&[1, 2, 5, 57, 158, 165, 170, 181, 194, 196, 203, 230]), CpuSDR::from_slice(&[1, 3, 27, 40, 62, 72, 74, 105, 136, 159, 171, 204]), ]
    //     ];
    //     let o = CpuSDR::vote_conv2d_arr(12,1,(1,1),(2,2),(2,3),&sdr_grid);
    //     assert_eq!(o.len(),1);
    //     assert_eq!(o[0].len(),2);
    //     let intersection = sdr_grid[0][0]
    //         .intersection(&sdr_grid[0][1])
    //         .intersection(&sdr_grid[1][0])
    //         .intersection(&sdr_grid[1][1]);
    //     assert!(intersection.subset(&o[0][0]));
    //     assert_eq!(o[0][1],CpuSDR::from_slice(&[1, 3, 4, 5, 86, 105, 158, 203, 230, 251, 252, 255]));
    // }
    #[test]
    fn test31() {
        let mut sdr = CpuSDR::from_slice(&[5, 41, 50, 51, 125, 157, 192, 220, 225, 230, 245, 253]);
        let votes = CpuSDR::new();
        sdr.randomly_extend_from(&votes, sdr.len());
        assert_eq!(sdr,CpuSDR::from_slice(&[5, 41, 50, 51, 125, 157, 192, 220, 225, 230, 245, 253]));
    }
    #[test]
    fn test32() {
        let o = [5, 41, 50, 51, 125, 157, 192, 220, 225, 230, 245, 253];
        let mut sdr = CpuSDR::from_slice(&o);
        let votes = CpuSDR::from_slice(&[34]);
        sdr.randomly_extend_from(&votes, sdr.len());
        assert_eq!(sdr.len(),o.len());
        assert!(sdr.contains(34));
    }
    #[test]
    fn test33() {
        let o = [5, 41, 50, 51, 125, 157, 192, 220, 225, 230, 245, 253];
        let mut sdr = CpuSDR::from_slice(&o);
        let votes = CpuSDR::from_slice(&[0,1,2,3,4,6,7,8,9,10,11,12]);
        sdr.randomly_extend_from(&votes, sdr.len());
        assert_eq!(sdr, votes);
    }
    #[test]
    fn test34() {
        let o = [5, 21, 78, 99, 101, 150, 168, 188, 189, 211, 217, 246];
        let mut sdr = CpuSDR::from_slice(&o);
        let votes = CpuSDR::from_slice(&[97]);
        sdr.randomly_extend_from(&votes, sdr.len());
        assert_eq!(sdr.len(),o.len());
        assert!(sdr.contains(97));
    }

    #[test]
    fn test35() -> Result<(),String>{
        let c = Context::default()?;
        let p = HtmProgram::new(c.clone())?;
        let mut dg = CpuDG2::new_2d(DgCoord2d::new_yx(32, 32), DgCoord2d::new_yx(4, 4), 8);
        dg.add_globally_uniform_prob(64, 8, 42354);
        let mut b = dg.make_bitset();
        let mut ocl_b = OclBitset::from_cpu(&b,p.clone())?;
        let mut ocl_dg = OclDG2::new(&dg,p)?;
        for trial in 0..8{
            b.clear_all();
            for _ in 0..32{
                let i = rand::random::<u32>() % b.size();
                b.set_bit_on(i);
            }
            ocl_b.copy_from(&b);
            let a1 = dg.compute_translation_invariant(&b, (1, 1));
            let ocl_a = ocl_dg.compute_translation_invariant(&ocl_b, (1, 1))?;
            let mut a2 = ocl_a.to_cpu()?;
            a2.sort();
            assert!(a1.cardinality() > 0);
            assert!(a1.cardinality() <= dg.n);
            assert_eq!(a1,a2);
        }
        Ok(())
    }

    #[test]
    fn test36() -> Result<(),String>{
        let c = Context::default()?;
        let p = HtmProgram::new(c.clone())?;
        let i = [2u32,32,32];
        let n = 4;
        let mut rand_seed = 42354;
        let mut htm = CpuHTM2::new(i.size(), n);
        let stride = [4,4];
        let kernel = [4,4];
        let mini_per_col = 16;
        let inp_per_mini = 10;
        let columns = [i.height(),i.width()].conv_out_size(&stride,&kernel);
        let mut pop = Population::new_2d_column_grid_with_3d_input(mini_per_col,stride ,kernel,i,1);
        rand_seed = pop.add_2d_column_grid_with_3d_input(0..htm.input_size(),mini_per_col,inp_per_mini,stride ,kernel,i,rand_seed);
        htm.add_population(&pop,rand_seed);
        let mut htms:Vec<CpuHTM2> = (0..columns.size()).map(|_| CpuHTM2::new(i.size(), n)).collect();
        let mut rand_seed = 42354;
        let column_area = [i[0],kernel[0],kernel[1]];
        let columns3d = [mini_per_col,columns[0],columns[1]];
        let mut pops:Vec<Population> = (0..htms.len()).map(|_|Population::new(mini_per_col as usize,1)).collect();
        for y in 0..columns[0]{
            for x in 0..columns[1]{
                let mut pop = &mut pops[columns.index(y,x) as usize];
                let from= [0,y*stride[0],x*stride[1]];
                let to = from.add(&column_area);
                rand_seed = pop.add_uniform_rand_inputs_from_area(0..i.size(),i,from..to,inp_per_mini,rand_seed);
            }
        }
        for y in 0..columns[0] {
            for x in 0..columns[1] {
                let h= &mut htms[columns.index(y,x) as usize];
                let pop= &mut pops[columns.index(y,x) as usize];
                rand_seed = h.add_population(pop,rand_seed);
            }
        }
        for y in 0..columns[0] {
            for x in 0..columns[1] {
                let h= &mut htms[columns.index(y,x) as usize];
                for (z,m) in h.minicolumns_as_slice().iter().enumerate(){
                    let j = ((y*columns[1] + x)*mini_per_col) as usize + z;
                    let m2 = &htm.minicolumns_as_slice()[j as usize];
                    for (c,c2) in h.feedforward_connections_as_slice()[m.range()].iter().zip(htm.feedforward_connections_as_slice()[m2.range()].iter()){
                        assert_eq!(c.permanence,c2.permanence);
                    }
                }
            }
        }
        for y in 0..columns[0] {
            for x in 0..columns[1] {
                let h= &mut htms[columns.index(y,x) as usize];
                for z in 0..h.minicolumns_as_slice().len(){
                    let j = columns3d.idx([z as u32,y,x]);
                    let m2 = &htm.minicolumns_as_slice()[j as usize];
                    let r = h.minicolumns_as_slice()[z].range();
                    for (c,c2) in h.feedforward_connections_as_mut_slice()[r].iter_mut().zip(htm.feedforward_connections_as_slice()[m2.range()].iter()){
                        assert_eq!(c.input_id,c2.input_id);
                        c.permanence = c2.permanence;
                    }
                }
            }
        }
        let minicolumn_stride = columns.product();
        let mut ocl_htm = OclHTM2::new(&htm,p.clone())?;
        for trial in 0..8{
            let mut b = CpuBitset::rand(i.size(),4553466+432*trial);
            let mut a1 = htm.compute_and_group_into_columns(mini_per_col as usize,minicolumn_stride as usize,&b);
            let a2:Vec<CpuSDR> = htms.iter_mut().map(|htm|htm.compute(&b)).collect();
            for y in 0..columns[0] {
                for x in 0..columns[1] {
                    let h= &mut htms[columns.index(y,x) as usize];
                    for (z,m) in h.minicolumns_as_slice().iter().enumerate(){
                        let j = columns3d.idx([z as u32,y,x]);
                        let m2 = &htm.minicolumns_as_slice()[j as usize];
                        assert_eq!(m2.overlap,m.overlap);
                    }
                }
            }
            let mut joined = CpuSDR::new();
            let ocl_b = OclBitset::from_cpu(&b,p.clone())?;
            let ocl_a = ocl_htm.compute_and_group_into_columns(mini_per_col as usize,minicolumn_stride as usize,&ocl_b)?;
            let mut a3 = ocl_a.to_cpu()?;
            a3.sort();
            for (k,a) in a2.iter().enumerate(){
                let [ col0,col1] = columns.pos(k as u32);
                assert_eq!(a.len(),n as usize);
                assert!(a.is_normalized());
                let a_mapped:Vec<u32> = a.as_slice().iter().map(|&i|columns3d.idx([i,col0,col1])).collect();
                joined.extend_from_slice(&a_mapped);
                
            }
            joined.sort();
            assert!(joined.is_normalized(), "joined={:?}");
            a1.sort();
            assert_eq!(a1,joined);
            assert_eq!(a1,a3);
        }
        Ok(())
    }
    #[test]
    fn test37() -> Result<(),String>{
        let i = 32;
        let mut htm = CpuBigHTM::new(i, 32,4,643765);
        let mut htm2 = htm.clone();
        let b = CpuBitset::rand(i,4547856);
        let b = CpuInput::from_dense(b);
        let o = htm.infer(&b,true);
        let o2 = htm2.infer(&b,false);
        htm2.update_permanence(&b,&o2);
        assert_eq!(o,o2);
        for (m,m2) in htm.minicolumns_as_slice().iter().zip(htm2.minicolumns_as_slice().iter()){
            assert_eq!(m.segments.len(),m2.segments.len());
            for (s,s2) in m.segments.iter().zip(m2.segments.iter()){
                assert_eq!(s.synapses,s2.synapses);
            }
        }
        Ok(())
    }
}
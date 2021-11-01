
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
mod rand;

pub use crate::rand::auto_gen_seed;
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
    use crate::rand::xorshift32;

    #[test]
    fn test1() -> Result<(), String> {
        let c = Context::default()?;
        let p = HtmProgram::new(c.clone())?;
        let mut sdr = CpuInput::from_sparse_slice(&[4, 6, 14, 3], 16);
        let mut htm = CpuHTM::new_globally_uniform_prob(16, 16, 4, 12, 4);
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
        let cat_enc = encoder.add_categorical(5, 10);
        let number_of_minicolumns = 50;
        let mut htm = CpuHTM2::new_globally_uniform_prob(
            encoder.input_size(),
            number_of_minicolumns,
            16,
            25,
            544768,
        );
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
            let activated = htm.infer2(sdr.get_dense(),true);
            hom.infer(&activated,true);
        }
        hom.reset();
        let activated = htm.infer2(sdrs[0].get_dense(),false);
        let mut predicted = hom.infer(&activated,false);
        for sdr in &sdrs[1..]{
            let activated = htm.infer2(sdr.get_dense(),true);
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


    #[test]
    fn test12() -> Result<(), String> {
        let mut sdr = CpuInput::from_sparse_slice(&[4, 6, 14, 3], 16);
        let mut htm = CpuHTM::new_globally_uniform_prob(16, 16, 4, 12,36564);
        let output_sdr = htm.infer(&sdr, false);
        let mut htm2 = CpuHTM2::from(&htm);
        let output_sdr2 = htm2.infer2(sdr.get_dense(), false);

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
        let mut htm = CpuHTM2::new_globally_uniform_prob(
            encoder.input_size(),
            COLUMNS,
            N,
            COLUMNS/2,
            967786374
        );
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
            let activated = htm.infer2(sdr.get_dense(),true);
            hom.infer(&activated,true);
        }
        hom.reset();
        let activated = htm.infer2(sdrs[0].get_dense(),false);
        let mut predicted = hom.infer(&activated,false);
        for sdr in &sdrs[1..]{
            let activated = htm.infer2(sdr.get_dense(),true);
            assert_eq!(predicted, activated);
            predicted = hom.infer(&activated,true);
        }

        Ok(())
    }

    #[test]
    fn test15() -> Result<(), String> {
        let mut htm = CpuHTM4::new_globally_uniform_prob(8,64,8,4,0.8,464567);
        let in1 = CpuBitset::from_bools(&[false,false,false,true,false,false,true,false]);
        let in2 = CpuBitset::from_bools(&[false,false,false,true,false,true,true,false]);
        for _ in 0..128{
            let out1 = htm.infer4(&in1,true);
            let out2 = htm.infer4(&in2,true);
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

    #[test]
    fn test17() {
        let h = CpuHTM2::new_local_2d((4,4),(2,2),4,4,2.1, 64747);

    }

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
        let mut htm1 = CpuHTM2::new_globally_uniform_prob(htm_enc.input_size(), out_columns, 30, 28 * 4,648679);
        let mut hom = CpuHOM::new(1, hom_enc.input_size());


        for (img, &lbl) in data.iter().zip(labels.iter()){
            let active_columns = htm1.infer2(img,true);
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
}
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_extra)]
#![feature(array_map)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]

pub mod mat;
pub mod num;
pub mod kernel;


#[cfg(test)]
mod tests {
    use super::*;
    use ocl::{SpatialDims, Platform, Device, Context, Program, Queue, Buffer, flags, Kernel, ProQue};
    use ocl::core::BufferRegion;
    use crate::mat::{Mat, MatError};
    use crate::kernel::LinAlgProgram;

    #[test]
    fn test_mat0() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let m1 = Mat::array2(&p, [[1, 10], [100, 1000]])?;
        let m2 = Mat::array2(&p, [[2, 3], [4, 5]])?;
        assert_eq!(m1.strides(), &[2, 1], "m1 strides");
        assert_eq!(m2.strides(), &[2, 1], "m2 strides");
        let m3 = m1.mm(&m2)?;
        assert_eq!(m1.to_vec().unwrap(), vec![1, 10, 100, 1000], "m1");
        assert_eq!(m2.to_vec().unwrap(), vec![2, 3, 4, 5], "m2");
        assert_eq!(m3.to_vec().unwrap(), vec![42, 53, 4200, 5300], "m3");
        assert_eq!(m3, Mat::array2(&p, [[42, 53], [4200, 5300]])?);
        Ok(())
    }

    #[test]
    fn test_mat1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let m1 = Mat::array2(&p, [[1, 2], [3, 4]])?;
        let m2 = Mat::array2(&p, [[1, 2], [3, 4]])?;
        assert_eq!(m1.strides(), &[2, 1], "m1 strides");
        assert_eq!(m2.strides(), &[2, 1], "m2 strides");
        let m3 = m1.mm(&m2)?;
        assert_eq!(m1.to_vec().unwrap(), vec![1, 2, 3, 4], "m1");
        assert_eq!(m2.to_vec().unwrap(), vec![1, 2, 3, 4], "m2");
        assert_eq!(m3.to_vec().unwrap(), vec![7, 10, 15, 22], "m3");
        assert_eq!(m3, Mat::array2(&p, [[1, 2], [3, 4]])?);
        Ok(())
    }

    #[test]
    fn test_mat2() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let m1 = Mat::array3(&p, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])?;
        let m2 = Mat::array3(&p, [[[0, 1], [2, 3]], [[4, 5], [6, 7]]])?;
        assert_eq!(m1.strides(), &[4, 2, 1], "m1 strides");
        assert_eq!(m2.strides(), &[4, 2, 1], "m2 strides");
        let m3 = m1.mm(&m2)?;
        assert_eq!(m1.to_vec().unwrap(), vec![1, 2, 3, 4, 5, 6, 7, 8], "m1");
        assert_eq!(m2.to_vec().unwrap(), vec![0, 1, 2, 3, 4, 5, 6, 7], "m2");
        assert_eq!(m3.to_vec().unwrap(), vec![4, 7, 8, 15, 56, 67, 76, 91], "m3");
        assert_eq!(m3, Mat::array3(&p, [[[4, 7], [8, 15]], [[56, 67], [76, 91]]])?);
        Ok(())
    }

    #[test]
    fn test_mat3() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let m1 = Mat::array3(&p, [[[0, 1], [0, 0]], [[0, 0], [0, 0]]])?;
        let m2 = Mat::array3(&p, [[[0, 1], [0, 0]], [[0, 0], [0, 0]]])?;
        assert_eq!(m1.strides(), &[4, 2, 1], "m1 strides");
        assert_eq!(m2.strides(), &[4, 2, 1], "m1 strides");
        let m3 = m1.mm(&m2)?;
        assert_eq!(m1.to_vec().unwrap(), vec![0, 1, 0, 0, 0, 0, 0, 0], "m1");
        assert_eq!(m2.to_vec().unwrap(), vec![0, 1, 0, 0, 0, 0, 0, 0], "m2");
        assert_eq!(m3.to_vec().unwrap(), vec![0, 0, 0, 0, 0, 0, 0, 0], "m3");
        assert_eq!(m3, Mat::array3(&p, [[[0, 0], [0, 0]], [[0, 0], [0, 0]]])?);
        Ok(())
    }

    #[test]
    fn test_mat4() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let m1 = Mat::array3(&p, [[[0, 1], [0, 0]], [[0, 0], [0, 0]]])?;
        let m2 = Mat::array3(&p, [[[0, 0], [1, 0]], [[0, 0], [0, 0]]])?;
        assert_eq!(m1.strides(), &[4, 2, 1], "m1 strides");
        assert_eq!(m2.strides(), &[4, 2, 1], "m1 strides");
        let m3 = m1.mm(&m2)?;
        assert_eq!(m1.to_vec().unwrap(), vec![0, 1, 0, 0, 0, 0, 0, 0], "m1");
        assert_eq!(m2.to_vec().unwrap(), vec![0, 0, 1, 0, 0, 0, 0, 0], "m2");
        assert_eq!(m3.to_vec().unwrap(), vec![1, 0, 0, 0, 0, 0, 0, 0], "m3");
        assert_eq!(m3, Mat::array3(&p, [[[1, 0], [0, 0]], [[0, 0], [0, 0]]])?);
        Ok(())
    }

    #[test]
    fn test_mat5() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let m1 = Mat::array3(&p, [[[0, 0], [0, 0]], [[0, 1], [0, 0]]])?;
        let m2 = Mat::array3(&p, [[[0, 0], [0, 0]], [[0, 0], [1, 0]]])?;
        assert_eq!(m1.strides(), &[4, 2, 1], "m1 strides");
        assert_eq!(m2.strides(), &[4, 2, 1], "m1 strides");
        let m3 = m1.mm(&m2)?;
        assert_eq!(m1.to_vec().unwrap(), vec![0, 0, 0, 0, 0, 1, 0, 0], "m1");
        assert_eq!(m2.to_vec().unwrap(), vec![0, 0, 0, 0, 0, 0, 1, 0], "m2");
        assert_eq!(m3.to_vec().unwrap(), vec![0, 0, 0, 0, 1, 0, 0, 0], "m3");
        assert_eq!(m3, Mat::array3(&p, [[[0, 0], [0, 0]], [[1, 0], [0, 0]]])?);
        Ok(())
    }

    #[test]
    fn test_mat6() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m2 = Mat::array3(&p, [[[2, 3], [4, 5]], [[6, 7], [8, 9]]])?;
        assert_eq!(m1.strides(), &[4, 2, 1], "m1 strides");
        assert_eq!(m2.strides(), &[4, 2, 1], "m1 strides");
        let m3 = m1.mm(&m2)?;
        assert_eq!(m1.to_vec().unwrap(), vec![1, 10, 100, 1000, 10000, 100000, 1000000, 10000000], "m1");
        assert_eq!(m2.to_vec().unwrap(), vec![2, 3, 4, 5, 6, 7, 8, 9], "m2");
        assert_eq!(m3.to_vec().unwrap(), vec![42, 53, 4200, 5300, 860000, 970000, 86000000, 97000000], "m3");
        assert_eq!(m3, Mat::array3(&p, [[[42, 53], [4200, 5300]], [[860000, 970000], [86000000, 97000000]]])?);
        Ok(())
    }

    #[test]
    fn test_add_assign_mat1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m2 = Mat::array3(&p, [[[2, 3], [4, 5]], [[6, 7], [8, 9]]])?;
        m1.add_mat(&m2)?;
        assert_eq!(m1.to_vec().unwrap(), vec![3, 13, 104, 1005, 10006, 100007, 1000008, 10000009], "m1");
        Ok(())
    }

    #[test]
    fn test_add_assign_mat2() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m2 = Mat::array3(&p, [[[2, 3], [4, 5]], [[6, 7], [8, 9]]])?;
        let m1 = m1 + &m2;
        assert_eq!(m1.to_vec().unwrap(), vec![3, 13, 104, 1005, 10006, 100007, 1000008, 10000009], "m1");
        Ok(())
    }

    #[test]
    fn test_add_assign_mat3() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m2 = Mat::array3(&p, [[[2, 3], [4, 5]], [[6, 7], [8, 9]]])?;
        m1 += &m2;
        assert_eq!(m1.to_vec().unwrap(), vec![3, 13, 104, 1005, 10006, 100007, 1000008, 10000009], "m1");
        Ok(())
    }


    #[test]
    fn test_sub_assign_mat1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m2 = Mat::array3(&p, [[[2, 3], [4, 5]], [[6, 7], [8, 9]]])?;
        m1 -= &m2;
        assert_eq!(m1.to_vec().unwrap(), vec![-1, 7, 96, 995, 9994, 99993, 999992, 9999991], "m1");
        Ok(())
    }

    #[test]
    fn test_sub_assign_mat2() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m2 = Mat::array3(&p, [[[2, 3], [4, 5]], [[6, 7], [8, 9]]])?;
        let m1 = m1 - &m2;
        assert_eq!(m1.to_vec().unwrap(), vec![-1, 7, 96, 995, 9994, 99993, 999992, 9999991], "m1");
        Ok(())
    }

    #[test]
    fn mul_assign_mat1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m2 = Mat::array3(&p, [[[2, 3], [4, 5]], [[6, 7], [8, 9]]])?;
        m1 *= &m2;
        assert_eq!(m1.to_vec().unwrap(), vec![2, 30, 400, 5000, 60000, 700000, 8000000, 90000000], "m1");
        Ok(())
    }

    #[test]
    fn mul_assign_mat2() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m2 = Mat::array3(&p, [[[2, 3], [4, 5]], [[6, 7], [8, 9]]])?;
        let m1 = m1 * &m2;
        assert_eq!(m1.to_vec().unwrap(), vec![2, 30, 400, 5000, 60000, 700000, 8000000, 90000000], "m1");
        Ok(())
    }

    #[test]
    fn div_assign_mat1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m2 = Mat::array3(&p, [[[2, 3], [4, 5]], [[6, 7], [8, 9]]])?;
        m1 /= &m2;
        assert_eq!(m1.to_vec().unwrap(), vec![0, 3, 25, 200, 1666, 14285, 125000, 1111111], "m1");
        Ok(())
    }

    #[test]
    fn div_assign_mat2() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m2 = Mat::array3(&p, [[[2, 3], [4, 5]], [[6, 7], [8, 9]]])?;
        let m1 = m1 / &m2;
        assert_eq!(m1.to_vec().unwrap(), vec![0, 3, 25, 200, 1666, 14285, 125000, 1111111], "m1");
        Ok(())
    }

    #[test]
    fn eq_assign_mat1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 5]], [[10000, 100000], [1000000, 9]]])?;
        let m2 = Mat::array3(&p, [[[2, 10], [4, 5]], [[6, 7], [8, 9]]])?;
        let m1 = m1.eq_mat(&m2)?;
        assert_eq!(m1.to_vec().unwrap(), vec![0,1,0,1,0,0,0,1], "m1");
        Ok(())
    }
    #[test]
    fn ne_assign_mat1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 5]], [[10000, 100000], [1000000, 9]]])?;
        let m2 = Mat::array3(&p, [[[2, 10], [4, 5]], [[6, 7], [8, 9]]])?;
        let m1 = m1.ne_mat(&m2)?;
        assert_eq!(m1.to_vec().unwrap(), vec![1,0,1,0,1,1,1,0], "m1");
        Ok(())
    }
    #[test]
    fn gt_assign_mat1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 5]], [[10000, 100000], [1000000, -9]]])?;
        let m2 = Mat::array3(&p, [[[2, 100], [4, 50]], [[10000, 100000], [8, 9]]])?;
        let m1 = m1.gt_mat(&m2)?;
        assert_eq!(m1.to_vec().unwrap(), vec![0, 0, 1, 0, 0, 0, 1, 0], "m1");
        Ok(())
    }
    #[test]
    fn ge_assign_mat1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 5]], [[10000, 100000], [1000000, -9]]])?;
        let m2 = Mat::array3(&p, [[[2, 100], [4, 50]], [[10000, 100000], [8, 9]]])?;
        let m1 = m1.ge_mat(&m2)?;
        assert_eq!(m1.to_vec().unwrap(), vec![0,0,1,0,1,1,1,0], "m1");
        Ok(())
    }

    #[test]
    fn test_add_scalar1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m1 = m1 + 4;
        assert_eq!(m1.to_vec().unwrap(), vec![5, 14, 104, 1004, 10004, 100004, 1000004, 10000004], "m1");
        Ok(())
    }

    #[test]
    fn test_sub_scalar1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m1 = m1 - 4;
        assert_eq!(m1.to_vec().unwrap(), vec![-3, 6, 96, 996, 9996, 99996, 999996, 9999996], "m1");
        Ok(())
    }

    #[test]
    fn test_mul_scalar1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m1 = m1 * 4;
        assert_eq!(m1.to_vec().unwrap(), vec![4, 40, 400, 4000, 40000, 400000, 4000000, 40000000], "m1");
        Ok(())
    }

    #[test]
    fn test_div_scalar1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        let m1 = m1 / 4;
        assert_eq!(m1.to_vec().unwrap(), vec![0, 2, 25, 250, 2500, 25000, 250000, 2500000], "m1");
        Ok(())
    }


    #[test]
    fn test_add_assign_scalar1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        m1 += 4;
        assert_eq!(m1.to_vec().unwrap(), vec![5, 14, 104, 1004, 10004, 100004, 1000004, 10000004], "m1");
        Ok(())
    }

    #[test]
    fn test_sub_assign_scalar1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        m1 -= 4;
        assert_eq!(m1.to_vec().unwrap(), vec![-3, 6, 96, 996, 9996, 99996, 999996, 9999996], "m1");
        Ok(())
    }

    #[test]
    fn test_mul_assign_scalar1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        m1 *= 4;
        assert_eq!(m1.to_vec().unwrap(), vec![4, 40, 400, 4000, 40000, 400000, 4000000, 40000000], "m1");
        Ok(())
    }

    #[test]
    fn test_div_assign_scalar1() -> Result<(), String> {
        let p = LinAlgProgram::gpu()?;
        let mut m1 = Mat::array3(&p, [[[1, 10], [100, 1000]], [[10000, 100000], [1000000, 10000000]]])?;
        m1 /= 4;
        assert_eq!(m1.to_vec().unwrap(), vec![0, 2, 25, 250, 2500, 25000, 250000, 2500000], "m1");
        Ok(())
    }
}

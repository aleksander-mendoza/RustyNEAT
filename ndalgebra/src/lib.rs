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
}

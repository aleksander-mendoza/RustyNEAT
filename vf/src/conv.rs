use crate::*;
use std::ops::{Range, Mul, Add, Sub, Div};
use num_traits::Zero;

pub fn in_range_begin<T: Mul + Copy, const DIM: usize>(out_position: &[T; DIM], stride: &[T; DIM]) -> [T; DIM] {
    out_position.mul(stride)
}

/**returns the range of inputs that connect to a specific output neuron*/
pub fn in_range<T: Copy + Mul + Add + Zero, const DIM: usize>(out_position: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> Range<[T; DIM]> {
    let from = in_range_begin(out_position, stride);
    let to = from.add(kernel_size);
    from..to
}

/**returns the range of inputs that connect to a specific patch of output neuron.
That output patch starts this position specified by this vector*/
pub fn in_range_with_custom_size<T: Copy + Mul + Add + Zero, const DIM: usize>(out_position: &[T; DIM], output_patch_size: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> Range<[T; DIM]> {
    if output_patch_size.all_gt_scalar(T::zero()) {
        let from = in_range_begin(out_position, stride);
        let to = in_range_begin(&out_position.add(output_patch_size)._sub_scalar(T::one()), stride);
        let to = to._add(kernel_size);
        from..to
    } else {
        [T::zero(); DIM]..[T::zero(); DIM]
    }
}

/**returns the range of outputs that connect to a specific input neuron*/
pub fn out_range<T: Div + Add + Sub, const DIM: usize>(in_position: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> Range<[T; DIM]> {
    //out_position * stride .. out_position * stride + kernel
    //out_position * stride ..= out_position * stride + kernel - 1
    //
    //in_position_from == out_position * stride
    //in_position_from / stride == out_position
    //round_down(in_position / stride) == out_position_to
    //
    //in_position_to == out_position * stride + kernel - 1
    //(in_position_to +1 - kernel)/stride == out_position
    //round_up((in_position +1 - kernel)/stride) == out_position_from
    //round_down((in_position +1 - kernel + stride - 1)/stride) == out_position_from
    //round_down((in_position - kernel + stride)/stride) == out_position_from
    //
    //(in_position - kernel + stride)/stride ..= in_position / stride
    //(in_position - kernel + stride)/stride .. in_position / stride + 1
    let to = in_position.div(stride)._add_scalar(T::one());
    let from = in_position.add(stride)._sub(kernel_size)._div(stride);
    from..to
}

pub fn out_transpose_kernel<T: Div + Add + Sub + Ord, const DIM: usize>(kernel: &[T; DIM], stride: &[T; DIM]) -> [T; DIM] {
    // (in_position - kernel + stride)/stride .. in_position / stride + 1
    //  in_position / stride + 1 - (in_position - kernel + stride)/stride
    //  (in_position- (in_position - kernel + stride))/stride + 1
    //  (kernel - stride)/stride + 1
    debug_assert!(kernel.all_ge(stride));
    kernel.sub(stride)._div(stride)._add_scalar(T::one())
}

/**returns the range of outputs that connect to a specific input neuron.
output range is clipped to 0, so that you don't get overflow on negative values when dealing with unsigned integers.*/
pub fn out_range_clipped<T: Div + Add + Sub + Ord, const DIM: usize>(in_position: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> Range<[T; DIM]> {
    let to = in_position.div(stride)._add_scalar(T::one());
    let from = in_position.add(stride)._max(kernel_size)._sub(kernel_size)._div(stride);
    from..to
}

pub fn out_range_clipped_both_sides<T: Div + Add + Sub + Ord, const DIM: usize>(in_position: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM], max_bounds: &[T; DIM]) -> Range<[T; DIM]> {
    let mut r = out_range_clipped(in_position, stride, kernel_size);
    r.end.min_(max_bounds);
    r
}

pub fn out_size<T: Div + Add + Sub + Ord, const DIM: usize>(input: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> [T; DIM] {
    assert!(kernel_size.all_le(input), "Kernel size {:?} is larger than the input shape {:?} ", kernel_size, input);
    let input_sub_kernel = input.sub(kernel_size);
    assert!(input_sub_kernel.rem(stride).all_eq_scalar(T::zero()), "Convolution stride {:?} does not evenly divide the input shape {:?}-{:?}={:?} ", stride, input, kernel_size, input_sub_kernel);
    input_sub_kernel._div(stride)._add_scalar(T::one())
    //(input-kernel)/stride+1 == output
}

pub fn in_size<T: Div + Add + Sub + Ord, const DIM: usize>(output: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> [T; DIM] {
    assert!(output.all_gt_scalar(T::zero()), "Output size {:?} contains zero", output);
    output.sub_scalar(T::one())._mul(stride)._add(kernel_size)
    //input == stride*(output-1)+kernel
}

pub fn stride<T: Div + Add + Sub + Ord, const DIM: usize>(input: &[T; DIM], out_size: &[T; DIM], kernel_size: &[T; DIM]) -> [T; DIM] {
    assert!(kernel_size.all_le(input), "Kernel size {:?} is larger than the input shape {:?}", kernel_size, input);
    let input_sub_kernel = input.sub(kernel_size);
    let out_size_minus_1 = out_size.sub_scalar(T::one());
    assert!(input_sub_kernel.rem_default_zero(&out_size_minus_1, T::zero()).all_eq_scalar(T::zero()), "Output shape {:?}-1 does not evenly divide the input shape {:?}", out_size, input);
    input_sub_kernel._div_default_zero(&out_size_minus_1, T::one())
    //(input-kernel)/(output-1) == stride
}

pub fn compose<T: Div + Add + Sub + Ord, const DIM: usize>(self_stride: &[T; DIM], self_kernel: &[T; DIM], next_stride: &[T; DIM], next_kernel: &[T; DIM]) -> ([T; DIM], [T; DIM]) {
    //(A-kernelA)/strideA+1 == B
    //(B-kernelB)/strideB+1 == C
    //((A-kernelA)/strideA+1-kernelB)/strideB+1 == C
    //(A-kernelA+(1-kernelB)*strideA)/(strideA*strideB)+1 == C
    //(A-(kernelA-(1-kernelB)*strideA))/(strideA*strideB)+1 == C
    //(A-(kernelA+(kernelB-1)*strideA))/(strideA*strideB)+1 == C
    //    ^^^^^^^^^^^^^^^^^^^^^^^^^^^                    composed kernel
    //                                   ^^^^^^^^^^^^^^^ composed stride
    let composed_kernel = next_kernel.sub_scalar(T::one())._mul(self_stride)._add(self_kernel);
    let composed_stride = self_stride.mul(next_stride);
    (composed_stride, composed_kernel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test5() {
        for x in 1..3 {
            for y in 1..4 {
                for sx in 1..2 {
                    for sy in 1..2 {
                        for ix in 1..3 {
                            for iy in 1..4 {
                                let kernel = [x, y];
                                let stride = [x, y];
                                let output_size = [ix, iy];
                                let input_size = in_size(&output_size, &stride, &kernel);
                                assert_eq!(output_size, out_size(&input_size, &stride, &kernel));
                                for ((&expected, &actual), &out) in stride.iter().zip(stride(&input_size, &output_size, &kernel).iter()).zip(output_size.iter()) {
                                    if out != 1 {
                                        assert_eq!(expected, actual);
                                    } else {
                                        assert_eq!(1, actual);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test6() {
        for output_idx in 0..24 {
            for x in 1..5 {
                for sx in 1..5 {
                    let i = in_range(&[output_idx], &[sx], &[x]);
                    let i_r = i.start[0]..i.end[0];
                    for i in i_r.clone() {
                        let o = out_range(&[i], &[sx], &[x]);
                        let o_r = o.start[0]..o.end[0];
                        assert!(o_r.contains(&output_idx), "o_r={:?}, i_r={:?} output_idx={} sx={} x={}", o_r, i_r, output_idx, sx, x)
                    }
                }
            }
        }
    }

    #[test]
    fn test7() {
        for input_idx in 0..24 {
            for x in 1..5 {
                for sx in 1..5 {
                    let o = out_range(&[input_idx], &[sx], &[x]);
                    let o_r = o.start[0]..o.end[0];
                    for o in o_r.clone() {
                        let i = in_range(&[o], &[sx], &[x]);
                        let i_r = i.start[0]..i.end[0];
                        assert!(i_r.contains(&input_idx), "o_r={:?}, i_r={:?} input_idx={} sx={} x={}", o_r, i_r, input_idx, sx, x)
                    }
                }
            }
        }
    }
}
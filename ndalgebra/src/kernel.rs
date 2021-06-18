use ocl::{ProQue, SpatialDims, Error, Platform, Device, DeviceType};
use std::fmt::{Formatter, Display};
use std::marker::PhantomData;
use ocl::builders::KernelBuilder;
use crate::num::Num;
use ocl::core::{DeviceInfoResult, DeviceInfo};

pub const MAX_MAT_DIMS: usize = 20;

fn source<N: Num>(fmt: &mut Formatter<'_>, _p: PhantomData<N>) -> std::fmt::Result {
    for dims in 2..MAX_MAT_DIMS {
        write!(fmt, "\
__kernel void {t}_mm{dims}(
                 __global {t} * lhs, // lhs.shape==(s*,j,i)
                 __global {t} * rhs, // rhs.shape==(s*,i,k)
                 __global {t} * out, // out.shape==(s*,j,k)
                 size_t i_len,
                 size_t lhs_j_stride,
                 size_t lhs_i_stride,
                 size_t rhs_i_stride,
                 size_t rhs_k_stride,
                 size_t out_j_stride,
                 size_t out_k_stride", dims = dims, t = N::opencl_type_str())?;
        for s in 0..(dims - 2) {
            write!(fmt, ",
                 size_t dim_s{},
                 size_t lhs_s{}_stride,
                 size_t rhs_s{}_stride,
                 size_t out_s{}_stride", s, s, s, s)?;
        }
        write!(fmt, "){{
    size_t j = get_global_id(0);
    size_t k = get_global_id(1);
    size_t s = get_global_id(2);
    size_t elements_in_hyper_plane0 = 1;")?;
        for s in 0..(dims - 2) {
            writeln!(fmt, "
    size_t elements_in_hyper_plane{} = dim_s{} * elements_in_hyper_plane{};", s + 1, s, s)?;
        }
        for s in (0..(dims - 2)).rev() {
            write!(fmt, "
    size_t s{} = (s % elements_in_hyper_plane{}) / elements_in_hyper_plane{};", s, s + 1, s)?;
        }

        for var in ["out", "lhs", "rhs"] {
            write!(fmt, "
    size_t {var}_s_offset = 0", var = var)?;
            for s in (0..(dims - 2)).rev() {
                write!(fmt, " + s{} * {var}_s{}_stride", s, s, var = var)?;
            }
            write!(fmt, ";")?;
        }

        write!(fmt, "
    {t} sum = 0;
    size_t out_offset = out_s_offset+j*out_j_stride+k*out_k_stride;
    size_t lhs_offset = lhs_s_offset+j*lhs_j_stride;
    size_t rhs_offset = rhs_s_offset+k*rhs_k_stride;
    for(size_t i=0;i<i_len;i++){{
       sum += lhs[lhs_offset+i*lhs_i_stride]*rhs[rhs_offset+i*rhs_i_stride];
    }}
    out[out_offset] = sum;
}}", t = N::opencl_type_str())?;
    }
    write!(fmt, "\
__kernel void {t}_aggregate_sum(__global const {t} *input, // https://dournac.org/info/gpu_sum_reduction
                         __global {t} *partialSums,
                         __local {t} *localSums)
 {{
  uint local_id = get_local_id(0);
  uint group_size = get_local_size(0);

  // Copy from global to local memory
  localSums[local_id] = input[get_global_id(0)];

  // Loop for computing localSums : divide WorkGroup into 2 parts
  for (uint stride = group_size/2; stride>0; stride /=2)
     {{
      // Waiting for each 2x2 addition into given workgroup
      barrier(CLK_LOCAL_MEM_FENCE);

      // Add elements 2 by 2 between local_id and local_id + stride
      if (local_id < stride)
        localSums[local_id] += localSums[local_id + stride];
     }}

  // Write result into partialSums[nWorkGroups]
  if (local_id == 0)
    partialSums[get_group_id(0)] = localSums[0];
 }}
__kernel void {t}_scalar_mat_mul(__global {t} * mat, {t} scalar){{
    mat[get_global_id(0)] *= scalar;
}}
__kernel void {t}_scalar_mat_clamp(__global {t} * mat, {t} min_val, {t} max_val){{
    size_t i = get_global_id(0);
    mat[i] = max(min(mat[i], max_val), min_val);
}}
__kernel void {t}_scalar_mat_fill(__global {t} * mat, {t} scalar){{
    mat[get_global_id(0)] = scalar;
}}
__kernel void {t}_to_lhs_add(__global {t} * lhs, __global {t} * rhs){{
    size_t i = get_global_id(0);
    lhs[i] += rhs[i];
}}
__kernel void {t}_to_lhs_sub(__global {t} * lhs, __global {t} * rhs){{
    size_t i = get_global_id(0);
    lhs[i] -= rhs[i];
}}
__kernel void {t}_to_lhs_hadamard(__global {t} * lhs, __global {t} * rhs){{
    size_t i = get_global_id(0);
    lhs[i] *= rhs[i];
}}
__kernel void {t}_to_lhs_div(__global {t} * lhs, __global {t} * rhs){{
    size_t i = get_global_id(0);
    lhs[i] /= rhs[i];
}}
__kernel void {t}_to_lhs_copy(__global {t} * lhs, __global {t} * rhs){{
    size_t i = get_global_id(0);
    lhs[i] = rhs[i];
}}", t = N::opencl_type_str())?;
    for (for_int,built_in) in [(true,"min"), (true,"max")] {
        if N::IS_FLOAT || for_int {
            write!(fmt, "
__kernel void {t}_scalar_mat_{built_in}(__global {t} * mat, {t} scalar){{
    size_t i = get_global_id(0);
    mat[i] = {built_in}(mat[i], scalar);
}}
", built_in = built_in, t = N::opencl_type_str())?;
        }
    }
    for (for_int, for_float,built_in) in [(false,true,"sin"), (false,true,"cos"), (false,true,"tan"),
        (false,true,"tanh"), (true,false,"abs"), (false, true,"fabs"), (false,true, "exp"), (false,true, "exp2"), (false, true,"exp10"),
        (false, true, "log"), (false, true, "log2"), (false, true, "log10")] {
        if (N::IS_FLOAT && for_float) || (N::IS_INT && for_int) {
            write!(fmt, "
__kernel void {t}_unary_mat_{built_in}(__global {t} * mat){{
    size_t i = get_global_id(0);
    mat[i] = {built_in}(mat[i]);
}}
", built_in = built_in, t = N::opencl_type_str())?;
        }
    }

    for (name,op) in [("eq","=="), ("lt","<"), ("le","<="), ("gt",">"), ("ge",">="), ("ne","!=")] {
            write!(fmt, "
__kernel void {t}_cmp_{name}(__global {t} * lhs, __global {t} * rhs, __global uchar * out){{
    size_t i = get_global_id(0);
    out[i] = lhs[i] {op} rhs[i];
}}
__kernel void {t}_scalar_cmp_{name}(__global {t} * mat, {t} scalar, __global uchar * out){{
    size_t i = get_global_id(0);
    out[i] = mat[i] {op} scalar;
}}", name = name, op=op, t = N::opencl_type_str())?;
    }
    Ok(())
}

fn source_for_all_types(fmt: &mut Formatter<'_>) -> std::fmt::Result {
    source::<f32>(fmt,PhantomData)?;
    source::<i8>(fmt,PhantomData)?;
    source::<i16>(fmt,PhantomData)?;
    source::<i32>(fmt,PhantomData)?;
    source::<i64>(fmt,PhantomData)?;
    source::<u8>(fmt,PhantomData)?;
    source::<u16>(fmt,PhantomData)?;
    source::<u32>(fmt,PhantomData)?;
    source::<u64>(fmt,PhantomData)
}

struct SourceCode {
}

impl Display for SourceCode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        source_for_all_types(f)
    }
}


#[derive(Clone)]
pub struct LinAlgProgram {
    pub pro_que: ProQue,
}

impl LinAlgProgram {
    pub fn device_list(platform: &Platform) -> Vec<Device> {
        Device::list_all(platform).unwrap_or_else(|_| vec![])
    }


    pub fn device_by_type(platform: &Platform, dev_type: DeviceType) -> Option<Device> {
        Device::list_all(platform).ok().and_then(|dl| dl.into_iter().find(|d| match d.info(DeviceInfo::Type) {
            Ok(DeviceInfoResult::Type(dev_type)) => true,
            _ => false
        }))
    }
    pub fn gpu() -> Result<LinAlgProgram, Error> {
        let p = Platform::default();
        Self::device_by_type(&p, DeviceType::GPU).ok_or_else(|| Error::from(format!("No GPU device"))).and_then(|d| Self::new(p, d))
    }
    pub fn cpu() -> Result<LinAlgProgram, Error> {
        let p = Platform::default();
        Self::device_by_type(&p, DeviceType::CPU).ok_or_else(|| Error::from(format!("No CPU device"))).and_then(|d| Self::new(p, d))
    }
    pub fn new(platform: Platform, device: Device) -> Result<LinAlgProgram, Error> {
        let src = format!("{}", SourceCode { });
        ProQue::builder()
            .platform(platform)
            .device(device)
            .src(src)
            .dims(SpatialDims::Unspecified)
            .build().map(|pro_que| Self { pro_que })
    }
}

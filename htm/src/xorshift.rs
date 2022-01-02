use std::time::SystemTime;

pub fn xorshift32(mut x:u32) ->u32{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}


pub fn xorshift64(mut x:u64) ->u64{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

#[cfg(target_pointer_width = "64")]
pub fn xorshift(mut x:usize) ->usize{
    xorshift64(x as u64) as usize
}
#[cfg(target_pointer_width = "32")]
pub fn xorshift(mut x:usize) ->usize{
    xorshift32(x as u32) as usize
}
#[cfg(target_pointer_width = "64")]
pub fn auto_gen_seed()->usize{
    auto_gen_seed64() as usize
}
#[cfg(target_pointer_width = "32")]
pub fn auto_gen_seed()->usize{
    auto_gen_seed32() as usize
}
pub fn rand_u64_to_random_f32( rand:u64)->f32{
    (u64::MAX - rand) as f32 / u64::MAX as f32
}
pub fn rand_u32_to_random_f32( rand:u32)->f32{
    (u32::MAX - rand) as f32 / u32::MAX as f32
}
pub fn auto_gen_seed32()->u32{
    auto_gen_seed64() as u32
}
pub fn auto_gen_seed64()->u64{
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).expect("System clock may have gone backwards").as_millis() as u64
}
use std::time::SystemTime;

pub fn xorshift32(mut x:u32) ->u32{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

pub fn rand_u32_to_random_f32( rand:u32)->f32{
    (u32::MAX - rand) as f32 / u32::MAX as f32
}

pub fn auto_gen_seed()->u32{
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).expect("System clock may have gone backwards").subsec_millis()
}
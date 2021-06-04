const OPENCL_SRC: &'static str = r#"
        __kernel void sin32(__global float * buffer) {
            buffer[get_global_id(0)] = sin(buffer[get_global_id(0)]);
        }
        __kernel void cos32(__global float * buffer) {
            buffer[get_global_id(0)] = cos(buffer[get_global_id(0)]);
        }
        __kernel void tan32(__global float * buffer) {
            buffer[get_global_id(0)] = tan(buffer[get_global_id(0)]);
        }
        __kernel void tanh32(__global float * buffer) {
            buffer[get_global_id(0)] = tanh(buffer[get_global_id(0)]);
        }
        __kernel void relu32(__global float * buffer) {
            buffer[get_global_id(0)] = max(buffer[get_global_id(0)], 0.f);
        }
        __kernel void sigmoid32(__global float * buffer) {
            buffer[get_global_id(0)] = 1.0 / (1.0 + exp(-buffer[get_global_id(0)]));
        }
        __kernel void abs32(__global float * buffer) {
            buffer[get_global_id(0)] = fabs(buffer[get_global_id(0)]);
        }
        __kernel void square32(__global float * buffer) {
            float f = buffer[get_global_id(0)];
            buffer[get_global_id(0)] = f*f;
        }
        __kernel void const32(__global float * buffer, float val) {
            buffer[get_global_id(0)] = val;
        }
        __kernel void identity32(__global float * buffer) {
        }
        __kernel void add32(__global float * in_buffer,__global float * out_buffer) {
            out_buffer[get_global_id(0)] += in_buffer[get_global_id(0)];
        }
"#;




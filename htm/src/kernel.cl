
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
////// SDR & BITSET
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
bool is_input_active(__global uint * inputs, uint input_id);
bool is_input_active_at(__global uint * inputs, uint input_y, uint input_x, uint input_w);
uint3 pos(const uint3 shape, uint index);
uint idx(const uint3 shape, uint3 pos);
uint4 conv_out_range_clipped(uint2 in_position, uint2 stride, uint2 kernel_size);
int4 conv_out_range(uint2 in_position, uint2 stride, uint2 kernel_size);
uint2 conv_in_range_begin(uint2 out_position, uint2 stride);


bool is_input_active(__global uint * inputs, uint input_id){
    return ( inputs[input_id>>5] & (2147483648 >> (input_id & 31)) ) != 0;
}

bool is_input_active_at(__global uint * inputs, uint input_y, uint input_x, uint input_w){
    return is_input_active(inputs, input_y*input_w+input_x);
}

__kernel void bitset_to_sdr(
                  __global uint * sdr_cardinality,
                  __global uint * sdr_input,
                  __global uint * bitset_input){
    const size_t bit_idx = get_global_id(0);
    if(is_input_active(bitset_input, bit_idx)){
        uint idx = atomic_add(sdr_cardinality,1);
        sdr_input[idx] = bit_idx;
    }
}
__kernel void bitset_set_active_inputs(
                  __global uint * sdr_input,
                  __global uint * bitset_input){
    const size_t input_idx = get_global_id(0);
    const uint input_neuron_idx = sdr_input[input_idx];
    atomic_or(&bitset_input[input_neuron_idx>>5],2147483648>>(input_neuron_idx&31));
}

__kernel void bitset_clean_up_active_inputs(__global uint * sdr_input,__global uint * bitset_input){
    const size_t input_idx = get_global_id(0);
    const uint input_neuron_idx = sdr_input[input_idx];
    bitset_input[input_neuron_idx/32] = 0;
}

uint3 pos(const uint3 shape, uint index) {
    uint z = index % shape.z;
    index = index / shape.z;
    uint y = index % shape.y;
    uint x = index / shape.y;
    return (uint3)(x,y,z);
}

uint idx(const uint3 shape, uint3 pos) {
    return (pos.x * shape.y + pos.y) * shape.z + pos.z;
}
uint4 conv_out_range_clipped(uint2 in_position, uint2 stride, uint2 kernel_size) {
    const uint2 to = in_position/stride+1;
    const uint2 from = (max(in_position+stride,kernel_size)-kernel_size)/stride;
    return (uint4)(from,to);
}
int4 conv_out_range(uint2 in_position, uint2 stride, uint2 kernel_size) {
    const int2 to = (int2)in_position/(int2)stride+1;
    const int2 from = ((int2)in_position+(int2)stride-(int2)kernel_size)/(int2)stride;
    return (int4)(from,to);
}
uint2 conv_in_range_begin(uint2 out_position, uint2 stride) {
    return out_position * stride;
}
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
////// ECC Dense
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

uint w_idx(uint output_idx, uint idx_within_kernel_column,uint output_volume);
uint2 kernel_offset(uint3 output_pos, uint2 stride);
uint3 sub_kernel_offset(uint3 input_pos,uint2 offset);
uint3 pos_within_kernel(uint3 input_pos, uint3 output_pos, uint2 stride);
uint idx_within_kernel(uint3 kernel_column, uint2 stride, uint3 input_pos, uint3 output_pos);

#define MARGIN_OF_SAFETY 10
#define TOTAL_SUM (1 << (10 + MARGIN_OF_SAFETY))
#define ACTIVITY_PENALTY (1 << MARGIN_OF_SAFETY)

uint w_idx(uint output_idx, uint idx_within_kernel_column,uint output_volume) {
    return output_idx + idx_within_kernel_column * output_volume;
}
uint2 kernel_offset(uint3 output_pos, uint2 stride) {
    return conv_in_range_begin(output_pos.xy, stride);
}
uint3 sub_kernel_offset(uint3 input_pos,uint2 offset) {
    return (uint3)(input_pos.xy - offset.xy, input_pos.z);
}
uint3 pos_within_kernel(uint3 input_pos, uint3 output_pos, uint2 stride) {
    return sub_kernel_offset(input_pos, kernel_offset(output_pos, stride));
}
uint idx_within_kernel(uint3 kernel_column, uint2 stride, uint3 input_pos, uint3 output_pos) {
    return idx(kernel_column, pos_within_kernel(input_pos, output_pos, stride));
}

__kernel void ecc_dense_sums(
        const uint3 output_kernel_column,
        const uint3 output_shape,
        const uint3 input_shape,
        const uint2 stride,
        const uint2 kernel_size,
        const uint v,
        __global uint * input_sdr,
        __global uint * sums,
        __global uint * w) {
    const uint input_idx = input_sdr[get_global_id(0)];
    const size_t output_within_kernel_idx = get_global_id(1);
    const uint3 input_pos = pos(input_shape, input_idx);
    uint3 output_pos = pos(output_kernel_column, (uint) output_within_kernel_idx);
    const int4 output_range = conv_out_range(input_pos.xy,stride,kernel_size);
    output_pos.xy += output_range.xy; //output_range might be negative.
    // Then the addition might produce a negative number.
    // Because output_pos is unsigned, it might lead to overflow.
    // We do not realistically expect to build network so large the entire 32bit
    // inter would be necessary to encode output_pos. We assume that output_shape
    // is much much less than that limit. Hence we do not need to worry about checking
    // the overflow. The wrap-around semantics of modular integers, will produce
    // unreasonably large values that will not pass the out_pos<output_shape check.
    if(all(output_pos<output_shape)){
        const uint3 kernel_column = (uint3)(kernel_size,input_shape.z);
        const uint out_idx = idx(output_shape,output_pos);
        const uint input_idx_within_kernel = idx_within_kernel(kernel_column,stride,input_pos,output_pos);
        atomic_add(&sums[out_idx],w[w_idx(out_idx,input_idx_within_kernel,v)]);
    }
}

__kernel void ecc_dense_zero_out_sums_for_sdr(__global uint * sdr, __global uint * sums){
    sums[get_global_id(0)] = 0;
}

__kernel void ecc_dense_decrement_activities_for_sdr(__global uint * sdr, __global uint * activity){
    activity[sdr[get_global_id(0)]] -= ACTIVITY_PENALTY;
}


__kernel void ecc_dense_incoming_weights_sum(
        const uint v,
        __global uint * output_sdr,
        __global uint * sums,
        __global uint * w) {
    const size_t input_within_kernel_idx = get_global_id(0);
    const uint output_idx = output_sdr[get_global_id(1)];
    const uint weight_index = w_idx(output_idx,input_within_kernel_idx,v);
    atomic_add(&sums[output_idx],w[weight_index]);
}


__kernel void ecc_dense_normalize(
        const uint v,
        __global uint * output_sdr,
        __global uint * sums,
        __global uint * w) {
    const size_t input_within_kernel_idx = get_global_id(0);
    const uint output_idx = output_sdr[get_global_id(1)];
    const uint incoming_weights_sum = sums[output_idx];
    const uint weight_index = w_idx(output_idx,input_within_kernel_idx,v);
    const float weight = (float)w[weight_index];
    const float w_factor = (float)TOTAL_SUM/(float)incoming_weights_sum;
    w[weight_index] = (uint)(weight*w_factor);
}


__kernel void ecc_dense_increment_weights(
        const uint v,
        const uint3 input_shape,
        const uint3 output_shape,
        const uint3 kernel_column,
        const uint2 stride,
        const uint plasticity,
        __global uint * output_sdr,
        __global uint * input_sdr,
        __global uint * w) {
    const uint input_idx = input_sdr[get_global_id(0)];
    const uint output_idx = output_sdr[get_global_id(1)];
    const uint3 input_pos = pos(input_shape,input_idx);
    const uint3 output_pos = pos(output_shape,output_idx);
    const uint input_within_kernel_idx = idx_within_kernel(kernel_column,stride,input_pos,output_pos);
    const uint weight_index = w_idx(output_idx,input_within_kernel_idx,v);
    w[weight_index] += plasticity;
}


__kernel void ecc_dense_max_r(
        const uint threshold,
        __global uint * activity,
        __global uint * top_values,
        __global uint * sums){
    const uint channel_idx = get_global_id(0);
    const uint output_column_idx = get_global_id(1);
    const uint output_idx = output_column_idx*get_global_size(0)+channel_idx;
    const uint s = sums[output_idx];
    if(s >= threshold){
        const uint a = activity[output_idx];
        const uint r = a + s;
        if(top_values[output_column_idx]<r){
            atomic_max(&top_values[output_column_idx], r);
        }
    }
}

__kernel void ecc_dense_top_1(
        const uint threshold,
        __global uint * activity,
        __global uint * top_values,
        __global uint * output_sdr,
        __global uint * sums){
    const uint channel_idx = get_global_id(0);
    const uint output_column_idx = get_global_id(1);
    const uint output_idx = output_column_idx*get_global_size(0)+channel_idx;
    const uint s = sums[output_idx];
    if(s >= threshold){
        const uint a = activity[output_idx];
        const uint r = a + s;
        if(top_values[output_column_idx]==r &&
           r==atomic_cmpxchg(&top_values[output_column_idx], r, 0)){
            const uint offset = atomic_add(&top_values[get_global_size(1)],1);
            output_sdr[offset] = output_idx;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
////// ECC Sparse
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


__kernel void ecc_sparse_sums(
        __global uint * input_sdr,
        __global uint * connections,
        __global uint2 * connection_ranges,
        __global uint * sums) {
    const uint input_idx = input_sdr[get_global_id(0)];
    const uint connection_idx = get_global_id(1);
    const uint2 connection_range = connection_ranges[input_idx];
    if(connection_idx<connection_range.y){
        atomic_inc(&sums[connections[connection_range.x + connection_idx]]);
    }
}


__kernel void ecc_sparse_elements_per_sum(
        const uint threshold,
        __global int * candidates_per_sum,
        __global uint * sums) {
    const uint output_column_idx = get_global_id(0);
    const uint channel = get_global_id(1);
    const uint output_idx = output_column_idx*get_global_size(1)+channel;
    const uint sum = sums[output_idx];
    if(threshold<=sum){
        atomic_inc(&candidates_per_sum[output_column_idx+(sum-threshold)*get_global_size(0)+1]);
    }
}

__kernel void ecc_sparse_retain_top_k_candidates(
        uint k,
        const uint threshold,
        const uint max_value,
        __global int * candidates_per_sum) {
    const uint output_column_idx = get_global_id(0);
    int i = max_value;
    for(; i>=threshold; i--){
        const uint j = output_column_idx+(i-threshold)*get_global_size(0)+1;
        if(k <= candidates_per_sum[j]){
            candidates_per_sum[j] = k;
            break;
        }
        k -= candidates_per_sum[j];
    }
    while(i-->threshold){
       candidates_per_sum[output_column_idx+(i-threshold)*get_global_size(0)+1] = 0;
    }
    if(output_column_idx==0){ // here we reset the output_sdr cardinality counter
        candidates_per_sum[0] = 0; // this will be later incremented in ecc_sparse_top_k
    }
}


__kernel void ecc_sparse_top_k(
        const uint threshold,
        __global int * candidates_per_sum,
        __global uint * sums,
        __global uint * output_sdr) {
    const uint output_column_idx = get_global_id(0);
    const uint channel = get_global_id(1);
    const uint output_idx = output_column_idx*get_global_size(1)+channel;
    const uint sum = sums[output_idx];
    if(threshold<=sum ){
        const uint j = output_column_idx+(sum-threshold)*get_global_size(0)+1;
        if(atomic_dec(&candidates_per_sum[j])>0){
            output_sdr[atomic_inc(&candidates_per_sum[0])] = output_idx;
        }
    }
}


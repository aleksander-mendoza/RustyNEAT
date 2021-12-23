

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
////// SDR & BITSET
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
bool is_input_active(__global uint * inputs, uint input_id);

bool is_input_active_at(__global uint * inputs, uint input_y, uint input_x, uint input_w);

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

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
////// HTM
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


typedef struct _HtmFeedforwardConnection{
    float permanence;
    uint input_id;
} HtmFeedforwardConnection;

typedef struct _HtmMinicolumn{
    uint connection_offset;
    uint connection_len;
    int overlap;
} HtmMinicolumn;

uint htm_calculate_overlap_for_minicolumn(const size_t minicolumn_idx,
                                          const float permanence_threshold,
                                          __global HtmMinicolumn  * minicolumns,
                                          __global uint * inputs,
                                          __global HtmFeedforwardConnection * feedforward_connections);

uint htm_calculate_overlap_for_minicolumn(const size_t minicolumn_idx,
                                          const float permanence_threshold,
                                          __global HtmMinicolumn  * minicolumns,
                                          __global uint * inputs,
                                          __global HtmFeedforwardConnection * feedforward_connections){
    const uint connection_offset = minicolumns[minicolumn_idx].connection_offset;
    const uint connection_len = minicolumns[minicolumn_idx].connection_len;
    uint overlap = 0;
    for(uint feedforward_connection_idx = connection_offset;feedforward_connection_idx<connection_offset+connection_len;feedforward_connection_idx++){
        if(feedforward_connections[feedforward_connection_idx].permanence > permanence_threshold){
            const uint input_id = feedforward_connections[feedforward_connection_idx].input_id;
            overlap+=(uint)is_input_active(inputs,input_id);
        }
    }
    minicolumns[minicolumn_idx].overlap = overlap;
    return overlap;
}

__kernel void htm_calculate_overlap(
                  float permanence_threshold,
                  __global HtmMinicolumn  * minicolumns,
                  __global uint * inputs,
                  __global HtmFeedforwardConnection * feedforward_connections,
                  __global int * number_of_minicolumns_per_overlap
){
    const size_t minicolumn_idx = get_global_id(0);
    uint overlap = htm_calculate_overlap_for_minicolumn(minicolumn_idx,
       permanence_threshold,
       minicolumns,inputs,
       feedforward_connections);
    if(overlap > 0){
        atomic_add(&number_of_minicolumns_per_overlap[overlap], 1);
    }
}

__kernel void htm_calculate_overlap_and_group_into_columns(
                  size_t max_overlap,
                  size_t column_stride,
                  size_t minicolumn_stride,
                  float permanence_threshold,
                  __global HtmMinicolumn  * minicolumns,
                  __global uint * inputs,
                  __global HtmFeedforwardConnection * feedforward_connections,
                  __global int * number_of_minicolumns_per_overlap
){
    const size_t minicolumn_idx = get_global_id(0);
    const size_t column_idx = column_stride==1?minicolumn_idx%minicolumn_stride:minicolumn_idx / column_stride;
    const size_t offset = (max_overlap+1)*column_idx;
    uint overlap = htm_calculate_overlap_for_minicolumn(minicolumn_idx,
           permanence_threshold,
           minicolumns,inputs,
           feedforward_connections);
    atomic_add(&number_of_minicolumns_per_overlap[offset+overlap], 1);
}
__kernel void htm_find_number_of_minicolumns_per_overlap_that_made_it_to_top_n_and_group_into_columns(
    const size_t n,
    const size_t max_overlap,
    __global int * number_of_minicolumns_per_overlap
){
    const size_t column_idx = get_global_id(0);
    const size_t offset = column_idx*(max_overlap+1);
    int total_minicolumns = 0;
    number_of_minicolumns_per_overlap = number_of_minicolumns_per_overlap+offset;
    int overlap=max_overlap;
    for(;overlap>=0;overlap--) {
        int number_of_minicolumns = number_of_minicolumns_per_overlap[overlap];
        total_minicolumns += number_of_minicolumns;
        if(total_minicolumns > (int)n){
            number_of_minicolumns_per_overlap[overlap] = n - (total_minicolumns - number_of_minicolumns);
            overlap--;
            break;
        }
    }
    for(;overlap>=0;overlap--) {
        number_of_minicolumns_per_overlap[overlap] = 0;
    }
}

/**This function does the exact same thing as htm_find_top_minicolumns, but that function works
optimally when the input is so sparse that only a tiny fraction of minicolumns has even a single
connection to some active input. In cases where vast majority minicolumns is expected to have
at least one connection to some active input, then htm_find_top_minicolumns will be much more optimal.
*/
__kernel void htm_find_top_minicolumns(
                  __global HtmMinicolumn  * minicolumns,
                  __global int * number_of_minicolumns_per_overlap_that_made_it_to_top_n,
                  uint smallest_overlap_that_made_it_to_top_n,
                  __global uint * top_n_minicolumns,
                  __global uint * current_top_n_minicolumn_idx // precodntion: equals 0 ; postcondition: less than or equal n
){
    const size_t minicolumn_idx = get_global_id(0);
    const int overlap = minicolumns[minicolumn_idx].overlap;
    minicolumns[minicolumn_idx].overlap = 0;
    if(overlap>=(int)smallest_overlap_that_made_it_to_top_n){ // the array number_of_minicolumns_per_overlap_that_made_it_to_top_n holds rubbish for any overlap lower than smallest_overlap_that_made_it_to_top_n
        if(atomic_add(&number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap],-1)>0){ // only add those columns that made it to top n
            top_n_minicolumns[atomic_add(current_top_n_minicolumn_idx,1)] = minicolumn_idx;
        }
    }
}
__kernel void htm_find_top_minicolumns_and_group_into_columns(size_t n, size_t max_overlap,
                      size_t column_stride,
                      size_t minicolumn_stride,
                      __global HtmMinicolumn  * minicolumns,
                      __global int * number_of_minicolumns_per_overlap_that_made_it_to_top_n,
                      __global uint * top_n_minicolumns,
                      __global uint * current_top_n_minicolumn_idx // precodntion: equals 0 ; postcondition: equals n
){
    const size_t minicolumn_idx = get_global_id(0);
    const size_t column_idx = column_stride==1?minicolumn_idx%minicolumn_stride:minicolumn_idx / column_stride;
    size_t overlap_offset = (max_overlap+1)*column_idx;
    const int overlap = minicolumns[minicolumn_idx].overlap;
    if(atomic_add(&number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap_offset + overlap],-1)>0){ // only add those columns that made it to top n
        top_n_minicolumns[column_idx*n + atomic_add(current_top_n_minicolumn_idx + column_idx,1)] = minicolumn_idx;
    }
}


__kernel void htm_update_permanence(
                  float permanence_increment,
                  float permanence_decrement,
                  __global HtmMinicolumn  * minicolumns,
                  __global uint * inputs,
                  __global uint * top_n_minicolumns,
                  __global HtmFeedforwardConnection * feedforward_connections
){
    const size_t top_minicolumn_idx = get_global_id(0);
    const uint minicolumn_idx = top_n_minicolumns[top_minicolumn_idx];
    const uint connection_offset = minicolumns[minicolumn_idx].connection_offset;
    const uint connection_len = minicolumns[minicolumn_idx].connection_len;
    const float permanence_decrement_increment[2] = {permanence_decrement,permanence_increment};
    for(uint feedforward_connection_idx = connection_offset;feedforward_connection_idx<connection_offset+connection_len;feedforward_connection_idx++){
        const uint input_id = feedforward_connections[feedforward_connection_idx].input_id;
        const float permanence_change = permanence_decrement_increment[(uint)is_input_active(inputs,input_id)];
        const float old_permanence = feedforward_connections[feedforward_connection_idx].permanence;
        const float new_permanence = clamp(old_permanence+permanence_change,0.f,1.f);
        feedforward_connections[feedforward_connection_idx].permanence = new_permanence;
    }
}



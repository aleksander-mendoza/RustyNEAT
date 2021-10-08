
typedef struct _HtmFeedforwardConnection2{
    float permanence;
    uint input_id;
} HtmFeedforwardConnection2;

typedef struct _HtmMinicolumn2{
    uint connection_offset;
    uint connection_len;
    int overlap;
} HtmMinicolumn2;

bool is_input_active(__global uint * inputs, uint input_id){
    return ( inputs[input_id>>5] & (1 << (input_id & 31)) ) != 0;
}

/**This function does the exact same thing as htm_calculate_overlap, but that function works
optimally when the input is so sparse that only a tiny fraction of minicolumns has even a single
connection to some active input. In cases where vast majority minicolumns is expected to have
at least one connection to some active input, then htm_calculate_overlap2 will be much more optimal.
The htm_calculate_overlap2 is implemented in two parts. First you call htm_calculate_overlap2_active_inputs
and then you call htm_calculate_overlap2_overlap_per_minicolumn*/
__kernel void htm_calculate_overlap2_active_inputs(
                  float permanence_threshold,
                  __global uint * sdr_input,
                  __global uint * inputs){
      const size_t input_idx = get_global_id(0);
      const uint input_neuron_idx = sdr_input[input_idx];
      atomic_or(&inputs[input_neuron_idx>>5],1<<(input_neuron_idx&31));
}

__kernel void htm_calculate_overlap2_overlap_per_minicolumn(
                  float permanence_threshold,
                  __global HtmMinicolumn2  * minicolumns,
                  __global uint * inputs,
                  __global HtmFeedforwardConnection2 * feedforward_connections,
                  __global int * number_of_minicolumns_per_overlap
){
    const size_t minicolumn_idx = get_global_id(0);
    const uint connection_offset = minicolumns[minicolumn_idx].connection_offset;
    const uint connection_len = minicolumns[minicolumn_idx].connection_len;
    uint overlap = 0;
    for(uint feedforward_connection_idx = connection_offset;feedforward_connection_idx<connection_offset+connection_len;feedforward_connection_idx++){
        if(feedforward_connections[feedforward_connection_idx].permanence > permanence_threshold){
            const uint input_id = feedforward_connections[feedforward_connection_idx].input_id;
            overlap+=(uint)is_input_active(inputs,input_id);
        }
    }
    if(overlap > 0){
        atomic_add(&number_of_minicolumns_per_overlap[overlap], 1);
    }
    minicolumns[minicolumn_idx].overlap = overlap;
}

__kernel void htm_clean_up_active_inputs(__global uint * sdr_input,__global uint * inputs){
    const size_t input_idx = get_global_id(0);
    const uint input_neuron_idx = sdr_input[input_idx];
    inputs[input_neuron_idx/32] = 0;
}


/**This function does the exact same thing as htm_find_top_minicolumns, but that function works
optimally when the input is so sparse that only a tiny fraction of minicolumns has even a single
connection to some active input. In cases where vast majority minicolumns is expected to have
at least one connection to some active input, then htm_find_top_minicolumns2 will be much more optimal.
*/
__kernel void htm_find_top_minicolumns2(
                  float permanence_threshold,
                  __global HtmMinicolumn2  * minicolumns,
                  __global int * number_of_minicolumns_per_overlap_that_made_it_to_top_n,
                  uint smallest_overlap_that_made_it_to_top_n,
                  __global uint * top_n_minicolumns,
                  __global uint * current_top_n_minicolumn_idx // precodntion: equals 0 ; postcondition: less than or equal n
){
    const size_t minicolumn_idx = get_global_id(0);
    const int overlap = minicolumns[minicolumn_idx].overlap;
    minicolumns[minicolumn_idx].overlap = 0;
    if(overlap>=smallest_overlap_that_made_it_to_top_n){ // the array number_of_minicolumns_per_overlap_that_made_it_to_top_n holds rubbish for any overlap lower than smallest_overlap_that_made_it_to_top_n
        if(atomic_add(&number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap],-1)>0){ // only add those columns that made it to top n
            top_n_minicolumns[atomic_add(current_top_n_minicolumn_idx,1)] = minicolumn_idx;
        }
    }
}


__kernel void htm_update_permanence(
                  float permanence_increment,
                  float permanence_decrement,
                  __global HtmMinicolumn2  * minicolumns,
                  __global uint * inputs,
                  __global uint * top_n_minicolumns,
                  __global HtmFeedforwardConnection2 * feedforward_connections
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
        const float new_permanence = clamp(old_permanence+permanence_change,0.,1.);
        feedforward_connections[feedforward_connection_idx].permanence = new_permanence;
    }
}





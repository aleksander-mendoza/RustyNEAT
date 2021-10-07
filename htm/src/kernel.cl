
typedef struct _HtmFeedforwardConnection{
    uint minicolumn_id;
    float permanence;
    uint input_id;
} HtmFeedforwardConnection;
typedef struct _HtmInput{
    uint connection_offset;
    uint connection_len;
    bool is_active;
} HtmInput;
typedef struct _HtmMinicolumn{
    uint connection_index_offset;
    uint connection_index_len;
    int overlap;
} HtmMinicolumn;

__kernel void htm_calculate_overlap(
                  float permanence_threshold,
                  __global uint * sdr_input,
                  __global HtmMinicolumn * minicolumns,
                  __global HtmInput * inputs,
                  __global HtmFeedforwardConnection * feedforward_connections
){
    const size_t input_idx = get_global_id(0);
    const uint input_neuron_idx = sdr_input[input_idx];
    const HtmInput input_neuron = inputs[input_neuron_idx];
    inputs[input_neuron_idx].is_active = true;
    for(uint i = 0;i<input_neuron.connection_len;i++){
        const uint connection_idx = input_neuron.connection_offset + i;
        if(feedforward_connections[connection_idx].permanence > permanence_threshold){
            const uint minicolumn_id = feedforward_connections[connection_idx].minicolumn_id;
            atomic_add(&minicolumns[minicolumn_id].overlap, 1);
        }
    }
}

__kernel void htm_clean_up_active_inputs(__global uint * sdr_input,__global HtmInput * inputs){
    const size_t input_idx = get_global_id(0);
    const uint input_neuron_idx = sdr_input[input_idx];
    inputs[input_neuron_idx].is_active = false;
}


__kernel void htm_clean_up_overlap(
                  __global uint * sdr_input,
                  __global HtmMinicolumn * minicolumns,
                  __global HtmInput * inputs,
                  __global HtmFeedforwardConnection * feedforward_connections
){
    const size_t input_idx = get_global_id(0);
    const uint input_neuron_idx = sdr_input[input_idx];
    const HtmInput input_neuron = inputs[input_neuron_idx];
    inputs[input_neuron_idx].is_active = true;
    for(uint i = 0;i<input_neuron.connection_len;i++){
        const uint connection_idx = input_neuron.connection_offset + i;
        const uint minicolumn_id = feedforward_connections[connection_idx].minicolumn_id;
        minicolumns[minicolumn_id].overlap = 0;
    }
}


__kernel void htm_calculate_number_of_minicolumns_per_overlap(
                  float permanence_threshold,
                  __global uint * sdr_input,
                  __global HtmMinicolumn * minicolumns,
                  __global HtmInput * inputs,
                  __global int * number_of_minicolumns_per_overlap,
                  __global HtmFeedforwardConnection * feedforward_connections
){
    const size_t input_idx = get_global_id(0);
    const uint input_neuron_idx = sdr_input[input_idx];
    const uint connection_len = inputs[input_neuron_idx].connection_len;
    const uint connection_offset = inputs[input_neuron_idx].connection_offset;
    for(uint i = 0;i<connection_len;i++){
        const uint connection_idx = connection_offset + i;
        if(feedforward_connections[connection_idx].permanence > permanence_threshold){
            const uint minicolumn_idx = feedforward_connections[connection_idx].minicolumn_id;
            const int overlap = minicolumns[minicolumn_idx].overlap;
            if(overlap>0){
                if(atomic_cmpxchg(&minicolumns[minicolumn_idx].overlap,overlap,-overlap)==overlap){
                    atomic_add(&number_of_minicolumns_per_overlap[overlap],1);
                }
            }
        }
    }

}

__kernel void htm_find_top_minicolumns(
                  float permanence_threshold,
                  __global uint * sdr_input,
                  __global HtmMinicolumn * minicolumns,
                  __global HtmInput * inputs,
                  __global int * number_of_minicolumns_per_overlap_that_made_it_to_top_n,
                  uint smallest_overlap_that_made_it_to_top_n,
                  __global uint * top_n_minicolumns,
                  __global uint * current_top_n_minicolumn_idx, // precodntion: equals 0 ; postcondition: less than or equal n
                  __global HtmFeedforwardConnection * feedforward_connections
){
    const size_t input_idx = get_global_id(0);
    const uint input_neuron_idx = sdr_input[input_idx];
    const uint connection_len = inputs[input_neuron_idx].connection_len;
    const uint connection_offset = inputs[input_neuron_idx].connection_offset;
    for(uint i = 0;i<connection_len;i++){
        const uint connection_idx = connection_offset + i;
        if(feedforward_connections[connection_idx].permanence > permanence_threshold){
            const uint minicolumn_idx = feedforward_connections[connection_idx].minicolumn_id;
            const int overlap_negative = minicolumns[minicolumn_idx].overlap;
            const int overlap = -overlap_negative;
            if(overlap>=smallest_overlap_that_made_it_to_top_n && // the array number_of_minicolumns_per_overlap_that_made_it_to_top_n holds rubbish for any overlap lower than smallest_overlap_that_made_it_to_top_n
                atomic_cmpxchg(&minicolumns[minicolumn_idx].overlap,overlap_negative,0)!=0){ //avoid adding the same column multiple times
                if(atomic_add(&number_of_minicolumns_per_overlap_that_made_it_to_top_n[overlap],-1)>0){ // only add those columns that made it to top n
                    top_n_minicolumns[atomic_add(current_top_n_minicolumn_idx,1)] = minicolumn_idx;
                }
            }
        }
    }
}

__kernel void htm_update_permanence(
                  float permanence_increment,
                  float permanence_decrement,
                  __global uint * connection_indices,
                  __global HtmMinicolumn * minicolumns,
                  __global HtmInput * inputs,
                  __global uint * top_n_minicolumns,
                  __global HtmFeedforwardConnection * feedforward_connections
){
    const size_t top_minicolumn_idx = get_global_id(0);
    const uint minicolumn_idx = top_n_minicolumns[top_minicolumn_idx];
    const uint connection_index_offset = minicolumns[minicolumn_idx].connection_index_offset;
    const uint connection_index_len = minicolumns[minicolumn_idx].connection_index_len;
    const float permanence_decrement_increment[2] = {permanence_decrement,permanence_increment};
    for(uint i = 0;i<connection_index_len;i++){
        const uint feedforward_connection_idx = connection_indices[connection_index_offset + i];
        const uint input_id = feedforward_connections[feedforward_connection_idx].input_id;

        const float permanence_change = permanence_decrement_increment[inputs[input_id].is_active];
        const float old_permanence = feedforward_connections[feedforward_connection_idx].permanence;
        const float new_permanence = clamp(old_permanence+permanence_change,0.,1.);
        feedforward_connections[feedforward_connection_idx].permanence = new_permanence;
    }
}




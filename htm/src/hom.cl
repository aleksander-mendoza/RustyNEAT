
typedef struct _HomDistalConnection{
    uint minicolumn_id;
    float permanence;
    uint input_id;
    bool is_inhibitory;
} HomDistalConnection;


__kernel void hom_active_minicolumns(
                  __global uint * active_minicolumns_sdr,
                  __global uint * minicolumns){
      const size_t active_minicolumn_idx = get_global_id(0);
      const uint minicolumn_idx = active_minicolumns_sdr[active_minicolumn_idx];
      atomic_or(&minicolumns[minicolumn_idx>>5],1<<(minicolumn_idx&31));
}

bool is_minicolumn_active(__global uint * minicolumns, uint minicolumn_id){
    return ( minicolumns[minicolumn_id>>5] & (1 << (minicolumn_id & 31)) ) != 0;
}

__kernel void hom_(
                  uint cells_per_minicolumn,
                  uint max_segments_per_minicolumn,
                  __global uint * segments_per_cell,
                  __global uint * segment_lengths,
                  __global HomDistalConnection * synapses,
                  float permanence_threshold,
){
    const size_t minicolumn_idx = get_global_id(0);
    const bool is_active = is_minicolumn_active(minicolumns, minicolumn_idx);
    const uint cell_offset = cells_per_minicolumn * minicolumn_idx;
    const uint segment_offset = max_segments_per_minicolumn * minicolumn_idx;
    for(uint cell_idx=cell_offset;cell_idx < cell_offset+cells_per_minicolumn;cell_idx++){
        const uint segments = segments_per_cell[cell_idx];
        for(uint segment=0;segment<segments;segment++){
            cells[cell_idx]
        }
    }

}



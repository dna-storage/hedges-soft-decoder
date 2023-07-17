
extern "C" __global__ void gather_scores(
					    FLOAT* __restrict__ target,
					    const long* __restrict__ dim0,
					    const long* __restrict__ dim1,
                        FLOAT* __restrict__ out,
					    int dim0_size,
					    int dim1_size,
                        int target_dim0_size,
                        int target_dim1_size
					)
{
    int dim0_idx = threadIdx.y+blockDim.y*blockIdx.x, dim1_idx = threadIdx.x;
    if(dim0_idx>=dim0_size) return; //don't do anything
    long target_dim0 = dim0[dim0_idx*dim1_size+dim1_idx];
    long target_dim1 = dim1[dim0_idx*dim1_size+dim1_idx];
    out[dim0_idx*dim1_size+dim1_idx] = target[target_dim0*target_dim1_size+target_dim1];
}



//function to help minimize inefficiencies when creating the next F matrix
extern "C" __global__ void F_copy(
    FLOAT* __restrict__ temp_f_outgoing,
    long* __restrict__ trellis_incoming_indexes,
    long* __restrict__ trellis_incoming_value,
    long* __restrict__ value_of_max_scores,
    FLOAT* return_F,
    int H,
    int T)   
{
    int h_index = blockIdx.x*blockDim.x+threadIdx.x;
    if(h_index>=H) return;
    long max_index = value_of_max_scores[h_index];




}


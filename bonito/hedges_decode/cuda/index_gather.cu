
extern "C" __global__ void gather_scores(
					    FLOAT* __restrict__ target,
					    const long* restrict dim0,
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


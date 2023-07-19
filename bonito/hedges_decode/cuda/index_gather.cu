
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

typedef long long int int64_t;


//function to help minimize inefficiencies when creating the next F matrix
extern "C" __global__ void F_copy(
    const FLOAT* __restrict__ temp_f_outgoing,
    const int64_t* __restrict__ trellis_incoming_indexes,
    const int64_t* __restrict__ trellis_incoming_value,
    const int64_t* __restrict__ value_of_max_scores,
    FLOAT* __restrict__ return_F,
    int H,
    int T,
    int trellis_dim_1,
    int E)   
{   
    //h_index refers to return_F state we are working on
    int h_index = blockIdx.x*blockDim.x+threadIdx.x;	
    if(h_index>=H) return;
        
    int64_t max_index = value_of_max_scores[h_index];
    int64_t incoming_index = trellis_incoming_indexes[h_index*trellis_dim_1+max_index];
    int64_t incoming_value = trellis_incoming_value[h_index*trellis_dim_1+max_index];

    
    //printf(" GPU ------ H %d incoming index %ld incoming_value %ld  T %d E %d \n",h_index,incoming_index,incoming_value,T,E); 
    
    //perform copy from outgoing to return F
    for(int t=0; t<T; t++){
    	//if(h_index==128) printf("H %d t %d v %f\n",h_index,t,temp_f_outgoing[t*H*E+incoming_index*E+incoming_value]);
        return_F[t*H+h_index] = temp_f_outgoing[t*H*E+incoming_index*E+incoming_value];
	//if(h_index==128) printf("STORE: H %d t %d v %f HT %d \n",h_index,t,return_F[t*H+h_index],H);
    }
    

}



__device__ __forceinline__ FLOAT max2(FLOAT a, FLOAT a1) {
    return a > a1 ? a : a1; 
}


__device__ __forceinline__ FLOAT logsumexp2(FLOAT a, FLOAT a1) {
  FLOAT maxa = max2(a, a1); 
  return maxa + log(exp(a-maxa) + exp(a1-maxa));
}



extern "C" __global__ void dot_reduce(
				      FLOAT* input,
				      FLOAT* output,
				      int T,
				      int stride,
              int H,
              int E
				      )
{
  extern __shared__ FLOAT smem[];
  int Eidx = threadIdx.y;
  int Hidx = threadIdx.z+blockDim.z*blockIdx.x;
  int Tidx = threadIdx.x+blockDim.x*blockIdx.y;
  int smem_idx = threadIdx.z*blockDim.y*blockDim.x+threadIdx.y*blockDim.x+threadIdx.x;
  int start=Tidx*2*stride;
  if(start+stride<T*stride) smem[smem_idx]=REDUCE(input[start*H*E+Hidx*E+Eidx],input[(start+stride)*H*E+Hidx*E+Eidx]);
  else if(start<T*stride) {
    smem[smem_idx]=input[start*H*E+Hidx*E+Eidx];
  }
  else {
    smem[smem_idx]=ZERO;
  }
  __syncthreads();
  for(int s = blockDim.x/2; s>0; s=s>>1){
    if(threadIdx.x<s){
      smem[smem_idx]=REDUCE(smem[smem_idx],smem[smem_idx+s]);
    }
    __syncthreads();
    //if(blockIdx.z==0 && Hidx==0 && Eidx==0) printf("s %d Tidx %d start %d T %d  %f \n",s, Tidx,start,T,smem[ThrIdx]);
  }
  if(threadIdx.x==0) {
    output[start*H*E+Hidx*E+Eidx]=smem[smem_idx];
  }
}

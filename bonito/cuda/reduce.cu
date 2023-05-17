
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
				      int stride
				      )
{
  extern __shared__ FLOAT smem[];
  int Eidx = blockIdx.y;
  int Hidx = blockIdx.x;
  int Tidx = threadIdx.x+blockIdx.z*blockDim.x;
  int ThrIdx = threadIdx.x;
  int start=Tidx*2*stride;
  if(start+stride<T*stride) smem[ThrIdx]=REDUCE(input[start*gridDim.x*gridDim.y+Hidx*gridDim.y+Eidx],input[(start+stride)*gridDim.x*gridDim.y+Hidx*gridDim.y+Eidx]);
  else if(start<T*stride) {
    smem[ThrIdx]=input[start*gridDim.x*gridDim.y+Hidx*gridDim.y+Eidx];
  }
  else {
    smem[ThrIdx]=ZERO;
  }
  __syncthreads();
  for(int s = 512; s>0; s=s>>1){
    if(ThrIdx<s){
      smem[ThrIdx]=REDUCE(smem[ThrIdx],smem[ThrIdx+s]);
    }
    __syncthreads();
    //if(blockIdx.z==0 && Hidx==0 && Eidx==0) printf("s %d Tidx %d start %d T %d  %f \n",s, Tidx,start,T,smem[ThrIdx]);
  }
  if(ThrIdx==0) {
    output[start*gridDim.x*gridDim.y+Hidx*gridDim.y+Eidx]=smem[0];
  }
}

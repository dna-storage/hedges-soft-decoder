__device__ __forceinline__ FLOAT max3(FLOAT a, FLOAT a1, FLOAT a2) {
    FLOAT maxa = a > a1 ? a : a1; 
    return maxa > a2 ? maxa : a2;
}

__device__ __forceinline__ FLOAT max2(FLOAT a, FLOAT a1) {
    return a > a1 ? a : a1; 
}

__device__ __forceinline__ FLOAT logsumexp3(FLOAT a, FLOAT a1, FLOAT a2) {
    FLOAT maxa = max3(a, a1, a2); 
    return maxa + log(exp(a-maxa) + exp(a1-maxa) + exp(a2-maxa));
}

__device__ __forceinline__ FLOAT logsumexp2(FLOAT a, FLOAT a1) {
  FLOAT maxa = max2(a, a1); 
  return maxa + log(exp(a-maxa) + exp(a1-maxa));
}


__device__ __forceinline__ FLOAT sum3(FLOAT a, FLOAT a1, FLOAT a2) {return a + a1 + a2;}
__device__ __forceinline__ FLOAT add(FLOAT a, FLOAT b) {return a + b;}
__device__ __forceinline__ FLOAT mul(FLOAT a, FLOAT b) {return a * b;}

extern "C" __global__ void fwd_logspace(
					    const FLOAT* __restrict__ target_scores,
					    FLOAT* __restrict__ alpha_t,
					    const FLOAT* __restrict__ mask,
					    const FLOAT* __restrict__ F,
					    int lower_t_range,
					    int upper_t_range,
					    int H,
					    int E,
					    int L,
					    int T,
					    int L_pad,
					    int target_score_pad)



{
  int bx = blockIdx.x, Lidx = threadIdx.x, Eidx = threadIdx.z, Hidx=threadIdx.y;
  int total_L = L+L_pad;
  int HEL_stride = H*E*total_L;
  int EL_stride = E*total_L;
  extern __shared__ FLOAT smem[];
  //smem needs to be initialized to time -1 so forward algorithm can go ahead
  if(Lidx==0) smem[(lower_t_range%2)*HEL_stride+ Hidx*EL_stride + Eidx*(total_L) + Lidx] = ZERO;
  if(Lidx==1) smem[(lower_t_range%2)*HEL_stride+ Hidx*EL_stride + Eidx*(total_L) + Lidx] = F[(lower_t_range-1)*H+Hidx];
  smem[(lower_t_range%2)*HEL_stride+Hidx*EL_stride+Eidx*(total_L)+(Lidx+L_pad)] = ZERO;
  __syncthreads();
  for(int t=lower_t_range; t<upper_t_range;t++){
    //perform core calculations for forward algorithm
    FLOAT a,a1,a2,final_score,score; //a->current string step, a1-> one string step back, a2->two string steps back
    a = smem[(t%2)*HEL_stride+ Hidx*EL_stride+ Eidx*total_L+ (Lidx+L_pad)];
    a1 = smem[(t%2)*HEL_stride+ Hidx*EL_stride+ Eidx*total_L + (Lidx+L_pad-1)];
    a2 =  MUL(smem[(t%2)*HEL_stride + Hidx*EL_stride + Eidx*total_L+ (Lidx+L_pad-2)],mask[Hidx*E*L+Eidx*L+Lidx]);
    score = target_scores[t*H*E*(L+target_score_pad)+ Hidx*E*(L+target_score_pad)+ Eidx*(L+target_score_pad)+ (Lidx+target_score_pad)];
    //if(Hidx==0 && Eidx==0) printf("t %d L %d score %f \n",t,Lidx,score);
    final_score = MUL(score,SUM(a,a1,a2));
    smem[(((t+1)%2))*HEL_stride + Hidx*EL_stride + Eidx*total_L+(Lidx+L_pad)]=final_score;
    if(Lidx==0) smem[(((t+1)%2))*HEL_stride+ Hidx*EL_stride+ Eidx*(total_L)+ Lidx] = ZERO;
    else
      {
	smem[((t+1)%2)*HEL_stride + Hidx*EL_stride + Eidx*total_L + Lidx] = F[t*H + Hidx];
	alpha_t[ t*H*E+ Hidx*E+ Eidx] = final_score;
      }
    __syncthreads();
  }
}


//this is an optimized verision of fwd_logspace, seems like it may not be most important to use right now
extern "C" __global__ void fwd_logspace_opt(
					    const FLOAT* __restrict__ target_scores,
					    FLOAT* __restrict__ alpha_t,
					    const FLOAT* __restrict__ mask,
					    const FLOAT* __restrict__ F,
					    int lower_t_range,
					    int upper_t_range,
					    int H,
					    int E,
					    int L,
					    int T,
					    int L_pad)



{
  int tid = threadIdx.x;
  int smem_tid;
  int total_L = L+L_pad;
  int HEL_stride = H*E*total_L;
  extern __shared__ FLOAT smem[];
  FLOAT* buff_0 = smem;
  FLOAT* buff_1 = smem+HEL_stride;
  int Hidx = tid/(L*E);
  int Eidx = (tid/L)-Hidx*E;
  int Lidx = tid - (Eidx*L) - (Hidx*L*E);
  if (Lidx==0) smem_tid=tid*2;
  else smem_tid=tid*2-1;
  //smem needs to be initialized to time -1 so forward algorithm can go ahead
  buff_0[smem_tid] = F[(lower_t_range-1)*H*L+Hidx*L_pad+Lidx];
  buff_0[smem_tid+L_pad] = ZERO;
  __syncthreads();
  for(int t=lower_t_range; t<upper_t_range;t++){
    FLOAT* temp_buff;
    //perform core calculations for forward algorithm
    FLOAT a,a1,a2,final_score,score; //a->current string step, a1-> one string step back, a2->two string steps back
    a = buff_0[smem_tid+L_pad];
    a1 = buff_0[smem_tid+L_pad-1];
    a2 =  MUL(buff_0[smem_tid],mask[tid]);
    score = target_scores[t*H*E*L+tid];
    final_score = MUL(score,SUM(a,a1,a2));
    buff_1[smem_tid+L_pad]=final_score;
    buff_1[smem_tid] = F[t*H*L_pad + Hidx*L_pad+ Lidx];
    if(Lidx==1) alpha_t[t*H*E+Hidx*E+Eidx] = final_score;
    //rotate buffers
    temp_buff=buff_0;
    buff_0=buff_1;
    buff_1=buff_0;
    __syncthreads();
  }
}

extern "C" __global__ void dot_mul(
					    const FLOAT* __restrict__ target_scores,
					    const FLOAT* __restrict__ alpha_t,
					    FLOAT* __restrict__ output,
					    int T,
					    int target_scores_L
				   )
{
  int Hidx = threadIdx.y;
  int Eidx = threadIdx.x;
  int Tidx = threadIdx.z+blockDim.z*blockIdx.x;
  if(Tidx<T){
    int H = blockDim.y;
    int E = blockDim.x;
    int idx = Tidx*H*E+Hidx*E+Eidx;
    int idx2= (idx+H*E)*target_scores_L+target_scores_L-1;
    if(Tidx+1<T)
      output[idx] = MUL(alpha_t[idx],SUM2(0,target_scores[idx2]));
    else
      output[idx] = alpha_t[idx];
  }
}


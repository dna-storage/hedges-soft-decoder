__device__ __forceinline__ FLOAT max3(FLOAT a, FLOAT a1, FLOAT a2) {
    FLOAT maxa = a > a1 ? a : a1; 
    return maxa > a2 ? maxa : a2;
}

__device__ __forceinline__ FLOAT logsumexp3(FLOAT a, FLOAT a1, FLOAT a2) {
    FLOAT maxa = max3(a, a1, a2); 
    return maxa + log(exp(a-maxa) + exp(a1-maxa) + exp(a2-maxa));
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
					    int L_pad)



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
    score = target_scores[t*H*E*L+ Hidx*E*L+ Eidx*L+ Lidx];
    final_score = MUL(score,SUM(a,a1,a2));
    //printf("%d %d %d %d %d %d %d\n",Hidx,Eidx,Lidx,((((t+1)%2))*HEL_stride) ,  (Hidx*EL_stride) , (Eidx*total_L),(Lidx+L_pad));

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
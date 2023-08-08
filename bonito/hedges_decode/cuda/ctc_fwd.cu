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


#define NBASE 5
typedef long long int int64_t;

#define PREFETCH 4

__device__ __forceinline__ void pfL1(const FLOAT* a){
    asm("prefetch.global.L1 [%0];"
        ::"l"(a));
    return;
}



extern "C" __global__ void fwd_logspace(
					    const int64_t* __restrict__ targets,
					    const FLOAT* __restrict__ scores,
					    FLOAT* __restrict__ alpha_t,
					    const FLOAT* __restrict__ mask,
					    const FLOAT* __restrict__ F,
					    int lower_t_range,
					    int upper_t_range,
					    int H,
					    int E,
					    int L,
					    int L_pad,
					    int target_score_pad,
					    int F_offset,
					    int F_T,
					    int abs_lower_t_range_offset
					)



{
  int Lidx = threadIdx.x, Eidx = threadIdx.z, Hidx=threadIdx.y+blockIdx.x*blockDim.y , blockHidx=threadIdx.y;
  int total_L = L+L_pad;
  int HEL_stride = blockDim.y*E*total_L;
  int EL_stride = E*total_L;
  int blockH_EL = blockHidx*EL_stride;
  



  int64_t target = targets[Hidx*E*(L+target_score_pad)+ Eidx*(L+target_score_pad)+ (Lidx+target_score_pad)];
  extern __shared__ FLOAT smem[];
  if(Hidx>=H) return; //get rid of dead threads
  //smem needs to be initialized to time -1 so forward algorithm can go ahead
  if(Lidx==0) smem[(lower_t_range%2)*HEL_stride+ blockHidx*EL_stride + Eidx*(total_L) + Lidx] = ZERO;
  if(Lidx==1) smem[(lower_t_range%2)*HEL_stride+ blockHidx*EL_stride + Eidx*(total_L) + Lidx] = F[(lower_t_range+F_offset)*H+Hidx];
  smem[(lower_t_range%2)*HEL_stride+blockHidx*EL_stride+Eidx*(total_L)+(Lidx+L_pad)] = ZERO; 
  __syncthreads();
   int fetch_ptr = 0;//index to keep track of what prefetch we use/need to update	
  if(Lidx==1){ //try some asynchroniouse mem copies
       for(int i=0; i<PREFETCH; i++) pfL1(&F[(lower_t_range+F_offset+1+i)*H+Hidx]);
  }  
  float mask_value = mask[Hidx*E*L+Eidx*L+Lidx];
  int smem_select = lower_t_range%2;
  for(int t=lower_t_range; t<upper_t_range;t++){
    //perform core calculations for forward algorithm
    FLOAT a,a1,a2,final_score,score; //a->current string step, a1-> one string step back, a2->two string steps back
    int next_smem = ~smem_select&0x01;
    int f_t = (t+1+F_offset);
    a = smem[(smem_select)*HEL_stride+ blockH_EL + Eidx*total_L+ (Lidx+L_pad)];
    a1 = smem[(smem_select)*HEL_stride+ blockH_EL + Eidx*total_L + (Lidx+L_pad-1)];
    a2 =  MUL(smem[(smem_select)*HEL_stride + blockH_EL + Eidx*total_L+ (Lidx+L_pad-2)],mask_value);
    score = scores[(abs_lower_t_range_offset+t)*NBASE+target];  	
    final_score = MUL(score,SUM(a,a1,a2));
    smem[(((next_smem)))*HEL_stride + blockHidx*EL_stride + Eidx*total_L+(Lidx+L_pad)]=final_score;
    if(Lidx==0) smem[(next_smem)*HEL_stride+ blockHidx*EL_stride+ Eidx*(total_L)+ Lidx] = ZERO;
    else if (Lidx==1){      
      pfL1(&F[(f_t+PREFETCH)*H+Hidx]);
      if(f_t<F_T) smem[(next_smem)*HEL_stride + blockHidx*EL_stride + Eidx*total_L + Lidx] = F[f_t*H + Hidx];
      else smem[(next_smem)*HEL_stride + blockHidx*EL_stride + Eidx*total_L + Lidx] = ZERO;
    }
    __syncthreads();
    //moved write to after sync 
    if (Lidx==L-1)alpha_t[ t*H*E+ Hidx*E+ Eidx] = final_score;
    smem_select=next_smem;
  }
}


extern "C" __global__ void fwd_logspace_align(
					    const FLOAT* __restrict__ target_scores,
					    FLOAT* __restrict__ F,
					    long* __restrict__ BT,
					    const FLOAT* __restrict__ mask,
					    int T,
					    int L,
					    int L_pad
					)
{
  int Lidx = threadIdx.x;
  int total_L = L+L_pad;
  extern __shared__ FLOAT smem[];
  if(Lidx==0 || Lidx==1) smem[Lidx] = ZERO;
  if(Lidx==0) smem[Lidx+L_pad]=ONE;
  else smem[Lidx+L_pad] = ZERO;
  __syncthreads();
  for(int t=0;t<T;t++){
    //perform core calculations for forward algorithm
    FLOAT a,a1,a2,final_score,score; //a->current string step, a1-> one string step back, a2->two string steps back
    score = target_scores[t*(L)+Lidx];
    a = MUL(score,smem[(t%2)*total_L+(Lidx+L_pad)]);
    a1 = MUL(score,smem[(t%2)*total_L+(Lidx+L_pad-1)]);
    a2 =  sum3(score,smem[(t%2)*total_L+(Lidx+L_pad-2)],mask[Lidx]);
final_score = max3(a,a1,a2);
    int a_ = (a>a1 && a>a2)*0;
    int a1_ = (a1>a && a1>a2)*1;
    int a2_ = (a2>a && a2>a1)*2; 
    F[(t*L+Lidx)]=final_score;
    
    if(Lidx>0) BT[t*(L-1)+Lidx-1]=Lidx-(a_+a1_+a2_);

    smem[(((t+1)%2))*total_L+(Lidx+L_pad)]=final_score;
    if(Lidx==0 || Lidx==1) smem[(((t+1)%2))*total_L+Lidx] = ZERO;
    __syncthreads();
  }
}


extern "C" __global__ void dot_mul(
					    const int64_t* __restrict__ targets,
					    const FLOAT* __restrict__ scores,
					    const FLOAT* __restrict__ alpha_t,
					    FLOAT* __restrict__ output,
					    int T,
  				            int target_scores_L,
              				    int H,
					    int E,
					    int abs_t_lower
				   )
{
  int Tidx = threadIdx.z+blockDim.z*blockIdx.x;
  int Hidx = threadIdx.y+blockDim.y*blockIdx.y;
  int Eidx = threadIdx.x;
  int target = targets[Hidx*E*target_scores_L+Eidx*target_scores_L+target_scores_L-1];
  if(Tidx<T){
    int idx = Tidx*H*E+Hidx*E+Eidx;
    int idx2= (Tidx+1+abs_t_lower)*NBASE+target;
    if(Tidx+1<T)
      output[idx] = MUL(alpha_t[idx],SUM2(0,scores[idx2]));
    else
      output[idx] = alpha_t[idx];
  }
}


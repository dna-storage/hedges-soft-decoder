/*
Code inspired from https://github.com/davidcpage/seqdist/blob/master/seqdist/cuda/sparse_logZ.cu
*/

#define LOG(x) log(x)
#define EXP(x) exp(x)

__device__ __forceinline__ FLOAT max3(FLOAT a, FLOAT a1, FLOAT a2) {
    FLOAT maxa = a > a1 ? a : a1; 
    return maxa > a2 ? maxa : a2;
}

__device__ __forceinline__ FLOAT max2(FLOAT a, FLOAT a1) {
    return a > a1 ? a : a1; 
}

__device__ __forceinline__ FLOAT logsumexp3(FLOAT a, FLOAT a1, FLOAT a2) {
    FLOAT maxa = max3(a, a1, a2); 
    return maxa + LOG(EXP(a-maxa) + EXP(a1-maxa) + EXP(a2-maxa));
}

__device__ __forceinline__ FLOAT logsumexp2(FLOAT a, FLOAT a1) {
  FLOAT maxa = max2(a, a1); 
  return maxa + LOG(EXP(a-maxa) + EXP(a1-maxa));
}

__device__ __forceinline__ FLOAT logdiffexp2(FLOAT a, FLOAT a1) {
  FLOAT maxa = max2(a, a1);
  return maxa + LOG(EXP(a-maxa) - EXP(a1-maxa));
}



__device__ __forceinline__ FLOAT add3(FLOAT a, FLOAT a1, FLOAT a2) {return a + a1 + a2;}
__device__ __forceinline__ FLOAT add(FLOAT a, FLOAT b) {return a + b;}
__device__ __forceinline__ FLOAT mul(FLOAT a, FLOAT b) {return a * b;}


#define NBASE 5
typedef long long int int64_t;
typedef int int32_t;

#define PREFETCH 0

__device__ __forceinline__ void pfL1(const FLOAT* a){
    asm("prefetch.global.L1 [%0];"
        ::"l"(a));
    return;
}



extern "C" __global__ void fwd_logspace_reduce(
					    const int32_t* __restrict__ targets, //bases being targetted for alignment
					    const FLOAT* __restrict__ scores, //raw emission scores
					    FLOAT* __restrict__ alpha_t,  //output alignments
					    const FLOAT* __restrict__ mask, //mask off matches
					    const FLOAT* __restrict__ F, //previous alignment
              				    FLOAT* __restrict__ out_scores, //output scores
					    const int32_t* __restrict__ lower_t_range_ptr, //start point for iterating
					    const int32_t* __restrict__ upper_t_range_ptr, //end point for iterating
					    int H, //number of hedges states
					    int E, //number of transitions
					    int L, //actual number of L positions calculated
					    int L_pad, //paddding needed for shared mem
					    int32_t target_score_pad, //padding from lower_t_range to actual index
					    const int32_t* __restrict__ F_offset_ptr, //offset between absolute position of new alignment and previous
					    int F_T, // time dimension of F matrix
              				    int T, //time dimension of scores matrix
					    
					    const int32_t*__restrict__ abs_lower_t_range_offset_ptr, //offset to absolute position of emissions scores
					    const int32_t* __restrict__ time_range_end, //endpoint of emissions for batch
					    const int32_t* __restrict__ F_end //endpoint of F for batch
					)



{ //identical kernel to the base fwd_logspace kernel, but instead I am trying out what it i looks like if reduction is done n the same kernel call
  const int Nidx = blockIdx.y;
  const int Lidx = threadIdx.x, Eidx = threadIdx.z, Hidx=threadIdx.y+blockIdx.x*blockDim.y , blockHidx=threadIdx.y;
  const int total_L = L+L_pad;
  const int HEL_stride = blockDim.y*E*total_L;
  const int EL_stride = E*total_L;
  const int blockH_EL = blockHidx*EL_stride;
  FLOAT reduction_value=ZERO;
  FLOAT next_t_score=ZERO;

  //load offsets for this batch
  const int lower_t_range = lower_t_range_ptr[Nidx];
  const int upper_t_range = upper_t_range_ptr[Nidx];
  const int F_offset=F_offset_ptr[Nidx];
  const int abs_lower_t_range_offset=abs_lower_t_range_offset_ptr[Nidx];
  const int abs_time_endpoint = time_range_end[Nidx];
  const int f_endpoint = F_end[Nidx];

  //if(Nidx>0) printf("  %d %d %d %d %d %d \n",lower_t_range,upper_t_range,F_offset,abs_lower_t_range_offset,abs_time_endpoint,f_endpoint);

  int64_t target = targets[Nidx*H*E*(L+target_score_pad)+Hidx*E*(L+target_score_pad)+ Eidx*(L+target_score_pad)+ (Lidx+target_score_pad)];
  extern __shared__ FLOAT smem[];
  if(Hidx>=H) return; //get rid of dead threads
  //smem needs to be initialized to time -1 so forward algorithm can go ahead
  if(Lidx==0) smem[(lower_t_range%2)*HEL_stride+ blockHidx*EL_stride + Eidx*(total_L) + Lidx] = ZERO;
  if(Lidx==1) smem[(lower_t_range%2)*HEL_stride+ blockHidx*EL_stride + Eidx*(total_L) + Lidx] = F[Nidx*H*F_T+(lower_t_range+F_offset)*H+Hidx];
  smem[(lower_t_range%2)*HEL_stride+blockHidx*EL_stride+Eidx*(total_L)+(Lidx+L_pad)] = ZERO; 
  __syncthreads();
   int fetch_ptr = 0;//index to keep track of what prefetch we use/need to update	
  if(Lidx==1){ //try some asynchroniouse mem copies
       for(int i=0; i<PREFETCH; i++) pfL1(&F[Nidx*F_T*H+(lower_t_range+F_offset+1+i)*H+Hidx]);
  }   
  float mask_value = mask[Nidx*H*E*L+Hidx*E*L+Eidx*L+Lidx];
  int smem_select = lower_t_range%2;
  for(int t=lower_t_range; t<upper_t_range;t++){
    if((abs_lower_t_range_offset+t)>=abs_time_endpoint) break; //if a strand reaches it's practical enpoit, just exit and don't waste compute
    FLOAT a,a1,a2,final_score,score; //a->current string step, a1-> one string step back, a2->two string steps back    
    score = scores[Nidx*T*NBASE+(abs_lower_t_range_offset+t)*NBASE+target];  	
    if(t<upper_t_range-1) next_t_score=scores[Nidx*T*NBASE+(abs_lower_t_range_offset+t+1)*NBASE+target];
    else  next_t_score=ZERO;
    //perform core calculations for forward algorithm
    int next_smem = ~smem_select&0x01;
    int f_t = (t+1+F_offset);
    a = smem[(smem_select)*HEL_stride+ blockH_EL + Eidx*total_L+ (Lidx+L_pad)];
    a1 = smem[(smem_select)*HEL_stride+ blockH_EL + Eidx*total_L + (Lidx+L_pad-1)];
    a2 =  MUL(smem[(smem_select)*HEL_stride + blockH_EL + Eidx*total_L+ (Lidx+L_pad-2)],mask_value);
    final_score = MUL(score,SUM(a,a1,a2));
    smem[(((next_smem)))*HEL_stride + blockHidx*EL_stride + Eidx*total_L+(Lidx+L_pad)]=final_score;
    if(Lidx==0) smem[(next_smem)*HEL_stride+ blockHidx*EL_stride+ Eidx*(total_L)+ Lidx] = ZERO;
    else if (Lidx==1){      
      pfL1(&F[Nidx*H*F_T+(f_t+PREFETCH)*H+Hidx]);
      if(f_t<f_endpoint) smem[(next_smem)*HEL_stride + blockHidx*EL_stride + Eidx*total_L + Lidx] = F[Nidx*H*F_T+f_t*H + Hidx];
      else smem[(next_smem)*HEL_stride + blockHidx*EL_stride + Eidx*total_L + Lidx] = ZERO;
    }
    __syncthreads();
    //moved write to after sync 
    if (Lidx==L-1){
      alpha_t[Nidx*H*F_T*E+t*H*E+ Hidx*E+ Eidx] = final_score; 
    }

    //reduce final score 
    reduction_value = SUM2(reduction_value,MUL(final_score,LOG(1-EXP(next_t_score))));
    //if(Nidx==1 && Lidx==L-1) printf("reduction value %f\n",reduction_value);
    smem_select=next_smem;
  }
  if(Lidx==L-1) out_scores[Nidx*H*E+Hidx*E+Eidx]=reduction_value;

}

extern "C" __global__ void fwd_logspace_align(
					    const FLOAT* __restrict__ scores,
              const int64_t* __restrict__ targets,
					    FLOAT* __restrict__ F,
					    long* __restrict__ BT,
					    const FLOAT* __restrict__ mask,
					    int T,
					    int L,
					    int L_pad,
              int N
					)
{
  int Lidx = threadIdx.x;
  int Nidx = threadIdx.y+blockDim.y*blockIdx.x;
  int Nidx_t = threadIdx.y;
  if(Nidx>=N) return;
  int64_t target = targets[Nidx*L+Lidx];
  int total_L = L+L_pad;
  extern __shared__ FLOAT smem[];
  FLOAT mask_value = mask[Nidx*L+Lidx];
  if(Lidx==0){
    smem[Nidx_t*2*total_L+Lidx] = ONE; //set this position to constant ONE, immitates behavior of "start" symbol
    smem[Nidx_t*2*total_L+total_L+Lidx]=ONE;
    mask_value = ONE; //first position needs to be able to use 2-back "start" symbol
  }
  else if(Lidx==1){
    smem[Nidx_t*2*total_L+Lidx] = ZERO; 
    smem[Nidx_t*2*total_L+total_L+Lidx]=ZERO;
  } 
  smem[Nidx_t*2*total_L+Lidx+L_pad] = ZERO;
  __syncthreads();
  for(int t=0;t<T;t++){
    //perform core calculations for forward algorithm
    FLOAT a,a1,a2,final_score,score; //a->current string step, a1-> one string step back, a2->two string steps back
    score = scores[Nidx*T*NBASE+t*NBASE+target];
    a = MUL(score,smem[Nidx_t*2*total_L+(t%2)*total_L+(Lidx+L_pad)]);
    a1 = MUL(score,smem[Nidx_t*2*total_L+(t%2)*total_L+(Lidx+L_pad-1)]);
    a2 =  add3(score,smem[Nidx_t*2*total_L+(t%2)*total_L+(Lidx+L_pad-2)],mask_value);
    final_score = max3(a,a1,a2);
    int a_ = (a>a1 && a>a2)*0;
    int a1_ = (a1>a && a1>a2)*1;
    int a2_ = (a2>a && a2>a1)*2; 
    F[(Nidx*T*L+t*L+Lidx)]=final_score;
    BT[Nidx*T*(L)+t*(L)+Lidx]=Lidx-(a_+a1_+a2_);
    smem[Nidx_t*total_L*2+(((t+1)%2))*total_L+(Lidx+L_pad)]=final_score;
    __syncthreads();
  }
}

extern "C" __global__ void longstrand_fwd_logspace_align(
					    const FLOAT* __restrict__ scores, //raw CTC score matrix
              const int64_t* __restrict__ targets, //indexes representing symbols of the string being aligned
					    FLOAT* __restrict__ F, //Score matrix
					    long* __restrict__ BT, //Back trace matrix
					    const FLOAT* __restrict__ mask, //mask for knowing repeats
					    int T, //Time dimension size
					    int L, //This is complete global Length of the actual strand being aligned
					    int L_pad, //Padding used to handle low indexes looking back
              int offset //offset of L being used
					)
{
  int Lidx_t = threadIdx.x;
  int Lidx_g = Lidx_t+offset;
  int Lidx_dim = blockDim.x+L_pad;
  int64_t target = targets[Lidx_g];
  extern __shared__ FLOAT smem[];
  FLOAT mask_value = mask[Lidx];
  if(offset==0){
    if(Lidx_t==0){
      smem[Lidx_t] = ONE; //set this position to constant ONE, immitates behavior of "start" symbol
      smem[Lidx_dim+Lidx_t]=ONE;
      mask_value = ONE; //first position needs to be able to use 2-back "start" symbol
    }
    else if(Lidx_t==1){
      smem[Lidx_t] = ZERO; 
      smem[Lidx_dim+Lidx_t] = ZERO;
    } 
  }
  else{
      //Just set the first two previous to Log(zero), reasoning is that offset>0, and scores should be undefined for t=-1 for Lidx_g-1 and Lidx_g-2
      smem[0] = ZERO; 
      smem[1] = ZERO;
  }
  smem[Lidx_t+L_pad] = ZERO;
  __syncthreads();
  for(int t=0;t<T;t++){
    //perform core calculations for forward algorithm
    FLOAT a,a1,a2,final_score,score; //a->current string step, a1-> one string step back, a2->two string steps back
    score = scores[t*NBASE+target];
    a = MUL(score,smem[(t%2)*Lidx_dim+(Lidx_t+L_pad)]);
    a1 = MUL(score,smem[(t%2)*Lidx_dim+(Lidx_t+L_pad-1)]);
    a2 =  add3(score,smem[(t%2)*Lidx_dim+(Lidx_t+L_pad-2)],mask_value);
    final_score = max3(a,a1,a2);
    int a_ = (a>a1 && a>a2)*0;
    int a1_ = (a1>a && a1>a2)*1;
    int a2_ = (a2>a && a2>a1)*2; 
    F[(t*L+Lidx_g)]=final_score;
    BT[t*(L)+Lidx_g]=Lidx_g-(a_+a1_+a2_);
    smem[(((t+1)%2))*Lidx_dim+(Lidx_t+L_pad)]=final_score;
    if(offset>0){ //need to make sure results are propagated from previous parts of the calculation when GPU-blocking is done
      if(Lidx_t==0){
        smem[(((t+1)%2))*Lidx_dim+Lidx_t] = F[t*L+Lidx_g-2];
      }
      else if(Lidx_t==1){
        smem[(((t+1)%2))*Lidx_dim+Lidx_t] = F[t*L+Lidx_g-1];
      } 
    }
    __syncthreads();
  }
}
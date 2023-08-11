#include "viterbi.cuh"
#include "viterbi_1.hpp"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#include <thrust/advance.h>
#include <thrust/copy.h>

__device__ __forceinline__ uint8_t base2int(char base) {
  switch (base)
  {
  case 'A':
    return 0;  
  case 'C':
    return 1;
  case 'G':
    return 2;
  case 'T':
    return 3;
  default:
    return 0;
  }
}
__device__ __forceinline__ char complement(char base) {
  switch (base)
  {
  case 'A':
    return 'T';  
  case 'C':
 return 'G';
  case 'G':
    return 'C';
  case 'T':
    return 'A';
  default:
    return 'A';
  }
}

__global__ void beam_kernel_1(kernel_values_t* args){
  uint32_t st_pos = threadIdx.x+blockDim.x*blockIdx.x;
  uint32_t st_conv = threadIdx.y+blockDim.y*blockIdx.y;
  uint32_t nstate_conv = args->nstate_conv;
  uint32_t nstate_pos = args->nstate_pos;
  uint32_t list_size = args->nlist_pos;
  uint32_t candidate_iter=0;
  uint32_t t = args->trellis_time;
  LVA_path_t_SOA curr_best_paths=args->curr_best_paths;
  LVA_path_t_SOA prev_best_paths=args->prev_best_paths;
  LVA_candidate_t_SOA candidate_paths=args->candidate_paths;
  float* post = args->post;
  if(st_pos>=args->end_pos || st_pos < args->start_pos || st_conv>=args->nstate_conv) return;
  //do stay transitions
  uint32_t num_stay_candidates = 0;
  for(uint32_t i = 0; i<list_size; i++){
    uint32_t st = get_state_idx(st_pos,st_conv,i);
    if(prev_best_paths.score[st]<=ZERO) break;
    num_stay_candidates++;
    float new_score_blank = logsumexp2(prev_best_paths.score_blank[st]+ post[get_post_idx(t,0)],prev_best_paths.score_nonblank[st] + post[get_post_idx(t,0)]);
    float new_score_nonblank = prev_best_paths.score_nonblank[st] + post[get_post_idx(t,prev_best_paths.last_base[st])];

    uint32_t candidate_pos = get_candidate_idx(st_pos,st_conv,candidate_iter);
    candidate_paths.assign(candidate_pos,
            prev_best_paths.msg[st],new_score_nonblank,new_score_blank,prev_best_paths.last_base[st],true,
            prev_best_paths.hedge_context[st]);
    candidate_iter++;
  } 
	
	uint8_t nbits=0;
	uint8_t msg_value=0;
  // Now go through non-stay transitions.
  // For each case, first look in the stay transitions to check if the msg has
  // already appeared before
  // start with psidx = 1, since 0 corresponds to stay (already done above)        
  if (st_pos != 0) { // otherwise only stay transition makes sense
	  nbits = gpu_pattern_vector.pattern[(st_pos+args->offset-1)%gpu_pattern_vector.pattern_length];
    uint32_t number_previous = (1ULL<<nbits)-1;
    uint32_t start_index = number_previous; 
    for (uint32_t psidx = number_previous; psidx < 2*(number_previous)+1; psidx++)
    {
      msg_value = gpu_previous_states[st_conv*TOTAL_PREVIOUS+psidx].guess_value; // Moved msg_value here, should be the same for all incoming states
      uint32_t prev_st_conv = gpu_previous_states[st_conv*TOTAL_PREVIOUS+psidx].st_conv;
      uint32_t prev_st_pos = st_pos - 1;
      for (uint32_t i = 0; i < list_size; i++)
      {
        uint32_t prev_st = get_state_idx(prev_st_pos,prev_st_conv,i);
        if (prev_best_paths.score[prev_st] <=ZERO)
          break;
        // KV: NOTE: new_base will be fully determined only once we look at each previous candidate
        char new_base = prev_best_paths.hedge_context[prev_st].getNextSymbol(nbits,msg_value);
        if (args->rc_flag) new_base = complement(new_base); // make sure to complement HEDGES output if necessary


        uint32_t candidate_index = get_candidate_idx(st_pos,st_conv,candidate_iter);
        // KV: NOTE: I'm making msg the base message instead of bits message
        bitset_t& msg = prev_best_paths.msg[prev_st];
        candidate_paths.msg[candidate_index] = msg;
        uint8_t msg_newbits = base2int(new_base);
        candidate_paths.msg[candidate_index]<<=2;
        candidate_paths.msg[candidate_index] |=msg_newbits;

        float new_score_blank = ZERO;
        float new_score_nonblank = ZERO;
        if (msg_newbits != prev_best_paths.last_base[prev_st])
        {
          new_score_nonblank =
              logsumexp2(prev_best_paths.score_blank[prev_st] + post[get_post_idx(t, msg_newbits + 1)],
                         prev_best_paths.score_nonblank[prev_st] + post[get_post_idx(t, msg_newbits + 1)]);

        }
        else
        {
          // the newly added base is same as last base so we can't have the
          // thing ending with non_blank (otherwise it gets collapsed)
          new_score_nonblank = prev_best_paths.score_blank[prev_st] + post[get_post_idx(t, msg_newbits + 1)];

        }
        if (new_score_nonblank <= ZERO)
        {
          continue;
          // overall score is -INF (this might happen if
          // score_blank for previous path is -INF and we are
          // in second case above)
        }

        // now check if this is already present in the stay transitions.
        // first try to match the new_base for speed, then look at full msg.
        // if already present, update the nonblank score
        bool match_found = false;
        for (uint32_t j = 0; j < num_stay_candidates; j++)
        {
          uint32_t stay_candidate_index= get_candidate_idx(st_pos,st_conv,j);
          if (msg_newbits == candidate_paths.last_base[stay_candidate_index])
          {
            if (msg == candidate_paths.msg[stay_candidate_index])
            {
              match_found = true;
                candidate_paths.score_nonblank[stay_candidate_index] =
                  logsumexp2(candidate_paths.score_nonblank[stay_candidate_index], new_score_nonblank);
              break;
            }
          }
        }
        if (!match_found)
        {
          uint32_t candidate_pos = get_candidate_idx(st_pos,st_conv,candidate_iter);
          //NOTE: optimized message assignment out, already computed message in candidate memory
          candidate_paths.assign(candidate_pos,new_score_nonblank,new_score_blank,msg_newbits+1,false,
            prev_best_paths.hedge_context[prev_st]);
          candidate_iter++;
        }
      }
    }
  }

  // update scores based on score_blank and score_nonblank
  for (uint32_t i = 0; i < candidate_iter; i++){
    uint32_t candidate_pos = get_candidate_idx(st_pos,st_conv,i);
    candidate_paths.compute_score_gpu(candidate_pos); 
  } 

  uint32_t num_candidates_to_keep = list_size<candidate_iter ? list_size : candidate_iter;

  
  uint32_t index[MAX_CANDIDATES];
  for(int i=0; i<candidate_iter; i++) index[i]=i;
  
  //simple bubble sort
  for(int i=0; i<(int)candidate_iter-1;i++){
      for(int j=0; j<(int)candidate_iter-1-i;j++){
      	 if(candidate_paths.score[get_candidate_idx(st_pos,st_conv,index[j])]<candidate_paths.score[get_candidate_idx(st_pos,st_conv,index[j+1])]){
	    uint32_t temp=index[j+1];
	    index[j+1]=index[j];
	    index[j]=temp;
	 }
      }
  }
  for(uint32_t i =0; i<candidate_iter;i++){
    uint32_t candidate_pos=get_candidate_idx(st_pos,st_conv,index[i]);
  }


  // The reason is that the candidates don't "own" their pointer to the context, its just copied from previous states
  // we cannot directly update the previous state context, else there may be a conflict if it is inherited in several places
  // thus, we need to copy candidate by candidate more carefully
  for (int i = 0; i < num_candidates_to_keep; i++)
  {
    uint32_t st =get_state_idx(st_pos,st_conv,i);
    uint32_t candidate_pos = get_candidate_idx(st_pos,st_conv,index[i]);
    curr_best_paths.copy_candidate(candidate_paths,nbits,msg_value,st,candidate_pos);
  }

  // fill any remaining positions in list with score -INF so they are not used later
  for (uint32_t i = num_candidates_to_keep; i < list_size; i++) {
    uint32_t st =get_state_idx(st_pos,st_conv,i);
    curr_best_paths.score[st] = ZERO;
  }
}

__host__ void call_beam_kernel_1(kernel_values_t& args){
    //Determine the sizing of kernel grid
    int dim  = 16;
    dim3 threads_per_block(dim,dim,1); //using 32x32 blocks (x-> positions) (y-> convolutional)
    assert(args.nstate_conv%dim==0);
    uint32_t H_blocks = args.nstate_conv/dim;
    uint32_t pos_blocks = (args.nstate_pos/dim)+1; //+1 to ensure there's enough
    dim3 blocks(pos_blocks,H_blocks,1);
    //copy args to gpu
    kernel_values_t* gpu_args=NULL;
    cudaMalloc((void**)&gpu_args,sizeof(kernel_values_t));
    cudaMemcpy((void*)gpu_args,(void*)&args,sizeof(kernel_values_t),cudaMemcpyHostToDevice);
    beam_kernel_1<<< blocks, threads_per_block>>>(gpu_args);
    cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess)  printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();	
    cudaFree((void*)gpu_args);
}
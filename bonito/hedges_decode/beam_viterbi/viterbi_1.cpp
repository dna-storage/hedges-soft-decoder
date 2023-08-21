/*
This code is largely adapted from
https://github.com/shubhamchandak94/nanopore_dna_storage

Changes made to the original include fitting it to work with Hedges,
so that an appropriate comparison of approaches can be made 
*/
#include <Python.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "hedges_hooks_c.h"
#include "viterbi_1.hpp"
#include "viterbi.cuh"
#include <assert.h>


/*GLOBAL CONSTANTS*/
__device__ __constant__  prev_state_info_t gpu_previous_states[MAX_NSTATE_CONV*TOTAL_PREVIOUS]; 
__device__ __constant__  pattern_info_t gpu_pattern_vector;
/*END GLOBAL CONSTANTS*/



std::map<char,char> complement = {{'A','T'},
                                          {'T','A'},
                                          {'G','C'},
                                          {'C','G'}};


const char int2base[NBASE] = {'A', 'C', 'G', 'T'};


/*GLOBAL DEFS*/
uint32_t nstate_conv;
uint8_t mem_conv;
uint32_t initial_state_conv;
uint32_t nstate_pos;
bool rc_flag = false;
/*END GLOBAL DEFS*/

std::vector<bitset_t> decode_post_conv_parallel_LVA(
    float* post, const uint32_t msg_len,
    const uint32_t list_size, const uint32_t num_thr,
    const uint32_t post_T,
    void* hedge_state,
    uint32_t offset,
    const uint32_t max_deviation);


std::string bases_from_bits(bitset_t b,uint32_t message_length){
  std::string ret_string="";
  uint32_t bitset_index=0;
  //assuming big endian in the bitset
  for(int i=message_length*2-1; i>=1;i-=2){ //iterate in reverse because of bitset ordering
    uint8_t bit1 = b[i];
    uint8_t bit2 = b[i-1];
    ret_string += int2base[bit1*2+bit2];
  }
  //std::cout<<"returning string "<<std::endl;
  return ret_string;
}

std::vector<prev_state_info_t> find_prev_states(void* h,const uint32_t &st2_conv,const uint8_t &nbits) ;

PyObject* beam_viterbi_1(
  uint32_t conv_mem, //number of bits for convolutional code
  uint32_t initial_state, //initial state to start code at
  uint32_t message_length, //length of the message (in bases)
  uint32_t list_size, //number of items per list
  uint32_t T, //total size of data block
  bool rc, //flag for reverse complement
  float* ctc_data, //pointer to flattened data representing CTC matrix, organized as [T]
  PyObject* hedges_state_pointer,//pyobject that actually refers to a hedges state pointer which can derive necessary contexts
  uint32_t offset,
  uint32_t omp_threads //number of threads to launch
){
  //take in some inititalizing information
  nstate_conv=1ULL<<conv_mem;
  mem_conv=(uint8_t)conv_mem;
  initial_state_conv = initial_state;
  nstate_pos=message_length+1;
  rc_flag=rc;
  void* hedges_head_state = PyLong_AsVoidPtr(hedges_state_pointer); //NOTE: this is a hedges global object, not a context object

  std::vector<bitset_t> decode_result = decode_post_conv_parallel_LVA(ctc_data,
                                                                                    message_length,
                                                                                    list_size,
                                                                                    omp_threads,
                                                                                    T,
                                                                                    hedges_head_state,
								                                                                    offset,
                                                                                    0); 
  std::string return_string = bases_from_bits(decode_result[0],message_length);

  return Py_BuildValue("s",return_string.c_str());

}

std::vector<bitset_t> decode_post_conv_parallel_LVA(
    float* post, const uint32_t msg_len,
    const uint32_t list_size, const uint32_t num_thr,
    const uint32_t post_T,
    void* hedge_state,
    uint32_t offset,
    const uint32_t max_deviation)  {
  
    omp_set_num_threads(num_thr);
    uint64_t nstate_total_64 = nstate_pos * nstate_conv;
    if (nstate_total_64 >= ((uint64_t)1 << 32)) throw std::runtime_error("Too many states, can't fit in 32 bits");
    uint32_t nstate_total = (uint32_t)nstate_total_64;
    uint32_t nblk = post_T;


  // instead of traceback, store the msg till now as a bitset
  if (msg_len*2 > BITSET_SIZE)
    throw std::runtime_error("msg_len can't be above BITSET_SIZE");


  //get pattern vector from HEDGES
  std::vector<uint8_t> pattern = get_pattern__c(hedge_state);
  assert(pattern.size()>0);
  pattern_info_t pattern_info(pattern);

  //make a prototype context to copy to every state object
  std::vector<uint32_t> context_vect = get_context_data__c(hedge_state);
  char* constraint_ptr;
  get_constraint_data__c(hedge_state,(void**)&constraint_ptr);
  context proto_context(context_vect,(void*)constraint_ptr);
  free(constraint_ptr);
  assert(list_size<=MAX_LIST_SIZE);

  LVA_path_t_SOA curr_best_paths(nstate_total*list_size,true);
  LVA_path_t_SOA prev_best_paths(nstate_total*list_size,true);
  curr_best_paths.set_context(proto_context);
  prev_best_paths.set_context(proto_context);
  //std::cout<<"make CPU paths"<<std::endl;

// precompute the previous states and associated info for all states now
// note that this is valid only for st_pos > 0 (if st_pos = 0, only previous
// state allowed is same state - which is always first entry in the
// prev_state_vector)

// KV NOTE: flatten this to an array so it can be passed off to the GPU easier
GPU_CONST_CHECK((TOTAL_PREVIOUS*nstate_conv));
prev_state_info_t* prev_state_flat_vector = new prev_state_info_t[TOTAL_PREVIOUS*nstate_conv];
//std::cout<<std::dec<<"TOTAL_PREVIOUS*nstate_conv "<<TOTAL_PREVIOUS*nstate_conv<<std::endl;
for (uint8_t nbits = 0; nbits < NBIT_RANGE; nbits++) {
  uint32_t nbit_start_index  = (uint32_t)((1ULL<<nbits)-1);
  #pragma omp parallel
  #pragma omp for
  for (uint32_t st_conv = 0; st_conv < nstate_conv; st_conv++){
      std::vector<prev_state_info_t> p = find_prev_states(hedge_state,st_conv, nbits);
      uint32_t previous_offset=0;
      for (auto& previous:p){
        uint32_t previous_index = nbit_start_index+previous_offset;
        prev_state_flat_vector[st_conv*TOTAL_PREVIOUS+previous_index]=previous;
	previous_offset+=1;
      }
  }
}

  // set score_blank to zero for initial state
  uint32_t initial_st = get_state_idx(0, initial_state_conv,0);
  curr_best_paths.score_blank[initial_st] = 0.0;
  curr_best_paths.score_nonblank[initial_st] = ZERO;
  curr_best_paths.last_base[initial_st]=0;
  curr_best_paths.compute_score_cpu(initial_st);

  //std::cout<<"CPU Init"<<std::endl;
  //Move initialized trellis states to GPU 
  LVA_path_t_SOA gpu_curr_best_paths(nstate_total*list_size,false);
  LVA_path_t_SOA gpu_prev_best_paths(nstate_total*list_size,false);
  LVA_candidate_t_SOA candidate_paths(nstate_total*MAX_CANDIDATES,false);
  host_to_dev(curr_best_paths,gpu_curr_best_paths);
  host_to_dev(prev_best_paths,gpu_prev_best_paths);
  //copy prev_state_vector and pattern information into constant memory
  cudaMemcpyToSymbol(gpu_previous_states, (void*)prev_state_flat_vector, TOTAL_PREVIOUS*nstate_conv*sizeof(prev_state_info_t));
  cudaMemcpyToSymbol(gpu_pattern_vector, (void*)&pattern_info, sizeof(pattern_info_t));

  //std::cout<<"Init GPU mem"<<std::endl;
  //kernel argument loading
  kernel_values_t kernel_args;
  kernel_args.nstate_conv=nstate_conv;
  kernel_args.nstate_pos=nstate_pos;
  kernel_args.nlist_pos=list_size;
  kernel_args.post=post;
  kernel_args.post_T = post_T;
  kernel_args.post_NB = NBASE;
  kernel_args.curr_best_paths=gpu_curr_best_paths;
  kernel_args.prev_best_paths=gpu_prev_best_paths;
  kernel_args.candidate_paths=candidate_paths;
  kernel_args.offset=offset;
  kernel_args.rc_flag=rc_flag;
  //std::cout<<"Kernel Initialized"<<std::endl;
  // forward Viterbi pass
  for (uint32_t t = 0; t < nblk; t++) {
    //std::cout<<std::dec<<"Block index "<<(int)t<<std::endl;
    //if(t>600) break;
    // swap prev and curr arrays
    std::swap(kernel_args.curr_best_paths, kernel_args.prev_best_paths);
    // only allow pos which can have non -INF scores or will lead to useful
    // final states initially large pos is not allowed, and at the end small
    // pos not allowed (since those can't lead to correct st_pos at the end).

    // st is current state
    //KV: NOTE: these bounds being made here are pretty standard CTC boundaries
    uint32_t st_pos_start =
        std::max((int64_t)nstate_pos  - (nblk - t), (int64_t)0);
    uint32_t st_pos_end = std::min(t+2, nstate_pos);

    kernel_args.start_pos=st_pos_start;
    kernel_args.end_pos=st_pos_end;
    kernel_args.trellis_time=t;

    //call kernel to calculate updated trellis
    call_beam_kernel_1(kernel_args);

  }

  //transfer back to host and free gpu memory
  dev_to_host(curr_best_paths,gpu_curr_best_paths);
  dev_to_host(prev_best_paths,gpu_prev_best_paths);
  gpu_prev_best_paths.free();
  gpu_curr_best_paths.free();
  candidate_paths.free();
  prev_best_paths.free();

  //std::cout<<"Dev to host"<<std::endl;

  //reframe the SOA to an AOS to make data better for CPU
  LVA_path_t* curr_best_paths_AOS = new LVA_path_t[nstate_total*list_size];
  curr_best_paths.to_AOS(curr_best_paths_AOS,nstate_total*list_size);
  curr_best_paths.free();

  static auto LVA_path_t_compare = [](const LVA_path_t &a, const LVA_path_t &b) {
                              return a.score > b.score;
                            };

  //consider the best state at the end of the trellis
  uint32_t st_pos = nstate_pos - 1;  
  std::vector<LVA_path_t> final_path_set;
  for(int i=0; i<nstate_conv;i++){
    std::vector<LVA_path_t> conv_paths;
    for(int j=0; j<list_size;j++){
      uint32_t st_idx = get_state_idx(st_pos,i,j); //need to account for list positions not being addressed next to each other
      conv_paths.push_back(curr_best_paths_AOS[st_idx]);
    }
    std::sort(conv_paths.begin(),conv_paths.begin()+list_size,LVA_path_t_compare);
    final_path_set.push_back(conv_paths[0]);
  }
  
  uint32_t st_conv = std::min_element(final_path_set.begin(),final_path_set.end(),LVA_path_t_compare)-final_path_set.begin();


  std::vector<LVA_path_t> LVA_path_list_final;
  for(int i=0; i<list_size; i++) LVA_path_list_final.push_back(curr_best_paths_AOS[get_state_idx(st_pos, st_conv,i)]);


  // sort in decreasing order by score 
  // NOTE: the curr_best_paths list is not sorted since we use nth_element partial sorting
  std::sort(LVA_path_list_final.begin(), LVA_path_list_final.end(), LVA_path_t_compare);

  std::vector<bitset_t> decoded_msg_list;

  for (uint32_t list_pos = 0; list_pos < list_size; list_pos++) {
    decoded_msg_list.push_back(LVA_path_list_final[list_pos].msg);
    // FOR DEBUGGING
    //std::cout << "score: " << LVA_path_list_final[list_pos].score << "\n";
        //for (auto b : decoded_msg_list.back()) std::cout << b;
        ///std::cout << "\n\n";
  }
  delete[] curr_best_paths_AOS;
  return decoded_msg_list;
}

std::vector<prev_state_info_t> find_prev_states(void* h,
                                                const uint32_t &st2_conv,
                                                const uint8_t &nbits) {
  std::vector<prev_state_info_t> prev_vector;
  //for a given nbits, determine the previous states for st2_conv
  previous_states_t prev_states = get_previous_states__c(h,nbits,st2_conv);
  //std::cout<<std::dec<<"Get prev states for "<<st2_conv<<std::endl;
  prev_vector.resize(prev_states.states.size());
  for(int i=0; i< prev_vector.size(); i++){
    prev_vector[i].st_conv=prev_states.states[i];
    prev_vector[i].guess_value=prev_states.guess_value;
    //std::cout<<std::dec<<"Prev state is "<<prev_vector[i].st_conv<<std::endl;
  }
  return prev_vector;
}

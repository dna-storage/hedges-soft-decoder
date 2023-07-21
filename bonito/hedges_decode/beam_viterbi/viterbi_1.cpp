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
#include <bitset>
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

std::map<char,char> complement = {{'A','T'},
                                          {'T','A'},
                                          {'G','C'},
                                          {'C','G'}};

std::map<char,uint8_t> base2int = {{'A',0},{'C',1},{'G',2},{'T',3}};                                          


const uint8_t NBASE = 4;
const uint8_t NBIT_RANGE = 2; //range of total bits that can be transferred per base
const char int2base[NBASE] = {'A', 'C', 'G', 'T'};



typedef std::array<float, NBASE + 1> ctc_mat_t;
// at each time step, we have probabilities for A,C,G,T,blank
// uint8_t base: 0->A, 1->C, 2->G, 3->T


// for parallel LVA
const uint32_t BITSET_SIZE = 256;  // 32 bytes, TODO: KV: Need to set this size to work for HEDGES, make it so it is at least long enough to encode the message butwith bases
typedef std::bitset<BITSET_SIZE> bitset_t;

float logsumexpf(float x, float y);

// this is the main stucture to store the top paths for each state
struct LVA_path_t {
  bitset_t msg; //KV:Note: Msg is now the DNA-message, not the binary message. helps analysis to be easier later
  float score_nonblank; // score for path ending with non-blank base
  float score_blank; // score for path ending with non-blank base
  float score; // logsumexp(score_nonblank,score_blank), used for sorting
  // If score is -INF, score_nonblank and score_blank can be garbage 
  uint8_t last_base; // for computing updated score_nonblank for stay transition
                     // and to compare with new_base in non-stay transitions 
                     // for checking whether nonblank->nonblank makes sense
  void* context; //for keeping track of contexts
  bool is_stay=false; //keeps track on whether path being considered is a stay path, allows avoiding context update

  LVA_path_t() {
    float INF = std::numeric_limits<float>::infinity();
    score_nonblank = -INF;
    score_blank = -INF;
    compute_score();
    context=NULL;
  }

  LVA_path_t(const bitset_t &msg_, const float &score_nonblank_, 
             const float &score_blank_, const uint8_t &last_base_,bool stay,void* context_) {
    msg = msg_;
    score_nonblank = score_nonblank_;
    score_blank = score_blank_;
    last_base = last_base_;
    is_stay=stay;
    context=context_; //copy context pointer, but don't really do anything with it yet.
  }

  // update score based on nonblank and blank score.
  // NOTE: this must be called externally
  void compute_score() {
    score = logsumexpf(score_nonblank, score_blank);
  }

  //copies in a candidate's value and makes hedges context update
  void copy_candidate(LVA_path_t& candidate, uint8_t nbits, uint8_t msg_value){
    this->msg = candidate.msg;
    this->score_nonblank=candidate.score_nonblank;
    this->score_blank=candidate.score_blank;
    this->score=candidate.score;
    this->last_base=candidate.last_base;
    if(!candidate.is_stay)update_context__c(this->context,candidate.context,nbits,msg_value);
    else copy_context_no_update__c(this->context,candidate.context); //copy candidate conext information without advancing
    this->is_stay=false;
  }
};

// struct for storing information about previous state for a current state and
// the transition
struct prev_state_info_t {
  uint32_t st_conv;
  uint32_t guess_value;
};


void write_bit_array(const std::vector<bool> &outvec,
                     const std::string &outfile);


template <class T>
void write_vector(const std::vector<T> &outvec, const std::string &outfile) {
  // write values in vector, one per line
  std::ofstream fout(outfile);
  for (auto v : outvec) {
    fout << v << "\n";
  }
  fout.close();
}

float logsumexpf(float x, float y) {
  float INF = std::numeric_limits<float>::infinity();
  if (x == -INF && y == -INF)
    return -INF; // the formula below returns nan
  float max_x_y = std::max(x, y);
  return max_x_y + logf(expf(x - max_x_y) + expf(y - max_x_y));
}


std::vector<std::vector<bitset_t>> decode_post_conv_parallel_LVA(
    const std::vector<ctc_mat_t> &post, const uint32_t msg_len,
    const uint32_t list_size, const uint32_t num_thr,
    const uint32_t max_deviation);

void write_bit_array(const std::vector<bool> &outvec,
                     const std::string &outfile) {
  std::ofstream fout(outfile);
  for (bool b : outvec) fout << (b ? '1' : '0');
  fout.close();
}

void write_char_array(const std::vector<char> &vec,
                      const std::string &outfile) {
  std::ofstream fout(outfile);
  for (char c : vec) fout << c;
  fout.close();
}


uint32_t get_state_idx(const uint32_t st_pos, const uint32_t st_conv) {
  return st_pos * nstate_conv + st_conv;
}

template
<uint32_t N>
std::string bases_from_bits(std::bitset<N> b,uint32_t message_length){
  std::string ret_string="";
  uint32_t bitset_index=0;
  assert(N>message_length*2);
  //assuming big endian in the bitset
  for(int i=message_length-1; i>=0;i--){ //iterate in reverse because of bitset ordering
    uint32_t bit1 = b[i];
    uint32_t bit2 = b[i-1];
    ret_string += int2base[bit1*2+bit2];
    bitset_index+=2;
  }
  return ret_string;
}

std::vector<prev_state_info_t> find_prev_states(void* h,const uint32_t &st2_conv,const uint8_t &nbits) ;


bool rc_flag = false;

uint8_t mem_conv;
uint32_t nstate_conv;
uint32_t initial_state_conv;
uint32_t nstate_pos;

PyObject* beam_viterbi_1(
  uint32_t conv_mem, //number of bits for convolutional code
  uint32_t initial_state, //initial state to start code at
  uint32_t message_length, //length of the message (in bases)
  uint32_t list_size, //number of items per list
  uint32_t T, //total size of data block
  bool rc, //flag for reverse complement
  float* ctc_data, //pointer to flattened data representing CTC matrix, organized as [T]
  PyObject* hedges_state_pointer,//pyobject that actually refers to a hedges state pointer which can derive necessary contexts
  uint32_t omp_threads //number of threads to launch
){
  //take in some inititalizing information
  nstate_conv=1ULL<<conv_mem;
  mem_conv=(uint8_t)conv_mem;
  initial_state_conv = initial_state;
  nstate_pos=message_length;
  nstate_pos = message_length; 
  rc_flag=rc;
  void* hedges_head_state = PyLong_AsVoidPtr(hedges_state_pointer); //NOTE: this is a hedges global object, not a context object

  std::vector<bitset_t> decode_result = decode_post_conv_parallel_LVA(ctc_data,
                                                                                    message_length,
                                                                                    list_size,
                                                                                    omp_threads,
                                                                                    T,
                                                                                    hedges_head_state,
                                                                                    0); 
  std::string return_string = bases_from_bits<BITSET_SIZE>(decode_result[0],message_length);

  return Py_BuildValue("s",return_string.c_str());

}

std::vector<bitset_t> decode_post_conv_parallel_LVA(
    float* post, const uint32_t msg_len,
    const uint32_t list_size, const uint32_t num_thr,
    const uint32_t post_T,
    void* hedge_state,
    const uint32_t max_deviation) {
  omp_set_num_threads(num_thr);
  float INF = std::numeric_limits<float>::infinity();
  uint64_t nstate_total_64 = nstate_pos * nstate_conv;
  if (nstate_total_64 >= ((uint64_t)1 << 32))
    throw std::runtime_error("Too many states, can't fit in 32 bits");
  uint32_t nstate_total = (uint32_t)nstate_total_64;
  uint32_t nblk = post_T;


  // instead of traceback, store the msg till now as a bitset
  if (msg_len*2 > BITSET_SIZE)
    throw std::runtime_error("msg_len can't be above BITSET_SIZE");

  // arrays for storing previous and current best paths
  // [nstate_total][list_size]
  LVA_path_t **curr_best_paths = new LVA_path_t *[nstate_total];
  LVA_path_t **prev_best_paths = new LVA_path_t *[nstate_total];
  for (uint32_t i = 0; i < nstate_total; i++) {
    curr_best_paths[i] = new LVA_path_t[list_size]();
    prev_best_paths[i] = new LVA_path_t[list_size]();
    //KV: make sure contexts are started up
    for(uint32_t j=0; j<list_size; j++){
      //need to create 2 contexts so that overwriting does not happen on state updates
      curr_best_paths[i][j].context = make_context__c(hedge_state);
      prev_best_paths[i][j].context = make_context__c(hedge_state);
    }
  }

  // lambda expression to compare paths (decreasing in score)
  auto LVA_path_t_compare = [](const LVA_path_t &a, const LVA_path_t &b) {
                              return a.score > b.score;
                            };


  // precompute the previous states and associated info for all states now
  // note that this is valid only for st_pos > 0 (if st_pos = 0, only previous
  // state allowed is same state - which is always first entry in the
  // prev_state_vector)
  std::vector<std::vector<std::vector<prev_state_info_t>>> prev_state_vector(NBIT_RANGE);
#pragma omp parallel
#pragma omp for
  for (uint8_t nbits = 0; nbits < NBIT_RANGE; nbits++) {
    prev_state_vector[nbits].resize(nstate_conv);
    for (uint32_t st_conv = 0; st_conv < nstate_conv; st_conv++)
        prev_state_vector[nbits][st_conv] =
            find_prev_states(hedge_state,st_conv, nbits);
  }

  // set score_blank to zero for initial state
  uint32_t initial_st = get_state_idx(0, initial_state_conv);
  curr_best_paths[initial_st][0].score_blank = 0.0;
  curr_best_paths[initial_st][0].score_nonblank = -INF;
  curr_best_paths[initial_st][0].compute_score();
  // forward Viterbi pass

  for (uint32_t t = 0; t < nblk; t++) {
    // swap prev and curr arrays
    std::swap(curr_best_paths, prev_best_paths);

    // only allow pos which can have non -INF scores or will lead to useful
    // final states initially large pos is not allowed, and at the end small
    // pos not allowed (since those can't lead to correct st_pos at the end).

    // st is current state
    //KV: NOTE: these bounds being made here are pretty standard CTC boundaries
    uint32_t st_pos_start =
        std::max((int64_t)nstate_pos - 2 - (nblk - 1 - t), (int64_t)0);
    uint32_t st_pos_end = std::min(t + 2, nstate_pos);

    /* KV: NOTE: Removed this, seems to be a heuristic that limits numbers of calculations
    st_pos_start = std::max(
        (int64_t)st_pos_start, (int64_t)((double)(t) / nblk * nstate_pos - max_deviation));
    st_pos_end = std::min(st_pos_start + 2 * max_deviation, st_pos_end);
    */

#pragma omp parallel
#pragma omp for schedule(dynamic)
    for (uint32_t st_pos = st_pos_start; st_pos < st_pos_end; st_pos++) {

      // vector containing the candidate items for next step list
      std::vector<LVA_path_t> candidate_paths;

      uint8_t nbits = get_nbits__c(hedge_state,st_pos); //KV: this needs to ask hedges what the nbits is for this length

      for (uint32_t st_conv = 0; st_conv < nstate_conv; st_conv++) {
        // check if this is a valid state, otherwise continue
        //if (!valid_state_array[nstate_conv * st_pos + st_conv]) continue;

        // clear candidate_paths
        candidate_paths.clear();

        // first do stay transition
        uint32_t st = get_state_idx(st_pos, st_conv);
        uint32_t num_stay_candidates = 0;
        for (uint32_t i = 0; i < list_size; i++) {
          if (prev_best_paths[st][i].score == -INF)
            break;
          num_stay_candidates++;
          float new_score_blank = logsumexpf(prev_best_paths[st][i].score_blank + post[t*(NBASE+1)+NBASE], 
                                   prev_best_paths[st][i].score_nonblank + post[t*(NBASE+1)+NBASE]);
          float new_score_nonblank = prev_best_paths[st][i].score_nonblank +
                                     post[t*(NBASE+1)+prev_best_paths[st][i].last_base];
          candidate_paths.emplace_back(prev_best_paths[st][i].msg,new_score_nonblank,
                                       new_score_blank,prev_best_paths[st][i].last_base,true,prev_best_paths[st][i].context);
        }

        // Now go through non-stay transitions.
        // For each case, first look in the stay transitions to check if the msg has
        // already appeared before
        // start with psidx = 1, since 0 corresponds to stay (already done above)
        const auto &prev_states_st = prev_state_vector[nbits][st_conv];
        uint8_t msg_value = prev_states_st[0].guess_value; //Moved msg_value here, should be the same for all incoming states 
        if (st_pos != 0) { // otherwise only stay transition makes sense
          for (uint32_t psidx = 1; psidx < prev_states_st.size(); psidx++) {
            uint32_t prev_st_pos = st_pos - 1;
            uint32_t prev_st = get_state_idx(prev_st_pos, prev_states_st[psidx].st_conv);
           
            for (uint32_t i = 0; i < list_size; i++) {
              if (prev_best_paths[prev_st][i].score == -INF)
                break;

              //KV: NOTE: new_base will be fully determined only once we look at each previous candidate
              char new_base = peek_context__c(prev_best_paths[prev_st][i].context,nbits,msg_value);
              if(rc_flag) new_base= complement[new_base]; //make sure to complement HEDGES output if necessary

              //KV: NOTE: I'm making msg the base message instead of bits message
              bitset_t msg = prev_best_paths[prev_st][i].msg;
              uint8_t msg_newbits = base2int[new_base];
              msg = (msg << 2) | bitset_t(msg_newbits);

              float new_score_blank = -INF;
              float new_score_nonblank;
              if (new_base != prev_best_paths[prev_st][i].last_base) {
                new_score_nonblank = 
                    logsumexpf(prev_best_paths[prev_st][i].score_blank + post[t*(NBASE+1)+new_base],
                             prev_best_paths[prev_st][i].score_nonblank + post[t*(NBASE+1)+new_base]);
              } else {
                // the newly added base is same as last base so we can't have the 
                // thing ending with non_blank (otherwise it gets collapsed)
                new_score_nonblank = prev_best_paths[prev_st][i].score_blank + post[t*(NBASE+1)+new_base];
              }
              if (new_score_nonblank == -INF)
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
              for (uint32_t j = 0; j < num_stay_candidates; j++) {
                if (new_base == candidate_paths[j].last_base) {
                  if (msg == candidate_paths[j].msg) {
                    match_found = true;
                    candidate_paths[j].score_nonblank = 
                            logsumexpf(candidate_paths[j].score_nonblank,new_score_nonblank);
                    break;
                  }
                }
              }
              if (!match_found) {
                candidate_paths.emplace_back(msg,new_score_nonblank,new_score_blank,new_base,false,msg_value,prev_best_paths[prev_st][i].context);
              }
            }
          }
        }
       
        uint32_t num_candidates = candidate_paths.size(); 
        // update scores based on score_blank and score_nonblank
        for (uint32_t i = 0; i < num_candidates; i++)
           candidate_paths[i].compute_score();

        auto num_candidates_to_keep = std::min(list_size, num_candidates);
        // use nth_element to to do partial sorting if num_candidates_to_keep < num_candidates
        if (num_candidates_to_keep < num_candidates && num_candidates_to_keep > 0)
          std::nth_element(candidate_paths.begin(),
                           candidate_paths.begin()+num_candidates_to_keep-1,
                           candidate_paths.end(),
                           LVA_path_t_compare);
        
        //KV: TODO: Candidate paths cannot simply be copied into the curr_best_paths set
        //The reason is that the candidates don't "own" their pointer to the context, its just copied from previous states
        //we cannot directly update the previous state context, else there may be a conflict if it is inherited in several places
        //thus, we need to copy candidate by candidate more carefully
        for(int candidate_iter = 0; candidate_iter<num_candidates_to_keep;candidate_iter++){
          curr_best_paths[st][candidate_iter].copy_candidate(candidate_paths[candidate_iter],nbits,msg_value);
        }



        // copy over top candidate paths to curr_best
        std::copy(candidate_paths.begin(),candidate_paths.begin()+num_candidates_to_keep,
                    curr_best_paths[st]);
        // fill any remaining positions in list with score -INF so they are not used later
        for (uint32_t i = num_candidates_to_keep; i < list_size; i++)
          curr_best_paths[st][i].score = -INF;
      }
    }
  }

  uint32_t st_pos = nstate_pos - 1, st_conv = 0;  // last state
  LVA_path_t *LVA_path_list_final = curr_best_paths[get_state_idx(st_pos, st_conv)];

  // sort in decreasing order by score 
  // NOTE: the curr_best_paths list is not sorted since we use nth_element partial sorting
  std::sort(LVA_path_list_final, LVA_path_list_final+list_size, LVA_path_t_compare);

  std::vector<bitset_t> decoded_msg_list;

  // reverse bitset so 
  for (uint32_t list_pos = 0; list_pos < list_size; list_pos++) {
    decoded_msg_list.push_back(LVA_path_list_final[list_pos].msg);
    // FOR DEBUGGING
    /*
        std::cout << "score: " << LVA_path_list_final[list_pos].score << "\n";
        for (auto b : decoded_msg_list.back()) std::cout << b;
        std::cout << "\n\n";
    */
  }
  // std::cout << "Final list size: " << decoded_msg_list.size() << "\n";

  for (uint32_t i = 0; i < nstate_total; i++) {
    delete[] curr_best_paths[i];
    delete[] prev_best_paths[i];
  }
  delete[] curr_best_paths;
  delete[] prev_best_paths;
  return decoded_msg_list;
}

std::vector<prev_state_info_t> find_prev_states(void* h,
                                                const uint32_t &st2_conv,
                                                const uint8_t &nbits) {
  std::vector<prev_state_info_t> prev_vector;
  //for a given nbits, determine the previous states for st2_conv
  previous_states_t prev_states = get_previous_states__c(h,nbits,st2_conv);

  prev_vector.resize(prev_states.states.size());
  for(int i=0; i< prev_vector.size(); i++){
    prev_vector[i].st_conv=prev_states.states[i];
    prev_vector[i].guess_value=prev_states.guess_value;
  }
  return prev_vector;
}
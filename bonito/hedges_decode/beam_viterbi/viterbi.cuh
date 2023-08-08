#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#define ZERO -1.000000E+38 // this effectively replacing INF of original beam implementation
#define NBASE 4
#define NBIT_RANGE  2 // range of total bits that can be transferred per base (non inclusive)
#define BITSET_SIZE 2160 * 2
#define MAX_NSTATE_CONV ((uint32_t)1ULL << 11)
#define MAX_LIST_SIZE 8 //set a max list size for mem safety
#define MAX_CANDIDATES (8*((1ULL<<NBIT_RANGE)-1)) //set the max number of candidates so we can allocate memory
#define GPU_CONST_CHECK(x) (assert(x <= MAX_NSTATE_CONV)) // safety checker to make sure constant memory isn't over-subbed
#define get_state_idx(st_pos, st_conv, list_pos) (st_conv * ((uint32_t)nstate_pos) * list_size + ((uint32_t)list_pos) * ((uint32_t)nstate_pos) + st_pos) // indexing for both cuda and c++ code
#define get_candidate_idx(st_pos,st_conv,can_pos)  (st_conv * ((uint32_t)nstate_pos) * MAX_CANDIDATES  + ((uint32_t)can_pos) * ((uint32_t)nstate_pos) + st_pos) 
#define get_post_idx(t, b) ((uint32_t)t * (NBASE + 1) + (uint32_t)b)
#define TOTAL_PREVIOUS ((uint32_t)(1ULL << NBIT_RANGE) - 1)

__device__ __forceinline__ float max2(float a, float a1)
{
  return a > a1 ? a : a1;
}

__device__ __forceinline__ float logsumexp2(float a, float a1)
{
  float maxa = max2(a, a1);
  return maxa + log(exp(a - maxa) + exp(a1 - maxa));
}

static float logsumexpf(float x, float y)
{
  float max_x_y = std::max(x, y);
  return max_x_y + logf(expf(x - max_x_y) + expf(y - max_x_y));
}



/**********************************************************************
  Hedges code snippets so that updating contexts can be seen by the GPU
***********************************************************************/

inline CUDA_CALLABLE_MEMBER
 uint64_t ranhash(uint64_t u)
  {
    /* Logic adapted from Press et al. */
    uint64_t v = u * 3935559000370003845ul + 2691343689449507681ul;
    v ^= v >> 21;
    v ^= v << 37;
    v ^= v >> 4;
    v *= 4768777513237032717ul;
    v ^= v << 20;
    v ^= v >> 41;
    v ^= v << 5;
    return  v;
  }

 inline CUDA_CALLABLE_MEMBER
  uint64_t digest(uint64_t prev, uint64_t prev_bits,
		  uint64_t index, uint64_t index_bits,
		  uint64_t salt, uint64_t salt_bits,
		  uint64_t mod)
  {
    uint64_t prev_mask = (1ULL << prev_bits) - 1;
    uint64_t index_mask = (1ULL << index_bits) - 1;
    uint64_t salt_mask = (1ULL << salt_bits) - 1;
    uint64_t t =  ((((index&index_mask) << prev_bits) | (prev & prev_mask)) << salt_bits) | (salt&salt_mask);
    t = ranhash(t) % mod;
    return t;
  }

class Constraint {
public:
  char last[12]={0};
  uint8_t index=11;
  uint8_t run;
  uint8_t AT;
  uint8_t GC;
  CUDA_CALLABLE_MEMBER void next(char c){
    if ( last[index] == c ) run++;
    else run = 1;
    if ( c == 'A' || c == 'T')AT++;
    else GC++;
    index = (index+1)%12;  
    if (last[index] == 'A' || last[index] == 'T') AT--;
    // we need the check due to initialization
    else if (last[index] == 'C' || last[index] == 'G')GC--;
    last[index] = c;
  };
  CUDA_CALLABLE_MEMBER int get_gc_difference()const{return (int)AT-(int)GC;};
  CUDA_CALLABLE_MEMBER uint32_t get(int mod,char* choose) const
  {
    uint32_t num_choose = 4;
    choose[0]='A';choose[1]='C';choose[2]='G';choose[3]='T';
    if (mod > 2)return num_choose;
    else
    {
      if (run > 2)
      {
        switch (last[index])
        {
        case 'A':
          choose[0]='C';choose[1]='G';choose[2]='T';
          num_choose=3;
          break;
        case 'C':
          choose[0]='A';choose[1]='G';choose[2]='T';
          num_choose=3;
          break;
        case 'G':
          choose[0]='A';choose[1]='C';choose[2]='T';
          num_choose=3;
          break;
        case 'T':
          choose[0]='A';choose[1]='C';choose[2]='G';
          num_choose=3;
          break;
        }
      }
      int diff = get_gc_difference();
      if (num_choose == 4 && (diff>3 || diff<-3))
      {
        if (AT > GC)
        {
          choose[0]='C';choose[1]='G';
          num_choose=2;
        }
        else
        {
          choose[0]='A';choose[1]='T';
          num_choose=2;
        }
      }
      return num_choose;
    }
  }
  CUDA_CALLABLE_MEMBER int get_run(){return run;}
   Constraint(void* constraint_ptr){
    //install constraint from serial memory
    char* ptr = (char*)constraint_ptr;
    memcpy(last,ptr,12*sizeof(char));
    ptr+=12*sizeof(char);
    memcpy((char*)&run,ptr,sizeof(uint8_t));
    ptr+=sizeof(uint8_t);
    memcpy((char*)&AT,ptr,sizeof(uint8_t));
    ptr+=sizeof(uint8_t);
    memcpy((char*)&GC,ptr,sizeof(uint8_t));
  }
  Constraint(){}
};


class context
{
public:
  uint32_t prev_bits;
  uint32_t prev;
  uint32_t salt_bits;
  uint32_t salt;
  uint32_t index_bits;
  uint32_t index;
  uint32_t prev_mask;
  // uint32_t bits_accounted_for;
  Constraint constraint;
  context(){}
  context(std::vector<uint32_t> context_data, void* constraint_ptr)
  {
    prev_bits=context_data[0];prev=context_data[1];
    salt_bits=context_data[2]; salt=context_data[3];
    index_bits=context_data[4]; index=context_data[5];
    prev_mask=context_data[6];
    constraint = Constraint(constraint_ptr);
  }
  CUDA_CALLABLE_MEMBER uint32_t get_prev() { return this->prev_mask && this->prev; }
  CUDA_CALLABLE_MEMBER char getNextSymbol(int num_bits, int val)
  {
    uint32_t mod = 1;
    char choose[4];
    switch (num_bits)
    {
    case 0:
      assert(val == 0);
    case 1:
      mod = 2;
      break;
    case 2:
      mod = 4;
      break;
    default:
      break;
    }
    mod = constraint.get(mod,choose);
    uint64_t res = (digest(prev, prev_bits, index, index_bits, salt, salt_bits, mod) + val) % mod;
    return choose[res];
  }
  CUDA_CALLABLE_MEMBER char nextSymbolWithUpdate(int num_bits, uint32_t val, char base)
  {
    uint32_t mask = (num_bits == 2) ? 3 : ((num_bits == 0) ? 0 : 1);
    char c = getNextSymbol(num_bits, val);
    prev = ((prev << num_bits) | (val & mask));
    index++;
    constraint.next(c);
    return c;
  }
};
/**********************************************************************/


//TODO: change this to something like a bit matrix for more efficient cuda handling?
template <int N>
class cuda_bitset_t
{ // basic bitset class that can be used by CUDA

public:
  uint8_t _a[N];

  CUDA_CALLABLE_MEMBER cuda_bitset_t()
  { // clear bitset
    for (int i = 0; i < N; i++)
      _a[0] = 0;
  }
  CUDA_CALLABLE_MEMBER cuda_bitset_t &operator<<=(int n)
  {
    uint8_t mask = ~(0x01 << n);
    for (int i = N - 1; i >= 0; i--)
    {
      this->_a[i] = this->_a[i] << n;
      if (i > 0)
        this->_a[i] = this->_a[i] | (this->_a[i - 1] & mask);
    }
    return *this;
  }

  CUDA_CALLABLE_MEMBER cuda_bitset_t operator<<(int n)
  {
    cuda_bitset_t r;
    r <<= n;
    return r;
  }

  CUDA_CALLABLE_MEMBER cuda_bitset_t &operator|=(const cuda_bitset_t &rhs)
  {
    for (int i = 0; i < N; i++) this->_a[i] = this->_a[i] | rhs._a[i];
    return *this;
  }

  CUDA_CALLABLE_MEMBER cuda_bitset_t &operator|=(uint8_t b)
  { // simple fast call that just ORs at the bottom of the array
    this->_a[0] |= b;
    return *this;
  }


  CUDA_CALLABLE_MEMBER uint8_t operator[](uint32_t i)
  {
    // return bit at position i (0 is least sig bit, this is so to work similarly to bitset)
    uint32_t byte_index = i / sizeof(uint8_t);
    uint32_t bit_index = i % sizeof(uint8_t);
    return (this->_a[byte_index] >> bit_index) & 0x01;
  }


  CUDA_CALLABLE_MEMBER bool operator==(const cuda_bitset_t &c)
  {
    for(uint32_t i=0;i<N;i++){
      if(_a[i]!=c._a[i]) return false;
    }
    return true;
  }

  CUDA_CALLABLE_MEMBER bool operator!=(const cuda_bitset_t &c)
  {
    return !(*this==c);
  }

  CUDA_CALLABLE_MEMBER cuda_bitset_t(uint8_t b)
  {
    _a[0] = b; // just put the incoming byte at the bottom of the array, bits should simply match up
    for (int i = 1; i < N; i++)
      _a[i] = 0;
  }
};

template<int N>
CUDA_CALLABLE_MEMBER cuda_bitset_t<N> operator|(const cuda_bitset_t<N> &lhs, const cuda_bitset_t<N> &rhs)
  {
    cuda_bitset_t<N> r;
    for (int i = 0; i < N; i++)
      r[i] = lhs._a[i] | rhs._a[i];
    return r;
  }



typedef cuda_bitset_t<BITSET_SIZE> bitset_t;

// struct for storing information about previous state for a current state and
// the transition
struct prev_state_info_t
{
  uint32_t st_conv;
  uint32_t guess_value;
  prev_state_info_t(){}
};

struct pattern_info_t
{
  uint8_t pattern[16];    // assuming pattern is not longer than 16 values, likely very safe bet
  uint8_t pattern_length; // length of pattern
  pattern_info_t(std::vector<uint8_t> p)
  {
    assert(p.size() <= 16);
    pattern_length = p.size();
    for (int i = 0; i < pattern_length; i++)
      pattern[i] = p[i];
  }
  pattern_info_t(){}
};


extern __device__ __constant__  prev_state_info_t gpu_previous_states[MAX_NSTATE_CONV*TOTAL_PREVIOUS]; 
extern __device__ __constant__  pattern_info_t gpu_pattern_vector;

//struct used as SOA at end of algoritm
struct LVA_path_t
{
  bitset_t msg;         // KV:Note: Msg is now the DNA-message, not the binary message. helps analysis to be easier later
  float score_nonblank; // score for path ending with non-blank base
  float score_blank;    // score for path ending with non-blank base
  float score;          // logsumexp(score_nonblank,score_blank), used for sorting
  uint8_t last_base;    // for computing updated score_nonblank for stay transition
                        // and to compare with new_base in non-stay transitions
                        // for checking whether nonblank->nonblank makes sense
  void *context;        // for keeping track of contexts. TODO: this needs to be fixed so that CUDA can properly interact with it
  bool is_stay; // keeps track on whether path being considered is a stay path, allows avoiding context update
};



struct LVA_candidate_t_SOA;
// this is the main stucture to store the top paths for each state
struct LVA_path_t_SOA
{
  public:
    bitset_t *msg;         // KV:Note: Msg is now the DNA-message, not the binary message. helps analysis to be easier later
    float *score_nonblank; // score for path ending with non-blank base
    float *score_blank;    // score for path ending with non-blank base
    float *score;          // logsumexp(score_nonblank,score_blank), used for sorting
    // If score is -INF, score_nonblank and score_blank can be garbage
    uint8_t *last_base;    // for computing updated score_nonblank for stay transition
                          // and to compare with new_base in non-stay transitions
                          // for checking whether nonblank->nonblank makes sense
    context *hedge_context;        // for keeping track of contexts. TODO: this needs to be fixed so that CUDA can properly interact with it
    bool is_cpu; //track where data was allocated
    uint32_t size; //bookeep size of data allocated
    
    
    __host__ void compute_score_cpu(uint32_t i)
    {
      score[i] = logsumexpf(score_nonblank[i], score_blank[i]);
    }
    LVA_path_t_SOA(){}
    __host__ LVA_path_t_SOA(uint32_t n, bool is_host)
    {
      size=n;
      if (is_host == true)
      { // cpu
        msg = new bitset_t[n];
        score_blank = new float[n];
        score_nonblank = new float[n];
        score = new float[n];
        last_base = new uint8_t[n];
        hedge_context = new context[n];
        is_cpu = true;
        for(uint32_t i=0; i<n;i++){
          score_nonblank[i]=ZERO;score_blank[i]=ZERO;compute_score_cpu(i);
          }

      }
      else
      {
        is_cpu = false;
        cudaMalloc((void **)&msg, sizeof(bitset_t) * n);
        cudaMalloc((void **)&score_nonblank, sizeof(float) * n);
        cudaMalloc((void **)&score_blank, sizeof(float) * n);
        cudaMalloc((void **)&score, sizeof(float) * n);
        cudaMalloc((void **)&last_base, sizeof(uint8_t) * n);
        cudaMalloc((void **)&hedge_context, sizeof(context) * n);
      }
    }

    __host__ void free()
    {
      if (is_cpu)
      {
        delete[] msg;
        delete[] score_blank;
        delete[] score_nonblank;
        delete[] score;
        delete[] last_base;
        delete[] hedge_context;
      }
      else
      {
        cudaFree((void *)msg);
        cudaFree((void *)score_blank);
        cudaFree((void *)score_nonblank);
        cudaFree((void *)score);
        cudaFree((void *)last_base);
        cudaFree((void *)hedge_context);
      }
    }

    __host__ void to_AOS(LVA_path_t* AOS, uint32_t n){
      assert(size==n);
      for(uint32_t i=0; i<n;i++){
        AOS[i].msg=msg[i];
        AOS[i].score_blank=score_blank[i];
        AOS[i].score_nonblank=score_nonblank[i];
        AOS[i].score=score[i];
        AOS[i].last_base=last_base[i];
      }
    }

    __host__ void set_context(context prototype){
      for(int i=0;i<size;i++)hedge_context[i]=prototype;
    }

    // copies in a candidate's value and makes hedges context update
    CUDA_CALLABLE_MEMBER void copy_candidate(const LVA_candidate_t_SOA &candidate, uint8_t nbits, uint8_t msg_value,
                                             uint32_t dest_index,uint32_t candidate_index);
    
    // update score based on nonblank and blank score.
    // NOTE: this must be called externally
    __device__ void compute_score_gpu(uint32_t i)
    {
      score[i] = logsumexp2(score_blank[i], score_nonblank[i]);
    }
 
};


struct LVA_candidate_t_SOA:public LVA_path_t_SOA{//class to hold candidate information 
  public:
    bool* is_stay;
    LVA_candidate_t_SOA(uint32_t n,bool is_host):LVA_path_t_SOA(n,is_host){
      assert(is_host==false); //should be only for gpu
      cudaMalloc((void **)&msg, sizeof(bitset_t) * n);
    }
    LVA_candidate_t_SOA(){}
    __host__ void free(){
      cudaFree((void*)is_stay);
      LVA_candidate_t_SOA::free();
    }
    __device__ void assign(uint32_t pos, const bitset_t &msg_, const float &score_nonblank_,
                            const float &score_blank_, const uint8_t &last_base_, bool stay, const context& hedge_context_ )
    { //assign values to candidate memory
      msg[pos]=msg_;         
      score_nonblank[pos]=score_nonblank_; 
      score_blank[pos]=score_nonblank_;   
      last_base[pos]=last_base_;
      hedge_context[pos]=hedge_context_;    
      is_stay[pos]=stay;    
    }
    __device__ void assign(uint32_t pos,const float &score_nonblank_,
                        const float &score_blank_, const uint8_t &last_base_, bool stay, const context& hedge_context_ )
    { //assign values to candidate memory
      score_nonblank[pos]=score_nonblank_; 
      score_blank[pos]=score_nonblank_;   
      last_base[pos]=last_base_;
      hedge_context[pos]=hedge_context_;    
      is_stay[pos]=stay;    
    }
};

 // copies in a candidate's value and makes hedges context update
  inline CUDA_CALLABLE_MEMBER void LVA_path_t_SOA::copy_candidate(const LVA_candidate_t_SOA &candidate, uint8_t nbits, uint8_t msg_value,
                                             uint32_t dest_index,uint32_t candidate_index)
    {
      this->msg[dest_index] = candidate.msg[candidate_index];
      this->score_nonblank[dest_index] = candidate.score_nonblank[candidate_index];
      this->score_blank[dest_index] = candidate.score_blank[candidate_index];
      this->score[dest_index] = candidate.score[candidate_index];
      this->last_base[dest_index] = candidate.last_base[candidate_index];
      this->hedge_context[dest_index] = candidate.hedge_context[candidate_index];
      if (!candidate.is_stay) this->hedge_context[dest_index].nextSymbolWithUpdate(nbits,msg_value,0xff);
    }


//helper function to transfer SOAs
inline __host__ void transfer(LVA_path_t_SOA &dst, LVA_path_t_SOA &src, cudaMemcpyKind c)
{
  assert(dst.size == src.size);
  cudaMemcpy(dst.msg, src.msg, sizeof(bitset_t) * src.size, c);
  cudaMemcpy(dst.score_nonblank, src.score_nonblank, sizeof(float) * src.size, c);
  cudaMemcpy(dst.score_blank, src.score_blank, sizeof(float) * src.size, c);
  cudaMemcpy(dst.score, src.score, sizeof(float) * src.size, c);
  cudaMemcpy(dst.last_base, src.last_base, sizeof(uint8_t) * src.size, c);
  cudaMemcpy(dst.hedge_context, src.hedge_context, sizeof(void *) * src.size, c);
}

inline __host__ void host_to_dev(LVA_path_t_SOA &host, LVA_path_t_SOA &device)
{
  transfer(device, host, cudaMemcpyHostToDevice);
}

inline __host__ void host_to_dev(LVA_candidate_t_SOA &host, LVA_candidate_t_SOA &device)
{
  transfer((LVA_path_t_SOA&)device, (LVA_path_t_SOA&)host, cudaMemcpyHostToDevice);
  cudaMemcpy(device.is_stay, host.is_stay, sizeof(bool) * host.size, cudaMemcpyHostToDevice);
}

inline __host__ void dev_to_host(LVA_path_t_SOA &host, LVA_path_t_SOA &device)
{
  transfer(host, device, cudaMemcpyDeviceToHost);
}


struct kernel_values_t
{
  uint32_t nstate_conv;        // total number of convolutional states
  uint32_t nstate_pos;         // total number of position states
  uint32_t nlist_pos;          // total number of list positions
  float *post;                 // pointer to CTC data
  uint32_t post_T;             // size of post array time dim
  uint32_t post_NB;            // size of post array base dim
  LVA_path_t_SOA curr_best_paths; // SOA struct for current states
  LVA_path_t_SOA prev_best_paths; // SOA struct for previous states
  LVA_candidate_t_SOA candidate_paths; //SOA for candidates 
  uint32_t trellis_time;       // time point we are calculating in the trellis
  uint32_t start_pos;          // starting point for pos
  uint32_t end_pos;            // end point for pos
  bool rc_flag;                // reverse complement flag
  kernel_values_t(){}
};                             // struct for packaging kernel parameters so that signaturs don't need to keep being made

void call_beam_kernel_1(kernel_values_t &args);

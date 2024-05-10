#ifndef HEDGES_CPP
#define HEDGES_CPP
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <queue>
#include <memory>
#include <vector>
#include <list>
#include <limits>
#include <algorithm>

#define NULL_SCORE 1000000

namespace hedges {

  template <typename unit, typename container = std::list<unit>>
  class bit_tree {
  public:
    using container_type = container;
    std::shared_ptr<bit_tree> parent;
    unit val;  
    bit_tree(std::shared_ptr<bit_tree> _parent, unit _val):parent(_parent),val(_val){}

    container get_n(int n) {
      container c;
      bit_tree<unit,container> *tmp = this;
      while (tmp && n>0 ) {
	c.push_front(tmp->val);
	tmp = tmp->parent.get();
	n--;
      }
      return c;
    }

    bool check_for_val(int n, uint val)
    {
      bit_tree<unit,container> *tmp = this;    
      tmp = this;
      while (n>0) {
	if (tmp->val != val) {
	  return false;
	}
	tmp = tmp->parent.get();
	n--;
      }
      return true;
    }
  
    container get_prev_n(int n) {
      bit_tree<unit,container> *tmp = parent.get();
      if (tmp != nullptr) {
	return tmp->get_n(n);
      } else {
	container c;
	return c;	
      }
    }

    container get_all() {
      container c;
      bit_tree<unit,container> *tmp = this;
      while (tmp) {
	c.push_front(tmp->val);
	tmp = tmp->parent.get();
      }
      return c;
    }
  };

  enum class hedge_rate
    {
      codewords=0,
      three_fourths = 1,
      six_tenths = 2,
      one_half = 3,
      one_third = 4,     
      one_fourth = 5,
      one_fifth = 6,
      one_sixth = 7,
      one_eighth = 8     
    };

  class bitwrapper {
  public:
    std::vector<uint8_t> &bits;
    uint8_t operator[] (uint32_t index) const
    {
      if (index >= bits.size()*8)
	return 0; // this will handle the padding without error
    
      uint8_t b = bits[index/8];
      uint8_t m = index % 8;
      return (b >> m) & 1;
    }
  
    uint32_t get_bits(uint32_t lower, uint32_t upper ) const { //bits are returned in increasing order e.g. bit 0, 1 , ... First bits are the least sigbits of bytes
      uint32_t offset = 0;
      uint32_t val = 0;
      assert((upper-lower)<=32); //can only fit 32 bits into a single value right now
      for (uint32_t k = lower; k < upper; k++)
	{
	  val |= (operator[](k) << offset);
	  offset ++;
	}
      return val;
    }

    uint32_t get_num_bits_from_data(uint32_t lower,uint32_t upper) const{
      if(upper>=bits.size()*8 && lower<bits.size()*8){
	return bits.size()*8-lower; //lower in range, upper out of range
      }
      else if(upper<bits.size()*8 && lower<bits.size()*8){
	return upper-lower; //lower and upper in range
      }
      else{
	return 0; //neither is in range
      }
    }
    void set_bit(uint32_t index, uint32_t v)
    {
      //if (index/8 >= bits.size())
      //  bits.resize(index/8*2);

      uint8_t b = bits[index/8];
      uint8_t m = index % 8;
      uint8_t mask = ~(v << (m));
    
      bits[index/8] = ~mask | (b&mask);
    }
  
    bitwrapper(std::vector<uint8_t> &abits):bits(abits){}

  };

  uint64_t ranhash(uint64_t u);
  uint64_t digest(uint64_t prev, uint64_t prev_bits,
		  uint64_t index, uint64_t index_bits,
		  uint64_t salt, uint64_t salt_bits,
		  uint64_t mod);

  
class Constraint {
public:
  char last[12]={0};
  uint8_t index=11;
  uint8_t run;
  uint8_t AT;
  uint8_t GC;
  void next(char c);
  std::vector<char> get(int mod) const;
  int get_gc_difference(){return (int)AT-(int)GC;};
  int get_run(){return run;}
  Constraint():index(11),run(0),AT(0),GC(0){}
};

class Reward {
public:
  using hr = hedge_rate;
  hedge_rate rate;
  double _wild_card;
  double _run_cost;
  int _using_custom_reward;
  double getReward() {
    if(_using_custom_reward>0) return _wild_card;
    switch(rate) {
    case hr::three_fourths: return -0.035;
    case hr::six_tenths: return -0.082;
    case hr::one_half: return -0.127;
    case hr::one_third: return -0.229;
    case hr::one_fourth: return -0.265;
    case hr::one_fifth: return -0.300;
    case hr::one_sixth: return -0.324;
    case hr::one_eighth: return -.410;
    default: 
      return _wild_card; //allow for modification of reward for codewords
    }
  }

  double exp_cost_function(uint32_t length){
    if(length<=10) return _run_cost;
    else return 10000;
  }
  template<typename SearchTree>
  double getInsPenalty(SearchTree* p) {
    char previous=p->kind;
    uint32_t length = p->error_run;
    if(_run_cost==1 && !_using_custom_reward) return 1; //base case
    if(previous=='s') return NULL_SCORE;
    if(previous=='i') return exp_cost_function(length);
    if(previous=='d') return NULL_SCORE; 
    return 1.0;
  }
  template<typename SearchTree>
  double getDelPenalty(SearchTree* p) {
    char previous=p->kind;
    uint32_t length = p->error_run;
    if(_run_cost==1 && !_using_custom_reward) return 1;
    if(previous=='s') return NULL_SCORE;
    if(previous=='i') return NULL_SCORE;
    if(previous=='d') return exp_cost_function(length); 
    return 1.0;
  }
  template<typename SearchTree>
  double getSubPenalty(SearchTree* p) {
    char previous=p->kind;
    uint32_t length = p->error_run;
    if(_run_cost==1 && !_using_custom_reward) return 1;
    if(previous=='s') return exp_cost_function(length);
    if(previous=='i') return exp_cost_function(length);
    if(previous=='d') return exp_cost_function(length);
    return 1.0;
  }
  
  Reward(hr r=hr::one_half):rate(r),_wild_card(0){}
  Reward(hr r=hr::one_half,double wild_card=-0.31,double run_cost=1.0,int use_custom_reward=0):rate(r),_wild_card(wild_card),_run_cost(run_cost),
											       _using_custom_reward(use_custom_reward){}
};


  template<typename DNAConstraint = Constraint>
  class context {
  public:
    uint32_t prev_bits;
    uint32_t prev;
    uint32_t salt_bits;
    uint32_t salt;
    uint32_t index_bits;
    uint32_t index;
    uint32_t prev_mask;

    //uint32_t bits_accounted_for;
    DNAConstraint constraint;

    context(int _prev_bits, int _salt_bits)
      :prev_bits(_prev_bits),prev(0),salt_bits(_salt_bits),salt(0),index_bits(16),index(0)
    {
      this->prev_mask = (1ULL<<_prev_bits)-1;
    }

    uint32_t get_prev(){ return this->prev_mask&&this->prev;}

    void print(){
      std::cout << "Printing context"<<std::endl;
      std::cout<<"Prev Bits "<<this->prev_bits<<std::endl;
      std::cout<<"Salt Bits "<<this->salt_bits<<std::endl;
      std::cout<<"Index Bits "<<this->index_bits<<std::endl;
      std::cout<<"History State "<<std::hex<<this->prev<<std::endl;
      std::cout<<"Salt State "<<std::hex<<this->salt<<std::endl;
      std::cout<<"Index State "<<std::hex<<this->index<<std::endl;
    }
    
    
    char getNextSymbol(int num_bits, int val)
    {
      int mod = 1;
      switch(num_bits) {
      case 0: assert (val==0);
      case 1: mod = 2; break;
      case 2: mod = 4; break;
      default:
	assert(false && "should have been 0, 1 or 2"); break;
      }
      std::vector<char> choose = constraint.get(mod);
      mod = choose.size();
      uint64_t res = (digest(prev,prev_bits, index, index_bits, salt, salt_bits, mod) + val)%mod;
      //std::cout << "digest = " << res << " " << choose[res] << std::endl;
      return choose[res];
    }

    char nextSymbolWithUpdate(int num_bits, uint32_t val, char base)
    {
      uint32_t mask = (num_bits==2)?3:((num_bits==0)?0:1);
      char c = getNextSymbol(num_bits, val);
      // perhaps we don't need to mask here, redundant with digest
      prev = ((prev << num_bits) | (val&mask));   
      index++;
      constraint.next(c);
      //bits_accounted_for+=num_bits;
      return c;
    }  

    bool at_codeword_end(void){ return true;} //always at codeword end
  };



  
  //moved the guess infrastructure out so that different search trees can be used
  template<typename Reward=Reward, typename SearchTree>
  std::vector< SearchTree> makeGuesses(SearchTree* s);
  template<typename Reward=Reward, typename SearchTree>
  std::vector< SearchTree> makeNLengthGuesses(SearchTree* s, uint32_t guess_length);
  template<typename Reward = Reward, typename SearchTree>
  std::vector< SearchTree> makeCWGuesses(SearchTree* s);
  template<typename Reward = Reward, typename SearchTree>
  std::vector< SearchTree> make2bitGuesses(SearchTree* s);
  template<typename Reward = Reward, typename SearchTree>
  std::vector< SearchTree> make1bitGuesses(SearchTree* s);
  template<typename Reward = Reward, typename SearchTree>
  std::vector< SearchTree > make0bitGuesses(SearchTree* s);
  template<typename Reward = Reward, typename SearchTree>
  void guessHelper(SearchTree* s, std::vector<SearchTree>& ret, char c, int nbits, uint32_t val);  
  template<typename Reward=Reward, typename SearchTree>
  SearchTree addMatch(SearchTree* s, char c, uint8_t bits, uint32_t val);
  template<typename Reward=Reward, typename SearchTree>
  SearchTree addSubst(SearchTree* s, char c, uint8_t bits, uint32_t val);
  template<typename Reward=Reward, typename SearchTree>
  SearchTree addDel(SearchTree*s, char c, uint8_t bits, uint32_t val);
  template<typename Reward=Reward, typename SearchTree>
  SearchTree addIns2(SearchTree* s,char c, uint8_t bits, uint32_t val, double penalty);
  template<typename Reward=Reward, typename SearchTree>
  SearchTree addIns(SearchTree* s);


  
#define MAX_SCORE std::numeric_limits<float>::max()
 




  
  template <typename node>
  struct BestScore {
    bool operator() (const node &a, const node &b) {
      return a.score > b.score;
    }
  };

 
  struct decode_return_t{
    float score;
    uint32_t return_bytes;
    decode_return_t(float score,uint32_t ret_bytes):score(score),return_bytes(ret_bytes){}
    decode_return_t():score(0),return_bytes(0){}
  };



  template< typename Context = context<Constraint>>
  class search_tree;
  
  template<typename DNAConstraint = Constraint >
  class hedge {
  public:
    using hr = hedge_rate;
    context<DNAConstraint> encoding_context; //move encoding context inside class 
    double raw_rate;
    hedge_rate rate;
    int seq_bytes;
    int message_bytes;
    int pad_bits;
    int prev_bits;
    int salt_bits;

    int adj_message_bits;
    int adj_seq_bits;
    int adj_pad_bits;
    int codeword_sync_period;
    int parity_period;
    int parity_history;
    int parity_history_mask;
    int max_index;
    double wild_card_reward;

    int guess_length;
    double run_cost;
    int use_custom_reward;
    
    //TODO: add some fields for padding index ranges for convenience
    std::vector< std::vector<int> > patterns =
      {{0},
       {2,1},
       {2,1,1,1,1},
       {1},
       {1,1,0},
       {1,0},
       {1,0,1,0,0},
       {1,0,0},
       {1,0,0,0}     
      };
    std::vector< int > pattern_sum = {0,3,6,1,2,1,2,1,1};
    std::vector< int > pattern_length = {1,2,5,1,3,2,5,3,4};

    hedge_rate pick_rate(double r)  {
      std::vector<double> rates = { 3.0/4, 6.0/10, 1.0/2, 1.0/3, 1.0/4, 1.0/5, 1.0/6, 1.0/8 };
      std::vector<hedge_rate> hrate = {hr::three_fourths,
				       hr::six_tenths,
				       hr::one_half,
				       hr::one_third,
				       hr::one_fourth,
				       hr::one_fifth,
				       hr::one_sixth,
				       hr::one_eighth };

      int i=0;
      for (auto f : rates) {
	if ( std::abs(r - f) < 0.01 ) {
	  return hrate[i];
	}
	i++;
      }

      //Make a "rate" for codewords
      // patterns --> {x,x,x,y} where x is bits per codeword, and y=0 when using synchronzation points
      // pattern_sum --> {x}
      // pattern_length --> {sync_period+1}, sync_period is the number of codewords before a 0-cw synchronization point
      //this hack should let the rest of the methods to work like padding, index range, etc.
    
    
      pattern_sum[0] = r;
      patterns[0].clear();
      pattern_length[0]=1+this->codeword_sync_period; //allow sync points to exist within codeword strings
      for(int i=0; i<pattern_length[0]; i++){
	patterns[0].push_back(r); 
      }
      if(this->codeword_sync_period>0) patterns[0][pattern_length[0]-1]=0; //set the last point in the array as the sync point
      return hr::codewords;
    }

    int check_if_padding_needed(int bits)
    {
      int s = pattern_sum[(int)rate];    
      if (bits % s == 0) {
	return 0;
      } else {
	return s - bits%s;
      }
    }
    int get_index_range(int bits) 
    {
      int i = 0;
      int j = 0;
      uint8_t len = pattern_length[int(rate)];
      std::vector<int> pat = patterns[int(rate)];
      while (bits > 0)
	{
	  bits -= pat[j % len];
	  i += 1;
	  j += 1;
	}
      if (j % len != 0) {
	i += len - j % len;
      }    
      assert (bits == 0);
      return i;
    }

    uint8_t get_n_bits(uint32_t index) {
      int len = pattern_length[ (int) rate ];
      return patterns[(int)rate][index%len];
    }

    hedge_rate get_rate(){ return rate;}
  
    hedge(double rate,
	  int seq_bytes,
	  int message_bytes,
	  int pad_bits,
	  int prev_bits,
	  int salt_bits,
	  int codeword_sync_period,
	  int parity_period,
	  int parity_history,
	  double wild_card_reward,
	  int guess_length=1,
	  int use_custom_reward=0,
	  double run_cost = 1.0);

    std::string encode(std::vector<uint8_t> seqId, std::vector<uint8_t> message, int bytes_to_encode);


    template <typename Reward = Reward, template <typename> class Context = context,
	      template<typename> class SearchTree = search_tree>
    decode_return_t  decode(std::string &observed,
			    std::vector<uint8_t> &seqId,
			    std::vector<uint8_t> &message,
			    int max_guesses=100000);


    void print(bool extra=false);
  };



  template<typename Context>
  class search_tree {
  public:
    hedge<Constraint> *h;
    Context c;
    float score;
    uint32_t offset;
    uint16_t error_run;
    std::string *observed;
    char kind;
    char guess;

    std::shared_ptr<bit_tree<uint8_t>> bits;
    std::shared_ptr<bit_tree<char>>    bases;

    search_tree(hedge<Constraint> *_h,
		search_tree *_parent,
		const Context &_c,
		float _score,
		uint32_t _offset,
		std::string *_observed,
		int nbits, uint32_t bit, char base, bool insertion=false, char _kind='m');
    
    search_tree(hedge<Constraint>*_h,
		const Context &_c,
		float _score,
		uint32_t _offset,
		std::string *_observed)  :h(_h),c(_c),score(_score),offset(_offset), observed(_observed), bits(nullptr), bases(nullptr){}

    search_tree(hedge<Constraint> *_h,
		const Context &_c,
		float _score,
		uint32_t _offset,
		std::string *_observed, char _kind, char base)
      :h(_h),c(_c),score(_score),offset(_offset), observed(_observed), kind(_kind), guess(base),  bits(nullptr), bases(nullptr){}

  
  
    bool checkPad(); 
    bool isIncomplete();
    bool isWinner()
    {
      return !isIncomplete() && checkPad();
    }
    bool giveUp()
    {
      return false;
    }

    char observedAt(uint32_t i) {
      if (i >= observed->size())
	return 'A';
      else
	return (*observed)[i];
    }
  };
 
  
  template<typename Context = context<Constraint>>
  class search_tree_parity : public hedges::search_tree<Context>{
  public:
    //Inherit all the other bookeeping stuff from the base search tree
    search_tree_parity(hedge<Constraint> *_h,
		       search_tree_parity *_parent,
		       const Context &_c,
		       float _score,
		       uint32_t _offset,
		       std::string *_observed,
		       int nbits, uint32_t bit, char base, bool insertion=false, char _kind='m');

    search_tree_parity(hedge<Constraint> *_h,
		       const Context &_c,
		       float _score,
		       uint32_t _offset,
		       std::string *_observed):
      search_tree<Context>(_h,_c,_score,_offset,_observed)
    {}
    void _update_parity(uint8_t bit);

  protected:
    uint32_t _bit_counter = 0; //track which bit we are at, helps with knowing when to compare parity
    uint8_t _parity = 0; //parity bit for this part of the search tree
    uint64_t _history =0; //parity history
  };



template <typename node>
struct BestScoreSort {
  bool operator() (const node &a, const node &b) {
    return a.score < b.score;
  }
};
 


#ifndef NULL_VALUE
#define NULL_VALUE 0xffffffff
#endif

#include "fast_hedges.tpp"
  
}

#endif //HEDGES_CPP

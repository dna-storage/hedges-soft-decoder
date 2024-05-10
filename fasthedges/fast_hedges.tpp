
template<typename Context>
bool search_tree<Context>::isIncomplete()
{
  return c.index < h->max_index;
}


template<typename Context>
search_tree<Context>::search_tree(hedge<Constraint> *_h,
				  search_tree *parent,
				  const Context  &_c,
				  float _score,
				  uint32_t _offset,
				  std::string *_observed,
				  int nbits,
				  uint32_t bit,
				  char base,
				  bool insertion,
				  char _kind )
  :h(_h),c(_c),score(_score),offset(_offset), observed(_observed), kind(_kind), guess(base)
{
  if(_kind!='m') this->error_run=parent->error_run+1; //add on errors
  else if(_kind=='m') this->error_run=0; //start no error run
  if (!insertion) {    
    char tmp = c.nextSymbolWithUpdate(nbits, bit, base);
    assert (tmp == base);
    if (c.index == h->get_index_range(h->adj_seq_bits))
      {
	c.salt = c.prev;
	c.prev = 0;
      }
    if(nbits>0){
      std::shared_ptr<bit_tree<uint8_t>> p = parent->bits;
      for(int i=0; i<nbits; i++){
	bits=std::make_shared<bit_tree<uint8_t>>(p,(bit>>i)&0x01);
	p=bits;
      }
    }
    else{
      bits = parent->bits;
    }
    
    bases = std::make_shared<bit_tree<char>>(parent->bases,base);
  } else {
    // no update to corrected strand or message bits
    bases = parent->bases;
    bits = parent->bits;
    kind = 'i';
   }  
}


template <typename Context>
void search_tree_parity<Context>::_update_parity(uint8_t bit){
  if(this->h->parity_history==0){
    _parity^=bit; //not really using a moving history, so we just use complete history
  }
  else{
    //we need to take into account the parity_history bits in this calculation
    _parity = _parity ^ bit ^ (_history>>(this->h->parity_history-1)&1ULL);
  }

}

template<typename Context>
search_tree_parity<Context>::search_tree_parity(hedge<Constraint> *_h,
						search_tree_parity *parent,
						const Context &_c,
						float _score,
						uint32_t _offset,
						std::string *_observed,
						int nbits,
						uint32_t bit,
						char base,
						bool insertion,
						char _kind ) :
  search_tree<Context>(_h,_c,_score,_offset,_observed,_kind,base)
						   	  
{

  this->_bit_counter = parent->_bit_counter;
  this->_parity = parent->_parity;
  this->_history=parent->_history;
  if(_kind!='m') this->error_run=parent->error_run+1; //add on errors
  else if(_kind=='m') this->error_run=0; //start no error run
  if (!insertion) {    
    char tmp = this->c.nextSymbolWithUpdate(nbits, bit, base);
    assert (tmp == base);
    if (this->c.index == this->h->get_index_range(this->h->adj_seq_bits))
      {
	this->c.salt = this->c.prev;
	this->c.prev = 0;
      }
    if(nbits>0){
      //take into account the embedded parity bits to weed out wrong choices
      std::shared_ptr<bit_tree<uint8_t>> p = parent->bits;
      for(int i=0; i<nbits; i++){
	uint8_t bit_iter =(bit>>i)&0x01;
	if((_bit_counter)%this->h->parity_period==0 && _bit_counter!=0){
	  //found a parity bit
	  if(bit_iter!=_parity) this->score = MAX_SCORE; //parity bit does not match, this path must fail
	  
	  else this->_update_parity(bit_iter); //calculate the parity with new bit
	}
	else{
	  this->bits=std::make_shared<bit_tree<uint8_t>>(p,bit_iter);
	  p=this->bits;
	  this->_update_parity(bit_iter); //calculate updated parity with new bit
	}
	_bit_counter++; //increment along the bit counter
	_history=(_history<<1) | bit_iter; 
      }
    }
    else{
      this->bits = parent->bits;
    }
    
    this->bases = std::make_shared<bit_tree<char>>(parent->bases,base);
  } else {
    // no update to corrected strand or message bits
    this->bases = parent->bases;
    this->bits = parent->bits;
    this->kind = 'i';
  }  
}





template<typename Reward, typename SearchTree>
SearchTree
addMatch(SearchTree* s,char base, uint8_t nbits, uint32_t val)
{
  Reward r(s->h->rate,s->h->wild_card_reward,s->h->run_cost,s->h->use_custom_reward);  
  return SearchTree(s->h,    // use same decoder settings
		    s, // this node is parent node, needed for
		    // bit_trees
		    s->c,
		    s->score+r.getReward(), // reward for matching
		    s->offset+1, // move forward in string
		    s->observed, // unchanged
		    nbits,
		    val,
		    base,
		    false // not an insertion
		    );
}


template<typename Reward, typename SearchTree>
SearchTree
addSubst(SearchTree* s,char base, uint8_t nbits, uint32_t val)
{
  Reward r(s->h->rate,s->h->wild_card_reward,s->h->run_cost,s->h->use_custom_reward);
    
  return SearchTree(s->h,    // use same decoder settings
		    s, // this node is parent node, needed for bit_trees
		    s->c,
		    s->score+r.getSubPenalty(s), // reward for matching
		    s->offset+1, // move forward in string
		    s->observed, // unchanged
		    nbits,
		    val,
		    base,
		    false, // not an insertion
		    's'
		    );
}

template<typename Reward, typename SearchTree>
SearchTree
addDel(SearchTree* s,char base, uint8_t nbits, uint32_t val)
{
  
  Reward r(s->h->rate,s->h->wild_card_reward,s->h->run_cost,s->h->use_custom_reward);
  
  return SearchTree(s->h,    // use same decoder settings
		    s, // this node is parent node, needed for bit_trees
		    s->c,
		    s->score+r.getDelPenalty(s), // penalty
		    s->offset,   // deletion, so offset stays fixed
		    s->observed, // unchanged
		    nbits,
		    val,
		    base,
		    false, // not an insertion
		    'd'
		    );
}


template<typename Reward, typename SearchTree>
SearchTree
addIns(SearchTree* s)
{
  Reward r(s->h->rate,s->h->wild_card_reward,s->h->run_cost,s->h->use_custom_reward);  
  return SearchTree(s->h,    // use same decoder settings
		    s, // this node is parent node, needed for bit_trees
		    s->c,
		    s->score+r.getInsPenalty(s), // penalty
		    s->offset+1, // insertion, move down 1 in array
		    s->observed, // unchanged
		    0,
		    0,
		    '_',
		    true, // not an insertion
		    'i'
		    );
}

template<typename Reward, typename SearchTree>
SearchTree
addIns2(SearchTree* s,char base, uint8_t nbits, uint32_t val, double penalty)
{
  Reward r(s->h->rate,s->h->wild_card_reward,s->h->run_cost,s->h->use_custom_reward);
  return SearchTree(s->h,    // use same decoder settings
		    s, // this node is parent node, needed for
		    // bit_trees
		    s->c,
		    // penalty + reward for matching			
		    s->score+penalty,
		    s->offset+2,   // insertion, jump 2 indices
		    s->observed,   // unchanged
		    nbits,
		    val,
		    base,
		    false, // not an insertion
		    'I'
		    );
}


template<typename Reward, typename SearchTree>
std::vector<SearchTree> make0bitGuesses(SearchTree* s)
{
  char c = s->c.getNextSymbol(0, 0);
  std::vector<SearchTree> ret;
  bool inserted = false;
  Reward r(s->h->rate,s->h->wild_card_reward,s->h->run_cost,s->h->use_custom_reward);
  guessHelper<Reward>(s,ret, c, 0, 0);
  //guess insertion
  if (r.getInsPenalty(s)!=NULL_SCORE) ret.push_back(addIns<Reward>(s));
  return ret;  
}



template<typename Reward, typename SearchTree>
void guessHelper(SearchTree* s, std::vector<SearchTree> &ret,
		 char c,
		 int nbits,
		 uint32_t val)
{
  Reward r(s->h->rate,s->h->wild_card_reward,s->h->run_cost,s->h->use_custom_reward);
  
  if ( c == s->observedAt(s->offset) )
    { 
      ret.push_back( addMatch<Reward>(s,c,nbits,val) );
    }
  else if(r.getSubPenalty(s)!=NULL_SCORE)
    {
      // guess substitution
      ret.push_back( addSubst<Reward>(s,c,nbits,val) );
    }
  if (r.getDelPenalty(s)!=NULL_SCORE) ret.push_back(addDel<Reward>(s,c,nbits,val) );  // guess deletion
  
}


template<typename Reward, typename SearchTree>
std::vector<SearchTree> make1bitGuesses(SearchTree* s)		 
{
  char c0 = s->c.getNextSymbol(1, 0);
  char c1 = s->c.getNextSymbol(1, 1);
  std::vector<SearchTree> ret;
  bool inserted = false;
  Reward r(s->h->rate,s->h->wild_card_reward,s->h->run_cost,s->h->use_custom_reward);
  guessHelper<Reward>(s,ret, c0, 1, 0);
  guessHelper<Reward>(s,ret, c1, 1, 1);
  //guess insertion
  if (r.getInsPenalty(s)!=NULL_SCORE) ret.push_back(addIns<Reward>(s));
  return ret;  
}

template<typename Reward, typename SearchTree>
std::vector<SearchTree> makeCWGuesses(SearchTree* s){
  std::vector<SearchTree> ret;
  char next_guess;
  uint32_t val = NULL_VALUE;
  //this bits value identifies how much the current codeword counts towards the number of stored bits
  //we need to check if we are transitioning from a complete codeword
  uint32_t bits_next_state = s->c.at_codeword_end() ? s->h->get_n_bits(s->c.index+1) : s->h->get_n_bits(s->c.index);
  uint32_t bits_this_state = s->h->get_n_bits(s->c.index);
  Reward r(s->h->rate,s->h->wild_card_reward,s->h->run_cost,s->h->use_custom_reward);
  while((next_guess = s->c.getNextSymbol(bits_next_state,val))!=0){
    uint32_t bits = bits_this_state;
    //do something with this guess
    if(val == NULL_VALUE) bits = 0; //nullify the number of bits, we haven't reached a codeword yet
    guessHelper<Reward>(s,ret,next_guess,bits,val);
    //guess insertion
    if(r.getInsPenalty(s)!=NULL_SCORE) ret.push_back(addIns<Reward>(s));
  }
  return ret;
}



// SearchTree - parent node
// guess_length - length (in bases) of the guesses to make, guess_length=1 is the basic case of adding a guess that is just the next base
template<typename Reward, typename SearchTree>
std::vector<SearchTree> makeNLengthGuesses(SearchTree* s,uint32_t guess_length){

  std::vector<SearchTree> ret; //this is going to be the return list
  std::vector<SearchTree> q; //queue of nodes we want to add on
  q.push_back(*s);
  for (int index=0; index<guess_length; index++){
    std::vector<SearchTree> tmp_q;
    while(q.size()>0){
      SearchTree x = q.back();
      q.pop_back();
      uint8_t nbits = x.h->get_n_bits(x.c.index);
      std::vector<SearchTree> tmp;
      if ( nbits == 0 ) {
	tmp=make0bitGuesses<Reward>(&x);    
      } else if ( nbits == 1 ) {
	tmp=make1bitGuesses<Reward>(&x);
      } else if ( nbits == 2 ) {
	tmp=make2bitGuesses<Reward>(&x);
      } else {
	assert (false && "invalid case");
      }
      for( auto t : tmp){
	ret.push_back(t);
	tmp_q.push_back(t);
      }
    }
    q=tmp_q;
  }
  return ret;
}


template<typename Reward, typename SearchTree>
std::vector<SearchTree> makeGuesses(SearchTree* s)
{
  hedges::hedge_rate hedge_type = s->h->get_rate();
  if(hedge_type==hedges::hedge_rate::codewords){
    //base guesses for codewords
    return makeCWGuesses<Reward>(s);
  }
  else{
    return makeNLengthGuesses(s,s->h->guess_length);
  }
}

template<typename Context>
bool search_tree<Context>::checkPad()
{
  return bits.get()->check_for_val(h->adj_pad_bits,0);
}



template<template <typename> class Context = context>
std::ostream& operator << (std::ostream& os, Context<Constraint> &c)
{
  os << std::hex;
  os << c.prev << " " << c.salt << " " << c.index ;
  return os;
}

template<typename DNAConstraint = Constraint, typename Reward = Reward, template <typename> class Context=context>
std::ostream & operator << (std::ostream &o, search_tree<Context<DNAConstraint>> &node)
{
  o << "i:" << std::dec << node.c.index << " " << "score:" << node.score;

  if (node.offset < node.observed->size())
    o << " obs[" << node.offset << "]=" << (*node.observed)[node.offset] ;
  else
    o << " [X]";

  o  << " " << node.kind << ":" << node.guess ;

  o << " context: " << node.c << " ";
  
  o << " bits: ";
  
  std::list<uint8_t> res2 = node.bits.get()->get_n(16);
  for(auto r : res2)
    std::cout << int(r) ;

  o << " bases: ";
  std::list<char> res3 = node.bases.get()->get_all();
  for(auto r : res3)
    std::cout << r ;

  std::cout << std::endl;
  
  return o;
}


template<typename Reward, typename SearchTree>
std::vector<SearchTree>make2bitGuesses(SearchTree*s)
{
  char c0 = s->c.getNextSymbol(2, 0);
  char c1 = s->c.getNextSymbol(2, 1);
  char c2 = s->c.getNextSymbol(2, 2);
  char c3 = s->c.getNextSymbol(2, 3);

  std::vector<SearchTree> ret; 
  Reward r(s->h->rate,s->h->wild_card_reward,s->h->run_cost,s->h->use_custom_reward);
  guessHelper<Reward>(s,ret, c0, 2, 0);
  guessHelper<Reward>(s,ret, c1, 2, 1);
  guessHelper<Reward>(s,ret, c2, 2, 2);
  guessHelper<Reward>(s,ret, c3, 2, 3);
  //guess insertion
  if(r.getInsPenalty(s)!=NULL_SCORE) ret.push_back(addIns<Reward>(s));
  return ret;  
}

template <typename DNAConstraint>
template < typename Reward, template <typename> class Context,
	   template <typename> class SearchTree>
hedges::decode_return_t hedge<DNAConstraint>::decode(std::string &observed,
						     std::vector<uint8_t> &seqId,
						     std::vector<uint8_t> &message,
						     int max_guesses)
{
  using node = SearchTree<Context<DNAConstraint>>;
  
  std::priority_queue<node,std::vector<node>,BestScore<node> > heap;

  while( observed.size() < max_index )
    observed.append("A");
  
  Context<DNAConstraint> state(prev_bits,salt_bits);
  state.salt = 0xA5A5A5A5;    
  heap.push(node(this,state,1000,0,&observed));
    
  while( !heap.empty() && max_guesses > 0 )
    {
      node tmp = heap.top();
      if ( tmp.isWinner() )
	{
	  break;	   
	}

      // pop so that new guesses don't rise to the top of the queue
      heap.pop();

      
      if ( tmp.isIncomplete() )
	{
	  std::vector<node> guesses = makeGuesses<Reward>(&tmp);
	  for(auto n : guesses) {
	    heap.push(n);
	  }
	}

      max_guesses--;
    }


  // may either find a winner or run out of guesses.
  // in either case, collect the result
  node best = heap.top();
  bool result = best.isWinner();

  decode_return_t ret(best.score,0);
  
  // collect seqId and message
  std::list<uint8_t> res2 = best.bits.get()->get_all();
  bitwrapper seq(seqId);
  bitwrapper mess(message);

  if ( res2.size() == (message_bytes + seq_bytes)*8 + pad_bits )
    ret.return_bytes = message_bytes + seq_bytes;
  else if (res2.size() > (message_bytes + seq_bytes)*8)
    ret.return_bytes = message_bytes + seq_bytes;
  else
    // Since we only take whole bytes, this result needs to be truncated to
    // the nearest byte. The last partial byte would be invalid.
    ret.return_bytes = res2.size() / 8;
  
  // Extract message and seqId
  int k=0;
  std::list<uint8_t>::iterator b = res2.begin();
  while(b != res2.end() && adj_seq_bits>0){
    if (k<seq_bytes*8)
      seq.set_bit(k,*b);
    k++;
    b++;
    if (k==adj_seq_bits)
      break;
  }
  
  k = 0;
  while(b != res2.end()) {    
    if (k<message_bytes*8)
      mess.set_bit(k,*b);
    else
      break;
    k++;
    b++;
  }
  
  return ret;
}


template<typename DNAConstraint>  
hedge<DNAConstraint>::hedge(double rate,
			    int seq_bytes,
			    int message_bytes,
			    int pad_bits,
			    int prev_bits,
			    int salt_bits,
			    int codeword_sync_period,
			    int parity_period,
			    int parity_history,
			    double wild_card_reward,
			    int guess_length,
			    int use_custom_reward,
			    double run_cost
			    ):raw_rate(rate),codeword_sync_period(codeword_sync_period),seq_bytes(seq_bytes),
			      message_bytes(message_bytes),pad_bits(pad_bits),prev_bits(prev_bits),
			      salt_bits(salt_bits),parity_period(parity_period),parity_history(parity_history),
			      wild_card_reward(wild_card_reward),encoding_context(prev_bits,salt_bits)
{
  this->guess_length=guess_length;
  this->run_cost = run_cost;
  this->use_custom_reward=use_custom_reward;

  this->rate = pick_rate(rate);  
  this->parity_history_mask =((1ULL)<<parity_history)-1; //generate mask for parity history, used by search tree nodes

  //Take into account parity when determining adjusted message bits so that the index can be correctly calcualted
  uint32_t total_parameter_data_bits = message_bytes*8+pad_bits;
  if(this->parity_period>0){
    assert(this->parity_period>1 && "Parity can't be set to a period of just 1");
    //parity is interleaved into pad_bits speced by user, so that needs to be taken into account in the total bits
    total_parameter_data_bits = (this->parity_period+1) + (total_parameter_data_bits-this->parity_period)/(this->parity_period-1) + total_parameter_data_bits;   
  }
  
  this->adj_seq_bits = seq_bytes*8 + check_if_padding_needed(seq_bytes*8);
  this->adj_pad_bits = this->pad_bits + check_if_padding_needed(total_parameter_data_bits);
  this->adj_message_bits = total_parameter_data_bits + (this->adj_pad_bits-this->pad_bits);
  this->max_index = get_index_range(adj_message_bits + adj_seq_bits);
  
}


template<typename DNAConstraint>
void hedge<DNAConstraint>::print(bool extra)
{
  std::cout << "-------- hedges --------" << std::endl;
  std::cout << "rate:               " << this->raw_rate << std::endl;
  std::cout << "rate id:            " << int(this->rate) << std::endl;
  std::cout << "seq_bytes:          " << this->seq_bytes << std::endl;
  std::cout << "message_bytes:      " << this->message_bytes << std::endl;
  std::cout << "salt bits:          " << this->salt_bits << std::endl;
  std::cout << "prev bits:          " << this->prev_bits << std::endl;
  std::cout << "index range:        " << 0 << " to " << max_index << std::endl;

  if (extra)
    {
      std::cout << "Extra details: " << std::endl;
      std::cout << "adj message bits:    " << this->adj_message_bits << std::endl;
      std::cout << "adj seq bits:        " << this->adj_seq_bits << std::endl;
      std::cout << "adj pad bits:        " << this->adj_pad_bits << std::endl;
    }
}



template<typename DNAConstraint>
std::string hedge<DNAConstraint>::encode(std::vector<uint8_t> seqId, std::vector<uint8_t> message,int bytes_to_encode)
{
  bitwrapper seq(seqId);    
  bitwrapper mess(message);
  std::string buff;
  
  std::vector<int> pat = patterns[ (int) rate ];
  int len = pattern_length[ (int) rate ];
  
  uint32_t bit = 0;
  uint32_t total_data_bits=0;
  uint32_t index = 0;
  
  context<Constraint>& state = this->encoding_context;
  state=context<Constraint>(this->prev_bits,this->salt_bits); //clear out context before encoding for safety
  // initial static salt; not really required
  // could be good to base this on the primer
  // or file id
  state.salt = 0xA5A5A5A5;

  std::cout << std::hex;
  
  // Encode seqId
  while(index < get_index_range(adj_seq_bits)) {
    int nbits = pat[index%len];
    int val = seq.get_bits(bit, bit+nbits);
    total_data_bits += seq.get_num_bits_from_data(bit,bit+nbits);
    char c = state.nextSymbolWithUpdate(nbits,val,0);
    buff.push_back(c);
    if(bytes_to_encode!=-1 && bytes_to_encode<total_data_bits/8) return buff;
    index ++;
    bit += nbits;
  }

  //std::cout << "after seq: " << std::dec << index << ": " << std::hex << state << std::endl;  
    
  // Copy prev over to be the salt
  state.salt = state.prev;
  // ? Should we set prev to 0 or not? Since we have the salt now, we don't need the prev.
  //   Probably doesn't make much of a difference since it's just redundant information.
  state.prev = 0;
  bit = 0;

  if(bytes_to_encode!=-1 && bytes_to_encode<=total_data_bits/8) return buff;
  // Encode message
  while(index < max_index) {
    int nbits = pat[index%len];
    int val = mess.get_bits(bit, bit+nbits);
    total_data_bits += mess.get_num_bits_from_data(bit,bit+nbits);
    char c = state.nextSymbolWithUpdate(nbits,val,0);
    buff.push_back(c);
    if(bytes_to_encode!=-1 && bytes_to_encode==total_data_bits/8) return buff;
    index++;
    bit += nbits;
  }

  std::cout << std::dec;
  
  return buff;
}

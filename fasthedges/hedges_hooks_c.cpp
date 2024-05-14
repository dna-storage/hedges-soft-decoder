#include "hedges_hooks_c.h"
#include "fast_hedges.hpp"
#include "shared_hedges.hpp"
#include <cstdlib>

const std::map<int,int> mod_map = {{287, 0},{222,1},{220,2},{216,3},{203,4},{138,5},{149,6}}; //maps mods to linear index


void update_context__c(void* c1, void* c2, int nbits, int value)
{
  context<Constraint>* c = (context<Constraint>*)c1;
  context<Constraint>* c2_ = (context<Constraint>*)c2;
  *c=*c2_; //copy in the context, makes it so we don't need to remake contexts constantly
  c->nextSymbolWithUpdate(nbits,value,(char)0xFF);
}

void copy_context_no_update__c(void* c1, void* c2){
  context<Constraint>* c = (context<Constraint>*)c1;
  context<Constraint>* c2_ = (context<Constraint>*)c2;
  *c=*c2_; //copy state of other context
}



char peek_context__c(void* c1, int nbits, int value)
{
    context<Constraint>* c = (context<Constraint>*)c1;
    return c->getNextSymbol(nbits,value); 
}

char peek_context__c(void* c1, int nbits, int value, int range)
{ 
    //This function assumes that the pattern is som {X,0,0,0,...} where value == X, and range indicates number of trailing zeros
    context<Constraint> tmp_c = context<Constraint>(*(context<Constraint>*)c1);
    char x = tmp_c.nextSymbolWithUpdate(nbits,value,(char)0xff);
    for(int i=0; i<range;i++) x=tmp_c.nextSymbolWithUpdate(0,0,(char)0xff);
    return x;
}

void* make_hedge__c(PyObject* h){
  hedge<Constraint>* h_new = new hedge<Constraint>(make_hedge_from_pyobject(h));
  return (void*)h_new;   
}


void* make_context__c(void* h1){
  hedge<Constraint>* h = (hedge<Constraint>*)h1;
  context<Constraint>* c  = new context<Constraint>(h->encoding_context);
  return (void*)c;
}


std::vector<uint8_t> get_pattern__c(void* h1){
  hedge<Constraint>* h = (hedge<Constraint>*)h1;
  hedge_rate r = h->get_rate();
  std::vector<int> pattern = h->patterns[(int)r];
  std::vector<uint8_t> return_vect;
  for(auto v: pattern) return_vect.push_back((uint8_t)v);
  return return_vect;
}

uint32_t get_nbits__c(void* h,uint32_t index){
  hedge<Constraint>* h1 = (hedge<Constraint>*)h;
  uint32_t nbits = h1->get_n_bits(index);
  return nbits;
}

uint32_t get_index__c(void* c){
  context<Constraint>* c1 = (context<Constraint>*) c;
  return c1->index;
}


std::vector<uint32_t> get_context_data__c(void* h){ //extract state of context
  hedge<Constraint>* h1 = (hedge<Constraint>*)h;
  std::vector<uint32_t> ret;
  context<Constraint>c = h1->encoding_context; 
  ret.push_back(c.prev_bits); 
  ret.push_back(c.prev);
  ret.push_back(c.salt_bits);
  ret.push_back(c.salt);
  ret.push_back(c.index_bits);
  ret.push_back(c.index);
  ret.push_back(c.prev_mask);
  return ret;
}

void get_constraint_data__c(void* h,void** m){ //extract state constraint
  hedge<Constraint>* h1 = (hedge<Constraint>*)h;
  std::vector<uint32_t> ret;
  Constraint c = h1->encoding_context.constraint; 
  //serialize out constraint structure 
  *m = malloc(sizeof(char)*12+sizeof(uint8_t)*4);
  char* copy_ptr = (char*)*m;
  memcpy((void*)copy_ptr,(void*)c.last,sizeof(char)*12);
  copy_ptr+=(12*sizeof(char));
  memcpy((void*)copy_ptr,(void*)&c.index,sizeof(uint8_t));
  copy_ptr+=(sizeof(uint8_t));
  memcpy((void*)copy_ptr,(void*)&c.run,sizeof(uint8_t));
  copy_ptr+=(sizeof(uint8_t));
  memcpy((void*)copy_ptr,(void*)&c.AT,sizeof(uint8_t));
  copy_ptr+=(sizeof(uint8_t));
  memcpy((void*)copy_ptr,(void*)&c.GC,sizeof(uint8_t));
  copy_ptr+=(sizeof(uint8_t));
  memcpy((void*)copy_ptr,(void*)&c.index,sizeof(char));
  copy_ptr+=(sizeof(uint8_t));
}







mod_struct_t get_valid_mods(void* c, int nbits,int* next_states_buffer,int* next_mods_buffer){
  context<Constraint>* current_context = (context<Constraint>*)c;
  int context_history = current_context->prev&current_context->prev_mask;
  int total_transitions = 1ULL<<nbits;
  int value_shifted_out = context_history>>(current_context->prev_bits-nbits);
  mod_struct_t mod_info;
  mod_info.val_to_next=value_shifted_out;
  mod_info.next_states = next_states_buffer;
  mod_info.next_mods = next_mods_buffer;
  mod_info.num_valid_mods = total_transitions;
  for(int i=0; i<total_transitions; i++){
    context<Constraint> tmp_context = *current_context;
    tmp_context.nextSymbolWithUpdate(nbits,i,'A');
    mod_info.next_states[i] = tmp_context.prev&tmp_context.prev_mask;
    int sum=0;
    for(auto& j: tmp_context.constraint.get(nbits)) sum+=(int)j;
    mod_info.next_mods[i]=mod_map.at(sum);
  }
  return mod_info;
}


//Thomas Wang Hash function
int hash6432shift(int64_t key)
{
  key = (~key) + (key << 18); // key = (key << 18) - key - 1;
  key = key ^ (key >> 31);
  key = key * 21; // key = (key + (key << 2)) + (key << 4);
  key = key ^ (key >> 11);
  key = key + (key << 6);
  key = key ^ (key >> 22);
  return (int) key;
}


mod_struct_t get_valid_mods_hash(void* c, int nbits,int* next_states_buffer,int* next_mods_buffer){
  context<Constraint>* current_context = (context<Constraint>*)c;
  int context_history = current_context->prev&current_context->prev_mask;
  int total_transitions = 1ULL<<nbits;
  int value_shifted_out = context_history>>(current_context->prev_bits-nbits);
  mod_struct_t mod_info;
  mod_info.val_to_next=value_shifted_out;
  mod_info.next_states = next_states_buffer;
  mod_info.next_mods = next_mods_buffer;
  mod_info.num_valid_mods = total_transitions;
  for(int i=0; i<total_transitions; i++){
    context<Constraint> tmp_context = *current_context;
    tmp_context.nextSymbolWithUpdate(nbits,i,'A');
    mod_info.next_states[i] = tmp_context.prev&tmp_context.prev_mask;
    int64_t run_balance = (((int64_t)tmp_context.constraint.get_run())<<32)|(((int64_t)tmp_context.constraint.get_gc_difference())&((((int64_t) 1)<<32)-1));
    int hash = hash6432shift(run_balance);
    mod_info.next_mods[i]=hash;
  }
  return mod_info;
}

previous_states_t get_previous_states__c(void* h, uint32_t nbits, uint32_t current_state)
{
  previous_states_t return_states;
  hedge<Constraint>* c = (hedge<Constraint>*)h;
  return_states.states.resize(1ULL<<nbits);
  return_states.values.resize(1ULL<<nbits);
  if(nbits==0){
    assert(return_states.states.size()==1);
    return_states.states[0] = current_state;
    return_states.values[0] = 0;
    return_states.guess_value=0;
  }
  else{ //nbits tells how many bits we shifted to get to current state, possible states that could have preceeded this is determined by reversing the shift
    uint32_t number_previous_states = 1ULL<<nbits;
    return_states.guess_value=((1ULL<<nbits)-1)&current_state;
    for(uint32_t i=0;i<number_previous_states;i++){
      uint32_t mask = (nbits==2)?3:((nbits==0)?0:1);
      uint32_t previous_state = (current_state>>nbits)|((i&mask)<<(c->prev_bits-nbits));
      return_states.states[i]=previous_state;
      return_states.values[i]=i;
    }
  }
  return return_states;
}

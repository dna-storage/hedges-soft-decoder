/*
Filename: codeword_hedges.hpp
Description: Support for hedges to allow for codeword based implementations. Support for codeword+assembly will also be included at some point.
Author: Kevin Volkel
*/

#ifndef CW_HEDGES_HPP
#define CW_HEDGES_HPP

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <string>
#include <map>
#include "fast_hedges.hpp"


namespace codeword_hedges{
  class DNAtrie;
  extern std::map<std::string,codeword_hedges::DNAtrie*> CodebookMap; //global codebook map, easiest thing I can think of right now
  extern std::vector<std::string> SyncBook; //global syncbook, holds a list of strings, each string should be considered in order as sync points

  //declare some utility functions
  uint8_t convert_base_to_bits(const char& base);
  char convert_bits_to_base(const uint8_t& base);
  void print_bitwrapper_as_string(uint32_t length, const hedges::bitwrapper& DNA);
  std::vector<uint8_t> convert_DNA_to_bytes(const std::string& DNA);


  
#define MAX_BASES 4
#ifndef NULL_VALUE
#define NULL_VALUE 0xffffffff
#endif

  //DNAtrie holds a tree of DNA strings as a compressed bit representation, necessary for tracking guesses along the tree for decoding 
  class DNAtrie{
  public:
    DNAtrie(std::vector<uint8_t> dna,uint8_t dna_length):_dna(dna),_prefix_length(dna_length){ //initialize node
      for(int i=0; i<MAX_BASES; i++) children[i] = nullptr;
    }
    DNAtrie():_prefix_length(0){ //initialize empty node
      for(int i=0; i<MAX_BASES; i++) children[i] = nullptr;
    }

    void print(){ //print out the tree
      assert(this->is_root()); //should start prints only from the root
      std::cout<<"PRINTING TREE"<<std::endl;
      std::vector<DNAtrie*> nodes;
      nodes.push_back(this);
      while(nodes.size()>0){
	DNAtrie* current_node = nodes.back();
	std::cout<<"NODE "<< std::hex << current_node<<std::endl;
	hedges::bitwrapper current_bitwrapper(current_node->_dna);
	print_bitwrapper_as_string(current_node->_prefix_length,current_bitwrapper);
	nodes.pop_back();
	for(int i=0; i<MAX_BASES; i++){
	  if(current_node->children[i].get()!=NULL){
	    std::cout<<"CHILD "<< std::hex << current_node->children[i].get()<<" base for direction " <<std::hex<< i << std::endl;
	    nodes.push_back(current_node->children[i].get());
	  }
	}
	if(current_node->is_leaf()) std::cout<<"IS LEAF, VALUE: "<<current_node->_value<<std::endl;
      }
    }

    //function to add simple nodes to the end of a leaf, really should only be used initially upon the root
    void _add_leaf_node(DNAtrie* parent,uint32_t total_indexes_covered, const std::string& DNA, const hedges::bitwrapper& DNA_bits, uint32_t value){
      //set up split node from DNA
      uint32_t DNA_base = DNA_bits.get_bits((total_indexes_covered)*2,(total_indexes_covered)*2+2);
      uint32_t bases_remaining_DNA = DNA.size()-total_indexes_covered;
      uint32_t DNA_pad_bits  = (bases_remaining_DNA*2)%8==0 ? 0 :  (8 - (bases_remaining_DNA*2)%8);
      std::vector<uint8_t> new_DNA_strand((bases_remaining_DNA*2+DNA_pad_bits)/8);
      hedges::bitwrapper new_DNA_bits(new_DNA_strand);
      //now make the new string for the input string
      for(int j=total_indexes_covered; j<DNA.size(); j++){
	uint32_t new_DNA_base = DNA_bits.get_bits(j*2,j*2+2);
	new_DNA_bits.set_bit((j-total_indexes_covered)*2,(new_DNA_base&0x01));
	new_DNA_bits.set_bit((j-total_indexes_covered)*2+1,(new_DNA_base)>>1 &0x01);
      }
      std::shared_ptr<DNAtrie> DNA_node = std::make_shared<DNAtrie>(new_DNA_strand,bases_remaining_DNA);
      DNA_node->_value = value;

      assert(parent->get_child(DNA_base)==NULL); //this should be Null, else there was an error where we didn't go down the tree
      parent->children[DNA_base] = DNA_node; //add the new node to the parent
    }

    //function to handle node splits 
    void _split_node(uint32_t total_indexes_covered, const std::string& DNA, uint32_t split_index, DNAtrie* current_node,
		      const hedges::bitwrapper& DNA_bits, const hedges::bitwrapper& node_bits,uint32_t value){
      uint32_t node_base = node_bits.get_bits(split_index*2,split_index*2+2);
      uint32_t DNA_base = DNA_bits.get_bits((total_indexes_covered)*2,(total_indexes_covered)*2+2);
      //set up split node from current ndoe
      uint32_t bases_remaining_node = current_node->_prefix_length-split_index;
      uint32_t node_pad_bits = (bases_remaining_node*2)%8==0 ? 0 : (8 - (bases_remaining_node*2)%8);
      std::vector<uint8_t> new_node_strand((bases_remaining_node*2+node_pad_bits)/8);
      hedges::bitwrapper new_node_bits(new_node_strand);
      //we need to split this node at position i
      for(int j=split_index; j<current_node->_prefix_length; j++){ //first make new DNA string for the new node
	uint32_t new_node_base = node_bits.get_bits(j*2,j*2+2);
	new_node_bits.set_bit((j-split_index)*2,(new_node_base&0x01)); //most sig bit should be first bitin new node
	new_node_bits.set_bit((j-split_index)*2+1,new_node_base>>1&0x01);
      }
      std::shared_ptr<DNAtrie> new_node = std::make_shared<DNAtrie>(new_node_strand,bases_remaining_node);

      //now connect to the new-internal node to everything that was on this current node
      for(int j=0; j<MAX_BASES; j++){
	new_node->set_child(current_node->children[j],j);
	current_node->children[j]=nullptr; //null out old connections
      }

      //set up split node from DNA
      uint32_t bases_remaining_DNA = DNA.size()-total_indexes_covered;
      uint32_t DNA_pad_bits  = (bases_remaining_DNA*2)%8==0 ? 0 :  (8 - (bases_remaining_DNA*2)%8);
      std::vector<uint8_t> new_DNA_strand((bases_remaining_DNA*2+DNA_pad_bits)/8);
      hedges::bitwrapper new_DNA_bits(new_DNA_strand);
      //now make the new string for the input string
      for(int j=total_indexes_covered; j<DNA.size(); j++){
	uint32_t new_DNA_base = DNA_bits.get_bits(j*2,j*2+2);
	new_DNA_bits.set_bit((j-total_indexes_covered)*2,(new_DNA_base&0x01));
	new_DNA_bits.set_bit((j-total_indexes_covered)*2+1,new_DNA_base>>1&0x01);
      }
      std::shared_ptr<DNAtrie> DNA_node = std::make_shared<DNAtrie>(new_DNA_strand,bases_remaining_DNA);
      DNA_node->_value = value;

      //do the split
      current_node->children[node_base] = new_node; //finally insert the new node as the split
      current_node->children[DNA_base] = DNA_node; //insert the node that arrisses from the input string
      if(current_node->_value != NULL_VALUE){
	new_node->_value = current_node->_value; //transfer value ownership down the tree
	current_node -> _value = NULL_VALUE; //this currentnode shouldn't have ownership of a value anymore 
      }
      current_node->_prefix_length= current_node->_prefix_length-bases_remaining_node; //this makes sure that the current node has a shorted string
      //should also remove unnesscary bytes from the current node's byte array 
      uint32_t final_pad = current_node->_prefix_length*2%8==0 ? 0 : 8-(current_node->_prefix_length*2)%8;
      uint32_t current_node_final_bytes = (final_pad+current_node->_prefix_length*2)/8;
      uint32_t current_node_og_bytes = current_node->_dna.size();
      for(int pop_counter =0; pop_counter<(current_node_og_bytes-current_node_final_bytes); pop_counter++) current_node->_dna.pop_back();
    }


    
    bool insert(const std::string& DNA,uint32_t value){
      assert(this->is_root()); //searches should start from the root
      std::vector<uint8_t> DNA_as_bytes = convert_DNA_to_bytes(DNA);
      hedges::bitwrapper DNA_bits(DNA_as_bytes);
      bool found=true;
      uint32_t input_length = DNA.size();
      uint32_t total_indexes_covered=0;
      DNAtrie* current_node=this;
      DNAtrie* parent=NULL;
      while(total_indexes_covered<input_length){
	parent=current_node;
	current_node = current_node->children[DNA_bits.get_bits(total_indexes_covered*2,total_indexes_covered*2+2)].get();

	//special case, node null, add string to this node
	if(current_node==NULL){
	  _add_leaf_node(parent,total_indexes_covered,DNA,DNA_bits,value);
	  found=false;
	  break; //this should be the endpoint given that we extended off of a leaf node
	}
	//if node is not null iterate thorugh string to see if node needs to be split
	hedges::bitwrapper node_bits(current_node->_dna);
	for(int i=0; i<current_node->_prefix_length; i++){
	  uint32_t node_base = node_bits.get_bits(i*2,i*2+2);
	  if(total_indexes_covered>=input_length) break;
	  uint32_t DNA_base = DNA_bits.get_bits((total_indexes_covered)*2,(total_indexes_covered)*2+2);
	  if((DNA_base^node_base)!=0){
	    _split_node(total_indexes_covered,DNA,i,current_node,DNA_bits, node_bits, value); //mismatch at base within node, split
	    found=false; //the addition of a node means that the string was indeed not found
	    break;
	  }
	  total_indexes_covered++; //tracking where at in the input strand we have traversed
	}
      }
      return found;
    }

    
    bool is_leaf(){
      return _value!=NULL_VALUE; //leaf nodes should only exist where non-NULL values exist
    }

    bool is_root(){
      return _prefix_length==0; //root should be only node empty
    }

    void set_child(std::shared_ptr<DNAtrie> child_node,uint8_t child_base){children[child_base]=child_node;}

    char get_index(uint8_t index){ //gets the base at the index for this node
      if(index>=_prefix_length) return 0x00; //return nothing if there is no more string to this node
      hedges::bitwrapper dna_bits(_dna);
      return convert_bits_to_base(dna_bits.get_bits(index*2,index*2+2)); //return the base
    }

    std::vector<char> get_transition_bases(){//get the transition bases for this node
      std::vector<char> return_bases;
      for(int i=0; i<MAX_BASES; i++){
	if(children[i]!=nullptr) return_bases.push_back(convert_bits_to_base(i&0x0f));
      }
      return return_bases;
    }

    uint8_t get_length(){return _prefix_length;} //length of the node's internal string

    uint32_t get_value(){ return _value;} //return the value of the complete codeword

    DNAtrie* get_child(char c){ return children[convert_base_to_bits(c)].get();} //get the pointer the child node
    DNAtrie* get_child(uint32_t c){return children[c].get();}//overload to allow for non-char child
    
  protected:
    std::vector<uint8_t> _dna; //local dna string for this node
    std::shared_ptr<DNAtrie> children[MAX_BASES];//pointer to children
    uint8_t _prefix_length; //length of the string at this node
    uint32_t _value=NULL_VALUE; //value stored for the DNA string
  };


  template<typename DNAConstraint = hedges::Constraint>
  class context : public hedges::context<DNAConstraint> {
  public:
    //this class is going to provide the main mechanism for traversing the tree while we make guesses in a strand
    context(int _prev_bits, int _salt_bits)
      :hedges::context<DNAConstraint>(_prev_bits,_salt_bits)
    {
      if(CodebookMap.find("codewords")==CodebookMap.end()){
	assert(0 && "Context fail initial root node, no codewords key found");
      }
      _root_node = CodebookMap["codewords"];
      _current_trie_position=_root_node;
      assert(_root_node->is_root() && "Root node of trie not actually a root node");
      
    }
    
    bool at_codeword_end(void){ //indicates whether a new guess would transfer from the end of the codeword
      if(_guessing_syncs){
	return _current_trie_prefix>=codeword_hedges::SyncBook[_sync_counter].size();
      }
      else{
	return _current_trie_position->is_leaf() && (_current_trie_prefix>=_current_trie_position->get_length());
      }
    }
             
    char getNextSymbol(int x, int y){ assert(0 && "not implemented for codeword context");} //unused function just to make constant calls happy

    char _getNextSymbolTrie(uint32_t& val){//handles the case where we are guessing from the Codewords Trie
      char ret_char=0x00;
      if(_done_guessing) return 0x00; //done guessing, give null character
      if(_current_trie_prefix>=_current_trie_position->get_length() && _current_transition_guesses.size()==0 && !_done_guessing){
	//we reached the end of this node, so we'll be guessing all transitions out
	if(_current_trie_position->is_leaf()){
	  _current_transition_guesses=_root_node->get_transition_bases();
	}
	else{
	  _current_transition_guesses=_current_trie_position->get_transition_bases();
	}
	ret_char=_current_transition_guesses.back();
	_current_transition_guesses.pop_back();
      }
      else if(_current_trie_prefix>=_current_trie_position->get_length() && _current_transition_guesses.size()>0){
	ret_char=_current_transition_guesses.back();
	_current_transition_guesses.pop_back();
	
      }
      else{
	//we just guess the next character
	ret_char=_current_trie_position->get_index(_current_trie_prefix);
      }

      if(_current_transition_guesses.size()==0) _done_guessing=true; //done with generating guesses from the current state

      return ret_char;
    }
    
    char _getNextSymbolSync(void){
      if(_done_guessing) return 0x00;
      _done_guessing=true; //just guess one value
      //simply return the character for the synchronization string, pretty much the same idea as guessing within a node
      std::string current_sync_string = codeword_hedges::SyncBook[_sync_counter];
      if(_current_trie_prefix>=current_sync_string.size()){
	current_sync_string = codeword_hedges::SyncBook[(_sync_counter+1)%codeword_hedges::SyncBook.size()];
	return current_sync_string[0];
      }
      else{
	return current_sync_string[_current_trie_prefix];
      }
    }
    
    char getNextSymbol(uint32_t num_bits, uint32_t& val){ //num_bits, val are references to be able to load with codeword values
      val = NULL_VALUE;
      if(at_codeword_end()){
	if(!this->_guessing_syncs) val = _current_trie_position->get_value();
	else _is_sync_transfer=true;
	_update_index=true;
      }
      if(num_bits==0){ //if num_bits is 0, hedges class is telling us to look at constant synchronization points
	assert(codeword_hedges::SyncBook.size()>0);
	if(!this->_guessing_syncs) {
	  this->_guessing_syncs=true;
	  _current_trie_prefix=0;
	}
	assert(_guessing_syncs);
	return _getNextSymbolSync(); //returns character based on sync codewords
      }
      else{
	if(this->_guessing_syncs) {
	  this->_guessing_syncs=false; //reset that we are no longer guessing syncs
	  _current_trie_position = _root_node;
	  _current_trie_prefix=0; 
	}
	return _getNextSymbolTrie(val); //returns character based on codeword Trie
      }
    }

    
    char _nextSymbolWithUpdateTrie(uint32_t num_bits,char c){
      char ret_symbol = 0x00;
      //we need to update the position we are in the trie based on what base was chosen as a guess
      if(_current_trie_prefix>=_current_trie_position->get_length()){
	_current_trie_prefix=1; //don't set to 0, we already guessed the base to get to this node, so we don't want to double count the base
	if(_current_trie_position->is_leaf()) {
	  _current_trie_position=_root_node->get_child(c); //this is a complete roll over,-->root-->child at c
	  assert(num_bits>0); //if you reach the end of the leaf, should not 
	}
	else{
	  _current_trie_position=_current_trie_position->get_child(c);
	}
	ret_symbol= _current_trie_position->get_index(_current_trie_prefix-1);
      }
      else{
	ret_symbol=_current_trie_position->get_index(_current_trie_prefix);
	_current_trie_prefix++; //stay in the same node
      }
      
      return ret_symbol;
    }

    char _nextSymbolWithUpdateSync(){
      std::string current_sync_string = codeword_hedges::SyncBook[this->_sync_counter];
	if(_current_trie_prefix>=current_sync_string.size()){
	  //We hit the end of the sync string, need to move on to the next
	  _current_trie_prefix=1;
	  return current_sync_string[_current_trie_prefix-1];
	}
	else{
	  _current_trie_prefix++; //keeping going along the sync string
	  return current_sync_string[_current_trie_prefix-1]; 
	}
    }
    
    char nextSymbolWithUpdate(int num_bits, uint32_t value, char c){
      if(_update_index){
	this->index++;
	if(_is_sync_transfer) _sync_counter= (_sync_counter+1)%codeword_hedges::SyncBook.size(); //need to make sure sync counter rolls
      }
      _state_clear(); //clear necessary state on this node
      if(this->_guessing_syncs){
	return _nextSymbolWithUpdateSync(); //use synchronization codeword state 
      }
      else{
	return _nextSymbolWithUpdateTrie(num_bits,c); //use trie codeword state 
      }
    }

    void _state_clear(){
      _done_guessing=false; //make sure this flag rolls over
      _current_transition_guesses.clear();//clearing the guesses vector so when the next node ends it doesn't use a stale vector
      _update_index=false;
      _is_sync_transfer=false;
    }
    
    
  private:
    DNAtrie* _root_node=nullptr; //the root node of the codeword trie
    DNAtrie* _current_trie_position=nullptr; //current position in the trie for this context
    uint8_t _current_trie_prefix=0; //tracks position within  a node
    bool _done_guessing=false; //flag to know when to stop guessing
    std::vector<char> _current_transition_guesses; //tracks the current transitions
    bool _guessing_syncs=false; //tracks whether we are guessing synchronization point or not
    uint32_t _sync_counter=0; // counter to track what sync number we are on
    bool _update_index=false; //flag to test whether to update index
    bool _is_sync_transfer=false; //flag for indicating sync transfer
  };
  
} //end namespace codeword_hedges

#endif

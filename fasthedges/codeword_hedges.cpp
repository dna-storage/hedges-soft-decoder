
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <string>
#include <map>
#include "codeword_hedges.hpp"


namespace codeword_hedges{

  std::map<std::string,codeword_hedges::DNAtrie*> CodebookMap; //global codebook map, easiest thing I can think of right now
  std::vector<std::string> SyncBook; //global syncbook, holds a list of strings, each string should be considered in order as sync points
  

  uint8_t convert_base_to_bits(const char& base){ //binary representation for bases
    switch(base){
    case 'A':
      return 0x00;
    case 'G':
      return 0x01;
    case 'C':
      return 0x02;
    case 'T': 
      return 0x03;
    }
    assert(0); //shouldn't reach here
    return 0xFF;
  }
  
  char convert_bits_to_base(const uint8_t& base){ //binary representation for bases
    switch(base){
    case 0x00:
      return 'A';
    case 0x01:
      return 'G';
    case 0x02:
      return 'C';
    case 0x03:
      return 'T';
    }
    assert(0); //shouldn't reach here
    return 0xFF;
  }

   void print_bitwrapper_as_string(uint32_t length, const hedges::bitwrapper& DNA){ //debug function for printing bit arrays as their string
    std::string DNA_string;
    std::cout<<"Printing a DNA bitwrapper as a string, length "<<length<<std::endl; 
    for(int i =0; i<length;i++){
      uint32_t base_bits = DNA.get_bits(i*2,i*2+2);
      const char base = convert_bits_to_base(base_bits);
      std::string temp(1,base);
      DNA_string = DNA_string + temp;
    }
    std::cout<<"Final DNA String: "<<DNA_string<<std::endl;
  }
  
  //convert DNA string to byte array that will be compatible with bitwrapper object
  std::vector<uint8_t> convert_DNA_to_bytes(const std::string& DNA){
    uint32_t pad_bits = 8 - (DNA.size()*2)%8;
    if (pad_bits==8){
      pad_bits=0; //pad_bits can only be 8 if % is 0
    }
    std::vector<uint8_t> DNA_bytes((DNA.size()*2+pad_bits)/8); //make a vector of bytes large enough to hold binary representation of DNA
    uint8_t shift=0;
    uint32_t byte_index=0;
    for(int i=0; i<DNA.size(); i++){
      assert(shift<8); //make sure shift doesn't fly off the rails
      uint8_t base_bits=convert_base_to_bits(DNA[i]);
      //base_bits = (base_bits&0x01)<<1 | (base_bits>>1); //flipping the endianess here so when we use bitwrapper it comes out the right way
      DNA_bytes[byte_index]|=(base_bits<<shift);
      shift+=2;
      if(shift%8==0){
	shift=0;
	byte_index++;
      }
    }


#ifdef DEBUG
    std::cout<<"Original DNA string before bits :"<<DNA<<std::endl;
    hedges::bitwrapper Test(DNA_bytes);
    print_bitwrapper_as_string(DNA.size(),Test);
#endif
    
    return DNA_bytes;
  }	       









  


  
  
};

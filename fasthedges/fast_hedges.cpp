#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <queue>
  
#include "fast_hedges.hpp"

using namespace hedges;

using hr = hedge_rate;

namespace hedges {

  uint64_t ranhash(uint64_t u)
  {
    /* Logic adapted from Press et al. */
    uint64_t v = u * 3935559000370003845ul + 2691343689449507681ul;
    v ^= v >> 21;
    v ^= v << 37;
    v ^= v >> 4;
    v *= 4768777513237032717;
    v ^= v << 20;
    v ^= v >> 41;
    v ^= v << 5;
    return  v;
  }

  uint64_t digest(uint64_t prev, uint64_t prev_bits,
		  uint64_t index, uint64_t index_bits,
		  uint64_t salt, uint64_t salt_bits,
		  uint64_t mod)
  {
    uint64_t prev_mask = (1ULL << prev_bits) - 1;
    uint64_t index_mask = (1ULL << index_bits) - 1;
    uint64_t salt_mask = (1ULL << salt_bits) - 1;

    //std::cout << "digest: " << std::hex << prev_mask << " " << index_mask << " " << salt_mask << " " << salt_bits << " " << (1ULL << salt_bits) << std::endl;
  
    uint64_t t =  ((((index&index_mask) << prev_bits) | (prev & prev_mask)) << salt_bits) | (salt&salt_mask);
    t = ranhash(t) % mod;
    return t;
  }

 


  void Constraint::next(char c) {
    if ( last[index] == c )
      run++;
    else
      run = 1;

    if ( c == 'A' || c == 'T')
      AT++;
    else
      GC++;

    index = (index+1)%12;  
    if (last[index] == 'A' || last[index] == 'T')
      AT--;
    // we need the check due to initialization
    else if (last[index] == 'C' || last[index] == 'G')
      GC--;
  
    last[index] = c;
  
    assert (run < 4);
  }

  std::vector<char> Constraint::get(int mod) const
  {
    auto choose = std::vector<char>({'A','C','G','T'});    
    if (mod>2) {
      return choose;
    } else {
      if ( run > 2 ) {
	switch(last[index]) {
	case 'A': choose = std::vector<char>({'C','G','T'}); break;
	case 'C': choose = std::vector<char>({'A','G','T'}); break;
	case 'G': choose = std::vector<char>({'A','C','T'}); break;
	case 'T': choose = std::vector<char>({'A','C','G'}); break;
	default: assert(false);
	}
      }      
      if ( choose.size() == 4 && std::abs(AT-GC)>3 ) {
	if ( AT > GC )
	  {
	    //std::cout << "here-GC!" << std::endl;
	    choose = std::vector<char>({'C','G'});
	  }
	else 
	  {
	    //std::cout << "here-AT!" << std::endl;		  
	    choose = std::vector<char>({'A','T'});
	  }
      }      
      return choose;
    }      
  }




} // end namespace
  
int main()
{
  int message_size = 15;
  
  hedge<Constraint> h(1/4.0,4,message_size,4,8,8,0,0,0,0.0);

  std::vector<uint8_t> arr = {100,102,103,106};
  std::string buff;

  std::vector<uint8_t> data;
  for(int i=0; i<message_size; i++) {
    uint8_t b = (uint8_t)(ranhash(i)&0xFF);
    //data[i] = b;
    data.push_back(b);
    std::cout << int(data[i]) << ", ";
  }
  std::cout << std::endl;
    
  for (int i=0; i<1; i++)
    buff = h.encode(arr,data,-1);

  std::cout << "correct: " << buff << std::endl;

  std::vector<uint8_t> mess(10), seq(4);

  std::string observed = buff;

  std::cout << "length of buff = " << buff.size() << std::endl;

  observed[1] = 'A';
  observed[30] = 'C';
  observed[35] = 'T';

  hedges::decode_return_t t(0,0);
  for (int i=0; i<1; i++)
    {
      t = h.decode(observed, seq, mess, 100000);
      if ((i+1)%100==0)
	std::cout << "." ;
    }


  for(auto s: seq)
    std::cout << int(s) << ", ";
  std::cout << std::endl;

  for(auto s: mess)
    std::cout << int(s) << ", ";
  std::cout << std::endl;
  
  return 0;
}

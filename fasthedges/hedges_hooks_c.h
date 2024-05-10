#ifndef HEDGES_HOOKS_DEF
#define HEDGES_HOOKS_DEF
#include<Python.h>
#include<map>
#include<vector>



//Header file for raw access to c functionality

void update_context__c(void* c1, void* c2, int nbits, int value);
void copy_context_no_update__c(void* c1, void* c2);
  
char peek_context__c(void* c1, int nbits, int value);
char peek_context__c(void* c1, int nbits, int value, int range);

uint32_t get_index__c(void* c);

void* make_hedge__c(PyObject* h);

void* make_context__c(void* h1);


std::vector<uint8_t> get_pattern__c(void*h1);
void get_constraint_data__c(void* h,void**c); //extract state constraint
std::vector<uint32_t> get_context_data__c(void* h); //extract state of context


struct previous_states_t{
    uint32_t guess_value;
    std::vector<uint32_t> states;
    std::vector<uint32_t> values;
};

previous_states_t get_previous_states__c(void* hedge, uint32_t nbits, uint32_t current_state);


uint32_t get_nbits__c(void* h,uint32_t index);

struct mod_struct{
    int* next_states; //array holding raw history integers
    int* next_mods; //array holding encodings of next mods
    int val_to_next; //mod indexes that are valid for a given context
    int num_valid_mods; //number of valid values in the arrays 
};

extern const std::map<int,int> mod_map;


typedef mod_struct mod_struct_t;

mod_struct_t get_valid_mods(void* c, int nbits,int* next_states_buffer,int* next_mods_buffer);
mod_struct_t get_valid_mods_hash(void* c, int nbits,int* next_states_buffer,int* next_mods_buffer);


#endif

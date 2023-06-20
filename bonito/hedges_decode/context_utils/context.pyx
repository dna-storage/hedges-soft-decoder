from libc.stdlib cimport malloc, free
cimport hedges_hooks_c
from hedges_hooks_c cimport mod_struct_t
import numpy as np
cimport numpy as cnp
import cython
from libcpp cimport bool
from cpython cimport PyLong_AsVoidPtr, PyDict_GetItem, PyLong_AsLong
ctypedef cnp.int64_t DTYPE_t
from libcpp.map cimport map

DTYPE=np.int64

cdef class ContextManager:
   cdef void** _contexts
   cdef int _H
   cdef map[char,int] letter_to_index 
   cdef map[char,char] complement
   def __cinit__(self,int H, object global_hedge_object):
        self._H=H
        cdef void* global_hedge = PyLong_AsVoidPtr(global_hedge_object)
        self._contexts = <void**>malloc(sizeof(void*)*self._H)
        cdef int i
        for i in range(self._H):
            self._contexts[i] = hedges_hooks_c.make_context__c(global_hedge)
            
        self.letter_to_index[<char>'A']=<int>1
        self.letter_to_index[<char>'T']=<int>4
        self.letter_to_index[<char>'C']=<int>2
        self.letter_to_index[<char>'G']=<int>3

        self.complement[<char>'A']=<char>'T'
        self.complement[<char>'T']=<char>'A'
        self.complement[<char>'C']=<char>'G'
        self.complement[<char>'G']=<char>'C'
   def __dealloc__(self):
       free(self._contexts)
   @cython.boundscheck(False)
   @cython.wraparound(False)
   def update_context(self,ContextManager c1, cnp.ndarray[DTYPE_t,ndim=2] BT, cnp.ndarray[DTYPE_t,ndim=2] Vals,
                      int update_index, int nbits):
       #Updates this context to the other context
       cdef void** c1_array = c1._contexts
       cdef int h
       cdef DTYPE_t value
       cdef DTYPE_t prev_state
       for h in range(self._H):
           value = Vals[h,0]
           prev_state = BT[h,update_index]
           hedges_hooks_c.update_context__c(self._contexts[h],c1_array[prev_state],nbits,value)

   @cython.boundscheck(False)
   @cython.wraparound(False)
   def const_update_context(self,ContextManager c1, cnp.ndarray[DTYPE_t,ndim=2] BT, int const_value,
                      int update_index, int nbits):
       #Updates this context to the other context
       cdef void** c1_array = c1._contexts
       cdef int h
       cdef DTYPE_t prev_state
       for h in range(self._H):
           prev_state = BT[h,update_index]
           hedges_hooks_c.update_context__c(self._contexts[h],c1_array[prev_state],nbits,const_value)

@cython.boundscheck(False)
@cython.wraparound(False)
def fill_base_transitions(int H, int n_edges, ContextManager c, int nbits, bool reverse):
    cdef cnp.ndarray[DTYPE_t,ndim=2] base_transitions = np.zeros([H, n_edges], dtype=DTYPE)
    cdef int i
    cdef int j
    cdef void* context
    cdef char next_base
    cdef int letter_index
    for i in range(H):
        context = c._contexts[i]
        for j in range(n_edges):
            next_base = hedges_hooks_c.peek_context__c(context,nbits,j)
            if reverse: next_base=c.complement[next_base]
            letter_index = c.letter_to_index[next_base]
            base_transitions[i,j]=letter_index
    return base_transitions

@cython.boundscheck(False)
@cython.wraparound(False)
def mod_mask_states(ContextManager c,int nbits, int num_mods):
    cdef cnp.ndarray[DTYPE_t,ndim=2] mod_mask= np.zeros([c._H, (1<<nbits)*num_mods], dtype=np.int32_t )
    cdef int i
    cdef int j
    cdef void* context
    cdef mod_struct_t mods
    cdef int value_to_next_state
    cdef int next_state
    cdef int current_mod_index   
    cdef int* next_states_buffer
    cdef int* next_mods_buffer
    next_states_buffer= <int*>malloc(sizeof(void*)*(1<<nbits))
    next_mods_buffer= <int*>malloc(sizeof(void*)*(1<<nbits))

    for i in range(c._H):
        context = c._contexts[i]
        current_mod_index = i%num_mods
        mods=hedges_hooks_c.get_valid_mods(context,nbits,next_states_buffer,next_mods_buffer)
        value_to_next_state = mods.val_to_next
        for j in range(mods.num_valid_mods):
            next_state = mods.next_states[j]*num_mods+mods.next_mods[j]
            mod_mask[next_state,value_to_next_state*num_mods+current_mod_index] = 1
    free(next_states_buffer)
    free(next_mods_buffer)
    return mod_mask



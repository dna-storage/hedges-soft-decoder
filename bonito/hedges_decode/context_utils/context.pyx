from libc.stdlib cimport malloc, free
cimport hedges_hooks_c
import numpy as np
cimport numpy as cnp
import cython
from libcpp cimport bool

ctypedef cnp.int64_t DTYPE_t

DTYPE=np.int64

cdef class ContextManager:
   cdef void** _contexts
   cdef int _H
   def __cinit__(self,int H):
       self._contexts = <void**>malloc(sizeof(void*)*H)
       self._H=H
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



cdef complement(char c):
    if c=='A': return 'T'
    elif c=='T': return 'A'
    elif c=='C': return 'G'
    elif c=='G': return 'C'

@cython.boundscheck(False)
@cython.wraparound(False)
def fill_base_transitions(int H, int n_edges, ContextManager c, int nbits, bool reverse,
                          dict letter_to_index):
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
            if reverse: next_base=complement(next_base)
            letter_index = letter_to_index[next_base]
            base_transitions[i,j]=next_base
    return base_transitions

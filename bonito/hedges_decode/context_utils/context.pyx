from libc.stdlib cimport malloc, free
cimport hedges_hooks_c
import numpy as np
cimport numpy as cnp
import cython
from libcpp cimport bool
from cpython cimport PyLong_AsVoidPtr, PyDict_GetItem, PyLong_AsLong
from hedges_hooks_c cimport Py_BuildValue
ctypedef cnp.int64_t DTYPE_t


DTYPE=np.int64

cdef class ContextManager:
   cdef void** _contexts
   cdef int _H
   def __cinit__(self,int H, object global_hedge_object):
        self._H=H
        cdef void* global_hedge = PyLong_AsVoidPtr(global_hedge_object)
        self._contexts = <void**>malloc(sizeof(void*)*self._H)
        cdef int i
        for i in range(self._H):
            self._contexts[i] = hedges_hooks_c.make_context__c(global_hedge)
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
    if c== (<char>'A'): return <char>'T'
    elif c==(<char>'T'): return <char>'A'
    elif c==(<char> 'C'): return <char> 'G'
    elif c==(<char>'G'): return <char>'C'


cdef letter_to_index(char c):
    if c==(<char>'A'): return <int>1
    elif c==(<char>'T'): return <int>4
    elif c==(<char> 'C'): return <int> 2
    elif c==(<char>'G'): return <int>3




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
            if reverse: next_base=complement(next_base)
            #this can be slow as hell, probably worth just using a static map 
            #letter_index = PyLong_AsLong(<object>PyDict_GetItem(letter_to_index,<object>Py_BuildValue("s#",<const char*>& next_base,1)))
            letter_index = letter_to_index(next_base)
            base_transitions[i,j]=letter_index
    return base_transitions

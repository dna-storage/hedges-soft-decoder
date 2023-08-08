cimport viterbi
from libc.stdlib cimport malloc, free
import cython
from cpython.ref cimport PyObject
from libcpp cimport bool
from libc.stdint cimport uint32_t
from cpython cimport PyLong_AsVoidPtr



def run_beam_1(uint32_t conv_mem, uint32_t initial_state, uint32_t message_length, uint32_t list_size, 
                uint32_t T, bool rc, object ctc_data, object hedges_pointer,uint32_t offset, uint32_t omp_threads):
    cdef float* ctc_data_ptr
    cdef void* tmp
    tmp = PyLong_AsVoidPtr(ctc_data)
    ctc_data_ptr = <float*> tmp
    return <object> viterbi.beam_viterbi_1(conv_mem, initial_state, message_length,list_size, T, 
                        rc, ctc_data_ptr, <PyObject*>hedges_pointer,offset,omp_threads)

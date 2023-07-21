from cpython.ref cimport PyObject
from libc.stdint cimport uint32_t
from libcpp cimport bool

cdef extern from "viterbi_1.hpp":
    PyObject* beam_viterbi_1(
        uint32_t conv_mem, 
        uint32_t initial_state, 
        uint32_t message_length,
        uint32_t list_size, 
        uint32_t T, 
        bool rc,
        float* ctc_data, 
        PyObject* hedges_state_pointer,
        uint32_t omp_threads)


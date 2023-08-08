/*
    Declarations for viterbi beam decoding to help with python interface
*/
#include <Python.h>
#include <cstdint>



PyObject* beam_viterbi_1(
  uint32_t conv_mem, //number of bits for convolutional code
  uint32_t initial_state, //initial state to start code at
  uint32_t message_length, //length of the message (in bases)
  uint32_t list_size, //number of items per list
  uint32_t T, //total size of data block
  bool rc, //flag for reverse complement
  float* ctc_data, //pointer to flattened data representing CTC matrix, organized as [T]
  PyObject* hedges_state_pointer,//pyobject that actually refers to a hedges state pointer which can derive necessary contexts
  uint32_t offset,
  uint32_t omp_threads //number of threads to launch
);

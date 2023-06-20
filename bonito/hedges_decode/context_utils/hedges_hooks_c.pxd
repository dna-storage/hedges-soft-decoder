from cpython.ref cimport PyObject

cdef extern from "hedges_hooks_c.h":
    void update_context__c(void* c1, void* c2, int nbits, int value)

    char peek_context__c(void* c1, int nbits, int value)

    void* make_hedge__c(PyObject* h)

    void* make_context__c(void* h1)

    struct mod_struct:
        int* next_states
        int* next_mods
        int val_to_next
        int num_valid_mods
    ctypedef mod_struct mod_struct_t

    mod_struct_t get_valid_mods(void*c, int nbits, int* next_states_buffer, int* next_mods_buffer)
from cpython.ref cimport PyObject

cdef extern from "hedges_hooks_c.h":
    void update_context__c(void* c1, void* c2, int nbits, int value)

    char peek_context__c(void* c1, int nbits, int value)

    void* make_hedge__c(PyObject* h)

    void* make_context__c(void* h1)

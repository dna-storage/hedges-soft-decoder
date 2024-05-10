#define PY_SSIZE_T_CLEAN
#include <cstring>
#include <cassert>
#include <cstdint>
#include <Python.h>
#include <limits>
#include "shared_hedges.hpp"
#include "fast_hedges.hpp"
#include "hedges_hooks_c.h"

using namespace hedges;

static PyObject *HooksError;


static PyObject *
make_hedge(PyObject *self, PyObject *args)
{
  PyObject *hObj;
  if (PyArg_ParseTuple(args, "O", &hObj)) {
    return PyLong_FromVoidPtr(make_hedge__c(hObj));      
  }
  return Py_BuildValue("s",NULL);
}

static PyObject *
make_context(PyObject *self, PyObject *args)
{
  PyObject* pylong_hedge_ptr;
  if (PyArg_ParseTuple(args, "O", &pylong_hedge_ptr)) {
    return PyLong_FromVoidPtr(make_context__c(PyLong_AsVoidPtr(pylong_hedge_ptr))); 
  }
  return Py_BuildValue("s",NULL);
}

static PyObject *
print_context(PyObject *self,PyObject *args){
  PyObject* pylong_context_ptr;
  if (PyArg_ParseTuple(args, "O", &pylong_context_ptr)) {
    context<Constraint>* c = (context<Constraint>*)PyLong_AsVoidPtr(pylong_context_ptr);
    c->print();
  }
  return Py_BuildValue("s",NULL);
}


static PyObject *
fastforward_context(PyObject *self, PyObject *args)
{
  PyObject* pylong_hedge_ptr;
  Py_buffer *buff = (Py_buffer *) malloc(sizeof(*buff));
  if (PyArg_ParseTuple(args, "y*O", buff, &pylong_hedge_ptr)) {
    uint8_t *buf = (uint8_t*) buff->buf;
    hedge<Constraint>* h = (hedge<Constraint>*)PyLong_AsVoidPtr(pylong_hedge_ptr);
    std::vector<uint8_t> index;
    std::vector<uint8_t> message;
    for(int i=0; i<buff->len && i<h->seq_bytes; i++) index.push_back(buf[i]);
    for(int i=h->seq_bytes; i<buff->len && i<h->message_bytes+h->seq_bytes; i++) message.push_back(buf[i]);
    std::string buff = h->encode(index,message,index.size()+message.size());
    return Py_BuildValue("s",buff.c_str());//return the string we encoded
  }
  return Py_BuildValue("s",NULL);
}

//while there is a __c version of this, I'll just keep this the way it is to reduce possible errors propagating
static PyObject *
get_incoming_states(PyObject *self, PyObject *args)
{
  PyObject* pylong_context_ptr;
  int nbits;
  uint32_t current_state;
  PyObject* outlist;
  PyObject* return_tuple;
  if (PyArg_ParseTuple(args, "OiI",&pylong_context_ptr,&nbits,&current_state)) {
    hedge<Constraint>* c = (hedge<Constraint>*)PyLong_AsVoidPtr(pylong_context_ptr);
    uint32_t guess_value; //value that was guessed to get to this state
    outlist = PyList_New(1ULL<<nbits);
    if(nbits==0){
      assert(PyList_Size(outlist)==1);
      PyList_SET_ITEM(outlist, 0, PyTuple_Pack(2,Py_BuildValue("I",0),Py_BuildValue("I",current_state))); 
      guess_value=0; //guess value is always 0 for nbits==0
    }
    else{ //nbits tells how many bits we shifted to get to current state, possible states that could have preceeded this is determined by reversing the shift
      uint32_t number_previous_states = 1ULL<<nbits;
      guess_value=((1ULL<<nbits)-1)&current_state; //deduce guess value by bottom order bits
      for(uint32_t i=0;i<number_previous_states;i++){
	uint32_t mask = (nbits==2)?3:((nbits==0)?0:1);
	uint32_t previous_state = (current_state>>nbits)|((i&mask)<<(c->prev_bits-nbits));
	PyList_SET_ITEM(outlist, i, PyTuple_Pack(2,Py_BuildValue("I",i),Py_BuildValue("I",previous_state)));
      }
    }
    return_tuple = PyTuple_Pack(2,Py_BuildValue("I",guess_value),outlist);
    return return_tuple;
  }
  return Py_BuildValue("s",NULL);
}



static PyObject *
update_context(PyObject *self, PyObject *args)
{
  int nbits;
  uint32_t value;
  PyObject* pylong_context_ptr;
  PyObject* pylong_context_ptr_to_copy;
  if (PyArg_ParseTuple(args, "OOiI",&pylong_context_ptr,&pylong_context_ptr_to_copy,&nbits,&value)) {
    update_context__c((context<Constraint>*)PyLong_AsVoidPtr(pylong_context_ptr),(context<Constraint>*)PyLong_AsVoidPtr(pylong_context_ptr_to_copy),
		      nbits,value);
  }
  return Py_BuildValue("s",NULL);
}


static PyObject *
peek_context(PyObject *self, PyObject *args)
{
  int nbits;
  uint32_t value;
  PyObject* pylong_context_ptr;
  if (PyArg_ParseTuple(args, "OiI",&pylong_context_ptr,&nbits,&value)) {
    context<Constraint>* c = (context<Constraint>*)PyLong_AsVoidPtr(pylong_context_ptr);
    char next_symbol = peek_context__c(c,nbits,value);
    return Py_BuildValue("s#",(const char*)&next_symbol,1);
  }
  return Py_BuildValue("s",NULL);
}



static PyObject * 
delete_context(PyObject *self, PyObject *args) //WARNING: take care to not double-delete the same memory
{
  PyObject* pylong_context_ptr;
  if (PyArg_ParseTuple(args, "O",&pylong_context_ptr)) {
    context<Constraint>* c = (context<Constraint>*)PyLong_AsVoidPtr(pylong_context_ptr);
    delete c;
  }
  return Py_BuildValue("s",NULL);
}


static PyObject *
delete_hedge(PyObject *self, PyObject *args)
{
  PyObject* pylong_hedge_ptr;
  if (PyArg_ParseTuple(args, "O",&pylong_hedge_ptr)) {
    hedge<Constraint>* h = (hedge<Constraint>*)PyLong_AsVoidPtr(pylong_hedge_ptr);
    delete h;
  }
  return Py_BuildValue("s",NULL);
}

static PyObject *
get_nbits(PyObject *self, PyObject *args)
{
  PyObject* pylong_hedge_ptr;
  uint32_t index;
  if (PyArg_ParseTuple(args, "OI",&pylong_hedge_ptr,&index)){
    hedge<Constraint>* h = (hedge<Constraint>*)PyLong_AsVoidPtr(pylong_hedge_ptr);
    uint32_t nbits = get_nbits__c((void*)h,index);
    return Py_BuildValue("I",nbits);
  }
  return Py_BuildValue("s",NULL);
}


static PyObject *
get_max_index(PyObject *self, PyObject *args)
{
  PyObject* pylong_hedge_ptr;
  if (PyArg_ParseTuple(args, "O",&pylong_hedge_ptr)){
    hedge<Constraint>* h = (hedge<Constraint>*)PyLong_AsVoidPtr(pylong_hedge_ptr);
    return Py_BuildValue("I",h->max_index);
  }
  return Py_BuildValue("s",NULL);
}

static PyObject *
get_hedge_context_mod(PyObject *self, PyObject *args)
{
  PyObject* pylong_hedge_ptr;
  if (PyArg_ParseTuple(args, "O",&pylong_hedge_ptr)){
    hedge<Constraint>* h = (hedge<Constraint>*)PyLong_AsVoidPtr(pylong_hedge_ptr);
    std::vector<char> mod = h->encoding_context.constraint.get(1);
    int sum = 0;
    for (auto& c: mod) sum=sum+(int)c;
    return Py_BuildValue("I",mod_map.at(sum));
  }
  return Py_BuildValue("s",NULL);
}

static PyObject *
get_hedge_context_history(PyObject *self, PyObject *args)
{
  PyObject* pylong_hedge_ptr;
  if (PyArg_ParseTuple(args, "O",&pylong_hedge_ptr)){
    hedge<Constraint>* h = (hedge<Constraint>*)PyLong_AsVoidPtr(pylong_hedge_ptr);
    context<Constraint> c = h->encoding_context;
    return Py_BuildValue("I",h->encoding_context.prev&h->encoding_context.prev_mask);
  }
  return Py_BuildValue("s",NULL);
}



static PyMethodDef HookMethods[] = {
  {"make_hedge",  make_hedge, METH_VARARGS, "Gets a c-representation of the global hedges state"},
  {"make_context", make_context,METH_VARARGS, "Makes and returns a c-context object from a global initial state"},
  {"print_context", print_context,METH_VARARGS, "Prints the content of the provided context"},  
  {"get_incoming_states",  get_incoming_states, METH_VARARGS, "Encode into DNA."},
  {"update_context",update_context,METH_VARARGS,"Update context with value and number bits"},
  {"peek_context",peek_context,METH_VARARGS,"Look at context without updates"},
  {"delete_context",delete_context,METH_VARARGS,"Delete memory for context"},
  {"delete_hedge",delete_hedge,METH_VARARGS,"delete a hedge global context"},
  {"get_nbits",get_nbits,METH_VARARGS,"get the number of bits that should be at a certain index"},
  {"get_max_index",get_max_index,METH_VARARGS,"Get the length of the encoded strand"},
  {"fastforward_context",  fastforward_context, METH_VARARGS, "Encodes a set of bytes which advances the context of the given hedge state"},
  {"get_hedge_context_mod",get_hedge_context_mod,METH_VARARGS,"Gets the mod index for the global context"},
  {"get_hedge_context_history",get_hedge_context_history,METH_VARARGS,"Gets the history for the global context"},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef hookmodule = {
  PyModuleDef_HEAD_INIT,
  "hedges_hooks",   /* name of module */
  NULL, /* module documentation, may be NULL */
  -1,       /* size of per-interpreter state of the module,
	       or -1 if the module keeps state in global variables. */
  HookMethods
};

PyMODINIT_FUNC PyInit_hedges_hooks(void)
{
  PyObject *m;

  m = PyModule_Create(&hookmodule);
  if (m == NULL)
    return NULL;

  HooksError = PyErr_NewException("hedges_hooks.error", NULL, NULL);
  Py_XINCREF(HooksError);
  if (PyModule_AddObject(m, "error", HooksError) < 0) {
    Py_XDECREF(HooksError);
    Py_CLEAR(HooksError);
    Py_DECREF(m);
    return NULL;
  }

  return m;
}


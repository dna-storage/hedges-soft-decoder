#include <Python.h>
#include <limits>
#include "shared_hedges.hpp"
#include "fast_hedges.hpp"
#include "codeword_hedges.hpp"


static PyObject *CodewordhedgesError;


static PyObject *
codewordhedges_decode(PyObject *self, PyObject *args)
{
  return shared_decode<hedges::Constraint,hedges::Reward,codeword_hedges::context>(self,args);
}


//Define the function for initializing codebooks
static PyObject*
codewordhedges_codebook_init(PyObject *self, PyObject *args){
  PyObject* codebook;
  const char * name;
  if(PyArg_ParseTuple(args,"O|s",&codebook,&name)){
    return create_codebook(codebook,name,CodewordhedgesError);
    if(codebook!=NULL && !PyDict_Check(codebook)){
      PyErr_SetString(CodewordhedgesError, "Given Python object is not a dictionary");
      return NULL;
    }
  }
  else{
    PyErr_SetString(CodewordhedgesError, "Error in codebook init");
    return NULL;
  }
  return  Py_BuildValue("s",NULL);
}

static PyObject*
codewordhedges_codebook_destroy(PyObject *self, PyObject *args){
  const char * name;
  if(PyArg_ParseTuple(args,"s",&name)){
    return destroy_codebook(name,CodewordhedgesError);
  }
  else{
    PyErr_SetString(CodewordhedgesError, "Error in codebook destroy");
    return NULL;
  }
  return  Py_BuildValue("s",NULL);
}


//Define the function for initializing codebooks
static PyObject*
codewordhedges_syncbook_init(PyObject *self, PyObject *args){
  PyObject* syncbook;
  if(PyArg_ParseTuple(args,"O",&syncbook)){
    return create_syncbook(syncbook,CodewordhedgesError);
    if(syncbook!=NULL && !PyList_Check(syncbook)){
      PyErr_SetString(CodewordhedgesError, "Given Python object is not a list");
      return NULL;
    }
  }
  else{
    PyErr_SetString(CodewordhedgesError, "Error in syncbook init");
    return NULL;
  }
  return  Py_BuildValue("s",NULL);
}

static PyObject*
codewordhedges_syncbook_clear(PyObject *self, PyObject *args){
  return clear_syncbook(CodewordhedgesError);
}


static PyMethodDef CodewordhedgesMethods[] = { //codeword based hedges methods, only going to really need decode
    {"decode",  codewordhedges_decode, METH_VARARGS, "Decode from DNA back into bytes."},
    {"codebook_init", codewordhedges_codebook_init, METH_VARARGS, "Create objects to store codebook information"},
    {"codebook_destroy",codewordhedges_codebook_destroy, METH_VARARGS, "Destroy a given codebook"},
    {"syncbook_init",codewordhedges_syncbook_init,METH_VARARGS,"Initializes an array that will be used for regular sync points"},
    {"syncbook_clear",codewordhedges_syncbook_clear,METH_NOARGS,"Clears an array used for regular sync points"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef codewordhedgesmodule = {
    PyModuleDef_HEAD_INIT,
    "codewordhedges",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    CodewordhedgesMethods
};


PyMODINIT_FUNC PyInit_codewordhedges(void) //initialization for codeword hedges
{
    PyObject *m;

    m = PyModule_Create(&codewordhedgesmodule);
    if (m == NULL)
        return NULL;

    CodewordhedgesError = PyErr_NewException("codewordhedges.error", NULL, NULL);
    Py_XINCREF(CodewordhedgesError);
    if (PyModule_AddObject(m, "error", CodewordhedgesError) < 0) {
        Py_XDECREF(CodewordhedgesError);
        Py_CLEAR(CodewordhedgesError);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}

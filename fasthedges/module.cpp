#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits>
#include "shared_hedges.hpp"
#include "fast_hedges.hpp"

using namespace hedges;

static PyObject *FasthedgesError;


static PyObject *
fasthedges_encode2(PyObject *self, PyObject *args)
{
    const char *command;
    Py_buffer *buff = (Py_buffer *) malloc(sizeof(*buff));
    double rate;
    unsigned long num_index_bytes;
    unsigned long num_message_bytes;
    PyObject *hObj;

    if (PyArg_ParseTuple(args, "y*O", buff, &hObj)) {

      hedge<Constraint> h = make_hedge_from_pyobject(hObj);
      
      std::vector<uint8_t> index;
      std::vector<uint8_t> message;
      
      uint8_t *buf = (uint8_t*) buff->buf;
      
      for(int i=0; i<buff->len && i<h.seq_bytes; i++)
	{
	  //printf("%d,",buf[i]);
	  index.push_back(buf[i]);
	}
      for(int i=h.seq_bytes; i<buff->len && i<h.message_bytes+h.seq_bytes; i++)
	{
	  //printf("%d,",buf[i]);
	  message.push_back(buf[i]);
	}

      std::string buff = h.encode(index, message,-1);
      
      PyObject *o = Py_BuildValue("s",buff.c_str());
      return o;      
    }

    return Py_BuildValue("s",NULL);
}


static PyObject *
fasthedges_decode2(PyObject *self, PyObject *args)
{
  return shared_decode(self,args);
}

static PyObject * decode_item(const char *strand, hedge<Constraint> &h, int guesses=1000000)
{
  std::string sstrand(strand);

  std::vector<uint8_t> mess(h.message_bytes), seq(h.seq_bytes);
  hedges::decode_return_t t = h.decode(sstrand,seq,mess,guesses);

  // std::cout << "[";
  // for(auto i : seq)
  //   std::cout << int(i) << ", ";
  // for(auto i : mess)
  //   std::cout << int(i) << ", ";
  // std::cout << "]" << std::endl;

  uint32_t size = h.seq_bytes + h.message_bytes;
  PyObject *list = PyList_New(size);
  for(auto i=0; i<size; i++)
    {
      if (i<h.seq_bytes) {
	PyObject *item = Py_BuildValue("i",seq[i]);
	Py_INCREF(item);
	PyList_SetItem(list,i,item);
      } else {
	PyObject *item = Py_BuildValue("i",mess[i-seq.size()]);
	Py_INCREF(item);
	PyList_SetItem(list,i,item);
      }
      
    }
  return list;
}

 
static PyObject *
fasthedges_bulk_decode(PyObject *self, PyObject *args)
{
    const char *strand;
    PyObject *l;
    PyObject *hObj;
    int guesses = 1000000;
    
    if (PyArg_ParseTuple(args, "OO|i", &l, &hObj, &guesses)) {

      hedge<Constraint> h = make_hedge_from_pyobject(hObj);
      
      if (l!=NULL && !PyList_Check(l)) {
         PyErr_SetString(FasthedgesError, "Expected first argument to be a list of DNA strands.");
         return NULL;	
      }

      PyObject *bulk_list = PyList_New(PyList_Size(l));

      for(Py_ssize_t j=0; j<PyList_Size(l); j++)
	{
	  PyObject *item = PyList_GetItem(l, j);
	  
	  if (PyUnicode_Check(item))
	    {
	      const char *strand = PyUnicode_AsUTF8(item);
	      PyObject *decoded_list = decode_item(strand, h);
	      Py_INCREF(decoded_list);
	      PyList_SetItem(bulk_list,j,decoded_list);	      
	    }
	}

      return bulk_list;
    }

    return Py_BuildValue("s",NULL);
}


static PyObject *
fasthedges_echo(PyObject *self, PyObject *args)
{
    PyObject *hObj;
    
    if (PyArg_ParseTuple(args, "O", &hObj))
      {
	hedge<Constraint> h = make_hedge_from_pyobject(hObj);

	h.print(true);
      }
    return Py_BuildValue("s",NULL);
}

static PyMethodDef FasthedgesMethods[] = {
    {"encode",  fasthedges_encode2, METH_VARARGS, "Encode into DNA."},
    {"decode",  fasthedges_decode2, METH_VARARGS, "Decode from DNA back into bytes."},
    {"bulk_decode",  fasthedges_bulk_decode, METH_VARARGS, "Bulk decode a list of strands from DNA back into bytes."},
    {"echo",  fasthedges_echo, METH_VARARGS, "Echo hedges configuration."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef fasthedgesmodule = {
    PyModuleDef_HEAD_INIT,
    "fasthedges",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    FasthedgesMethods
};

PyMODINIT_FUNC PyInit_fasthedges(void)
{
    PyObject *m;

    m = PyModule_Create(&fasthedgesmodule);
    if (m == NULL)
        return NULL;

    FasthedgesError = PyErr_NewException("fasthedges.error", NULL, NULL);
    Py_XINCREF(FasthedgesError);
    if (PyModule_AddObject(m, "error", FasthedgesError) < 0) {
        Py_XDECREF(FasthedgesError);
        Py_CLEAR(FasthedgesError);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}


#ifndef SHARED_HPP
#define SHARED_HPP

#include <Python.h>
#include "fast_hedges.hpp"
#include "codeword_hedges.hpp"

using namespace hedges;

inline long getLong(PyObject *Object, const char *attr)
{
  if (PyObject_HasAttrString(Object, attr))
    {
      PyObject *a = PyObject_GetAttrString(Object,attr);
      if (PyLong_Check(a))
	{
	  return PyLong_AsLong(a);
	}
    }
  return -1;
}

inline double getDouble(PyObject *Object, const char *attr)
{
  if (PyObject_HasAttrString(Object, attr))
    {
      PyObject *a = PyObject_GetAttrString(Object,attr);
      if (PyFloat_Check(a))
	{
	  return PyFloat_AsDouble(a);
	}
    }
  return -1.0;
}

static hedges::hedge<Constraint> make_hedge_from_pyobject(PyObject *object)
{
  double rate = getDouble(object,"rate");
  int seq_bytes = getLong(object, "seq_bytes");
  int message_bytes = getLong(object, "message_bytes");
  int pad_bits = getLong(object, "pad_bits");
  int prev_bits = getLong(object, "prev_bits");
  int salt_bits = getLong(object, "salt_bits");
  int codeword_sync_period = getLong(object,"cw_sync_period");
  int parity_period = getLong(object,"parity_period");
  int parity_history = getLong(object,"parity_history");
  double wild_card_reward = getDouble(object,"custom_reward");
  int guess_length=getLong(object,"guess_length");
  int use_custom_reward =getLong(object,"use_custom_reward");
  double run_cost = getDouble(object,"run_cost");
  if (pad_bits == -1) {
    if (rate > 0.33)
      pad_bits = 8;
    else if (rate > 0.125)
      pad_bits = 4;
    else
      pad_bits = 1;
  }
  if (prev_bits == -1)
    prev_bits = 8;
  if (salt_bits == -1)
    salt_bits = 8;
    
  hedges::hedge<Constraint> h(rate, seq_bytes, message_bytes, pad_bits, prev_bits, salt_bits,
			      codeword_sync_period,parity_period,parity_history,wild_card_reward,guess_length,use_custom_reward,run_cost);

  return h;
}


template<typename DNAConstraint = hedges::Constraint, typename Reward = hedges::Reward,  template <typename> class Context = hedges::context>
static PyObject *
shared_decode(PyObject *self, PyObject *args)
{
    const char *strand;
    PyObject *hObj;
    int guesses = 100000;
    
    if (PyArg_ParseTuple(args, "sO|i", &strand, &hObj, &guesses)) {

      hedges::hedge<DNAConstraint> h = make_hedge_from_pyobject(hObj);
      
      std::vector<uint8_t> mess(h.message_bytes), seq(h.seq_bytes);

      std::string sstrand(strand);
      
      hedges::decode_return_t t(0,0);
      if(h.parity_period==0) t = h. template decode<Reward,Context,hedges::search_tree>(sstrand,seq,mess,guesses);
      else t  = h. template decode<Reward,Context,hedges::search_tree_parity>(sstrand,seq,mess,guesses); //parity search tree considers parity data during decoding
      int sz = seq.size() + mess.size();
      PyObject *list = PyList_New(sz);
      for(auto i=0; i<sz; i++)
	{	  
	  if (i<seq.size()) {
	    if (i >= t.return_bytes) {
	      PyObject *item = Py_BuildValue("s",NULL);
	      Py_INCREF(item);
	      PyList_SetItem(list,i,item);

	    } else {
	      PyObject *item = Py_BuildValue("i",seq[i]);
	      Py_INCREF(item);
	      PyList_SetItem(list,i,item);
	    }
	  } else {
	    if (i >= t.return_bytes) {
	      PyObject *item = Py_BuildValue("s",NULL);
	      Py_INCREF(item);
	      PyList_SetItem(list,i,item);
	    } else {
	      PyObject *item = Py_BuildValue("i",mess[i-seq.size()]);
	      Py_INCREF(item);
	      PyList_SetItem(list,i,item);
	    }
	  }
	  
	}
      PyObject* return_dict =Py_BuildValue("{s:O,s:f}","return_bytes",list,"score",t.score);
      return return_dict;
    }

    return Py_BuildValue("s",NULL);
}




static PyObject *
create_codebook(PyObject*codebook, const char* name, PyObject* exception)
{
  codeword_hedges::DNAtrie* trie_root = new codeword_hedges::DNAtrie();
  PyObject* PyDNA;
  PyObject* PyKey;
  Py_ssize_t pos = 0;
  while(PyDict_Next(codebook,&pos,&PyKey,&PyDNA)){
    if(!PyUnicode_Check(PyDNA)){
      PyErr_SetString(exception,"Dictionary values are not unicode");
      return NULL;
    }
    const char* strand = PyUnicode_AsUTF8(PyDNA);
    std::string DNA(strand);
    uint32_t value = (uint32_t)PyLong_AsLong(PyKey);
    trie_root->insert(DNA,value);
  }
  //add the trie to the module
  std::string codebook_name(name);

  if(codeword_hedges::CodebookMap.find(codebook_name)!=codeword_hedges::CodebookMap.end()){
    PyErr_WarnEx(PyExc_RuntimeWarning,"Codebook already exists, deleting old codebook to avoid memory leak",1);
    delete codeword_hedges::CodebookMap[codebook_name];
    codeword_hedges::CodebookMap.erase(codebook_name);
  }
  
  codeword_hedges::CodebookMap[codebook_name] = trie_root;

#ifdef DEBUG
  trie_root->print();
#endif
  
  return Py_BuildValue("s",NULL);
}

static PyObject* destroy_codebook(const char* name, PyObject* exception){ //removes a codebook associated with the module
  std::string codebook_name(name);
  if(codeword_hedges::CodebookMap.find(codebook_name) == codeword_hedges::CodebookMap.end()){
    PyErr_WarnEx(PyExc_RuntimeWarning,"Codebook does not exist on delete, returning to avoid double deletion",1);
    return NULL;
  }
  if(codeword_hedges::CodebookMap[codebook_name]==nullptr){
    PyErr_SetString(exception,"Nullptr being freed on codebook destroy operation");
    return NULL;
  }
  delete codeword_hedges::CodebookMap[codebook_name];
  codeword_hedges::CodebookMap.erase(codebook_name);			       
  return Py_BuildValue("s",NULL);
}

static PyObject *
create_syncbook(PyObject* syncbook,PyObject* exception)
{
  PyObject* PyDNA;
  Py_ssize_t syncbook_length = PyList_Size(syncbook);
    
  for(Py_ssize_t pos =0; pos<syncbook_length; pos++){
    PyDNA = PyList_GetItem(syncbook,pos);
    if(!PyUnicode_Check(PyDNA)){
      PyErr_SetString(exception,"Dictionarmy values are not unicode");
      return NULL;
    }
    const char* strand = PyUnicode_AsUTF8(PyDNA);
    std::string DNA(strand);
    codeword_hedges::SyncBook.push_back(DNA);
  }
  return Py_BuildValue("s",NULL);
}

static PyObject* clear_syncbook(PyObject* exception){ //removes a codebook associated with the module
  codeword_hedges::SyncBook.clear();
  return Py_BuildValue("s",NULL);
}



#endif

from os import listdir
import os
import torch
import math
import numpy as np
import json
import inspect
from collections import namedtuple
import dnastorage.codec.hedges as hedges
import dnastorage.codec.hedges_hooks as hedges_hooks
import cupy as cp
import gc
import sys
import traceback
import time

import bonito.hedges_decode.cuda_utils as cu
from bonito.hedges_decode.base_decode import *
from bonito.hedges_decode.decode_ctc import *
from bonito.hedges_decode.align import *

def check_hedges_params(hedges_params_dict)->None:
    """
    @brief      Checks parameters in dictionary to make sure they line up with class

    @param      hedges_params_dict  dictionary of parameters

    @return     No return value, raise exception if error
    """
    args = inspect.signature(hedges.hedges_state.__init__)
    for param in hedges_params_dict:
        if param not in args.parameters: raise KeyError("{} not legal".format(param))
        


#@profile
def hedges_decode(read_id,scores,hedges_params:str,hedges_bytes:bytes,
                  using_hedges_DNA_constraint:bool,alphabet:list,stride=1,
                  endpoint_seq:str="",window=0)->dict:
    """
        @brief      Top level function for decoding CTC-style outputes to hedges strands

        @details    Generates a base-call that should be a strand that satisfies the given hedges code

        @param      scores  Log-probabilities for bases at a particular point in the signal
        @param      hedges_params file that contains parameters for the hedges code
        @param      hedges_bytes optional string of bytes to fast forward the decode process to
        @param      using_hedges_DNA_cosntraint Boolean when True uses DNA constraint information of the hedges code in the trellis
        @param      stride parameter used to satisfy interface, has no purpose at the moment

        @return     Dictionary with entries related to the output seqeunce
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_time=time.time()
    try:
        with torch.no_grad():
            scores=scores["scores"].to("cpu")
            assert(hedges_params!=None and hedges_bytes!=None)

            try:
                hedges_params_dict = json.load(open(hedges_params,'r'))
                check_hedges_params(hedges_params_dict)
            except Exception as e:
                print(e)
                exit(1)
                

            decoder = HedgesBonitoCTCGPU(hedges_params_dict,hedges_bytes,using_hedges_DNA_constraint,alphabet,"cuda:0",window=window)
            #create aligner
            aligner = AlignCTC(alphabet,device="cpu")

            f_endpoint_upper_index=0
            r_endpoint_lower_index=len(scores)
            f_endpoint_score=Log.one
            r_endpoint_score=Log.one
            if len(endpoint_seq)>0:
                f_endpoint_lower_index,f_endpoint_upper_index,f_endpoint_score  = aligner.align(scores,endpoint_seq)
                r_endpoint_lower_index,r_endpoint_upper_index,r_endpoint_score  = aligner.align(scores,reverse_complement(endpoint_seq))

            f_hedges_bytes_lower_index,f_hedges_bytes_upper_index,f_hedges_score = aligner.align(scores,decoder.fastforward_seq[::-1])
            r_hedges_bytes_lower_index,r_hedges_bytes_upper_index,r_hedges_score = aligner.align(scores,complement(decoder.fastforward_seq))

            """
            We need to rearrange the scores based on alignments so that index is always at the beginning of the strand.
            Because we
            """
            seq=""
            #print("hedges f score {}".format(f_hedges_score))
            #print("endpoint f score {}".format(f_endpoint_score))
            #print("hedges r score {}".format(r_hedges_score))
            #print("endpoint r score {}".format(r_endpoint_score))
            if Log.mul(f_hedges_score,f_endpoint_score)>Log.mul(r_endpoint_score,r_hedges_score):
                #print("IS FORWARD")
                s=scores[f_endpoint_upper_index:f_hedges_bytes_upper_index]
                #print(" {} {}".format(f_endpoint_upper_index,f_hedges_bytes_upper_index))
                s=s.flip([0])
                complement_trellis=False
                if(s.size(0)==0 or s.size(0)<decoder._full_message_length): seq="N"
                else:
                    decode_start_time=time.time()
                    #print("Time up to decode {}".format(decode_start_time-start_time))
                    seq = decoder.decode(s,complement_trellis)
                    decode_end_time=time.time()
                    #print("Time to decode {}".format(decode_end_time-decode_start_time))
            else:
                #print("IS REVERSE")
                s=scores[r_hedges_bytes_lower_index:r_endpoint_lower_index]
                if(s.size(0)==0 or s.size(0)<decoder._full_message_length): seq="N"
                else:
                    #print("heges lower upper {} {}".format(r_hedes_bytes_lower_index,r_hedges_bytes_upper_index))
                    #print("{} {}".format(r_hedges_bytes_lower_index,r_endpoint_lower_index))
                    complement_trellis=True
                    decoder.fastforward_seq = complement(decoder.fastforward_seq)
                    decode_start_time=time.time()
                    #print("Time up to decode {}".format(decode_start_time-start_time))
                    seq = decoder.decode(s,complement_trellis)
                    decode_end_time=time.time()
                    #print("Time to decode {}".format(decode_end_time-decode_start_time))
            #try to clean up memory
            return {'sequence':seq,'qstring':"*"*len(seq),'stride':stride,'moves':seq}
    except Exception as e:
        traceback.print_exc()
        seq="N"
        print("\n\nOffending read: {}\n\n".format(read_id),file=sys.stderr)
        print("Scores size: {}\n".format(scores.size()),file=sys.stderr)
        print(e,file=sys.stderr)
        return {'sequence':seq,'qstring':"*"*len(seq),'stride':stride,'moves':seq}

        

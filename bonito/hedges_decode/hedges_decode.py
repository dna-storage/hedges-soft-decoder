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
import logging
logger=logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
        



def hedges_decode(read_id,scores_arg,hedges_params:str,hedges_bytes:bytes,
                  using_hedges_DNA_constraint:bool,alphabet:list,stride=1,
                  endpoint_seq:str="",window=0,trellis="base",mod_states=3)->dict:
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
    try:
        with torch.no_grad():
            alignment_scores=scores_arg["scores"]
            assert(hedges_params!=None and hedges_bytes!=None)
            logger.info("Scores Time Range at beginning: {}".format(alignment_scores.size(0)))
            try:
                hedges_params_dict = json.load(open(hedges_params,'r'))
                check_hedges_params(hedges_params_dict)
            except Exception as e:
                raise ValueError("Error in hedges params dictionary")
                exit(1)

            if trellis=="base":
                #basic trellis with base constraint information taken into account
                decoder = HedgesBonitoBase(hedges_params_dict,hedges_bytes,using_hedges_DNA_constraint,alphabet,"cuda:0","CTC",window=int(window))
            elif trellis=="mod":
                #trellis that accounts for constraints by using the modulo state of contexts
                decoder = HedgesBonitoDelayStates(hedges_params_dict,hedges_bytes,True,alphabet,"cuda:0","CTC",window=int(window),mod_states=mod_states)
            elif "beam" in trellis:
                #running beam trellis
                decoder = HedgesBonitoBeam(hedges_params_dict,hedges_bytes,using_hedges_DNA_constraint,alphabet,"cuda:0","CTC",trellis)
            else:
                raise ValueError("Trellis name error")

            logger.info("Batch size {}".format(alignment_scores.size(0)))
            #create aligner
            aligner = AlignCTCGPU(alphabet,device="cuda:0")
            f_endpoint_upper_index=torch.full((alignment_scores.size(0),),0,dtype=torch.long)
            r_endpoint_lower_index=torch.full((alignment_scores.size(0),),alignment_scores.size(1),dtype=torch.long)
            f_endpoint_score=torch.full((alignment_scores.size(0),),Log.one,dtype=torch.float32)
            r_endpoint_score=torch.full((alignment_scores.size(0),),Log.one,dtype=torch.float32)
            if len(endpoint_seq)>0:
                f_endpoint_index,f_endpoint_score  = aligner.align(alignment_scores,endpoint_seq)
                r_endpoint_index,r_endpoint_score  = aligner.align(alignment_scores,reverse_complement(endpoint_seq))

            f_hedges_index,f_hedges_score = aligner.align(alignment_scores,decoder.fastforward_seq[::-1])
            r_hedges_index,r_hedges_score = aligner.align(alignment_scores,complement(decoder.fastforward_seq))

            #logger.info("f hedges score {}".format(f_hedges_score))
            #logger.info("f endpoint score {}".format(f_endpoint_score))
            #logger.info("r hedges score {}".format(r_hedges_score))
            #logger.info("r endpoint score {}".format(r_endpoint_score))


            forward_scores = Log.mul(f_hedges_score,f_endpoint_score)
            #logger.info("f score: {}".format(forward_scores))
            reverse_scores = Log.mul(r_hedges_score,r_endpoint_score)
            #logger.info("r score: {}".format(reverse_scores))

            #logger.info("f hedges_index {}".format(f_hedges_index))
            #logger.info("r hedges_index {}".format(r_hedges_index))
            #logger.info("r endpoint index {}".format(r_endpoint_index))

            reverse_tensor= torch.argmax(torch.stack([forward_scores,reverse_scores]),dim=0).to(torch.bool)
            
            #logger.info("Reverse Tensor: {}".format(reverse_tensor))

            np_reverse = reverse_tensor.numpy()
            after_alignment_scores=[]
            for i in range(reverse_tensor.size(0)):
                if not np_reverse[i]:
                    after_alignment_scores.append(alignment_scores[i,int(f_endpoint_index[i,1]):int(f_hedges_index[i,1]),:].flip(0))
                else:
                    after_alignment_scores.append(alignment_scores[i,int(r_hedges_index[i,0]):int(r_endpoint_index[i,0]),:])
            max_score_length = max((s.size(0) for s in after_alignment_scores))
            padded_scores = (torch.nn.functional.pad(s,(0,0,0,max_score_length-s.size(0)),value=-1000) for s in after_alignment_scores)
            viterbi_scores = torch.stack(list(padded_scores)) #viterbi scores should hold all scores now, padded
            logger.info("Score tensor after align: {}".format(viterbi_scores.size()))

            if window>0 and window<1: decoder.window=int(window*viterbi_scores.size(1)/2) # 1 window for all scores
            seq_batch = decoder.decode(viterbi_scores,reverse_tensor)

            #try to clean up memory
            return [{'sequence':seq,'qstring':"*"*len(seq),'stride':stride,'moves':seq} for seq in seq_batch]
    except Exception as e:
        traceback.print_exc()
        seq="N"
        print("\n\nOffending read: {}\n\n".format(read_id),file=sys.stderr)
        print(e,file=sys.stderr)
        return [{'sequence':seq,'qstring':"*"*len(seq),'stride':stride,'moves':seq}]

        

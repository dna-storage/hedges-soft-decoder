"""
Bonito basecall
"""

import torch
import numpy as np
from functools import partial

from bonito.multiprocessing import process_map
from bonito.util import mean_qscore_from_qstring
from bonito.util import chunk, stitch, batchify, unbatchify, permute

from bonito.hedges_decode.hedges_decode import hedges_decode

import pickle
import sys
import gc
import os
from itertools import islice




def basecall(model, reads, beamsize=5, chunksize=0, overlap=0, batchsize=1, qscores=False, reverse=None,**kwargs):
    """
    Basecalls a set of reads.
    """
    chunks = (
        (read, chunk(torch.tensor(read.signal), chunksize, overlap)) for read in reads
    )
    scores = unbatchify(
        (k, compute_scores(model, v)) for k, v in batchify(chunks, batchsize)
    )
    scores = (
        (read, {'scores': stitch(v, chunksize, overlap, len(read.signal), model.stride)}) for read, v in scores
    )

    

    if kwargs.get("hedges_params",None)!=None:
        alphabet=model.alphabet
        decoder = partial(hedges_decode,hedges_params = kwargs["hedges_params"],hedges_bytes=kwargs["hedges_bytes"],
                          using_hedges_DNA_constraint=kwargs["hedges_using_DNA_constraint"],alphabet=alphabet,endpoint_seq=kwargs["strand_pad"],window=kwargs["window"],
                          trellis=kwargs["trellis"],mod_states=kwargs["mod_states"])
    else:
        decoder = partial(decode, decode=model.decode, beamsize=beamsize, qscores=qscores, stride=model.stride)
    #pickle.dump([(read,s['scores']) for read,s in scores],open("debug_scores","wb+"))
    basecalls = process_map(decoder, scores, n_proc=1)
    return basecalls


def compute_scores(model, batch):
    """
    Compute scores for model.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        chunks = batch.to(torch.float32).to(device)
        probs = permute(model(chunks), 'TNC', 'NTC')
    return probs.cpu().to(torch.float32)


def decode(scores, decode, beamsize=5, qscores=False, stride=1):
    """
    Convert the network scores into a sequence.
    """
    # do a greedy decode to get a sensible qstring to compute the mean qscore from
    seq, path = decode(scores['scores'], beamsize=1, qscores=True, return_path=True)
    seq, qstring = seq[:len(path)], seq[len(path):]
    mean_qscore = mean_qscore_from_qstring(qstring)

    # beam search will produce a better sequence but doesn't produce a sensible qstring/path
    if not (qscores or beamsize == 1):
        try:
            seq = decode(scores['scores'], beamsize=beamsize)
            path = None
            qstring = '*'
        except:
            pass

    return {'sequence': seq, 'qstring': qstring, 'stride': stride, 'moves': path}


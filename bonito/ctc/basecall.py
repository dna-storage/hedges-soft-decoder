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
from bonito.hedges_decode.hedges_decode_utils import hedges_batch_scores,unpack_basecalls

import pickle
import sys
import gc
import os
from itertools import islice

import h5py

def dump_ctc(scores:np.ndarray,is_reverse:bool,read_id:str,h5_file:h5py.File):
    assert h5_file.name=="/"
    read_group = h5_file.create_group(read_id.read_id)
    read_group.create_dataset("reverse",data=is_reverse)
    read_group.create_dataset("scores",data=scores,chunks=True,compression="gzip")
    


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
    ctc_dump=None
    if kwargs.get("hedges_params",None)!=None:
        alphabet=model.alphabet
        if kwargs["ctc_dump"]:
            if os.path.exists(kwargs["ctc_dump"]):os.remove(kwargs["ctc_dump"])
            ctc_dump=h5py.File(kwargs["ctc_dump"],'w')
        decoder = partial(hedges_decode,hedges_params = kwargs["hedges_params"],hedges_bytes=kwargs["hedges_bytes"],
                          using_hedges_DNA_constraint=kwargs["hedges_using_DNA_constraint"],alphabet=alphabet,endpoint_seq=kwargs["strand_pad"],window=kwargs["window"],
                          trellis=kwargs["trellis"],mod_states=kwargs["mod_states"],ctc_dump=not(ctc_dump is None))
        scores=hedges_batch_scores(scores,kwargs["batch_size"])
    else:
        decoder = partial(decode, decode=model.decode, beamsize=beamsize, qscores=qscores, stride=model.stride)
    basecalls = process_map(decoder, scores, n_proc=kwargs["processes"])
    if ctc_dump:
        for k,v in basecalls:
            print("Finished CTC DUMP on {}".format(v[-1].read_id)) 
            dump_ctc(*v,ctc_dump) #intercept basecaller response and get the ctc dumps
            sys.stdout.flush()
        ctc_dump.close()
    basecalls = unpack_basecalls(basecalls)
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


def decode(key,scores, decode, beamsize=5, qscores=False, stride=1):
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


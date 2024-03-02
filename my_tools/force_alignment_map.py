import torch
import h5py
import os
import numpy as np
from bonito.hedges_decode.align import *
from bonito.hedges_decode.ctc_analysis_utils import *

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tool to subset reads from tar ball')
    parser.add_argument('-f', '--scores_file',
                        help="Input hdf5 file that has scores we will align to", type=str, required=True, default=None)
    parser.add_argument('-o', '--out_path',
                        help="Output path for the alignments",type=str,required=False,default=os.getcwd())
    parser.add_argument('-s', '--sequences',
                        help="path to file that specifies a .dna file",type=str,required=True,default=None)
    parser.add_argument('-i','--index',nargs="+",type=int,help="Index of strand from .dna file to use as the target")
    args = parser.parse_args()
  
    strand=None
    with open(args.sequences,"r") as dna_file:
        for l in dna_file:
            if "%" not in l:
                index = l.strip().split(":")[0].strip("(").strip(")").split(",")
                index = tuple((int(_) for _ in index))
                if index==tuple(args.index):
                    strand = l.strip().split(":")[1].strip()
                    break
    assert strand is not None

    #if os.path.exists(args.out_path): os.remove(args.out_path) #remove old hdf5 file

    scores_h5 = h5py.File(args.scores_file,mode="r")
    out_h5 = h5py.File(args.out_path,mode='w')
    
    out_h5.create_dataset("strand",data=string_to_num(strand))
    
    #allocate alignment object
    aligner = LongStrandAlignCTCGPU(alphabet,"cuda:0")

    #now we need to do some forced alignments using our strand
    for read_count,read_id in enumerate(scores_h5.keys()):
        print("Aligning Read {}".format(read_id))
        working_strand = strand
        read_group = scores_h5[read_id]
        reverse = np.array(scores_h5[read_id]["reverse"])
        if reverse:
            working_strand=reverse_strand(strand)
        print("Working Strand {}".format(working_strand))
        #get scores, convert to torch and place on GPU
        scores = np.array(scores_h5[read_id]["scores"])
        #print("Score Shape {}".format(scores.shape))
        torch_scores = torch.from_numpy(scores)
        torch_scores.to("cuda:0")
        try:
            alignment = aligner.align(torch_scores,working_strand,remove_end_blanks=True)
        except:
            alignment=Alignment()
        #alignment we are looking for here should be long as the scores, and will be a CTC encoding 

        read_g = out_h5.create_group(read_id)
        read_g.create_dataset("alignment",data=alignment.ctc_encoding,dtype=np.uint8)
        print("CTC Alignment {}\n".format(num_to_string(alignment.ctc_encoding)))
        #print("CTC Alignment Length {}".format(len(num_to_string(alignment.ctc_encoding))))
        print("CTC Decoded {}\n".format(CTC_decode(num_to_string(alignment.ctc_encoding))))
        read_g.create_dataset("aligment_start",data=alignment.alignment_start,dtype=np.uint32)
        read_g.create_dataset("alignment_end",data=alignment.alignment_end, dtype=np.uint32)
        read_g.create_dataset("alignment_score",data=alignment.alignment_score,dtype=np.float32)
        #print("Alignment Score {}\n\n\n".format(alignment.alignment_score))
        #if read_count>40: break
    out_h5.close()
    scores_h5.close()

    

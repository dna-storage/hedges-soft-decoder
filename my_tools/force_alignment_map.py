import torch
import h5py
import os
import numpy as np
from bonito.hedges_decode.align import LongStrandAlignCTCGPU

alphabet = ["N","A","C","G","T"] #TODO: should confirm this
alphabet_num_map = {"N":0,"A":1,"C":2,"G":3,"T":4}


def reverse_strand(s:str):
    return_string = ""
    d = {"A":"T","T":"A","C":"G","G":"C"}
    for i in s:
        return_string = return_string+d[i]
    return return_string

def CTC_to_numpy(s:str): #converts alignment string to a numpy array #TODO: maybe won't need this 
    return_array = np.zeros((len(s),),dtype=np.uint8)
    for index,i in enumerate(s):
        return_array[index]=alphabet_num_map[i]
    return return_array

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tool to subset reads from tar ball')
    parser.add_argument('-f', '--scores_file',
                        description="Input hdf5 file that has scores we will align to", type=str, required=True, default=None)
    parser.add_argument('-o', '--out_path',
                        description="Output path for the alignments",type=str,required=False,default=os.getcwd())
    parser.add_argument('-s', '--sequences',
                        description="path to file that specifies a .dna file",type=str,required=True,default=None)
    parser.add_argument('-i','--index',nargs="+",type=int,help="Index of strand from .dna file to use as the target")
    args = parser.parse_args()
  
    strand=None
    with open(args.sequences,"r") as dna_file:
        for l in dna_file:
            if "%" not in l:
                index = l.strip().split(":")[0].strip("(").strip(")").split(",")
                index = tuple((int(_) for _ in index))
                if index==tuple(args.index):
                    strand = l.strip().split(":")[1]
                    break
    assert strand is not None

    #if os.path.exists(args.out_path): os.remove(args.out_path) #remove old hdf5 file

    scores_h5 = h5py.File(args.scores_file,mode="r")
    out_h5 = h5py.File(args.out_file,mode='w')
    
    
    #allocate alignment object
    aligner = LongStrandAlignCTCGPU(alphabet,"cuda:0")

    #now we need to do some forced alignments using our strand
    for read_id in scores_h5.keys():
        print("Aligning Read {}".format(read_id))
        working_strand = strand
        read_group = out_h5[read_id]
        reverse = out_h5[read_id]["reverse"]
        if reverse:
            working_strand=reverse_strand(strand)
        #get scores, convert to torch and place on GPU
        scores = out_h5[read_id]["scores"]
        torch_scores = torch.from_numpy(scores)
        torch_scores.to("cuda:0")
        alignment = aligner.align(torch_scores,working_strand,remove_end_blanks=True)
        
        #alignment we are looking for here should be long as the scores, and will be a CTC encoding 
        #array = CTC_to_numpy(alignment.ctc_encoding)

        read_g = out_h5.create_group(read_id)
        read_g.create_dataset("alignment",data=alignment.ctc_encoding,dtype=np.uint8)

        read_g.create_dataset("aligment_start",data=alignment.start,dtype=np.uint32)
        read_g.create_dataset("alignment_end",data=alignment.end, dtype=np.uint32)
        read_g.create_dataset("alignment_score",data=alignment.score,dtype=np.float32)
    out_h5.close()
    scores_h5.close()

    
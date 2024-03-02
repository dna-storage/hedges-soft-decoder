import os

"""
Script to launch jobs for data obtained from paul in the 231220 data set
"""

import os
import pickle
import random
import re


byte_multiplier_map={
    "231206_dna_datastorage_1667_batch_full_1667":8,
    "231206_dna_datastorage_1667_batch_full_1667_s1":8,
    "231206_dna_datastorage_1667_batch_full_1667_s2":8,
    "231206_dna_datastorage_1667_batch_half_1667":16,
    "231206_dna_datastorage_1667_batch_quarter_1667":8,
    "231207_dna_datastorage_others_fl_batch_1250_fl":64,
    "231207_dna_datastorage_others_fl_batch_3333_fl":64,
    "231207_dna_datastorage_others_fl_batch_5000_fl":64    
}


def build_command(format_dict):
    return "sbatch --time \"5:00:00\"  --exclude=\"c32,c34,c[58-59],c68\" --exclusive -J {sim} -o {out_file} -e {err_file} -n 1  -p {GPU}   --wrap \"source ~/.bashrc && conda activate bonito_cuda && bonito basecaller dna_r9.4.1@v2 {fast5_path} --disable_koi --strand_pad GGCGACAGAAGAGTCAAGGTTC --hedges_bytes 204 {strand_byte} --hedges_params {hedges_params} --disable_half --trellis base --processes 1 --lower_index {lower_index} --upper_index {upper_index} --batch_size 50\"".format(**format_dict)



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="convert map to fastq")
    parser.add_argument('--batch_dir', default="/mnt/beegfs/kvolkel/221118_nanopore_dna/231220_experimental_fast5/dna/",
                        required=True,help="top path of batches which we will use for running ctc on")
    parser.add_argument('--fast5_dir', default="fast5_10k",required=True,help="fast 5 data set size to use")
    parser.add_argument("--lower_index",default=0,type=int,help="lower index to use on fast5")
    parser.add_argument("--upper_index",default=10000,type=int,help="upper index to use on fast5")
    parser.add_argument("--prefix",default="",type=str,help="prefix to add to output filesm, and fast5 directory")
    parser.add_argument("--gpu",default="rtx2060super",type=str,help="GPU model to use")
    parser.add_argument("--params_path",default="/home/kvolkel/231220_experiment_hedges_options/",type=str,help="options directory")
    parser.add_argument("--target_batch",default=None,nargs="+",help="Target batch to go for")
    args = parser.parse_args()
    for batch in os.listdir(args.batch_dir):
        if not args.target_batch is None:
            if batch not in args.target_batch:continue
        batch_path=os.path.join(args.batch_dir,batch)
        if not os.path.isdir(batch_path): continue
        fast5_path="{}{}".format(args.prefix,args.fast5_dir)
        out_dirname="{}hedges_{}".format(args.prefix,args.fast5_dir.replace("fast5","fastq"))
        fast5_path=os.path.join(batch_path,fast5_path)
        out_dirpath=os.path.join(batch_path,out_dirname)
        if not os.path.exists(out_dirpath):
            #print(out_dirpath)
            os.mkdir(out_dirpath)
        if not os.path.exists(fast5_path): continue
        for strand_f5 in os.listdir(fast5_path):
            if ".fast5" not in strand_f5: continue
            g=re.search("strand([0-9]+)",strand_f5)
            strand_ID=int(g[1])
            #if strand_ID!=2: continue
            #print(strand_ID)
            strand_byte=(strand_ID-1)*byte_multiplier_map[batch] #strand_ID-1 to get to 0 index base
            out_file=args.prefix+batch+"_"+strand_f5.split(".")[0]+".fastq"
            err_file=args.prefix+batch+"_"+strand_f5.split(".")[0]+".err"
            strand_fast5_path=os.path.join(fast5_path,strand_f5)
            assert os.path.isdir(strand_fast5_path) and os.path.exists(strand_fast5_path)
            assert os.path.isfile(os.path.join(args.params_path,batch+".json")) and os.path.exists(os.path.join(args.params_path,batch+".json"))
            format_dict={
                "sim":batch,
                "out_file":os.path.join(out_dirpath,out_file),
                "err_file":os.path.join(out_dirpath,err_file),
                "GPU":args.gpu,
                "fast5_path":strand_fast5_path,
                "strand_byte":str(strand_byte),
                "lower_index":str(args.lower_index),
                "upper_index":str(args.upper_index),
                "hedges_params":os.path.join(args.params_path,batch+".json")
            }
            command=build_command(format_dict)
            print(command)
            os.system(command)

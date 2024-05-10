import bonito.hedges_decode.hedges_decode as hd
from bonito.hedges_decode.hedges_decode_utils import hedges_batch_scores
import logging
import time
import argparse

if __name__ =="__main__":
    import pickle
    import sys
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('scores')
    parser.add_argument('hedges_parameters')
    parser.add_argument("--trellis",default="base",type=str,help="Trellis type")
    parser.add_argument("--window_size",default=0,type=float,help="Window Size")
    parser.add_argument("--mod_states",default=0,type=int,help="mod states to use, only does something when trellis==mod")
    parser.add_argument("--batch",default=1,type=int,help="number of strands to batch together")
    parser.add_argument("--rna",default=False,action="store_true",help="Use this option if you want decoding to occur based on RNA strands")
    args=parser.parse_args()


    root=logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    logger=logging.getLogger(__name__)
    debug_data = pickle.load(open(args.scores,"rb"))
    alphabet = ["N","A","C","G","T"]
    b = bytes([204,0])
    endpoint_str="GGCGACAGAAGAGTCAAGGTTC"
    #quick debug of decoding
    benchmark_start_time = time.perf_counter()
    counter=0
    for read,scores in hedges_batch_scores(debug_data,args.batch):
        logger.info("Decoding Read: {}".format(read))
        logger.info("Batch Set Size: {}".format(scores["scores"].size()))
        time_start=time.perf_counter()
        batch=hd.hedges_decode(read,scores,args.hedges_parameters,b,False,alphabet,1,endpoint_str,window=args.window_size,
                               trellis=args.trellis,mod_states=args.mod_states,rna=args.rna)
        time_end=time.perf_counter()
        logger.info("Batch Completed in: {} seconds".format(time_end-time_start))
        for x in batch:
            logger.info(x['sequence'])
        counter+=1
        if counter==401: exit(0)
    logger.info("Benchmark Completed in: {} seconds".format(time.perf_counter()-benchmark_start_time))
        

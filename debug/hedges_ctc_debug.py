import bonito.hedges_decode.hedges_decode as hd
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
    for read,scores in debug_data:
        logger.info("Decoding Read: {}".format(read))
        logger.info(scores.size())
        time_start=time.time()
        x=hd.hedges_decode(read,{"scores":scores},args.hedges_parameters,b,False,alphabet,1,endpoint_str,window=args.window_size,
                           trellis=args.trellis,mod_states=args.mod_states)
        time_end=time.time()
        logger.info("Read Completed in: {} seconds".format(time_end-time_start))
        logger.info(x['sequence'])
        

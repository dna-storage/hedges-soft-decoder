import bonito.hedges_decode.hedges_decode as hd
import logging
import time

if __name__ =="__main__":
    import pickle
    import sys
    import json

    root=logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    logger=logging.getLogger(__name__)

    assert(len(sys.argv)==3)
    debug_data = pickle.load(open(sys.argv[1],"rb"))
    alphabet = ["N","A","C","G","T"]
    b = bytes([204,0])
    endpoint_str="GGCGACAGAAGAGTCAAGGTTC"
    #quick debug of decoding
    for read,scores in debug_data:
        logger.info("Decoding Read: {}".format(read))
        logger.info(scores.size())
        time_start=time.time()
        x=hd.hedges_decode(read,{"scores":scores},sys.argv[2],b,False,alphabet,1,endpoint_str,window=0,trellis="base")
        time_end=time.time()
        logger.info("Read Completed in: {} seconds".format(time_end-time_start))
        logger.info(x['sequence'])
        

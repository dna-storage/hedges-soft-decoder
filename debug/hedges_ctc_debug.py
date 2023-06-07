import bonito.hedges_decode as hd

if __name__ =="__main__":
    import pickle
    import sys
    import json
    assert(len(sys.argv)==3)
    debug_data = pickle.load(open(sys.argv[1],"rb"))
    alphabet = ["N","A","C","G","T"]
    b = bytes([204,0])
    endpoint_str="GGCGACAGAAGAGTCAAGGTTC"
    #quick debug of decoding
    for read,scores in debug_data:
        print("Decoding Read {}".format(read))
        print(scores.size())
        x=hd.hedges_decode(read,{"scores":scores},sys.argv[2],b,False,alphabet,1,endpoint_str)
        print(x['sequence'])

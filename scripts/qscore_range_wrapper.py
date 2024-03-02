"""
Script that wraps all other workflow scripts in order to facilitate qscore subsetting
"""
import numpy as np 
import os
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a command for a range of Qscores")
    parser.add_argument("--range",nargs="+",required=True,type=np.float64,help="range to use for qscores")
    parser.add_argument("--step",type=np.float64,required=True,help="step to use in the qscore range")
    parser.add_argument("--command",type=str,required=True,help="Full command with parameters other than the prefix to run")
    args = parser.parse_args()

    assert len(args.range)==2
    assert args.step>0
    print(np.round((args.range[1]-args.range[0])/args.step))
    endpoints=np.linspace(args.range[0],args.range[1],int(np.round((args.range[1]-args.range[0])/args.step)+1))
    print("Endpoints {}".format(list(endpoints)))

    for i,_ in enumerate(endpoints):
        if i==(len(endpoints)-1): break
        q_low=endpoints[i]
        q_high=endpoints[i+1]
        prefix="QSCORE_{:.1f}_{:.1f}_".format(q_low,q_high)
        assert "--prefix" not in args.command
        command = "{command} --prefix {prefix}".format(command=args.command,prefix=prefix)

        os.system(command)
        #print(command)

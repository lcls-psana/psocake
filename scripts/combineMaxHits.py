# Combines maxHits.npy files for multiple runs for better virtual powder sum
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e','--exp', help="experiment name (e.g. cxic0415)", type=str)
parser.add_argument('-r','--runs', help="start and end run (e.g. 24 28)", nargs='+', type=int)
parser.add_argument('-p','--path', help="path to psocake analysis folder (e.g. '/reg/d/psdm/mfx/mfxp17318/scratch/yoon82/psocake')", type=str)
args = parser.parse_args()

## Modify these lines ##
expName=args.exp
if not len(args.runs) == 2:
    print("Provide start and end run numbers, e.g. -r 24 28")
    exit()
runs=np.arange(args.runs[0],args.runs[1])
if not os.path.exists(args.path):
    print("Path provided does not exist")
    exit()
path=args.path
########################

calib = None
for val in runs:
    p = os.path.join(path,"r"+str(val).zfill(4),expName+"_"+str(val).zfill(4)+"_maxHits.npy")
    if os.path.exists(p):
        if calib is None:
            calib = np.load(p)
        else:
            _c = np.load(p)
            print("Combining maxHits from run ", val)
            calib = np.maximum(calib,_c)
fname = os.path.join(path,expName+"_"+str(runs[0])+"_"+str(runs[-1])+"_maxHits.npy")
np.save(fname, calib)
print("Saved: ", fname)
fname = os.path.join(path,expName+"_"+str(runs[0])+"_"+str(runs[-1])+"_maxHits.txt")
np.savetxt(fname, calib.reshape((-1,calib.shape[-1])))
print("Saved: ", fname)
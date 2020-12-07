# Combines maxHits.npy files for multiple runs for better virtual powder sum
import numpy as np
import os

## Modify these lines ##
expName='cxih0115'
runs=np.arange(18,39)
path='/reg/data/ana03/scratch/yoon82/psocake/cxih0115/cxih0115/yoon82/psocake'
########################

calib = None
for val in runs:
    p = os.path.join(path,"r"+str(val).zfill(4),expName+"_"+str(val).zfill(4)+"_maxHits.npy")
    if os.path.exists(p):
        if calib is None:
            calib = np.load(p)
        else:
            _c = np.load(p)
            calib = np.maximum(calib,_c)
fname = os.path.join(path,expName+"_"+str(runs[0])+"_"+str(runs[-1])+"_maxHits.npy")
np.save(fname, calib)
fname = os.path.join(path,expName+"_"+str(runs[0])+"_"+str(runs[-1])+"_maxHits.txt")
np.savetxt(fname, calib.reshape((-1,calib.shape[-1])))

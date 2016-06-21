#!/usr/bin/env python
from xtc2cxidbMaster import runmaster
from xtc2cxidbClient import runclient
import h5py
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'At least 2 MPI ranks required'
numClients = size-1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-e","--exp", help="psana experiment name (e.g. cxic0415)", type=str)
parser.add_argument("-r","--run", help="psana run number (e.g. 15)", type=int)
parser.add_argument("-d","--det",help="psana detector name (e.g. DscCsPad)", type=str)
parser.add_argument("-i","--inDir",help="input directory where files_XXXX.lst exists (e.g. /reg/d/psdm/cxi/cxic0415/scratch)", type=str)
parser.add_argument("-o","--outDir",help="output directory (e.g. /reg/d/psdm/cxi/cxic0415/scratch)", type=str)
parser.add_argument("--sample",help="sample name (e.g. lysozyme)",default='', type=str)
parser.add_argument("--instrument",help="instrument name (e.g. CXI)", type=str)
parser.add_argument("--clen", help="camera length epics name (e.g. CXI:DS1:MMS:06.RBV or CXI:DS2:MMS:06.RBV)", type=str)
parser.add_argument("--coffset", help="camera offset, CXI home position to sample (m)",default=0, type=float)
parser.add_argument("--detectorDistance", help="detector distance from interaction point (m)",default=0, type=float)
parser.add_argument("--cxiVersion", help="cxi version",default=140, type=int)
parser.add_argument("--pixelSize", help="pixel size (m)", type=float)
parser.add_argument("--condition",help="comparator operation for choosing input data from an hdf5 dataset."
                                       "Must be double quotes and comma separated for now."
                                       "Available comparators are: gt(>), ge(>=), eq(==), le(<=), lt(<)"
                                       "e.g. /particleSize/corrCoef,ge,0.85 ",default='', type=str)
args = parser.parse_args()
print "123"
print "inDir: ", args.inDir

# Read list of files
runStr = "%04d" % args.run
filename = args.inDir+'/'+args.exp+'_'+runStr+'.cxi'
print "Reading file: %s" % (filename)

f = h5py.File(filename, "r+")

if "/status/xtc2cxidb" in f:
    del f["/status/xtc2cxidb"]
f["/status/xtc2cxidb"] = 'fail'

# Condition:
if args.condition:
    print "got here"
    import operator
    operations = {"lt":operator.lt,
                  "le":operator.le,
                  "eq":operator.eq,
                  "ge":operator.ge,
                  "gt":operator.gt,}

    s = args.condition.split(",")
    ds = s[0] # hdf5 dataset containing metric
    comparator = s[1] # operation
    cond = float(s[2]) # conditional value

    metric = f[ds].value
    hitInd = np.argwhere(operations[comparator](metric,cond))
    numHits = len(hitInd)
else:
    numHits = len(f["/entry_1/result_1/nPeaksAll"])
    hitInd = np.arange(numHits)
nPeaks = f["/entry_1/result_1/nPeaksAll"].value
posX = f["/entry_1/result_1/peakXPosRawAll"].value
posY = f["/entry_1/result_1/peakYPosRawAll"].value
atot = f["/entry_1/result_1/peakTotalIntensityAll"].value
f.close()

#if rank==0:
#    runmaster(args,numClients)
#else:
#    runclient(args)

MPI.Finalize()

#state = {'hitInd': hitInd,
         #'nPeaks': nPeaks,
         #'posX': posX,
         #'posY': posY,
         #'atot': atot,
#         'dim0': dim0,
#         'dim1': dim1}
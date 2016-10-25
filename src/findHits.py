# Find pixels with photons
from hitFinderMaster import runmaster
from hitFinderClient import runclient

import h5py, psana
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'At least 2 MPI ranks required'
numClients = size-1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-e','--exp', help="experiment name (e.g. cxic0415)", type=str)
parser.add_argument('-r','--run', help="run number (e.g. 24)", type=int)
parser.add_argument("-d","--detectorName",help="psana detector alias, (e.g. pnccdBack, DsaCsPad)", type=str)
parser.add_argument("-o","--outDir",help="output directory",default="", type=str)
parser.add_argument("-n","--noe",help="number of events, all events=-1",default=-1, type=int)
parser.add_argument("-a","--algorithm",help="algorithm number",default=2, type=int)
parser.add_argument("-p","--pruneInterval",help="number of events to update running background",default=-1, type=float)
parser.add_argument("-l","--litPixelThreshold",help="number of ADUs to be considered a lit pixel",default=-1, type=float)
parser.add_argument("--userMask_path",help="full path to user mask numpy array",default=None, type=str)
parser.add_argument("--streakMask_on",help="streak mask on",default="False", type=str)
parser.add_argument("--streakMask_sigma",help="streak mask sigma above background",default=0., type=float)
parser.add_argument("--streakMask_width",help="streak mask width",default=0, type=float)
parser.add_argument("--psanaMask_on",help="psana mask on",default="False", type=str)
parser.add_argument("--psanaMask_calib",help="psana calib on",default="False", type=str)
parser.add_argument("--psanaMask_status",help="psana status on",default="False", type=str)
parser.add_argument("--psanaMask_edges",help="psana edges on",default="False", type=str)
parser.add_argument("--psanaMask_central",help="psana central on",default="False", type=str)
parser.add_argument("--psanaMask_unbond",help="psana unbonded pixels on",default="False", type=str)
parser.add_argument("--psanaMask_unbondnrs",help="psana unbonded pixel neighbors on",default="False", type=str)
parser.add_argument("-v","--verbose",help="verbosity of output for debugging, 1=print, 2=print+plot",default=0, type=int)
parser.add_argument("--localCalib", help="use local calib directory, default=False", action='store_true')
args = parser.parse_args()

def getNoe(args):
    runStr = "%04d" % args.run
    ds = psana.DataSource("exp="+args.exp+":run="+runStr+':idx')
    run = ds.runs().next()
    times = run.times()
    # check if the user requested specific number of events
    if args.noe == -1:
        numJobs = len(times)
    else:
        if args.noe <= len(times):
            numJobs = args.noe
        else:
            numJobs = len(times)
    return numJobs

if args.localCalib: psana.setOption('psana.calib-dir','./calib')

if rank == 0:
    runStr = "%04d" % args.run
    fname = args.outDir +"/"+ args.exp +"_"+ runStr + ".cxi"
    # Get number of events to process
    numJobs = getNoe(args)

    # Create hdf5 and save psana input
    myHdf5 = h5py.File(fname, 'w')
    myHdf5['/status/findHits'] = 'fail'
    dt = h5py.special_dtype(vlen=bytes)
    myInput = ""
    for key,value in vars(args).iteritems():
        myInput += key
        myInput += " "
        myInput += str(value)
        myInput += "\n"
    dset = myHdf5.create_dataset("/psana/input",(1,), dtype=dt)
    dset[...] = myInput

    grpName = "/entry_1/result_1"
    dset_nHits = "/nHitsAll"
    if grpName in myHdf5:
        del myHdf5[grpName]
    myHdf5.flush()
    grp = myHdf5.create_group(grpName)
    myHdf5.create_dataset(grpName+dset_nHits, data=np.ones(numJobs,)*-1, dtype='int')
    myHdf5.flush()
    myHdf5.close()

comm.Barrier()

if rank==0:
    runmaster(args,numClients)
else:
    runclient(args)

MPI.Finalize()

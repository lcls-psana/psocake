# Find Bragg peaks
from peakFinderMaster import runmaster
from peakFinderClient import runclient

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
parser.add_argument('-d','--det', help="detector name (e.g. pnccdFront)", type=str)
parser.add_argument('-o','--outDir', help="output directory where .cxi will be saved (e.g. /reg/d/psdm/cxi/cxic0415/scratch)", type=str)
parser.add_argument("-p","--imageProperty",help="determines what preprocessing is done on the image",default=1, type=int)
parser.add_argument("--algorithm",help="number of events to process",default=1, type=int)
parser.add_argument("--alg_npix_min",help="number of events to process",default=1., type=float)
parser.add_argument("--alg_npix_max",help="number of events to process",default=45., type=float)
parser.add_argument("--alg_amax_thr",help="number of events to process",default=250., type=float)
parser.add_argument("--alg_atot_thr",help="number of events to process",default=330., type=float)
parser.add_argument("--alg_son_min",help="number of events to process",default=10., type=float)
parser.add_argument("--alg1_thr_low",help="number of events to process",default=80., type=float)
parser.add_argument("--alg1_thr_high",help="number of events to process",default=270., type=float)
parser.add_argument("--alg1_radius",help="number of events to process",default=3, type=int)
parser.add_argument("--alg1_dr",help="number of events to process",default=1., type=float)
parser.add_argument("--alg3_rank",help="number of events to process",default=3, type=int)
parser.add_argument("--alg3_r0",help="number of events to process",default=5., type=float)
parser.add_argument("--alg3_dr",help="number of events to process",default=0.05, type=float)
parser.add_argument("--alg4_thr_low",help="number of events to process",default=10., type=float)
parser.add_argument("--alg4_thr_high",help="number of events to process",default=150., type=float)
parser.add_argument("--alg4_rank",help="number of events to process",default=3, type=int)
parser.add_argument("--alg4_r0",help="number of events to process",default=5, type=int)
parser.add_argument("--alg4_dr",help="number of events to process",default=0.05, type=float)
parser.add_argument("--streakMask_on",help="streak mask on",default="False", type=str)
parser.add_argument("--streakMask_sigma",help="streak mask sigma above background",default=0., type=float)
parser.add_argument("--streakMask_width",help="streak mask width",default=0, type=float)
parser.add_argument("--userMask_path",help="full path to user mask numpy array",default=None, type=str)
parser.add_argument("--psanaMask_on",help="psana mask on",default="False", type=str)
parser.add_argument("--psanaMask_calib",help="psana calib on",default="False", type=str)
parser.add_argument("--psanaMask_status",help="psana status on",default="False", type=str)
parser.add_argument("--psanaMask_edges",help="psana edges on",default="False", type=str)
parser.add_argument("--psanaMask_central",help="psana central on",default="False", type=str)
parser.add_argument("--psanaMask_unbond",help="psana unbonded pixels on",default="False", type=str)
parser.add_argument("--psanaMask_unbondnrs",help="psana unbonded pixel neighbors on",default="False", type=str)
parser.add_argument("-m","--maxNumPeaks",help="maximum number of peaks to store per event",default=2048, type=int)
parser.add_argument("-n","--noe",help="number of events to process",default=-1, type=int)
parser.add_argument("--medianBackground",help="subtract median background",default=0, type=int)
parser.add_argument("--medianRank",help="median background window size",default=0, type=int)
parser.add_argument("--radialBackground",help="subtract radial background",default=0, type=int)
parser.add_argument("--distance",help="detector distance used for radial background",default=0, type=float)
parser.add_argument("--localCalib", help="Use local calib directory. A calib directory must exist in your current working directory.", action='store_true')

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
    myHdf5['/status/findPeaks'] = 'fail'
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
    dset_nPeaks = "/nPeaksAll"
    dset_posX = "/peakXPosRawAll"
    dset_posY = "/peakYPosRawAll"
    dset_atot = "/peakTotalIntensityAll"
    dset_maxRes = "/maxResAll"
    if grpName in myHdf5:
        del myHdf5[grpName]
    myHdf5.flush()
    grp = myHdf5.create_group(grpName)
    myHdf5.create_dataset(grpName+dset_nPeaks, data=np.ones(numJobs,)*-1, dtype='int')
    myHdf5.create_dataset(grpName+dset_posX, (numJobs,args.maxNumPeaks), dtype='float32', chunks=(1,args.maxNumPeaks))
    myHdf5.create_dataset(grpName+dset_posY, (numJobs,args.maxNumPeaks), dtype='float32', chunks=(1,args.maxNumPeaks))
    myHdf5.create_dataset(grpName+dset_atot, (numJobs,args.maxNumPeaks), dtype='float32', chunks=(1,args.maxNumPeaks))
    myHdf5.create_dataset(grpName+dset_maxRes, data=np.ones(numJobs,)*-1, dtype='int')
    myHdf5.flush()
    myHdf5.close()

comm.Barrier()

if rank==0:
    runmaster(args,numClients)
else:
    runclient(args)

MPI.Finalize()

# Find Bragg peaks
from peakFinderMaster import runmaster
from peakFinderClient import runclient
import h5py
import psana

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
parser.add_argument("-n","--noe",help="number of events to process",default=-1, type=int)
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
parser.add_argument("--streakMask_on",help="streak mask on",default=False, type=bool)
parser.add_argument("--streakMask_sigma",help="streak mask sigma above background",default=0., type=float)
parser.add_argument("--streakMask_width",help="streak mask width",default=0, type=float)
parser.add_argument("--userMask_path",help="full path to user mask numpy array",default=None, type=str)
parser.add_argument("--psanaMask_on",help="psana mask on",default=False, type=bool)
parser.add_argument("--psanaMask_calib",help="psana calib on",default=False, type=bool)
parser.add_argument("--psanaMask_status",help="psana status on",default=False, type=bool)
parser.add_argument("--psanaMask_edges",help="psana edges on",default=False, type=bool)
parser.add_argument("--psanaMask_central",help="psana central on",default=False, type=bool)
parser.add_argument("--psanaMask_unbond",help="psana unbonded pixels on",default=False, type=bool)
parser.add_argument("--psanaMask_unbondnrs",help="psana unbonded pixel neighbors on",default=False, type=bool)
parser.add_argument("-m","--maxNumPeaks",help="maximum number of peaks to store per event",default=2048, type=int)
args = parser.parse_args()

if rank==0:
    runmaster(args,numClients)
else:
    runclient(args)

MPI.Finalize()

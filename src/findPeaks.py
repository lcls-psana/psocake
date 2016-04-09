# bsub -q psanaq -a mympi -n 2 -o %J.log python findPeaks.py exp=cxic0415:run=24 -d DscCsPad
# mpirun -n 2 python findPeaks.py exp=cxic0415:run=24 -d DscCsPad
# FIXME: only works for cspad images
from psana import *
import numpy as np
import sys
from ImgAlgos.PyAlgos import PyAlgos # peak finding
import argparse
import h5py
import time
import myskbeam

class PeakFinder:
    def __init__(self,exp,run,detname,evt,detector,algorithm,hitParam_alg_npix_min,hitParam_alg_npix_max,
                 hitParam_alg_amax_thr,hitParam_alg_atot_thr,hitParam_alg_son_min,
                 windows=None,**kwargs):
        self.exp = exp
        self.run = run
        self.detname = detname
        self.evt = evt
        self.det = detector
        self.algorithm = algorithm
        self.windows = windows
        self.userMask = None
        self.psanaMask = None
        self.streakMask = None
        self.streak_sigma = None
        self.streak_width = None
        self.combinedMask = None

        # Mask user mask
        if kwargs["userMask_path"] is not None:
            self.userMask = np.load(kwargs["userMask_path"])

        # Make psana mask
        if kwargs["psanaMask_calib"] or kwargs["psanaMask_status"] or \
           kwargs["psanaMask_edges"] or kwargs["psanaMask_central"] or \
           kwargs["psanaMask_unbond"] or kwargs["psanaMask_unbondnrs"]:
            self.psanaMask = detector.mask(evt, calib=kwargs["psanaMask_calib"], status=kwargs["psanaMask_status"], edges=kwargs["psanaMask_edges"], central=kwargs["psanaMask_central"], unbond=kwargs["psanaMask_unbond"], unbondnbrs=kwargs["psanaMask_unbondnrs"])

        # Get streak mask parameters
        if kwargs["streakMask_sigma"] is not 0:
            self.streak_sigma = kwargs["streakMask_sigma"]
            self.streak_width = kwargs["strerakMask_width"]

        # Combine userMask and psanaMask
        if self.userMask is not None or self.psanaMask is not None:
            self.combinedMask = np.one_like(self.det.calib(self.evt))
        if self.userMask is not None:
            self.combinedMask *= self.userMask
        if self.psanaMask is not None:
            self.combinedMask *= self.psanaMask

        self.alg = PyAlgos(windows=self.windows, mask=self.combinedMask, pbits=0)
        # set peak-selector parameters:
        self.alg.set_peak_selection_pars(npix_min=hitParam_alg_npix_min, npix_max=hitParam_alg_npix_max, \
                                        amax_thr=hitParam_alg_amax_thr, atot_thr=hitParam_alg_atot_thr, \
                                        son_min=hitParam_alg_son_min)
        # set algorithm specific parameters
        if algorithm == 1:
            self.hitParam_alg1_thr_low = kwargs["alg1_thr_low"]
            self.hitParam_alg1_thr_high = kwargs["alg1_thr_high"]
            self.hitParam_alg1_radius = int(kwargs["alg1_radius"])
            self.hitParam_alg1_dr = kwargs["alg1_dr"]
        elif algorithm == 3:
            self.hitParam_alg3_rank = kwargs["alg3_rank"]
            self.hitParam_alg3_r0 = int(kwargs["alg3_r0"])
            self.hitParam_alg3_dr = kwargs["alg3_dr"]

    def findPeaks(self,calib):
        if kwargs["streakMask_sigma"] is not 0: # make new streak mask
            self.streakMask = myskbeam.getStreakMaskCalib(self.det,self.evt,width=self.streak_width,sigma=self.streak_sigma)
            
            if self.combinedMask is not None:
                self.combinedMask *= self.streakMask
            else:
                self.combinedMask = self.streakMask

            self.alg = PyAlgos(windows=self.windows, mask=self.combinedMask, pbits=0)
            # set peak-selector parameters:
            self.alg.set_peak_selection_pars(npix_min=hitParam_alg_npix_min, npix_max=hitParam_alg_npix_max, \
                                        amax_thr=hitParam_alg_amax_thr, atot_thr=hitParam_alg_atot_thr, \
                                        son_min=hitParam_alg_son_min)

            # set algorithm specific parameters
            if algorithm == 1:
                # v1 - aka Droplet Finder - two-threshold peak-finding algorithm in restricted region
                #                           around pixel with maximal intensity.
                self.peaks = self.alg.peak_finder_v1(calib, thr_low=self.hitParam_alg1_thr_low, thr_high=self.hitParam_alg1_thr_high, \
                                       radius=self.hitParam_alg1_radius, dr=self.hitParam_alg1_dr)
            elif algorithm == 3:
                self.peaks = self.alg.peak_finder_v3(calib, rank=self.hitParam_alg3_rank, r0=self.hitParam_alg3_r0, dr=self.hitParam_alg3_dr)
        
    def savePeaks(self,myHdf5,ind):
        if "cxi" in self.exp:
            nPeaks = self.peaks.shape[0]
            for i,peak in enumerate(self.peaks):
                seg,row,col,npix,amax,atot,rcent,ccent,rsigma,csigma,rmin,rmax,cmin,cmax,bkgd,rms,son = peak[0:17]
                cheetahRow,cheetahCol = self.convert_peaks_to_cheetah(seg,row,col)
                #print "nPeaks,cheetahRow,cheetahCol,atot: ", nPeaks,cheetahRow, cheetahCol, atot
                myHdf5[grpName+dset_posX][ind,i] = cheetahCol
                myHdf5[grpName+dset_posY][ind,i] = cheetahRow
                myHdf5[grpName+dset_atot][ind,i] = atot
        #print "found peaks: ", self.peaks, self.peaks.shape
        myHdf5[grpName+dset_nPeaks][ind] = nPeaks

    def convert_peaks_to_cheetah(self, s, r, c) :
        """Converts seg, row, col assuming (32,185,388)
           to cheetah 2-d table row and col (8*185, 4*388)
        """
        segs, rows, cols = (32,185,388)
        row2d = (int(s)%8) * rows + int(r) # where s%8 is a segment in quad number [0,7]
        col2d = (int(s)/8) * cols + int(c) # where s/8 is a quad number [0,3]
        return row2d, col2d

def getMyUnfairShare(numJobs,numWorkers,rank):
    """Returns number of events assigned to the slave calling this function."""
    assert(numJobs >= numWorkers)
    allJobs = np.arange(numJobs)
    jobChunks = np.array_split(allJobs,numWorkers)
    myChunk = jobChunks[rank]
    myJobs = allJobs[myChunk[0]:myChunk[-1]+1]
    return myJobs

def getMyFairShare(numJobs,numWorkers,rank):
    """Returns number of events assigned to the slave calling this function."""
    assert(numJobs >= numWorkers)
    numJobs = int(numJobs/numWorkers)*numWorkers # This is the numJobs that can be performed in parallel
    allJobs = np.arange(numJobs)
    jobChunks = np.array_split(allJobs,numWorkers)
    myChunk = jobChunks[rank]
    myJobs = allJobs[myChunk[0]:myChunk[-1]+1]
    return myJobs

def getRemainingShare(numJobs,numSlaves):
    """Returns number of remaining events"""
    assert(numJobs >= numSlaves)
    numJobsDone = int(numJobs/numSlaves)*numSlaves # This is the numJobs that can be performed in parallel
    myJobs = np.arange(numJobsDone,numJobs)
    return myJobs

def detList(s):
    try:
        return s.split(',')
    except:
        raise argparse.ArgumentTypeError("Detector list must be comma separated")

def findPeaksForMyChunk(myHdf5,ind,mytimes,single=False):
    for i,myTime in enumerate(mytimes):
        print "ind: ", ind[i]
        if i%100==0: print 'Rank',rank,'processing event', i, myTime
        evt = run.event(myTime)
        if evt is None:
            print '*** event fetch failed'
            continue
        for d in detlist:
            try:
                tic = time.time()
                detarr = d.calib_data(evt)
                tic1 = time.time()
            except ValueError:
                id = evt.get(EventId)
                print 'Value Error!'
                print id
                print id.time(),id.fiducials()
                continue
            if detarr is None:
                print '*** failed to get detarr'
                continue
            # Run hit finding
            if not hasattr(d,'peakFinder'):
                if args.algorithm == 1:
                    d.peakFinder = PeakFinder(env.experiment(),evt.run(),d.detname,evt,d,
                                              args.algorithm, args.alg_npix_min,
                                              args.alg_npix_max, args.alg_amax_thr,
                                              args.alg_atot_thr, args.alg_son_min,
                                              alg1_thr_low=args.alg1_thr_low, alg1_thr_high=args.alg1_thr_high,
                                              alg1_radius=args.alg1_radius, alg1_dr=args.alg1_dr)
                elif args.algorithm == 3:
                    d.peakFinder = PeakFinder(env.experiment(),evt.run(),d.detname,evt,d,
                                              args.algorithm, args.alg_npix_min,
                                              args.alg_npix_max, args.alg_amax_thr,
                                              args.alg_atot_thr, args.alg_son_min,
                                              alg3_rank=args.alg3_rank, alg3_r0=args.alg3_r0,
                                              alg3_dr=args.alg3_dr)
            tic2 = time.time()
            d.peakFinder.findPeaks(detarr)
            tic3 = time.time()
            d.peakFinder.savePeaks(myHdf5,ind[i])
            toc = time.time()

            print "calib,findPeaks,savePeaks: ", tic1-tic, tic3-tic2, toc-tic3

parser = argparse.ArgumentParser()
parser.add_argument('-e','--exp', help="experiment name (e.g. cxic0415)", type=str)
parser.add_argument('-r','--run', help="run number (e.g. 24)", type=int)
parser.add_argument('-d','--detList', help="list of detectors separated with comma (e.g. pnccdFront,pnccdBack)", dest="detList", type=detList, nargs=1)
parser.add_argument('-o','--outDir', help="output directory where .cxi will be saved (e.g. /reg/d/psdm/cxi/cxic0415/scratch)", type=str)
parser.add_argument("-n","--noe",help="number of events to process",default=0, type=int)
parser.add_argument("--algorithm",help="number of events to process",default=1, type=int)
parser.add_argument("--alg_npix_min",help="number of events to process",default=5., type=float)
parser.add_argument("--alg_npix_max",help="number of events to process",default=5000., type=float)
parser.add_argument("--alg_amax_thr",help="number of events to process",default=0., type=float)
parser.add_argument("--alg_atot_thr",help="number of events to process",default=0., type=float)
parser.add_argument("--alg_son_min",help="number of events to process",default=10., type=float)
parser.add_argument("--alg1_thr_low",help="number of events to process",default=10., type=float)
parser.add_argument("--alg1_thr_high",help="number of events to process",default=150., type=float)
parser.add_argument("--alg1_radius",help="number of events to process",default=5, type=int)
parser.add_argument("--alg1_dr",help="number of events to process",default=0.05, type=float)
parser.add_argument("--alg3_rank",help="number of events to process",default=3, type=int)
parser.add_argument("--alg3_r0",help="number of events to process",default=5., type=float)
parser.add_argument("--alg3_dr",help="number of events to process",default=0.05, type=float)
parser.add_argument("--streakMask_sigma",help="streak mask sigma above background",default=0., type=float)
parser.add_argument("--streakMask_width",help="streak mask width",default=0, type=float)
parser.add_argument("--userMask_path",help="full path to user mask numpy array",default=None, type=str)
parser.add_argument("--psanaMask_calib",help="psana calib on",default=False, type=bool)
parser.add_argument("--psanaMask_status",help="psana status on",default=False, type=bool)
parser.add_argument("--psanaMask_edges",help="psana edges on",default=False, type=bool)
parser.add_argument("--psanaMask_central",help="psana central on",default=False, type=bool)
parser.add_argument("--psanaMask_unbond",help="psana unbonded pixels on",default=False, type=bool)
parser.add_argument("--psanaMask_unbondnrs",help="psana unbonded pixel neighbors on",default=False, type=bool)
args = parser.parse_args()

experimentName = args.exp
runNumber = args.run

def main():

    ds = DataSource("exp="+experimentName+":run="+str(runNumber)+':idx')
    env = ds.env()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    detname = args.detList[0]
    detlist = [Detector(s, env) for s in detname]
    for d,n in zip(detlist,detname):
        d.detname = n

    run = ds.runs().next()

    # list of all events
    times = run.times()
    # check if the user requested specific number of events
    if args.noe == 0:
        numJobs = len(times)
    else:
        if args.noe <= len(times):
            numJobs = args.noe
        else:
            numJobs = len(times)

    ind = getMyUnfairShare(numJobs,size,rank)
    mytimes = times[ind[0]:ind[-1]+1]

    print "mytimes: ", rank, len(times), len(mytimes), ind[0], ind[-1]

    runStr = "%04d" % runNumber
    fname = args.outDir +"/"+ experimentName +"_"+ runStr + ".cxi"
    print fname

    # Create hdf5 and save
    if rank == 0:
        myHdf5 = h5py.File(fname, 'w')
        dt = h5py.special_dtype(vlen=bytes)

        myInput = ""
        for key,value in vars(args).iteritems():
            myInput += key
            myInput += " "
            myInput += str(value)
            myInput += "\n"
        dset = myHdf5.create_dataset("/psana/input",(1,), dtype=dt)
        dset[...] = myInput
        myHdf5.close()

    # Parallel process events
    myHdf5 = h5py.File(fname, 'r+', driver='mpio', comm=comm)
    grpName = "/entry_1/result_1"
    dset_nPeaks = "/nPeaksAll"
    dset_posX = "/peakXPosRawAll"
    dset_posY = "/peakYPosRawAll"
    dset_atot = "/peakTotalIntensityAll"
    if grpName in myHdf5:
        del myHdf5[grpName]
    grp = myHdf5.create_group(grpName)
    myHdf5.create_dataset(grpName+dset_nPeaks, (numJobs,), dtype='int')
    myHdf5.create_dataset(grpName+dset_posX, (numJobs,2048), dtype='float32', chunks=(1,2048))
    myHdf5.create_dataset(grpName+dset_posY, (numJobs,2048), dtype='float32', chunks=(1,2048))
    myHdf5.create_dataset(grpName+dset_atot, (numJobs,2048), dtype='float32', chunks=(1,2048))
    findPeaksForMyChunk(myHdf5,ind,mytimes,single=False)
    myHdf5.close()

    MPI.Finalize()

if __name__ == '__main__':
    main()

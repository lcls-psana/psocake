# psana and svn versions
#from __future__ import absolute_import, division#, print_function,unicode_literals

import subprocess
import numpy as np
import h5py
import psana
import argparse
import time
import os
import socket

parser = argparse.ArgumentParser()
parser.add_argument("exprun", help="psana experiment/run string (e.g. exp=xppd7114:run=43)", type=str)
parser.add_argument("-t","--tag",help="append tag to end of filename",default="", type=str)
parser.add_argument("-d","--detectorName",help="psana detector alias, (e.g. pnccdBack, DsaCsPad)", type=str)
parser.add_argument("-o","--outdir",help="output directory",default=0, type=str)
parser.add_argument("-n","--noe",help="number of events, all events=0",default=0, type=int)
parser.add_argument("-m","--mask",help="full path to binary mask numpy ndarray (psana shape)",default=None, type=str)
parser.add_argument("-l","--litPixelThreshold",help="number of ADUs to be considered a lit pixel",default=100, type=float)
parser.add_argument("-v","--verbose",help="verbosity of output for debugging, 1=print, 2=print+plot",default=0, type=int)
parser.add_argument("--localCalib", help="use local calib directory, default=False", action='store_true')
args = parser.parse_args()
assert os.path.isdir(args.outdir)

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>=2, 'Require at least two mpi ranks'
numslaves = size-1

if rank == 0:
    psana_version = subprocess.check_output(["ls","-lart","/reg/g/psdm/sw/releases/ana-current"]).split('-> ')[-1].split('\n')[0]
    svn_version = subprocess.check_output("svnversion")
    print("psana,svn versions: ", psana_version, svn_version)
    print("Running litPixels: ", args.exprun)

if args.localCalib:
    psana.setOption('psana.calib-dir','./calib')

def getAveragePhotonEnergy():
    """
    Get average photon energy of this experiment in keV
    """
    times = run.times()
    eventTotal = len(times)
    np.random.seed(2016)
    randomEvents = np.random.permutation(eventTotal)
    numSample = 100
    if numSample > eventTotal:
        numSample = eventTotal
    photonEnergySample = np.zeros(numSample)
    counter = 0
    for i,val in enumerate(randomEvents[:numSample]):
        evt = run.event(times[val])
        ebeam = evt.get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
        if ebeam is not None:
            photonEnergySample[counter] = ebeam.ebeamPhotonEnergy()
            counter += 1
    photonEnergy = np.median(photonEnergySample[:counter])/1000.
    return photonEnergy

grpName = "/litPixels"

myDatasource = args.exprun+':idx'
ds = psana.DataSource(myDatasource)
env = ds.env()
run = ds.runs().next()

if rank == 0:
    photonEnergy = getAveragePhotonEnergy()

aliasList = args.detectorName.split(",")
print("#### aliasList: ", aliasList)
myDetList = [psana.Detector(src) for src in aliasList]
numDet = len(aliasList)

def getGains(evt):
    for d in myDetList:
        d.gains = d.gain(evt) # CHUCK: gain is not deployed for amo87215, returns 1
        hybridGain = d.gain_mask(evt,gain=6.85)
        if hybridGain is not None:
            d.gains *= hybridGain

def getMasks(evt):
    for d in myDetList:
        d.spiMask = np.copy(d.mask(evt, calib=False, edges=True, central=True, unbond=True, unbondnbrs=True))  # calib is user-defined
        if d.spiMask.size==2296960:  # number of pixels of CSPAD
            d.spiMask = d.spiMask.reshape((32,185,388))
        if d.spiMask.size==1048576:  # number of pixels of PNCCD
            d.spiMask = d.spiMask.reshape((4,512,512))
        if d.spiMask is None:
            d.spiMask = np.ones_like(d.spiMask.shape)
        if args.mask and d == myDetList[0]:
            userMask = np.load(args.mask)
            if d.spiMask.shape == userMask.shape: # unassembled mask
                d.spiMask *= userMask
            elif d.spiMask.size == userMask.shape: # assembled mask
                userMask = d.ndarray_from_image(evt,userMask, pix_scale_size_um=None, xy0_off_pix=None)
                d.spiMask *= userMask # FIXME: cspad2x2 case
            else:
                print("Not applying user mask because the shape is inconsistent")

def getEventID(evt):
    evtid = evt.get(psana.EventId)
    seconds = evtid.time()[0]
    nanoseconds = evtid.time()[1]
    fiducials = evtid.fiducials()
    return seconds, nanoseconds, fiducials

def nunchakuAlgorithm(stickLength=15):
    # Generate output filename
    outname = os.path.join(args.outdir,
                           env.experiment()+'_'+str(run.run())+'_'+args.tag+'.h5')
    # Clean up hdf5 structure if it exists
    f = h5py.File(outname, 'r+')
    t             = f[grpName+'/eventTime'].value
    fid           = f[grpName+'/fiducials'].value
    hitMetric     = f[grpName+'/hitMetric'].value
    timeOrder   = t.argsort()
    hitMetric = hitMetric[timeOrder]
    numEvents = len(hitMetric)
    
    hitMetric_ds    = f.require_dataset(grpName+"/hitMetric", (numEvents,), dtype='float32')
    hitMetric_ds[...] = hitMetric
    event_ds    = f.create_dataset(grpName+"/event", (numEvents,), dtype='uint32')
    event_ds[...] = np.arange(numEvents)

    # Reorder other dataset in time
    evttime_ds = f.require_dataset(grpName+"/eventTime", (numEvents,),dtype='uint64')
    fids_ds    = f.require_dataset(grpName+"/fiducials", (numEvents,), dtype='uint32')
    evttime_ds[...] = t[timeOrder]
    fids_ds[...] = fid[timeOrder]

    f.close()

class bigMsg:
    """Message holding the events message and raw image"""
    def __init__(self):
        self.data = []
        self.smallMsg = smallMsg()
        self.reset()

    def reset(self):
        self.data = []

    def send(self):
        self.smallMsg.send()

        for res in self.data:
            for arr in res:
                comm.Send(arr.astype(float), dest=0, tag=1)
        self.reset()

class smallMsg:
    """Class to hold data for event (no image)"""
    def __init__(self):
        self.done = False
        self.events = []
        self.reset()

    def reset(self):
        self.events     = []
        #self.arraySizes = []
        self.done = False

    def send(self):
        comm.send(self, dest=0, tag=0)
        self.reset()

    def sendDone(self):
        self.done = True
        comm.send(self, dest=0, tag=0)
        self.reset()

    def sendNonResettingDone(self):
        self.done = True
        comm.send(self, dest=0, tag=0)

def getNumEventsToProc(run,noe):
    """Returns number of events to process."""
    if noe == 0:
        numEventsToProc = len(run.times())
    else:
        assert(noe <= len(run.times()))
        numEventsToProc = noe
    return numEventsToProc

def getMyUnfairShare(run,numslaves,rank,numOfEventsToProc=0):
    """Returns number of events assigned to the slave calling this function"""
    times = run.times()
    assert(len(times) > numOfEventsToProc)
    numEvents = getNumEventsToProc(run,numOfEventsToProc)
    jobChunks = np.array_split(np.arange(numEvents),numslaves)
    myChunk = jobChunks[rank-1]
    myJobs = times[myChunk[0]:myChunk[-1]+1]
    print("number of assigned events: ", len(myJobs))
    return myJobs

class slave_class(object):
    """
    Requires:
    Global variables:
        myDetList
        numDet
    """
    def __init__(self, run, numslaves, rank, noe, numDet):
        self.run            = run
        self.numslaves      = numslaves
        self.numDet         = numDet
        self.rank           = rank
        self.noe            = noe
        self.myJobs         = getMyUnfairShare(self.run, numslaves, rank, noe)
        self.tstart         = time.time()
        self.totalJobs      = len(self.myJobs)
        self.myMask         = [None for nd in range(numDet)]
        self.myMsg          = None

    def setup(self):
        evt = self.run.event(self.myJobs[0])
        getMasks(evt)
        getGains(evt)
        self.myMsg = smallMsg()

        for i in range(numDet):
            self.myMask[i]  = myDetList[i].spiMask.copy() # important to copy
        self.myMask[0] *= myDetList[0].gains # save time by applying gain to mask

    def process_run(self):
        print('Rank', self.rank, 'starts event loop at', time.strftime('%H:%M:%S'))
        self.setup()

        for nevent,t in enumerate(self.myJobs):
            evt = self.run.event(t)
            if nevent%100 == 0:
                rate = nevent/(time.time() - self.tstart)
                progress = np.around(nevent*100./self.totalJobs,1)
                print('Rank',self.rank,'at event', nevent,'(', progress,'%) with rate', rate)

            # Get event identifiers
            seconds, nanoseconds, fiducials = getEventID(evt)

            # Hit find on the first detector
            calib = myDetList[0].calib(evt)
            try:
                calib *= self.myMask[0] # myMask masks away high and low variance pixels (includes spiMask)
                hitMetric = np.where(calib>args.litPixelThreshold)[0].shape[0]
            except:
                hitMetric = 0

            # Sends smallMsg component to BigMsg
            self.myMsg.events = (seconds, nanoseconds, fiducials, hitMetric)

            # Sends to BigMsg a list of detector arrays if it's a hit
            # otherwise, sends an empty list
            self.myMsg.send()

        # Finished handling all assigned events
        # send a message when we're done
        self.myMsg.sendDone()
        print('Done rank: ',rank,'hostname: ',socket.gethostname(),'done at',time.strftime('%X'))

def master():
    slavecount = numslaves
    numEventsToProc = getNumEventsToProc(run,args.noe)
    junkevt = run.event(run.times()[0])  # to get calib store filled
    getMasks(junkevt)

    # Generate output filename
    outname = os.path.join(args.outdir,env.experiment()+'_'+str(run.run())+'_'+args.tag+'.h5')
    # Clean up hdf5 structure if it exists
    f = h5py.File(outname, 'a')
    if grpName in f:
        del f[grpName]
    grp = f.create_group(grpName)
    # Initialize event identifiers
    evttime_ds = grp.create_dataset("eventTime", (numEventsToProc,),dtype='uint64')
    fids_ds    = grp.create_dataset("fiducials", (numEventsToProc,), dtype='uint32')
    # Initialize hit metric
    hitMetric_ds    = grp.create_dataset("hitMetric", (numEventsToProc,), dtype='float32')

    # Loop over all detectors and initialize/save fields
    for i in np.arange(len(myDetList)):
        det = myDetList[i]
        # Save mask
        thisMask = grp.create_dataset(aliasList[i]+"/mask", np.shape(det.spiMask), dtype='uint8')
        thisMask[...] = det.spiMask

        # Fetch X,Y indices and coordinates
        iX = det.indexes_x(junkevt)
        iY = det.indexes_y(junkevt)
        coordsX = det.coords_x(junkevt)
        coordsY = det.coords_y(junkevt)
        coordsZ = det.coords_z(junkevt)

        # Save X,Y indices and coordinates
        if iX is not None:
            thisIX = grp.create_dataset(aliasList[i]+"/iX", np.shape(iX), dtype='uint32')
            thisIX[...] = iX
            thisIY = grp.create_dataset(aliasList[i]+"/iY", np.shape(iY), dtype='uint32')
            thisIY[...] = iY
            thisCoordsX = grp.create_dataset(aliasList[i]+"/coordsX", np.shape(coordsX), dtype='float32')
            thisCoordsX[...] = coordsX
            thisCoordsY = grp.create_dataset(aliasList[i]+"/coordsY", np.shape(coordsY), dtype='float32')
            thisCoordsY[...] = coordsY
            thisCoordsZ = grp.create_dataset(aliasList[i]+"/coordsZ", np.shape(coordsZ), dtype='float32')
            thisCoordsZ[...] = coordsZ

    # Receive results from slaves
    status = MPI.Status()
    nevts = 0
    while slavecount:
        # Receive event message (pickled object)
        smallMsg = comm.recv(source=MPI.ANY_SOURCE, tag=0, status=status)
        if smallMsg.done:
            print('Received done message from rank',status.Get_source(), \
                  'at',time.strftime('%H:%M:%S'))
            slavecount-=1
        else:
            seconds = smallMsg.events[0]
            nanoseconds = smallMsg.events[1]
            fiducials = smallMsg.events[2]
            hitMetric = smallMsg.events[3]

            evttime_ds[nevts] = (seconds << 32) | nanoseconds
            fids_ds[nevts] = fiducials
            hitMetric_ds[nevts] = hitMetric
            nevts+=1

    # Save attributes
    hitMetric_ds.attrs['numEvents'] = nevts
    hitMetric_ds.attrs['ROI'] = 'None'
    hitMetric_ds.attrs['photonEnergy'] = photonEnergy
    hitMetric_ds.attrs['exprun'] = args.exprun
    hitMetric_ds.attrs['detectorName'] = args.detectorName
    hitMetric_ds.attrs['psanaVersion'] = psana_version
    hitMetric_ds.attrs['svnVersion'] = svn_version

    f.close()

################################################################################
# Code for this mpi process continues here
################################################################################

curr_slave = slave_class(run, numslaves, rank, args.noe, numDet)
if rank==0:
    master()
    print("If you are running from an interactive node, you may ignore the 'mpi_warn_on_fork' message.")
else:
    print('Starting rank: ',rank,'hostname: ',socket.gethostname())
    curr_slave.process_run()

if rank==0:
    nunchakuAlgorithm() # reorder in time

MPI.Finalize()

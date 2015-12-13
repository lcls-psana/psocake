#bsub -q psanaq -a mympi -n 36 -o %J.log python generatePowder.py exp=amo87215:run=15 -d pnccdFront

from psana import *
import numpy as np
import sys

class Stats:
    def __init__(self,detarr,exp,run,detname):
        self.sum=detarr.astype(np.float64)
        self.sumsq=detarr.astype(np.float64)*detarr.astype(np.float64)
        self.maximum=detarr.astype(np.float64)
        self.nevent=1
        self.exp = exp
        self.run = run
        self.detname = detname
    def update(self,detarr):
        self.sum+=detarr
        self.sumsq+=detarr*detarr
        self.maximum=np.maximum(self.maximum,detarr)
        self.nevent+=1
    def store(self):
        self.totevent = comm.reduce(self.nevent)
        if rank==0:
            comm.Reduce(MPI.IN_PLACE,self.sum)
            comm.Reduce(MPI.IN_PLACE,self.sumsq)
            comm.Reduce(MPI.IN_PLACE,self.maximum,op=MPI.MAX)
            # Accumulating floating-point numbers introduces errors,
            # which may cause negative variances.  Since a two-pass
            # approach is unacceptable, the standard deviation is
            # clamped at zero.
            self.mean = self.sum / float(self.totevent)
            self.variance = (self.sumsq / float(self.totevent)) - (self.mean**2)
            self.variance[self.variance < 0] = 0
            self.stddev = np.sqrt(self.variance)
            file = '%s_%4.4d_%s'%(self.exp,self.run,self.detname)
            print 'writing file',file
            np.savez(file,mean=self.mean,stddev=self.stddev,max=self.maximum)
        else:
            comm.Reduce(self.sum,self.sum)
            comm.Reduce(self.sumsq,self.sumsq)
            comm.Reduce(self.maximum,self.maximum,op=MPI.MAX)

def getMyUnfairShare(numJobs,numWorkers,rank):
    """Returns number of events assigned to the slave calling this function."""
    assert(numJobs >= numWorkers)
    allJobs = np.arange(numJobs)
    jobChunks = np.array_split(allJobs,numWorkers)
    myChunk = jobChunks[rank]
    myJobs = allJobs[myChunk[0]:myChunk[-1]+1]
    return myJobs

def detList(s):
    try:
        return s.split(',')
    except:
        raise argparse.ArgumentTypeError("Detector list must be comma separated")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("exprun", help="psana experiment/run string (e.g. exp=xppd7114:run=43)")
parser.add_argument('-d','--detList', help="list of detectors separated with comma (e.g. pnccdFront,pnccdBack)", dest="detList", type=detList, nargs=1)
parser.add_argument("-n","--noe",help="number of events to process",default=0, type=int)
args = parser.parse_args()

ds = DataSource(args.exprun+':idx')
env = ds.env()

# set this to sys.maxint to analyze all events
maxevents = sys.maxint

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

detname = args.detList[0] #['Camp.0:pnCCD.0']#,'Camp.0:pnCCD.1']
detlist = [Detector(s, env) for s in detname]
for d,n in zip(detlist,detname):
    d.detname = n

nevent = np.array([0])

for run in ds.runs():
    runnumber = run.run()
    # list of all events
    times = run.times()
    if args.noe == 0:
        numJobs = len(times)
    else:
        if args.noe <= len(times):
            numJobs = args.noe
        else:
            numJobs = len(times)

    ind = getMyUnfairShare(len(times),size,rank)
    mytimes = times[ind[0]:ind[-1]+1]

    for i,time in enumerate(mytimes):
        if i%100==0: print 'Rank',rank,'processing event', i
        evt = run.event(time)
        # very useful for seeing what data is in the event
        #print evt.keys()
        if evt is None:
            print '*** event fetch failed'
            continue
        for d in detlist:
            try:
                detarr = d.calib_data(evt)
            except ValueError:
                id = evt.get(EventId)
                print 'Value Error!'
                print id
                print id.time(),id.fiducials()
                continue
            if detarr is None:
                print '*** failed to get detarr'
                continue
            if not hasattr(d,'stats'):
                d.stats = Stats(detarr,env.experiment(),evt.run(),d.detname)
            else:
                d.stats.update(detarr)
        nevent+=1

for d in detlist: d.stats.store()
MPI.Finalize()

#bsub -q psanaq -a mympi -n 36 -o %J.log python max_det_orig.py exp=amo87215:run=15 -d pnccdFront

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

def detList(s):
    try:
        return s.split(',')
    except:
        raise argparse.ArgumentTypeError("Detector list must be comma separated")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("exprun", help="psana experiment/run string (e.g. exp=xppd7114:run=43)")
parser.add_argument('-d','--detList', help="list of detectors separated with comma (e.g. pnccdFront,pnccdBack)", dest="detList", type=detList, nargs=1)
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
print "detname: ", detname
detlist = [Detector(s, env) for s in detname]
for d,n in zip(detlist,detname):
    d.detname = n

nevent = np.array([0])

for run in ds.runs():
    runnumber = run.run()
    # list of all events
    times = run.times()
    nevents = min(len(times),maxevents)
    mylength = nevents/size # easy but sloppy. lose few events at end of run.

    # chop the list into pieces, depending on rank
    mytimes= times[rank*mylength:(rank+1)*mylength]
    for i in range(mylength):
        if i%100==0: print 'Rank',rank,'processing event',rank*mylength+i
        evt = run.event(mytimes[i])
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

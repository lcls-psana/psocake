from psana import *
from mpidata import mpidata 

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def runclient(args):
    ds = DataSource(args.exprun+':smd')
    det1 = Detector(args.areaDetName)
    
    for nevent,evt in enumerate(ds.events()):
        if nevent == args.noe : break
        if nevent%(size-1)!=rank-1: continue # different ranks look at different events
        img = det1.image(evt)
        if img is None: continue
        intensity = img.sum()
        md=mpidata()
        md.addarray('img',img)
        md.small.intensity = intensity
        if ((nevent)%1 == 0): # send mpi data object to master when desired
            md.send()

    md.endrun()	

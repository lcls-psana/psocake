import psana
import numpy as np
from mpidata import mpidata 
import HitFinder as hf

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def runclient(args):
    ds = psana.DataSource("exp="+args.exp+":run="+str(args.run)+':idx')
    run = ds.runs().next()
    env = ds.env()
    times = run.times()
    d = psana.Detector(args.detectorName)
    d.do_reshape_2d_to_3d(flag=True)

    for nevent in np.arange(len(times)):
        if nevent == args.noe : break
        if nevent%(size-1) != rank-1: continue # different ranks look at different events
        evt = run.event(times[nevent])
        detarr = d.calib(evt)

        if detarr is None: continue

        # Initialize hit finding
        if not hasattr(d,'hitFinder'):
            d.hitFinder = hf.HitFinder(env.experiment(),
                                       evt.run(),
                                       args.detectorName,
                                       evt,
                                       d,
                                       args.litPixelThreshold,
                                       streakMask_on=args.streakMask_on,
                                       streakMask_sigma=args.streakMask_sigma,
                                       streakMask_width=args.streakMask_width,
                                       userMask_path=args.userMask_path,
                                       psanaMask_on=args.psanaMask_on,
                                       psanaMask_calib=args.psanaMask_calib,
                                       psanaMask_status=args.psanaMask_status,
                                       psanaMask_edges=args.psanaMask_edges,
                                       psanaMask_central=args.psanaMask_central,
                                       psanaMask_unbond=args.psanaMask_unbond,
                                       psanaMask_unbondnrs=args.psanaMask_unbondnrs)
        d.hitFinder.findHits(detarr,evt)
        md=mpidata()
        md.small.eventNum = nevent
        md.small.nPixels = d.hitFinder.nPixels
        md.small.powder = 0
        md.send() # send mpi data object to master when desired

    # At the end of the run, send the powder of hits and misses
    md = mpidata()
    #md.small.powder = 1
    #md.addarray('powderHits', d.hitFinder.powderHits)
    #md.addarray('powderMisses', d.hitFinder.powderMisses)
    #md.send()
    md.endrun()

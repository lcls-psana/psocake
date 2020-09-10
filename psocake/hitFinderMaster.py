from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import h5py, json
from mpidata import mpidata
import psana

def writeStatus(fname,d):
    json.dump(d, open(fname, 'w'))

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

def runmaster(args,nClients):

    runStr = "%04d" % args.run
    fname = args.outDir +"/"+ args.exp +"_"+ runStr + ".cxi"
    grpName = "/entry_1/result_1"
    dset_nHits = "/nHitsAll"
    statusFname = args.outDir + "/status_hits.txt"

    powderHits = None
    powderMisses = None

    numProcessed = 0
    fracDone = 0.0
    numEvents = getNoe(args)
    d = {"fracDone": fracDone}
    writeStatus(statusFname, d)

    myHdf5 = h5py.File(fname, 'r+')
    while nClients > 0:
        # Remove client if the run ended
        md = mpidata()
        md.recv()
        if md.small.endrun:
            nClients -= 1
        #elif md.small.powder == 1:
        #    if powderHits is None:
        #        powderHits = md.powderHits
        #        powderMisses = md.powderMisses
        #    else:
        #        powderHits = np.maximum(powderHits, md.powderHits)
        #        powderMisses = np.maximum(powderMisses, md.powderMisses)
        else:
            try:
                nPixels = md.small.nPixels
            except:
                continue
            myHdf5[grpName+dset_nHits][md.small.eventNum] = nPixels
            myHdf5.flush()
            numProcessed += 1
            # Update status
            fracDone = numProcessed * 100. / numEvents
            d = {"fracDone": fracDone}
            writeStatus(statusFname, d)

    if '/status/findHits' in myHdf5:
        del myHdf5['/status/findHits']
    myHdf5['/status/findHits'] = 'success'
    myHdf5.flush()
    myHdf5.close()

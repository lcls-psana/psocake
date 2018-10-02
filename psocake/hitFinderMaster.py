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

    # fnameHits = args.outDir +"/"+ args.exp +"_"+ runStr + "_maxHits.npy"
    # fnameMisses = args.outDir +"/"+ args.exp +"_"+ runStr + "_maxMisses.npy"
    # fnameHitsTxt = args.outDir +"/"+ args.exp +"_"+ runStr + "_maxHits.txt"
    # fnameMissesTxt = args.outDir +"/"+ args.exp +"_"+ runStr + "_maxMisses.txt"
    # fnameHitsNatural = args.outDir +"/"+ args.exp +"_"+ runStr + "_maxHits_natural_shape.npy"
    # fnameMissesNatural = args.outDir +"/"+ args.exp +"_"+ runStr + "_maxMisses_natural_shape.npy"
    #
    # if powderHits.size == 2 * 185 * 388:  # cspad2x2
    #     # DAQ shape
    #     asData2x2 = two2x1ToData2x2(powderHits)
    #     np.save(fnameHits, asData2x2)
    #     np.savetxt(fnameHitsTxt, asData2x2.reshape((-1, asData2x2.shape[-1])), fmt='%0.18e')
    #     # Natural shape
    #     np.save(fnameHitsNatural, powderHits)
    #     # DAQ shape
    #     asData2x2 = two2x1ToData2x2(powderMisses)
    #     np.save(fnameMisses, asData2x2)
    #     np.savetxt(fnameMissesTxt, asData2x2.reshape((-1, asData2x2.shape[-1])), fmt='%0.18e')
    #     # Natural shape
    #     np.save(fnameMissesNatural, powderMisses)
    # else:
    #     np.save(fnameHits, powderHits)
    #     np.savetxt(fnameHitsTxt, powderHits.reshape((-1, powderHits.shape[-1])), fmt='%0.18e')
    #     np.save(fnameMisses, powderMisses)
    #     np.savetxt(fnameMissesTxt, powderMisses.reshape((-1, powderMisses.shape[-1])), fmt='%0.18e')

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import h5py, json
from mpidata import mpidata
import psana, time
import numpy as np
from PSCalib.GeometryObject import two2x1ToData2x2

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
    dset_nPeaks = "/nPeaksAll"
    dset_posX = "/peakXPosRawAll"
    dset_posY = "/peakYPosRawAll"
    dset_atot = "/peakTotalIntensityAll"
    dset_maxRes = "/maxResAll"
    statusFname = args.outDir + "/status_peaks.txt"

    powderHits = None
    powderMisses = None

    numProcessed = 0
    numHits = 0
    hitRate = 0.0
    fracDone = 0.0
    numEvents = getNoe(args)
    d = {"numHits": numHits, "hitRate": hitRate, "fracDone": fracDone}
    try:
        writeStatus(statusFname, d)
    except:
        print "Couldn't update status"
        pass

    myHdf5 = h5py.File(fname, 'r+')
    while nClients > 0:
        # Remove client if the run ended
        md = mpidata()
        md.recv()
        if md.small.endrun:
            nClients -= 1
        elif md.small.powder == 1:
            if powderHits is None:
                powderHits = md.powderHits
                powderMisses = md.powderMisses
            else:
                powderHits = np.maximum(powderHits, md.powderHits)
                powderMisses = np.maximum(powderMisses, md.powderMisses)
        else:
            try:
                nPeaks = md.peaks.shape[0]
                maxRes = md.small.maxRes
                #print "### nPeaks, maxRes: ", nPeaks, maxRes
            except:
                continue
            if nPeaks > args.maxNumPeaks: # only save upto maxNumPeaks
                md.peaks = md.peaks[:args.maxNumPeaks]
                nPeaks = md.peaks.shape[0]
            for i,peak in enumerate(md.peaks):
                seg,row,col,npix,amax,atot,rcent,ccent,rsigma,csigma,rmin,rmax,cmin,cmax,bkgd,rms,son = peak[0:17]
                cheetahRow,cheetahCol = convert_peaks_to_cheetah(seg,row,col)
                myHdf5[grpName+dset_posX][md.small.eventNum,i] = cheetahCol
                myHdf5[grpName+dset_posY][md.small.eventNum,i] = cheetahRow
                myHdf5[grpName+dset_atot][md.small.eventNum,i] = atot
                myHdf5.flush()
            myHdf5[grpName+dset_nPeaks][md.small.eventNum] = nPeaks
            myHdf5[grpName+dset_maxRes][md.small.eventNum] = maxRes
            # Save image

            # Save mask

            myHdf5.flush()
            if nPeaks >= 15: numHits += 1
            numProcessed += 1
            # Update status
            try:
                hitRate = numHits * 100. / numProcessed
                fracDone = numProcessed * 100. / numEvents
                d = {"numHits": numHits, "hitRate": hitRate, "fracDone": fracDone}
                writeStatus(statusFname, d)
            except:
                print "Couldn't update status"
                pass

    if '/status/findPeaks' in myHdf5:
        del myHdf5['/status/findPeaks']
    myHdf5['/status/findPeaks'] = 'success'
    myHdf5.flush()
    myHdf5.close()

    fnameHits = args.outDir +"/"+ args.exp +"_"+ runStr + "_maxHits.npy"
    fnameMisses = args.outDir +"/"+ args.exp +"_"+ runStr + "_maxMisses.npy"
    fnameHitsTxt = args.outDir +"/"+ args.exp +"_"+ runStr + "_maxHits.txt"
    fnameMissesTxt = args.outDir +"/"+ args.exp +"_"+ runStr + "_maxMisses.txt"
    fnameHitsNatural = args.outDir +"/"+ args.exp +"_"+ runStr + "_maxHits_natural_shape.npy"
    fnameMissesNatural = args.outDir +"/"+ args.exp +"_"+ runStr + "_maxMisses_natural_shape.npy"
    #np.save(fnameHits,powderHits)
    #np.save(fnameMisses, powderMisses)

    if powderHits.size == 2 * 185 * 388:  # cspad2x2
        # DAQ shape
        asData2x2 = two2x1ToData2x2(powderHits)
        np.save(fnameHits, asData2x2)
        np.savetxt(fnameHitsTxt, asData2x2.reshape((-1, asData2x2.shape[-1])), fmt='%0.18e')
        # Natural shape
        np.save(fnameHitsNatural, powderHits)
        # DAQ shape
        asData2x2 = two2x1ToData2x2(powderMisses)
        np.save(fnameMisses, asData2x2)
        np.savetxt(fnameMissesTxt, asData2x2.reshape((-1, asData2x2.shape[-1])), fmt='%0.18e')
        # Natural shape
        np.save(fnameMissesNatural, powderMisses)
    else:
        np.save(fnameHits, powderHits)
        np.savetxt(fnameHitsTxt, powderHits.reshape((-1, powderHits.shape[-1])), fmt='%0.18e')
        np.save(fnameMisses, powderMisses)
        np.savetxt(fnameMissesTxt, powderMisses.reshape((-1, powderMisses.shape[-1])), fmt='%0.18e')

def convert_peaks_to_cheetah(s, r, c) :
    """Converts seg, row, col assuming (32,185,388)
       to cheetah 2-d table row and col (8*185, 4*388)
    """
    segs, rows, cols = (32,185,388)
    row2d = (int(s)%8) * rows + int(r) # where s%8 is a segment in quad number [0,7]
    col2d = (int(s)/8) * cols + int(c) # where s/8 is a quad number [0,3]
    return row2d, col2d

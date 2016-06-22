from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import h5py
from mpidata import mpidata
import psana
import numpy as np

def runmaster(args,nClients):

    runStr = "%04d" % args.run
    fname = args.outDir +"/"+ args.exp +"_"+ runStr + ".cxi"

    # Get number of events to process
    numJobs = getNoe(args)

    # Create hdf5 and save psana input
    #print "### Writing: ", fname
    myHdf5 = h5py.File(fname, 'w')
    #myHdf5.swmr_mode = True
    myHdf5['/status/findPeaks'] = 'fail'
    dt = h5py.special_dtype(vlen=bytes)
    myInput = ""
    for key,value in vars(args).iteritems():
        myInput += key
        myInput += " "
        myInput += str(value)
        myInput += "\n"
    dset = myHdf5.create_dataset("/psana/input",(1,), dtype=dt)
    dset[...] = myInput

    grpName = "/entry_1/result_1"
    dset_nPeaks = "/nPeaksAll"
    dset_posX = "/peakXPosRawAll"
    dset_posY = "/peakYPosRawAll"
    dset_atot = "/peakTotalIntensityAll"
    dset_maxRes = "/maxResAll"
    if grpName in myHdf5:
        del myHdf5[grpName]
    grp = myHdf5.create_group(grpName)
    myHdf5.create_dataset(grpName+dset_nPeaks, data=np.ones(numJobs,)*-1, dtype='int')
    myHdf5.create_dataset(grpName+dset_posX, (numJobs,args.maxNumPeaks), dtype='float32', chunks=(1,args.maxNumPeaks))
    myHdf5.create_dataset(grpName+dset_posY, (numJobs,args.maxNumPeaks), dtype='float32', chunks=(1,args.maxNumPeaks))
    myHdf5.create_dataset(grpName+dset_atot, (numJobs,args.maxNumPeaks), dtype='float32', chunks=(1,args.maxNumPeaks))
    myHdf5.create_dataset(grpName+dset_maxRes, data=np.ones(numJobs,)*-1, dtype='int')
    myHdf5.close()

    myHdf5 = h5py.File(fname, 'r+')
    #saveInterval = 10
    counter = 0
    #print "### nClients: ", nClients
    while nClients > 0:
        #print "GOT HERE!!!!!!!!!!!"
        # Remove client if the run ended
        md = mpidata()
        md.recv()
        if md.small.endrun:
            nClients -= 1
        else:
            #print "### Recv: ", md.peaks, md.small.maxRes, md.small.endrun
            #if counter == saveInterval:
            #    myHdf5 = h5py.File(fname, 'r+')
            #save to hdf5
            try:
                nPeaks = md.peaks.shape[0]
                maxRes = md.small.maxRes
                #print "### nPeaks, maxRes: ", nPeaks, maxRes
            except:
                continue
            if nPeaks > args.maxNumPeaks:
                md.peaks = md.peaks[:args.maxNumPeaks]
                nPeaks = md.peaks.shape[0]
            for i,peak in enumerate(md.peaks):
                seg,row,col,npix,amax,atot,rcent,ccent,rsigma,csigma,rmin,rmax,cmin,cmax,bkgd,rms,son = peak[0:17]
                cheetahRow,cheetahCol = convert_peaks_to_cheetah(seg,row,col)
                myHdf5[grpName+dset_posX][md.small.eventNum,i] = cheetahCol
                myHdf5[grpName+dset_posY][md.small.eventNum,i] = cheetahRow
                myHdf5[grpName+dset_atot][md.small.eventNum,i] = atot
            myHdf5[grpName+dset_nPeaks][md.small.eventNum] = nPeaks
            myHdf5[grpName+dset_maxRes][md.small.eventNum] = maxRes
            counter += 1
            #if counter == saveInterval:
            #    myHdf5.close()
    #print "### Done clients"
    if '/status/findPeaks' in myHdf5:
        del myHdf5['/status/findPeaks']
    myHdf5['/status/findPeaks'] = 'success'
    myHdf5.close()

def convert_peaks_to_cheetah(s, r, c) :
    """Converts seg, row, col assuming (32,185,388)
       to cheetah 2-d table row and col (8*185, 4*388)
    """
    segs, rows, cols = (32,185,388)
    row2d = (int(s)%8) * rows + int(r) # where s%8 is a segment in quad number [0,7]
    col2d = (int(s)/8) * cols + int(c) # where s/8 is a quad number [0,3]
    return row2d, col2d

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
    print "run, numJobs: ", args.run, numJobs
    return numJobs


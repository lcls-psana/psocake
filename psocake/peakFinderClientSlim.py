import numpy as np
import time
import os
import PeakFinder as pf
import h5py
from utils import *
import PSCalib.GlobalUtils as gu

facility = 'LCLS'
import psana

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def get_es_value(es, name, NoneCheck=False, exceptReturn=0):
    try:
        value = es.value(name)
        if NoneCheck and (value is None): value = exceptReturn
    except:
        value = exceptReturn
    return value

def calcPeaks(args, nHits, myHdf5, detarr, evt, d, nevent):
    tic = time.time()
    d.peakFinder.findPeaks(detarr, evt, args.minPeaks) # this will perform background subtraction on detarr
    toc = time.time()
    myHdf5["/entry_1/result_1/nPeaksAll"][nevent] = len(d.peakFinder.peaks)
    if nHits%500==0:
        print "time to find/write peaks: ", rank, toc-tic, time.time()-toc
    if len(d.peakFinder.peaks) >= args.minPeaks and \
       len(d.peakFinder.peaks) <= args.maxPeaks and \
       d.peakFinder.maxRes >= args.minRes:
        tic = time.time()
        evtId = evt.get(psana.EventId)
        myHdf5['/LCLS/eventNumber'][nHits] = nevent
        myHdf5['/LCLS/machineTime'][nHits] = evtId.time()[0]
        myHdf5['/LCLS/machineTimeNanoSeconds'][nHits] = evtId.time()[1]
        myHdf5['/LCLS/fiducial'][nHits] = evtId.fiducials()
        myHdf5['/entry_1/data_1/data'][nHits,:,:] = pct(args.det, detarr)
        segs = d.peakFinder.peaks[:,0]
        rows = d.peakFinder.peaks[:,1]
        cols = d.peakFinder.peaks[:,2]
        amaxs = d.peakFinder.peaks[:,4]
        atots = d.peakFinder.peaks[:,5]
        cheetahRows, cheetahCols = convert_peaks_to_cheetah(args.det, segs, rows, cols)
        myHdf5["/entry_1/result_1/nPeaks"][nHits] = len(d.peakFinder.peaks)
        myHdf5["/entry_1/result_1/peakXPosRaw"][nHits, :len(d.peakFinder.peaks)] = cheetahCols.astype('int')
        myHdf5["/entry_1/result_1/peakYPosRaw"][nHits, :len(d.peakFinder.peaks)] = cheetahRows.astype('int')
        myHdf5["/entry_1/result_1/peakTotalIntensity"][nHits, :len(d.peakFinder.peaks)] = atots
        myHdf5["/entry_1/result_1/peakMaxIntensity"][nHits, :len(d.peakFinder.peaks)] = amaxs
        """
        for j, peak in enumerate(d.peakFinder.peaks):
            seg,row,col,npix,amax,atot,rcent,ccent,rsigma,csigma,rmin,rmax,cmin,cmax,bkgd,rms,son = peak[0:17]
            cheetahRow, cheetahCol = convert_peaks_to_cheetah(args.det, seg, row, col)
            myHdf5["/entry_1/result_1/nPeaks"][nHits] = len(d.peakFinder.peaks)
            myHdf5["/entry_1/result_1/peakXPosRaw"][nHits,j] = cheetahCol
            myHdf5["/entry_1/result_1/peakYPosRaw"][nHits,j] = cheetahRow
            myHdf5["/entry_1/result_1/peakTotalIntensity"][nHits,j] = atot
            myHdf5["/entry_1/result_1/peakMaxIntensity"][nHits,j] = amax
        """
        if nHits % 500 == 0:
            print "time to write cxi peaks: ", len(d.peakFinder.peaks), time.time() - tic
        nHits += 1
        myHdf5.flush()
    return nHits

def readMask(args):
    # Init mask
    mask = None
    if args.mask is not None:
        f = h5py.File(args.mask, 'r')
        mask = f['/entry_1/data_1/mask'][()]
        f.close()
        mask = -1*(mask-1)
    return mask

def runclient(args,ds,run,times,det,numEvents):
    numHits = 0
    myJobs = getMyUnfairShare(numEvents, size, rank)
    numJobs = len(myJobs) # events per rank

    runStr = "%04d" % args.run
    fname = args.outDir + '/' + args.exp +"_"+ runStr +"_"+str(rank)
    if args.tag: fname += '_' + args.tag
    fname += ".cxi"
    myHdf5 = h5py.File(fname,"r+")

    for i, nevent in enumerate(myJobs):
        #print "nevent: ", rank, nevent
        if i%200 == 0 and i > 0: print "hits, hit rate: ", numHits, numHits*1./i
        evt = run.event(times[nevent])
        if evt is None: continue

        if not args.inputImages:
            if args.cm0 > 0: # override common mode correction
                if args.cm0 == 5:  # Algorithm 5
                    detarr = det.calib(evt, cmpars=(args.cm0, args.cm1))
                else:  # Algorithms 1 to 4
                    detarr = det.calib(evt, cmpars=(args.cm0, args.cm1, args.cm2, args.cm3))
            else:
                tic = time.time()
                detarr = det.calib(evt)
                if i % 200 == 0: print "time to fetch calib: ", rank, time.time() - tic
        else:
            f = h5py.File(args.inputImages)
            ind = np.where(f['eventNumber'][()] == nevent)[0][0]
            if len(f['/data/data'].shape) == 3:
                detarr = ipct(args.det, f['data/data'][ind, :, :])
            else:
                detarr = f['data/data'][ind, :, :, :]
            f.close()

        if detarr is None: continue

        # Initialize hit finding
        if not hasattr(det,'peakFinder'):
            if args.algorithm == 1:
                if facility == 'LCLS':
                    det.peakFinder = pf.PeakFinder(args.exp, args.run, args.det, evt, det,
                                              args.algorithm, args.alg_npix_min,
                                              args.alg_npix_max, args.alg_amax_thr,
                                              args.alg_atot_thr, args.alg_son_min,
                                              alg1_thr_low=args.alg1_thr_low,
                                              alg1_thr_high=args.alg1_thr_high,
                                              alg1_rank=args.alg1_rank,
                                              alg1_radius=args.alg1_radius,
                                              alg1_dr=args.alg1_dr,
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
                                              psanaMask_unbondnrs=args.psanaMask_unbondnrs,
                                              medianFilterOn=args.medianBackground,
                                              medianRank=args.medianRank,
                                              radialFilterOn=args.radialBackground,
                                              distance=args.detectorDistance,
                                              minNumPeaks=args.minPeaks,
                                              maxNumPeaks=args.maxPeaks,
                                              minResCutoff=args.minRes,
                                              clen=args.clen,
                                              localCalib=args.localCalib,
                                              access=args.access)
            elif args.algorithm >= 2:
                print "init alg: ", rank
                det.peakFinder = pf.PeakFinder(args.exp, args.run, args.det, evt, det,
                                             args.algorithm, args.alg_npix_min,
                                             args.alg_npix_max, args.alg_amax_thr,
                                             args.alg_atot_thr, args.alg_son_min,
                                             alg1_thr_low=args.alg1_thr_low,
                                             alg1_thr_high=args.alg1_thr_high,
                                             alg1_rank=args.alg1_rank,
                                             alg1_radius=args.alg1_radius,
                                             alg1_dr=args.alg1_dr,
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
                                             psanaMask_unbondnrs=args.psanaMask_unbondnrs,
                                             medianFilterOn=args.medianBackground,
                                             medianRank=args.medianRank,
                                             radialFilterOn=args.radialBackground,
                                             distance=args.detectorDistance,
                                             minNumPeaks=args.minPeaks,
                                             maxNumPeaks=args.maxPeaks,
                                             minResCutoff=args.minRes,
                                             clen=args.clen,
                                             localCalib=args.localCalib,
                                             access=args.access)
            ix = det.indexes_x(evt)
            iy = det.indexes_y(evt)
            det.iX = np.array(ix, dtype=np.int64)
            det.iY = np.array(iy, dtype=np.int64)
            #det.ipx, det.ipy = det.point_indexes(evt, pxy_um=(0, 0))
            det.ipx, det.ipyx = det.point_indexes(evt, pxy_um=(0, 0),
                                                  pix_scale_size_um=None,
                                                  xy0_off_pix=None,
                                                  cframe=gu.CFRAME_PSANA, fract=True)

        numHits = calcPeaks(args, numHits, myHdf5, detarr, evt, det, nevent)
        #if numHits%50==0:
        #    print "Rank " + str(rank) + " found " + str(numHits) + " hits (hit rate): " + str(numHits*1./numJobs)
    # Finished with peak finding
    # Fill in clen and photon energy (eV)
    es = ds.env().epicsStore()
    if 'mfxc00318' in args.exp:
        lclsDet = 303.8794  # mm
    else:
        lclsDet = es.value(args.clen)  # mm
        if lclsDet is None:
            lclsDet = 0.0

    if 'mfxc00318' in args.exp:
        photonEnergy = 9.8714e3 # eV
    elif 'cxic00318' in args.exp:
        photonEnergy = 9.25e3  # eV
    elif 'cxilv4418' in args.exp:
        photonEnergy = 9.86e3 # eV
    elif 'mfxp17218' in args.exp:
        photonEnergy = 9.86e3 # eV
    elif 'mfxp17118' in args.exp:
        photonEnergy = 9.86e3 # eV
    else:
        try:
            photonEnergy = 0
            wavelength = get_es_value(es, 'SIOC:SYS0:ML00:AO192', NoneCheck=False, exceptReturn=0)
            if wavelength > 0:
                h = 6.626070e-34  # J.m
                c = 2.99792458e8  # m/s
                joulesPerEv = 1.602176621e-19  # J/eV
                photonEnergy = (h / joulesPerEv * c) / (wavelength * 1e-9)
        except:
            ebeamDet = psana.Detector('EBeam')
            ebeam = ebeamDet.get(evt)
            photonEnergy = ebeam.ebeamPhotonEnergy()

    myHdf5['/LCLS/detector_1/EncoderValue'][...] = lclsDet
    myHdf5['/LCLS/photon_energy_eV'][...] = photonEnergy

    # crop
    cropHdf5(myHdf5, '/entry_1/result_1/nPeaks', numHits)
    cropHdf5(myHdf5, '/LCLS/photon_energy_eV', numHits)
    cropHdf5(myHdf5, '/LCLS/detector_1/EncoderValue', numHits)
    cropHdf5(myHdf5, '/LCLS/eventNumber', numHits)
    cropHdf5(myHdf5, '/LCLS/machineTime', numHits)
    cropHdf5(myHdf5, '/LCLS/machineTimeNanoSeconds', numHits)
    cropHdf5(myHdf5, '/LCLS/fiducial', numHits)

    # resize
    dataShape = myHdf5["/entry_1/data_1/data"].shape
    myHdf5["/entry_1/data_1/data"].resize((numHits, dataShape[1], dataShape[2]))
    myHdf5["/entry_1/result_1/peakXPosRaw"].resize((numHits,args.maxPeaks))
    myHdf5["/entry_1/result_1/peakYPosRaw"].resize((numHits,args.maxPeaks))
    myHdf5["/entry_1/result_1/peakTotalIntensity"].resize((numHits,args.maxPeaks))
    myHdf5["/entry_1/result_1/peakMaxIntensity"].resize((numHits,args.maxPeaks))

    # add powder
    myHdf5["/entry_1/data_1/powderHits"][...] = pct(args.det, det.peakFinder.powderHits)
    myHdf5["/entry_1/data_1/powderMisses"][...] = pct(args.det, det.peakFinder.powderMisses)

    runStr = "%04d" % args.run
    fname = args.outDir + '/' + args.exp + "_" + runStr + "_maxHits"
    if args.tag: fname += '_' + args.tag
    fname += ".npy"
    np.save(fname, det.peakFinder.powderHits)
    fname = args.outDir + '/' + args.exp + "_" + runStr + "_maxMisses"
    if args.tag: fname += '_' + args.tag
    fname += ".npy"
    np.save(fname, det.peakFinder.powderMisses)

    # attach attr
    myHdf5["/LCLS/eventNumber"].attrs["numEvents"] = numJobs

    # TODO: add mask
    if args.mask is not None:
        myHdf5['/entry_1/data_1/mask'][:,:] = readMask(args)

    if '/status/findPeaks' in myHdf5: del myHdf5['/status/findPeaks']
    myHdf5['/status/findPeaks'] = 'success'
    myHdf5.flush()
    myHdf5.close()

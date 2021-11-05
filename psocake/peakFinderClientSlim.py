import numpy as np
import h5py
import PSCalib.GlobalUtils as gu
import psana
import psocake.PeakFinder as pf
from psocake.utils import *
from psocake.cheetahUtils import readMask

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

facility = 'LCLS'

def get_es_value(es, name, NoneCheck=False, exceptReturn=0):
    try:
        value = es.value(name)
        if NoneCheck and (value is None): value = exceptReturn
    except:
        value = exceptReturn
    return value


def calcPeaks(args, nHits, myHdf5, detarr, evt, d, nevent, detDesc, es, evr0, evr1, evr2):
    """ Find peaks and writes to cxi file """
    d.peakFinder.findPeaks(detarr, evt, args.minPeaks) # this will perform background subtraction on detarr
    myHdf5["/entry_1/result_1/nPeaksAll"][nevent] = len(d.peakFinder.peaks)
    
    if len(d.peakFinder.peaks) >= args.minPeaks and \
       len(d.peakFinder.peaks) <= args.maxPeaks and \
       d.peakFinder.maxRes >= args.minRes:
        evtId = evt.get(psana.EventId)
        myHdf5['/LCLS/eventNumber'][nHits] = nevent
        myHdf5['/LCLS/machineTime'][nHits] = evtId.time()[0]
        myHdf5['/LCLS/machineTimeNanoSeconds'][nHits] = evtId.time()[1]
        myHdf5['/LCLS/fiducial'][nHits] = evtId.fiducials()
        _data = detDesc.pct(detarr)
        myHdf5['/entry_1/data_1/data'][nHits,:,:] = _data
        # https://confluence.slac.stanford.edu/display/PSDM/Hit+and+Peak+Finding+Algorithms#HitandPeakFindingAlgorithms-Peakfinders
        segs = d.peakFinder.peaks[:,0]
        rows = d.peakFinder.peaks[:,1]
        cols = d.peakFinder.peaks[:,2]
        amaxs = d.peakFinder.peaks[:,4]
        atots = d.peakFinder.peaks[:,5]
        rcentT = d.peakFinder.peaks[:,6] # row center of gravity
        ccentT = d.peakFinder.peaks[:,7] # col center of gravity
        rminT = d.peakFinder.peaks[:,10] # minimal row of pixel group accounted in the peak
        rmaxT = d.peakFinder.peaks[:,11] # maximal row "
        cminT = d.peakFinder.peaks[:,12] # minimal col "
        cmaxT = d.peakFinder.peaks[:,13] # maximal col "
        cheetahRows, cheetahCols = detDesc.convert_peaks_to_cheetah(segs, rows, cols) #,
        myHdf5["/entry_1/result_1/nPeaks"][nHits] = len(d.peakFinder.peaks)
        myHdf5["/entry_1/result_1/peakXPosRaw"][nHits, :len(d.peakFinder.peaks)] = cheetahCols.astype('int')
        myHdf5["/entry_1/result_1/peakYPosRaw"][nHits, :len(d.peakFinder.peaks)] = cheetahRows.astype('int')

        myHdf5["/entry_1/result_1/rcent"][nHits, :len(d.peakFinder.peaks)] = rcentT
        myHdf5["/entry_1/result_1/ccent"][nHits, :len(d.peakFinder.peaks)] = ccentT
        myHdf5["/entry_1/result_1/rmin"][nHits, :len(d.peakFinder.peaks)] = rminT
        myHdf5["/entry_1/result_1/rmax"][nHits, :len(d.peakFinder.peaks)] = rmaxT
        myHdf5["/entry_1/result_1/cmin"][nHits, :len(d.peakFinder.peaks)] = cminT
        myHdf5["/entry_1/result_1/cmax"][nHits, :len(d.peakFinder.peaks)] = cmaxT

        myHdf5["/entry_1/result_1/peakTotalIntensity"][nHits, :len(d.peakFinder.peaks)] = atots
        myHdf5["/entry_1/result_1/peakMaxIntensity"][nHits, :len(d.peakFinder.peaks)] = amaxs

        cenX = d.iX[np.array(d.peakFinder.peaks[:, 0], dtype=np.int64),
                    np.array(d.peakFinder.peaks[:, 1], dtype=np.int64),
                    np.array(d.peakFinder.peaks[:, 2], dtype=np.int64)] + 0.5
        cenY = d.iY[np.array(d.peakFinder.peaks[:, 0], dtype=np.int64),
                    np.array(d.peakFinder.peaks[:, 1], dtype=np.int64),
                    np.array(d.peakFinder.peaks[:, 2], dtype=np.int64)] + 0.5
        x = cenX - d.ipx
        y = cenY - d.ipy
        radius = np.sqrt((x ** 2) + (y ** 2))
        myHdf5["/entry_1/result_1/peakRadius"][nHits, :len(d.peakFinder.peaks)] = radius

        # Save epics variables
        # FIXME: Timetool variable name change (Nov/2021) TTSPEC -> TIMETOOL
        instrument = args.instrument.upper()
        timeToolDelay = get_es_value(es, instrument+':LAS:MMN:04.RBV', NoneCheck=True)
        laserTimeZero = get_es_value(es, 'LAS:FS5:VIT:FS_TGT_TIME_OFFSET', NoneCheck=True)
        laserTimeDelay = get_es_value(es, 'LAS:FS5:VIT:FS_TGT_TIME_DIAL', NoneCheck=True)
        laserTimePhaseLocked = get_es_value(es, 'LAS:FS5:VIT:PHASE_LOCKED', NoneCheck=True)
        ttspecAmpl = get_es_value(es, instrument+':TIMETOOL:AMPL', NoneCheck=True)
        ttspecAmplNxt = get_es_value(es, instrument+':TIMETOOL:AMPLNXT', NoneCheck=True)
        ttspecFltPos = get_es_value(es, instrument+':TIMETOOL:FLTPOS', NoneCheck=True)
        ttspecFltPosFwhm = get_es_value(es, instrument+':TIMETOOL:FLTPOSFWHM', NoneCheck=True)
        ttspecFltPosPs = get_es_value(es, instrument+':TIMETOOL:FLTPOS_PS', NoneCheck=True)
        ttspecRefAmpl = get_es_value(es, instrument+':TIMETOOL:REFAMPL', NoneCheck=True)
        myHdf5['/entry_1/result_1/timeToolDelay'][nHits] = timeToolDelay
        myHdf5['/entry_1/result_1/laserTimeZero'][nHits] = laserTimeZero
        myHdf5['/entry_1/result_1/laserTimeDelay'][nHits] = laserTimeDelay
        myHdf5['/entry_1/result_1/laserTimePhaseLocked'][nHits] = laserTimePhaseLocked
        myHdf5['/LCLS/ttspecAmpl'][nHits] = ttspecAmpl
        myHdf5['/LCLS/ttspecAmplNxt'][nHits] = ttspecAmplNxt
        myHdf5['/LCLS/ttspecFltPos'][nHits] = ttspecFltPos
        myHdf5['/LCLS/ttspecFltPosFwhm'][nHits] = ttspecFltPosFwhm
        myHdf5['/LCLS/ttspecFltPosPs'][nHits] = ttspecFltPosPs
        myHdf5['/LCLS/ttspecRefAmpl'][nHits] = ttspecRefAmpl

        if evr0:
            ec = evr0.eventCodes(evt)
            if ec is None: ec = [-1]
            myHdf5['/LCLS/detector_1/evr0'][nHits] = np.array(ec)
        if evr1:
            ec = evr1.eventCodes(evt)
            if ec is None: ec = [-1]
            myHdf5['/LCLS/detector_1/evr1'][nHits] = np.array(ec)
        if evr2:
            ec = evr2.eventCodes(evt)
            if ec is None: ec = [-1]
            myHdf5['/LCLS/detector_1/evr2'][nHits] = np.array(ec)
        nHits += 1

    return nHits

def getValidEvent(run, times, myJobs):
    for i, nevent in enumerate(myJobs):
        evt = run.event(times[nevent])
        if evt is None: continue
        return evt

def runclient(args,ds,run,times,det,numEvents,detDesc):
    numHits = 0
    es = ds.env().epicsStore()
    try:
        evr0 = psana.Detector('evr0')
    except:
        evr0 = None
    try:
        evr1 = psana.Detector('evr1')
    except:
        evr1 = None
    try:
        evr2 = psana.Detector('evr2')
    except:
        evr2 = None

    myJobs = getMyUnfairShare(numEvents, size, rank)
    numJobs = len(myJobs) # events per rank
    reportFreq = numJobs/2 + 1

    runStr = "%04d" % args.run
    fname = args.outDir + '/' + args.exp +"_"+ runStr +"_"+str(rank)
    if args.tag: fname += '_' + args.tag
    fname += ".cxi"
    myHdf5 = h5py.File(fname,"r+")

    evt = getValidEvent(run, times, myJobs)

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
        try:
            det.ipx, det.ipy = det.point_indexes(evt, pxy_um=(0, 0),
                                              pix_scale_size_um=None,
                                              xy0_off_pix=None,
                                              cframe=gu.CFRAME_PSANA, fract=True)
        except AttributeError:
            det.ipx, det.ipy = det.point_indexes(evt, pxy_um=(0, 0))

    for i, nevent in enumerate(myJobs):
        evt = run.event(times[nevent])
        if evt is None: continue

        if not args.inputImages:
            if args.cm0 > 0: # override common mode correction
                if args.cm0 == 5:  # Algorithm 5
                    detarr = det.calib(evt, cmpars=(args.cm0, args.cm1))
                else:  # Algorithms 1 to 4
                    detarr = det.calib(evt, cmpars=(args.cm0, args.cm1, args.cm2, args.cm3))
            else:
                detarr = det.calib(evt)
        else:
            f = h5py.File(args.inputImages)
            ind = np.where(f['eventNumber'][()] == nevent)[0][0]
            if len(f['/data/data'].shape) == 3:
                detarr = detDesc.ipct(f['data/data'][ind, :, :])
            else:
                detarr = f['data/data'][ind, :, :, :]
            f.close()
        if detarr is None: continue

        numHits = calcPeaks(args, numHits, myHdf5, detarr, evt, det, nevent, detDesc, es, evr0, evr1, evr2)


    # Finished with peak finding
    # Fill in clen and photon energy (eV)
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

    myHdf5["/entry_1/result_1/rcent"].resize((numHits, args.maxPeaks))
    myHdf5["/entry_1/result_1/ccent"].resize((numHits, args.maxPeaks))
    myHdf5["/entry_1/result_1/rmin"].resize((numHits, args.maxPeaks))
    myHdf5["/entry_1/result_1/rmax"].resize((numHits, args.maxPeaks))
    myHdf5["/entry_1/result_1/cmin"].resize((numHits, args.maxPeaks))
    myHdf5["/entry_1/result_1/cmax"].resize((numHits, args.maxPeaks))

    myHdf5["/entry_1/result_1/peakTotalIntensity"].resize((numHits,args.maxPeaks))
    myHdf5["/entry_1/result_1/peakMaxIntensity"].resize((numHits,args.maxPeaks))
    myHdf5["/entry_1/result_1/peakRadius"].resize((numHits, args.maxPeaks))

    # add powder
    myHdf5["/entry_1/data_1/powderHits"][...] = detDesc.pct(det.peakFinder.powderHits)
    myHdf5["/entry_1/data_1/powderMisses"][...] = detDesc.pct(det.peakFinder.powderMisses)

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
        myHdf5['/entry_1/data_1/mask'][:,:] = readMask(args.mask)

    if '/status/findPeaks' in myHdf5: del myHdf5['/status/findPeaks']
    myHdf5['/status/findPeaks'] = 'success'
    myHdf5.close()

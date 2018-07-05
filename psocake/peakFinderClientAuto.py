import numpy as np
from mpidata import mpidata
import time
import os
import PeakFinder as pf
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import h5py

#
if 'PSOCAKE_FACILITY' not in os.environ: os.environ['PSOCAKE_FACILITY'] = 'LCLS' # Default facility
if 'LCLS' in os.environ['PSOCAKE_FACILITY'].upper():
    facility = 'LCLS'
    import psanaWhisperer, psana
elif 'PAL' in os.environ['PSOCAKE_FACILITY'].upper():
    facility = 'PAL'
    import glob, h5py

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def str2bool(v): return v.lower() in ("yes", "true", "t", "1")

def runclient(args):
    pairsFoundPerSpot = 0.0
    highSigma = 3.5
    lowSigma = 2.5
    likelihoodThresh = 0.035

    if facility == 'LCLS':
        access = "exp="+args.exp+":run="+str(args.run)+':idx'
        if 'ffb' in args.access.lower(): access += ':dir=/reg/d/ffb/' + args.exp[:3] + '/' + args.exp + '/xtc'
        ds = psana.DataSource(access)
        run = ds.runs().next()
        env = ds.env()
        times = run.times()
        d = psana.Detector(args.det)
        d.do_reshape_2d_to_3d(flag=True)
        ps = psanaWhisperer.psanaWhisperer(args.exp, args.run, args.det, args.clen, args.localCalib, access=args.access)
        ps.setupExperiment()
        ebeamDet = psana.Detector('EBeam')
        try:
            evr0 = psana.Detector('evr0')
        except:
            evr0 = None
        try:
            evr1 = psana.Detector('evr1')
        except:
            evr1 = None

    elif facility == 'PAL':
        temp = args.dir + '/' + args.exp[:3] + '/' + args.exp + '/data/r' + str(args.run).zfill(4) + '/*.h5'
        _files = glob.glob(temp)
        numEvents = len(_files)
        if args.noe == -1 or args.noe > numEvents:
            times = np.arange(numEvents)
        else:
            times = np.arange(args.noe)

        class Object(object): pass
        d = Object()

    hasCoffset = False
    hasDetectorDistance = False
    if args.detectorDistance is not 0:
        hasDetectorDistance = True
    if args.coffset is not 0:
        hasCoffset = True

    if facility == 'LCLS':
        if hasCoffset:
            try:
                detectorDistance = args.coffset + ps.clen * 1e-3  # sample to detector in m
            except:
                detectorDistance = 0
        elif hasDetectorDistance:
            detectorDistance = args.detectorDistance
    
    for nevent in np.arange(len(times)):
        if nevent == args.noe : break
        if nevent%(size-1) != rank-1: continue # different ranks look at different events

        if args.profile: startTic = time.time()

        #print "rank, event: ", rank, nevent

        if facility == 'LCLS':
            evt = run.event(times[nevent])
            detarr = d.calib(evt)
            exp = env.experiment()
            #run = evt.run()
        elif facility == 'PAL':
            f = h5py.File(_files[nevent], 'r')
            detarr = f['/data'].value
            f.close()
            exp = args.exp
            run = args.run

        if args.profile: calibTime = time.time() - startTic # Time to calibrate per event

        if detarr is None: continue

        # Initialize hit finding
        if not hasattr(d,'peakFinder'):
            if args.algorithm == 1:
                if facility == 'LCLS':
                    if not str2bool(args.auto):
                        d.peakFinder = pf.PeakFinder(exp, args.run, args.det, evt, d,
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
                    else:
                        # Auto peak finder
                        d.peakFinder = pf.PeakFinder(exp, args.run, args.det, evt, d,
                                                     args.algorithm, args.alg_npix_min,
                                                     args.alg_npix_max, args.alg_amax_thr,
                                                     0, args.alg_son_min,
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
                        # Read in powder pattern and calculate pixel indices
                        powderSumFname = args.outDir + '/background.npy'
                        powderSum = np.load(powderSumFname)
                        powderSum1D = powderSum.ravel()
                        cx, cy = d.indexes_xy(evt)
                        d.ipx, d.ipy = d.point_indexes(evt, pxy_um=(0, 0))
                        r = np.sqrt((cx - d.ipx) ** 2 + (cy - d.ipy) ** 2).ravel().astype(int)
                        startR = 0
                        endR = np.max(r)
                        profile = np.zeros(endR - startR, )
                        for i, val in enumerate(np.arange(startR, endR)):
                            ind = np.where(r == val)[0].astype(int)
                            if len(ind) > 0:
                                profile[i] = np.mean(powderSum1D[ind])
                        myThreshInd = np.argmax(profile)
                        print "###################################################"
                        print "Solution scattering radius (pixels): ", myThreshInd
                        print "###################################################"
                        thickness = 10
                        indLo = np.where(r >= myThreshInd - thickness / 2.)[0].astype(int)
                        indHi = np.where(r <= myThreshInd + thickness / 2.)[0].astype(int)
                        d.ind = np.intersect1d(indLo, indHi)

                        ix = d.indexes_x(evt)
                        iy = d.indexes_y(evt)
                        d.iX = np.array(ix, dtype=np.int64)
                        d.iY = np.array(iy, dtype=np.int64)
                elif facility == 'PAL':
                    _geom = args.dir + '/' + args.exp[:3] + '/' + args.exp + '/scratch/' + os.environ['USER'] + \
                            '/psocake/r' + str(args.currentRun).zfill(4) + '/.temp.geom'
                    (detectorDistance, photonEnergy, _, _, _) = readCrystfelGeometry(_geom, facility)
                    evt = None
                    d.peakFinder = pf.PeakFinder(exp, run, args.det, evt, d,
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
                                              geom=_geom)
            elif args.algorithm == 2:
                d.peakFinder = pf.PeakFinder(exp, args.run, args.det, evt, d,
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
            ix = d.indexes_x(evt)
            iy = d.indexes_y(evt)
            d.iX = np.array(ix, dtype=np.int64)
            d.iY = np.array(iy, dtype=np.int64)
            d.ipx, d.ipy = d.point_indexes(evt, pxy_um=(0, 0))

        d.peakFinder.findPeaks(detarr, evt)
        # Likelihood
        numPeaksFound = d.peakFinder.peaks.shape[0]
        if numPeaksFound >= args.minPeaks and \
           numPeaksFound <= args.maxPeaks and \
           d.peakFinder.maxRes >= args.minRes:
            cenX = d.iX[np.array(d.peakFinder.peaks[:, 0], dtype=np.int64),
                        np.array(d.peakFinder.peaks[:, 1], dtype=np.int64),
                        np.array(d.peakFinder.peaks[:, 2], dtype=np.int64)] + 0.5
            cenY = d.iY[np.array(d.peakFinder.peaks[:, 0], dtype=np.int64),
                        np.array(d.peakFinder.peaks[:, 1], dtype=np.int64),
                        np.array(d.peakFinder.peaks[:, 2], dtype=np.int64)] + 0.5

            x = cenX - d.ipx  # args.center[0]
            y = cenY - d.ipy  # args.center[1]

            pixSize = float(d.pixel_size(evt))
            detdis = float(args.detectorDistance)
            z = detdis / pixSize * np.ones(x.shape)  # pixels

            ebeam = ebeamDet.get(evt)
            try:
                photonEnergy = ebeam.ebeamPhotonEnergy()
            except:
                photonEnergy = 1

            wavelength = 12.407002 / float(photonEnergy)  # Angstrom
            norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            qPeaks = (np.array([x, y, z]) / norm - np.array([[0.], [0.], [1.]])) / wavelength
            [meanClosestNeighborDist, pairsFoundPerSpot] = calculate_likelihood(qPeaks)
        else:
            pairsFoundPerSpot = 0.0

        md=mpidata()
        md.addarray('peaks', d.peakFinder.peaks)
        md.small.eventNum = nevent
        md.small.maxRes = d.peakFinder.maxRes
        md.small.powder = 0
        md.small.likelihood = pairsFoundPerSpot

        if args.profile:
            md.small.calibTime = calibTime
            md.small.peakTime = peakTime

        if facility == 'LCLS':
            # other cxidb data
            ps.getEvent(nevent)

            es = ps.ds.env().epicsStore()

            # timetool
            try:
                md.small.timeToolDelay = es.value(args.instrument+':LAS:MMN:04.RBV')
                if md.small.timeToolDelay is None: md.small.timeToolDelay = 0
            except:
                md.small.timeToolDelay = 0

            try:
                md.small.laserTimeZero = es.value('LAS:FS5:VIT:FS_TGT_TIME_OFFSET')
                if md.small.laserTimeZero is None: md.small.laserTimeZero = 0
            except:
                md.small.laserTimeZero = 0

            try:
                md.small.laserTimeDelay = es.value('LAS:FS5:VIT:FS_TGT_TIME_DIAL')
                if md.small.laserTimeDelay is None: md.small.laserTimeDelay = 0
            except:
                md.small.laserTimeDelay = 0

            try:
                md.small.laserTimePhaseLocked = es.value('LAS:FS5:VIT:PHASE_LOCKED')
                if md.small.laserTimePhaseLocked is None: md.small.laserTimePhaseLocked = 0
            except:
                md.small.laserTimePhaseLocked = 0

            try:
                md.small.ttspecAmpl = es.value(args.instrument+':TTSPEC:AMPL')
                if md.small.ttspecAmpl is None: md.small.ttspecAmpl = 0
            except:
                md.small.ttspecAmpl = 0

            try:
                md.small.ttspecAmplNxt = es.value(args.instrument+':TTSPEC:AMPLNXT')
                if md.small.ttspecAmplNxt is None: md.small.ttspecAmplNxt = 0
            except:
                md.small.ttspecAmplNxt = 0

            try:
                md.small.ttspecFltPos = es.value(args.instrument+':TTSPEC:FLTPOS')
                if md.small.ttspecFltPos is None: md.small.ttspecFltPos = 0
            except:
                md.small.ttspecFltPos = 0

            try:
                md.small.ttspecFltPosFwhm = es.value(args.instrument+':TTSPEC:FLTPOSFWHM')
                if md.small.ttspecFltPosFwhm is None: md.small.ttspecFltPosFwhm = 0
            except:
                md.small.ttspecFltPosFwhm = 0

            try:
                md.small.ttspecFltPosPs = es.value(args.instrument+':TTSPEC:FLTPOS_PS')
                if md.small.ttspecFltPosPs is None: md.small.ttspecFltPosPs = 0
            except:
                md.small.ttspecFltPosPs = 0

            try:
                md.small.ttspecRefAmpl = es.value(args.instrument+':TTSPEC:REFAMPL')
                if md.small.ttspecRefAmpl is None: md.small.ttspecRefAmpl = 0
            except:
                md.small.ttspecRefAmpl = 0

            if evr0:
                ec = evr0.eventCodes(evt)
                if ec is None: ec = [-1]
                md.addarray('evr0', np.array(ec))

            if evr1:
                ec = evr1.eventCodes(evt)
                if ec is None: ec = [-1]
                md.addarray('evr1', np.array(ec))

            # pulse length
            try:
                pulseLength = es.value('SIOC:SYS0:ML00:AO820') * 1e-15  # s
            except:
                pulseLength = 0

            md.small.pulseLength = pulseLength

            md.small.detectorDistance = detectorDistance

            md.small.pixelSize = args.pixelSize

            # LCLS
            if "cxi" in args.exp:
                md.small.lclsDet = es.value(args.clen)  # mm
            elif "mfx" in args.exp:
                md.small.lclsDet = es.value(args.clen)  # mm
            elif "xpp" in args.exp:
                md.small.lclsDet = es.value(args.clen)  # mm

            try:
                md.small.ebeamCharge = es.value('BEND:DMP1:400:BDES')
            except:
                md.small.ebeamCharge = 0

            try:
                md.small.beamRepRate = es.value('EVNT:SYS0:1:LCLSBEAMRATE')
            except:
                md.small.beamRepRate = 0

            try:
                md.small.particleN_electrons = es.value('BPMS:DMP1:199:TMIT1H')
            except:
                md.small.particleN_electrons = 0

            try:
                md.small.eVernier = es.value('SIOC:SYS0:ML00:AO289')
            except:
                md.small.eVernier = 0

            try:
                md.small.charge = es.value('BEAM:LCLS:ELEC:Q')
            except:
                md.small.charge = 0

            try:
                md.small.peakCurrentAfterSecondBunchCompressor = es.value('SIOC:SYS0:ML00:AO195')
            except:
                md.small.peakCurrentAfterSecondBunchCompressor = 0

            try:
                md.small.pulseLength = es.value('SIOC:SYS0:ML00:AO820')
            except:
                md.small.pulseLength = 0

            try:
                md.small.ebeamEnergyLossConvertedToPhoton_mJ = es.value('SIOC:SYS0:ML00:AO569')
            except:
                md.small.ebeamEnergyLossConvertedToPhoton_mJ = 0

            try:
                md.small.calculatedNumberOfPhotons = es.value('SIOC:SYS0:ML00:AO580') * 1e12  # number of photons
            except:
                md.small.calculatedNumberOfPhotons = 0

            try:
                md.small.photonBeamEnergy = es.value('SIOC:SYS0:ML00:AO541')
            except:
                md.small.photonBeamEnergy = 0

            try:
                md.small.wavelength = es.value('SIOC:SYS0:ML00:AO192')
            except:
                md.small.wavelength = 0

            try:
                md.small.injectorPressureSDS = es.value('CXI:LC20:SDS:Pressure')
            except:
                md.small.injectorPressureSDS = 0

            try:
                md.small.injectorPressureSDSB = es.value('CXI:LC20:SDSB:Pressure')
            except:
                md.small.injectorPressureSDSB = 0

            ebeam = ebeamDet.get(ps.evt)#.get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
            try:
                photonEnergy = md.small.photonBeamEnergy #ebeam.ebeamPhotonEnergy()
                pulseEnergy = ebeam.ebeamL3Energy()  # MeV
            except:
                photonEnergy = 0
                pulseEnergy = 0
                if md.small.wavelength > 0:
                    h = 6.626070e-34  # J.m
                    c = 2.99792458e8  # m/s
                    joulesPerEv = 1.602176621e-19  # J/eV
                    photonEnergy = (h / joulesPerEv * c) / (md.small.wavelength * 1e-9)

            md.small.photonEnergy = photonEnergy
            md.small.pulseEnergy = pulseEnergy

            evtId = ps.evt.get(psana.EventId)
            md.small.sec = evtId.time()[0]
            md.small.nsec = evtId.time()[1]
            md.small.fid = evtId.fiducials()

            if len(d.peakFinder.peaks) >= args.minPeaks and \
               len(d.peakFinder.peaks) <= args.maxPeaks and \
               d.peakFinder.maxRes >= args.minRes:
               #and pairsFoundPerSpot >= likelihoodThresh:
                # Write image in cheetah format
                img = ps.getCheetahImg()
                if img is not None:
                    md.addarray('data', img)

            if args.profile:
                totalTime = time.time() - startTic
                md.small.totalTime = totalTime
                md.small.rankID = rank
            md.send() # send mpi data object to master when desired
        elif facility == 'PAL':
            if len(d.peakFinder.peaks) >= args.minPeaks and \
               len(d.peakFinder.peaks) <= args.maxPeaks and \
               d.peakFinder.maxRes >= args.minRes:
                # Write image in cheetah format
                if detarr is not None: md.addarray('data', detarr)
            md.small.detectorDistance = detectorDistance
            md.small.photonEnergy = photonEnergy
            md.send()

    ###############################
    # Help your neighbor find peaks
    ###############################
    if facility == 'LCLS':
        print "HELPER!!! ", rank
        runStr = "%04d" % args.run
        fname = args.outDir + "/" + args.exp + "_" + runStr + ".cxi"
        dset_nPeaks = "/nPeaksAll"
        grpName = "/entry_1/result_1"
        try:
            myHdf5 = h5py.File(fname, 'r')
            nPeaksAll = myHdf5[grpName + dset_nPeaks].value
            myHdf5.close()
            ind = np.where(nPeaksAll == -1)[0]
            numLeft = len(ind)

            if numLeft > 0:
                import numpy.random
                ind = numpy.random.permutation(ind)

                for nevent in ind:
                    # check all done
                    try:
                        f = open(args.outDir+"/status_peaks.txt")
                        line = f.readline()
                        fracDone = float(line.split(',')[0].split(':')[-1])
                        f.close()
                        if fracDone >= 100:
                            md = mpidata()
                            md.small.eventNum = nevent
                            md.small.powder = 0
                            md.send()
                            break
                    except:
                        pass

                    if facility == 'LCLS':
                        evt = run.event(times[nevent])
                        detarr = d.calib(evt)
                        exp = env.experiment()
                        # run = evt.run()
                    elif facility == 'PAL':
                        f = h5py.File(_files[nevent], 'r')
                        detarr = f['/data'].value
                        f.close()
                        exp = args.exp
                        run = args.run

                    if detarr is None:
                        md = mpidata()
                        md.small.eventNum = nevent
                        md.small.powder = 0
                        md.send()
                        continue

                    d.peakFinder.findPeaks(detarr, evt)
                    # Likelihood
                    numPeaksFound = d.peakFinder.peaks.shape[0]
                    if numPeaksFound >= args.minPeaks and \
                                    numPeaksFound <= args.maxPeaks and \
                                    d.peakFinder.maxRes >= args.minRes:
                        cenX = d.iX[np.array(d.peakFinder.peaks[:, 0], dtype=np.int64),
                                    np.array(d.peakFinder.peaks[:, 1], dtype=np.int64),
                                    np.array(d.peakFinder.peaks[:, 2], dtype=np.int64)] + 0.5
                        cenY = d.iY[np.array(d.peakFinder.peaks[:, 0], dtype=np.int64),
                                    np.array(d.peakFinder.peaks[:, 1], dtype=np.int64),
                                    np.array(d.peakFinder.peaks[:, 2], dtype=np.int64)] + 0.5

                        x = cenX - d.ipx  # args.center[0]
                        y = cenY - d.ipy  # args.center[1]

                        pixSize = float(d.pixel_size(evt))
                        detdis = float(args.detectorDistance)
                        z = detdis / pixSize * np.ones(x.shape)  # pixels

                        ebeam = ebeamDet.get(evt)
                        try:
                            photonEnergy = ebeam.ebeamPhotonEnergy()
                        except:
                            photonEnergy = 1

                        wavelength = 12.407002 / float(photonEnergy)  # Angstrom
                        norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                        qPeaks = (np.array([x, y, z]) / norm - np.array([[0.], [0.], [1.]])) / wavelength
                        [meanClosestNeighborDist, pairsFoundPerSpot] = calculate_likelihood(qPeaks)
                    else:
                        pairsFoundPerSpot = 0.0

                    md = mpidata()
                    md.addarray('peaks', d.peakFinder.peaks)
                    md.small.eventNum = nevent
                    md.small.maxRes = d.peakFinder.maxRes
                    md.small.powder = 0
                    md.small.likelihood = pairsFoundPerSpot

                    if args.profile:
                        md.small.calibTime = calibTime
                        md.small.peakTime = peakTime

                    if facility == 'LCLS':
                        # other cxidb data
                        ps.getEvent(nevent)

                        es = ps.ds.env().epicsStore()

                        # timetool
                        try:
                            md.small.timeToolDelay = es.value(args.instrument+':LAS:MMN:04.RBV')
                            if md.small.timeToolDelay is None: md.small.timeToolDelay = 0
                        except:
                            md.small.timeToolDelay = 0

                        try:
                            md.small.laserTimeZero = es.value('LAS:FS5:VIT:FS_TGT_TIME_OFFSET')
                            if md.small.laserTimeZero is None: md.small.laserTimeZero = 0
                        except:
                            md.small.laserTimeZero = 0

                        try:
                            md.small.laserTimeDelay = es.value('LAS:FS5:VIT:FS_TGT_TIME_DIAL')
                            if md.small.laserTimeDelay is None: md.small.laserTimeDelay = 0
                        except:
                            md.small.laserTimeDelay = 0

                        try:
                            md.small.laserTimePhaseLocked = es.value('LAS:FS5:VIT:PHASE_LOCKED')
                            if md.small.laserTimePhaseLocked is None: md.small.laserTimePhaseLocked = 0
                        except:
                            md.small.laserTimePhaseLocked = 0

                        try:
                            md.small.ttspecAmpl = es.value(args.instrument + ':TTSPEC:AMPL')
                            if md.small.ttspecAmpl is None: md.small.ttspecAmpl = 0
                        except:
                            md.small.ttspecAmpl = 0

                        try:
                            md.small.ttspecAmplNxt = es.value(args.instrument + ':TTSPEC:AMPLNXT')
                            if md.small.ttspecAmplNxt is None: md.small.ttspecAmplNxt = 0
                        except:
                            md.small.ttspecAmplNxt = 0

                        try:
                            md.small.ttspecFltPos = es.value(args.instrument + ':TTSPEC:FLTPOS')
                            if md.small.ttspecFltPos is None: md.small.ttspecFltPos = 0
                        except:
                            md.small.ttspecFltPos = 0

                        try:
                            md.small.ttspecFltPosFwhm = es.value(args.instrument + ':TTSPEC:FLTPOSFWHM')
                            if md.small.ttspecFltPosFwhm is None: md.small.ttspecFltPosFwhm = 0
                        except:
                            md.small.ttspecFltPosFwhm = 0

                        try:
                            md.small.ttspecFltPosPs = es.value(args.instrument + ':TTSPEC:FLTPOS_PS')
                            if md.small.ttspecFltPosPs is None: md.small.ttspecFltPosPs = 0
                        except:
                            md.small.ttspecFltPosPs = 0

                        try:
                            md.small.ttspecRefAmpl = es.value(args.instrument + ':TTSPEC:REFAMPL')
                            if md.small.ttspecRefAmpl is None: md.small.ttspecRefAmpl = 0
                        except:
                            md.small.ttspecRefAmpl = 0

                        if evr0:
                            ec = evr0.eventCodes(evt)
                            if ec is None: ec = [-1]
                            md.addarray('evr0', np.array(ec))

                        if evr1:
                            ec = evr1.eventCodes(evt)
                            if ec is None: ec = [-1]
                            md.addarray('evr1', np.array(ec))

                        # pulse length
                        try:
                            pulseLength = es.value('SIOC:SYS0:ML00:AO820') * 1e-15  # s
                        except:
                            pulseLength = 0

                        md.small.pulseLength = pulseLength

                        md.small.detectorDistance = detectorDistance

                        md.small.pixelSize = args.pixelSize

                        # LCLS
                        if "cxi" in args.exp:
                            md.small.lclsDet = es.value(args.clen)  # mm
                        elif "mfx" in args.exp:
                            md.small.lclsDet = es.value(args.clen)  # mm
                        elif "xpp" in args.exp:
                            md.small.lclsDet = es.value(args.clen)  # mm

                        try:
                            md.small.ebeamCharge = es.value('BEND:DMP1:400:BDES')
                        except:
                            md.small.ebeamCharge = 0

                        try:
                            md.small.beamRepRate = es.value('EVNT:SYS0:1:LCLSBEAMRATE')
                        except:
                            md.small.beamRepRate = 0

                        try:
                            md.small.particleN_electrons = es.value('BPMS:DMP1:199:TMIT1H')
                        except:
                            md.small.particleN_electrons = 0

                        try:
                            md.small.eVernier = es.value('SIOC:SYS0:ML00:AO289')
                        except:
                            md.small.eVernier = 0

                        try:
                            md.small.charge = es.value('BEAM:LCLS:ELEC:Q')
                        except:
                            md.small.charge = 0

                        try:
                            md.small.peakCurrentAfterSecondBunchCompressor = es.value('SIOC:SYS0:ML00:AO195')
                        except:
                            md.small.peakCurrentAfterSecondBunchCompressor = 0

                        try:
                            md.small.pulseLength = es.value('SIOC:SYS0:ML00:AO820')
                        except:
                            md.small.pulseLength = 0

                        try:
                            md.small.ebeamEnergyLossConvertedToPhoton_mJ = es.value('SIOC:SYS0:ML00:AO569')
                        except:
                            md.small.ebeamEnergyLossConvertedToPhoton_mJ = 0

                        try:
                            md.small.calculatedNumberOfPhotons = es.value('SIOC:SYS0:ML00:AO580') * 1e12  # number of photons
                        except:
                            md.small.calculatedNumberOfPhotons = 0

                        try:
                            md.small.photonBeamEnergy = es.value('SIOC:SYS0:ML00:AO541')
                        except:
                            md.small.photonBeamEnergy = 0

                        try:
                            md.small.wavelength = es.value('SIOC:SYS0:ML00:AO192')
                        except:
                            md.small.wavelength = 0

                        try:
                            md.small.injectorPressureSDS = es.value('CXI:LC20:SDS:Pressure')
                        except:
                            md.small.injectorPressureSDS = 0

                        try:
                            md.small.injectorPressureSDSB = es.value('CXI:LC20:SDSB:Pressure')
                        except:
                            md.small.injectorPressureSDSB = 0

                        ebeam = ebeamDet.get(ps.evt)  # .get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
                        try:
                            photonEnergy = md.small.photonBeamEnergy #ebeam.ebeamPhotonEnergy()
                            pulseEnergy = ebeam.ebeamL3Energy()  # MeV
                        except:
                            photonEnergy = 0
                            pulseEnergy = 0
                            if md.small.wavelength > 0:
                                h = 6.626070e-34  # J.m
                                c = 2.99792458e8  # m/s
                                joulesPerEv = 1.602176621e-19  # J/eV
                                photonEnergy = (h / joulesPerEv * c) / (md.small.wavelength * 1e-9)

                        md.small.photonEnergy = photonEnergy
                        md.small.pulseEnergy = pulseEnergy

                        evtId = ps.evt.get(psana.EventId)
                        md.small.sec = evtId.time()[0]
                        md.small.nsec = evtId.time()[1]
                        md.small.fid = evtId.fiducials()

                        if len(d.peakFinder.peaks) >= args.minPeaks and \
                                        len(d.peakFinder.peaks) <= args.maxPeaks and \
                                        d.peakFinder.maxRes >= args.minRes:
                            #and pairsFoundPerSpot >= likelihoodThresh:
                            # Write image in cheetah format
                            img = ps.getCheetahImg()
                            if img is not None: md.addarray('data', img)

                        if args.profile:
                            totalTime = time.time() - startTic
                            md.small.totalTime = totalTime
                            md.small.rankID = rank
                        md.send()  # send mpi data object to master when desired
                    elif facility == 'PAL':
                        if len(d.peakFinder.peaks) >= args.minPeaks and \
                                        len(d.peakFinder.peaks) <= args.maxPeaks and \
                                        d.peakFinder.maxRes >= args.minRes:
                            # Write image in cheetah format
                            if detarr is not None: md.addarray('data', detarr)
                        md.small.detectorDistance = detectorDistance
                        md.small.photonEnergy = photonEnergy
                        md.send()

        except:
            print "I can't help you: ", rank
            pass

    # At the end of the run, send the powder of hits and misses
    if facility == 'LCLS':
        md = mpidata()
        md.small.powder = 1
        md.addarray('powderHits', d.peakFinder.powderHits)
        md.addarray('powderMisses', d.peakFinder.powderMisses)
        md.send()
        md.endrun()
        print "Done: ", rank
    elif facility == 'PAL':
        md = mpidata()
        md.small.powder = 1
        md.addarray('powderHits', d.peakFinder.powderHits)
        md.addarray('powderMisses', d.peakFinder.powderMisses)
        md.send()
        md.endrun()
        print "Done: ", rank

def calculate_likelihood(qPeaks):

    nPeaks = int(qPeaks.shape[1])
    selfD = distance.cdist(qPeaks.transpose(), qPeaks.transpose(), 'euclidean')
    sortedSelfD = np.sort(selfD)
    closestNeighborDist = sortedSelfD[:, 1]
    meanClosestNeighborDist = np.median(closestNeighborDist)
    closestPeaks = [None] * nPeaks
    coords = qPeaks.transpose()
    pairsFound = 0.

    for ii in range(nPeaks):
        index = np.where(selfD[ii, :] == closestNeighborDist[ii])
        closestPeaks[ii] = coords[list(index[0]), :].copy()
        p = coords[ii, :]
        flip = 2 * p - closestPeaks[ii]
        d = distance.cdist(coords, flip, 'euclidean')
        sigma = closestNeighborDist[ii] / 4.
        mu = 0.
        bins = d
        vals = np.exp(-(bins - mu) ** 2 / (2. * sigma ** 2))
        weight = np.sum(vals)
        pairsFound += weight

    pairsFound = pairsFound / 2.
    pairsFoundPerSpot = pairsFound / float(nPeaks)

    return [meanClosestNeighborDist, pairsFoundPerSpot]

def readCrystfelGeometry(geomFile, facility):
    if facility == 'PAL':
        with open(geomFile, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if 'clen' in line:
                detectorDistance = float(line.split('=')[-1])
            elif 'photon_energy' in line:
                photonEnergy = float(line.split('=')[-1])
            elif 'p0/res' in line:
                pixelSize = 1./float(line.split('=')[-1])
            elif 'p0/corner_x' in line:
                cy = -1 * float(line.split('=')[-1])
            elif 'p0/corner_y' in line:
                cx = float(line.split('=')[-1])
        return detectorDistance, photonEnergy, pixelSize, cx, cy

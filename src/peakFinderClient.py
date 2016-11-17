import psana
import numpy as np
from mpidata import mpidata 
import PeakFinder as pf
import psanaWhisperer

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def runclient(args):
    ds = psana.DataSource("exp="+args.exp+":run="+str(args.run)+':idx')
    run = ds.runs().next()
    env = ds.env()
    times = run.times()
    d = psana.Detector(args.det)
    d.do_reshape_2d_to_3d(flag=True)

    ps = psanaWhisperer.psanaWhisperer(args.exp, args.run, args.det, args.clen, args.localCalib)
    ps.setupExperiment()

    ebeamDet = psana.Detector('EBeam')

    hasCoffset = False
    hasDetectorDistance = False
    if args.detectorDistance is not 0:
        hasDetectorDistance = True
    if args.coffset is not 0:
        hasCoffset = True

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
        evt = run.event(times[nevent])
        detarr = d.calib(evt)
        if detarr is None: continue

        # Initialize hit finding
        if not hasattr(d,'peakFinder'):
            if args.algorithm == 1:
                d.peakFinder = pf.PeakFinder(env.experiment(),evt.run(),args.det,evt,d,
                                          args.algorithm, args.alg_npix_min,
                                          args.alg_npix_max, args.alg_amax_thr,
                                          args.alg_atot_thr, args.alg_son_min,
                                          alg1_thr_low=args.alg1_thr_low, alg1_thr_high=args.alg1_thr_high,
                                          alg1_radius=args.alg1_radius, alg1_dr=args.alg1_dr,
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
                                          localCalib=args.localCalib)
        d.peakFinder.findPeaks(detarr,evt)
        md=mpidata()
        md.addarray('peaks',d.peakFinder.peaks)
        md.small.eventNum = nevent
        md.small.maxRes = d.peakFinder.maxRes
        md.small.powder = 0

        # other cxidb data
        ps.getEvent(nevent)

        es = ps.ds.env().epicsStore()
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

        ebeam = ebeamDet.get(ps.evt)#.get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
        try:
            photonEnergy = ebeam.ebeamPhotonEnergy()
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
        md.small.evtNum = nevent

        if len(d.peakFinder.peaks) >= args.minPeaks and \
           len(d.peakFinder.peaks) <= args.maxPeaks and \
           d.peakFinder.maxRes >= args.minRes:
            # Write image in cheetah format
            img = ps.getCheetahImg()
            #assert (img is not None)
            if img is not None: md.addarray('data', img)
        md.send() # send mpi data object to master when desired
    # At the end of the run, send the powder of hits and misses
    md = mpidata()
    md.small.powder = 1
    md.addarray('powderHits', d.peakFinder.powderHits)
    md.addarray('powderMisses', d.peakFinder.powderMisses)
    md.send()
    md.endrun()

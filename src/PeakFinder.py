import numpy as np
from ImgAlgos.PyAlgos import PyAlgos # peak finding
import myskbeam
import time

class PeakFinder:
    def __init__(self,exp,run,detname,evt,detector,algorithm,hitParam_alg_npix_min,hitParam_alg_npix_max,
                 hitParam_alg_amax_thr,hitParam_alg_atot_thr,hitParam_alg_son_min,
                 streakMask_on,streakMask_sigma,streakMask_width,userMask_path,psanaMask_on,psanaMask_calib,
                 psanaMask_status,psanaMask_edges,psanaMask_central,psanaMask_unbond,psanaMask_unbondnrs,
                 windows=None,**kwargs):
        self.exp = exp
        self.run = run
        self.detname = detname
        self.det = detector
        self.algorithm = algorithm

        self.npix_min=hitParam_alg_npix_min
        self.npix_max=hitParam_alg_npix_max
        self.amax_thr=hitParam_alg_amax_thr
        self.atot_thr=hitParam_alg_atot_thr
        self.son_min=hitParam_alg_son_min

        self.streakMask_on = streakMask_on
        self.streakMask_sigma = streakMask_sigma
        self.streakMask_width = streakMask_width
        self.userMask_path = userMask_path
        self.psanaMask_on = psanaMask_on

        self.windows = windows

        self.userMask = None
        self.psanaMask = None
        self.streakMask = None
        self.userPsanaMask = None
        self.combinedMask = None

        # Make user mask
        if self.userMask_path is not None:
            self.userMask = np.load(self.userMask_path)

        # Make psana mask
        if self.psanaMask_on:
            self.psanaMask = detector.mask(evt, calib=psanaMask_calib, status=psanaMask_status,
                                           edges=psanaMask_edges, central=psanaMask_central,
                                           unbond=psanaMask_unbond, unbondnbrs=psanaMask_unbondnrs)

        # Combine userMask and psanaMask
        #if self.userMask is not None or self.psanaMask is not None:
        self.userPsanaMask = np.ones_like(self.det.calib(evt))
        if self.userMask is not None:
            self.userPsanaMask *= self.userMask
        if self.psanaMask is not None:
            self.userPsanaMask *= self.psanaMask

        self.alg = PyAlgos(windows=self.windows, mask=self.userPsanaMask, pbits=0)
        # set peak-selector parameters:
        self.alg.set_peak_selection_pars(npix_min=self.npix_min, npix_max=self.npix_max, \
                                        amax_thr=self.amax_thr, atot_thr=self.atot_thr, \
                                        son_min=self.son_min)
        # set algorithm specific parameters
        if algorithm == 1:
            self.hitParam_alg1_thr_low = kwargs["alg1_thr_low"]
            self.hitParam_alg1_thr_high = kwargs["alg1_thr_high"]
            self.hitParam_alg1_radius = int(kwargs["alg1_radius"])
            self.hitParam_alg1_dr = kwargs["alg1_dr"]
        elif algorithm == 3:
            self.hitParam_alg3_rank = kwargs["alg3_rank"]
            self.hitParam_alg3_r0 = int(kwargs["alg3_r0"])
            self.hitParam_alg3_dr = kwargs["alg3_dr"]
        elif algorithm == 4:
            self.hitParam_alg4_thr_low = kwargs["alg4_thr_low"]
            self.hitParam_alg4_thr_high = kwargs["alg4_thr_high"]
            self.hitParam_alg4_rank = int(kwargs["alg4_rank"])
            self.hitParam_alg4_r0 = int(kwargs["alg4_r0"])
            self.hitParam_alg4_dr = kwargs["alg4_dr"]

        self.maxNumPeaks = 2048
        self.StreakMask = myskbeam.StreakMask(self.det, evt, width=self.streakMask_width, sigma=self.streakMask_sigma)

    def findPeaks(self,calib, evt):
        if self.streakMask_on: # make new streak mask
            #tic = time.time()
            self.streakMask = self.StreakMask.getStreakMaskCalib(evt) #myskbeam.getStreakMaskCalib(self.det,self.evt,width=self.streakMask_width,sigma=self.streakMask_sigma)
            #tic1 = time.time()
        if self.streakMask is not None:
            self.combinedMask = self.userPsanaMask * self.streakMask
        else:
            self.combinedMask = self.userPsanaMask
        #tic2 = time.time()
        # set new mask
        self.alg.set_mask(self.combinedMask)
        #tic3 = time.time()
        # set algorithm specific parameters
        if self.algorithm == 1:
            # v1 - aka Droplet Finder - two-threshold peak-finding algorithm in restricted region
            #                           around pixel with maximal intensity.
            self.peaks = self.alg.peak_finder_v1(calib, thr_low=self.hitParam_alg1_thr_low, thr_high=self.hitParam_alg1_thr_high, \
                                   radius=self.hitParam_alg1_radius, dr=self.hitParam_alg1_dr)
        elif self.algorithm == 3:
            self.peaks = self.alg.peak_finder_v3(calib, rank=self.hitParam_alg3_rank, r0=self.hitParam_alg3_r0, dr=self.hitParam_alg3_dr)
        elif self.algorithm == 4:
            # v4 - aka iDroplet Finder - two-threshold peak-finding algorithm in restricted region
            #                            around pixel with maximal intensity.
            self.peaks = self.alg.peak_finder_v4(calib, thr_low=self.hitParam_alg4_thr_low, thr_high=self.hitParam_alg4_thr_high, \
                                   rank=self.hitParam_alg4_rank, r0=self.hitParam_alg4_r0, dr=self.hitParam_alg4_dr)
            #tic4 = time.time()
            #print "makeStreak, combineMask, setMask, peakFind: ", tic1-tic, tic2-tic1, tic3-tic2, tic4-tic3


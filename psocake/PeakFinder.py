import numpy as np
import myskbeam
import os

if 'LCLS' in os.environ['PSOCAKE_FACILITY'].upper():
    facility = 'LCLS'
    import psana
    #from ImgAlgos.PyAlgos import PyAlgos # peak finding
    from psalgos.pypsalgos import PyAlgos
    from pyimgalgos.RadialBkgd import RadialBkgd, polarization_factor
    from pyimgalgos.MedianFilter import median_filter_ndarr
elif 'PAL' in os.environ['PSOCAKE_FACILITY'].upper():
    facility = 'PAL'

def str2bool(v): return v.lower() in ("yes", "true", "t", "1")

class PeakFinder:
    def __init__(self, exp, run, detname, evt, detector, algorithm, hitParam_alg_npix_min, hitParam_alg_npix_max,
                 hitParam_alg_amax_thr, hitParam_alg_atot_thr, hitParam_alg_son_min,
                 streakMask_on, streakMask_sigma, streakMask_width, userMask_path, psanaMask_on, psanaMask_calib,
                 psanaMask_status, psanaMask_edges, psanaMask_central, psanaMask_unbond, psanaMask_unbondnrs,
                 generousMask=0, medianFilterOn=0, medianRank=5, radialFilterOn=0, distance=0.0, windows=None, **kwargs):
        self.exp = exp
        self.run = run
        self.detname = detname
        self.det = detector
        self.algorithm = algorithm
        self.maxRes = 0

        self.npix_min = hitParam_alg_npix_min
        self.npix_max = hitParam_alg_npix_max
        self.amax_thr = hitParam_alg_amax_thr
        self.atot_thr = hitParam_alg_atot_thr
        self.son_min = hitParam_alg_son_min

        self.streakMask_on = str2bool(streakMask_on)
        self.streakMask_sigma = streakMask_sigma
        self.streakMask_width = streakMask_width
        self.userMask_path = userMask_path
        self.psanaMask_on = str2bool(psanaMask_on)
        self.psanaMask_calib = str2bool(psanaMask_calib)
        self.psanaMask_status = str2bool(psanaMask_status)
        self.psanaMask_edges = str2bool(psanaMask_edges)
        self.psanaMask_central = str2bool(psanaMask_central)
        self.psanaMask_unbond = str2bool(psanaMask_unbond)
        self.psanaMask_unbondnrs = str2bool(psanaMask_unbondnrs)
        self.generousMaskOn = generousMask

        self.medianFilterOn = medianFilterOn
        self.medianRank = medianRank
        self.radialFilterOn = radialFilterOn
        self.distance = distance

        self.windows = windows

        self.userMask = None
        self.psanaMask = None
        self.streakMask = None
        self.userPsanaMask = None
        self.combinedMask = None
        self.generousMask = None

        # Make user mask
        if self.userMask_path is not None:
            self.userMask = np.load(self.userMask_path)

        if facility == 'LCLS':
            # Make psana mask
            if self.psanaMask_on:
                self.psanaMask = detector.mask(evt, calib=self.psanaMask_calib, status=self.psanaMask_status,
                                               edges=self.psanaMask_edges, central=self.psanaMask_central,
                                               unbond=self.psanaMask_unbond, unbondnbrs=self.psanaMask_unbondnrs)
                if self.generousMaskOn:
                    self.psanaMask = self.generousBadPixel(self.psanaMask)
            # Combine userMask and psanaMask
            self.userPsanaMask = np.ones_like(self.det.calib(evt), dtype=np.int16)
            if self.userMask is not None:
                self.userPsanaMask *= self.userMask
            if self.psanaMask is not None:
                self.userPsanaMask *= self.psanaMask
        elif facility == 'PAL':
            self.userPsanaMask = self.userMask

        # Powder of hits and misses
        self.powderHits = None
        self.powderMisses = None

        # set algorithm specific parameters
        if algorithm == 1:
            self.hitParam_alg1_thr_low = kwargs["alg1_thr_low"]
            self.hitParam_alg1_thr_high = kwargs["alg1_thr_high"]
            self.hitParam_alg1_rank = int(kwargs["alg1_rank"])
            self.hitParam_alg1_radius = int(kwargs["alg1_radius"])
            self.hitParam_alg1_dr = kwargs["alg1_dr"]
        elif algorithm == 2:
            self.hitParam_alg1_thr_low = kwargs["alg1_thr_low"]
            self.hitParam_alg1_thr_high = kwargs["alg1_thr_high"]
            self.hitParam_alg1_rank = int(kwargs["alg1_rank"])
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

        if facility == 'LCLS':
            self.access = kwargs["access"]
            if self.algorithm == 1:
                self.alg = PyAlgos(mask=None, pbits=0)
                self.peakRadius = int(self.hitParam_alg1_radius)
                self.alg.set_peak_selection_pars(npix_min=self.npix_min, npix_max=self.npix_max, \
                                                 amax_thr=self.amax_thr, atot_thr=self.atot_thr, \
                                                 son_min=self.son_min)
                #self.alg = myskbeam.DropletA(self.peakRadius, self.peakRadius+self.hitParam_alg1_dr) #PyAlgos(windows=self.windows, mask=None, pbits=0)
            elif self.algorithm == 2:
                # set peak-selector parameters:
                self.alg = PyAlgos(mask=None, pbits=0)
                self.peakRadius = int(self.hitParam_alg1_radius)
                self.alg.set_peak_selection_pars(npix_min=self.npix_min, npix_max=self.npix_max, \
                                                 amax_thr=self.amax_thr, atot_thr=self.atot_thr, \
                                                 son_min=self.son_min)
        elif facility == 'PAL':
            self.peakRadius = int(self.hitParam_alg1_radius)
            self.alg = myskbeam.DropletA(self.peakRadius, self.hitParam_alg1_dr)

        if facility == 'LCLS':
            self.StreakMask = myskbeam.StreakMask(self.det, evt, width=self.streakMask_width, sigma=self.streakMask_sigma)
            self.cx, self.cy = self.det.point_indexes(evt, pxy_um=(0, 0))
            self.iX = np.array(self.det.indexes_x(evt), dtype=np.int64)
            self.iY = np.array(self.det.indexes_y(evt), dtype=np.int64)
            if len(self.iX.shape) == 2:
                self.iX = np.expand_dims(self.iX, axis=0)
                self.iY = np.expand_dims(self.iY, axis=0)
            # Initialize radial background subtraction
            self.setupExperiment()
            if self.radialFilterOn:
                self.setupRadialBackground()
                self.updatePolarizationFactor()
        elif facility == 'PAL':
            self.geom = kwargs["geom"]
            with open(self.geom, 'r') as f: lines = f.readlines()
            for i, line in enumerate(lines):
                if 'p0/max_ss' in line:
                    dim0 = int(line.split('=')[-1]) + 1
                elif 'p0/max_fs' in line:
                    dim1 = int(line.split('=')[-1]) + 1
                elif 'p0/corner_x' in line:
                    self.cy = -1 * float(line.split('=')[-1])
                elif 'p0/corner_y' in line:
                    self.cx = float(line.split('=')[-1])
            self.iy = np.tile(np.arange(dim0), [dim1, 1])
            self.ix = np.transpose(self.iy)
            self.iX = np.array(self.ix, dtype=np.int64)
            self.iY = np.array(self.iy, dtype=np.int64)

    def generousBadPixel(self, unassemMask, n=10):
        generousBadPixelMask = unassemMask
        (numAsic, numFs, numSs) = unassemMask.shape
        for i in range(numAsic):
            for a in range(numSs):
                numBadPixels = len(np.where(unassemMask[i, :, a] == 0)[0])
                if numBadPixels >= n:
                    generousBadPixelMask[i, :, a] = 0
        return generousBadPixelMask

    def setupExperiment(self):
        access = 'exp=' + str(self.exp) + ':run=' + str(self.run) + ':idx'
        if 'ffb' in self.access.lower(): access += ':dir=/reg/d/ffb/' + self.exp[:3] + '/' + self.exp + '/xtc'
        self.ds = psana.DataSource(access)
        self.run = self.ds.runs().next()
        self.times = self.run.times()
        self.eventTotal = len(self.times)
        self.env = self.ds.env()
        self.evt = self.run.event(self.times[0])
        self.det = psana.Detector(str(self.detname), self.env)
        self.det.do_reshape_2d_to_3d(flag=True)

    def setupRadialBackground(self):
        self.geo = self.det.geometry(self.run)  # self.geo = GeometryAccess(self.parent.geom.calibPath+'/'+self.parent.geom.calibFile)
        self.xarr, self.yarr, self.zarr = self.geo.get_pixel_coords()
        self.ix = self.det.indexes_x(self.evt)
        self.iy = self.det.indexes_y(self.evt)
        if self.ix is None:
            self.iy = np.tile(np.arange(self.userMask.shape[1]), [self.userMask.shape[2], 1])
            self.ix = np.transpose(self.iy)
        self.iX = np.array(self.ix, dtype=np.int64)
        self.iY = np.array(self.iy, dtype=np.int64)
        if len(self.iX.shape) == 2:
            self.iX = np.expand_dims(self.iX, axis=0)
            self.iY = np.expand_dims(self.iY, axis=0)
        self.mask = self.geo.get_pixel_mask( mbits=0377)  # mask for 2x1 edges, two central columns, and unbound pixels with their neighbours
        self.rb = RadialBkgd(self.xarr, self.yarr, mask=self.mask, radedges=None, nradbins=100,
                             phiedges=(0, 360), nphibins=1)

    def updatePolarizationFactor(self):
        self.pf = polarization_factor(self.rb.pixel_rad(), self.rb.pixel_phi(), self.distance * 1e6)  # convert to um

    def findPeaks(self, calib, evt, thr_high=None, thr_low=None):

        if facility == 'LCLS':
            if self.streakMask_on: # make new streak mask
                self.streakMask = self.StreakMask.getStreakMaskCalib(evt)

            # Apply background correction
            if self.medianFilterOn:
                calib -= median_filter_ndarr(calib, self.medianRank)

            if self.radialFilterOn:
                self.pf.shape = calib.shape  # FIXME: shape is 1d
                calib = self.rb.subtract_bkgd(calib * self.pf)
                calib.shape = self.userPsanaMask.shape  # FIXME: shape is 1d

            if self.streakMask is not None:
                self.combinedMask = self.userPsanaMask * self.streakMask
            else:
                self.combinedMask = self.userPsanaMask

            # set new mask
            #self.alg.set_mask(self.combinedMask) # This doesn't work reliably
        elif facility == 'PAL':
            self.combinedMask = self.userMask

        # set algorithm specific parameters
        if self.algorithm == 1:
            if facility == 'LCLS':
                #print "param: ", self.npix_min, self.npix_max, self.atot_thr, self.son_min, thr_low, thr_high, np.sum(self.combinedMask)
                # v1 - aka Droplet Finder - two-threshold peak-finding algorithm in restricted region
                #                           around pixel with maximal intensity.
                if thr_high is None: # use gui input
                    self.peakRadius = int(self.hitParam_alg1_radius)
                    self.peaks = self.alg.peak_finder_v4r3(calib,
                                                    thr_low=self.hitParam_alg1_thr_low,
                                                    thr_high=self.hitParam_alg1_thr_high,
                                                    rank = self.hitParam_alg1_rank,
                                                    r0=self.hitParam_alg1_radius,
                                                    dr=self.hitParam_alg1_dr,
                                                    mask=self.combinedMask.astype(np.uint16))
#                    self.peaks = self.alg.peak_finder_v4r2(calib,
#                                                           thr_low=self.hitParam_alg1_thr_low,
#                                                           thr_high=self.hitParam_alg1_thr_high,
#                                                           rank=self.hitParam_alg1_rank,
#                                                           r0=self.hitParam_alg1_radius,
#                                                           dr=self.hitParam_alg1_dr)
                else:
                    self.peaks = self.alg.findPeaks(calib,
                                                    npix_min=self.npix_min,
                                                    npix_max=self.npix_max,
                                                    atot_thr=self.atot_thr,
                                                    son_min=self.son_min,
                                                    thr_low=thr_low,
                                                    thr_high=thr_high,
                                                    mask=self.combinedMask)
#                    self.peaks = self.alg.peak_finder_v4r2(calib,
#                                                           thr_low=thr_low,
#                                                           thr_high=thr_high,
#                                                           rank=self.hitParam_alg1_rank,
#                                                           r0=self.hitParam_alg1_radius,
#                                                           dr=self.hitParam_alg1_dr)
            elif facility == 'PAL':
                self.peakRadius = int(self.hitParam_alg1_radius)
                _calib = np.zeros((1, calib.shape[0], calib.shape[1]))
                _calib[0, :, :] = calib
                if self.combinedMask is None:
                    _mask = None
                else:
                    _mask = self.combinedMask.astype(np.uint16)

                self.peaks = self.alg.findPeaks(_calib,
                                                npix_min = self.npix_min,
                                                npix_max = self.npix_max,
                                                son_min = self.son_min,
                                                thr_low = self.hitParam_alg1_thr_low,
                                                thr_high=self.hitParam_alg1_thr_high,
                                                atot_thr = self.atot_thr,
                                                r0 = self.peakRadius,
                                                dr = int(self.hitParam_alg1_dr),
                                                mask = _mask)
        elif self.algorithm == 2:
            if facility == 'LCLS':
                #print "param: ", self.npix_min, self.npix_max, self.atot_thr, self.son_min, thr_low, thr_high, np.sum(self.combinedMask)
                # v1 - aka Droplet Finder - two-threshold peak-finding algorithm in restricted region
                #                           around pixel with maximal intensity.
                self.peakRadius = int(self.hitParam_alg1_radius)
                self.peaks = self.alg.peak_finder_v3r3(calib, rank=int(self.hitParam_alg1_rank),
                                                       r0=self.peakRadius, dr=self.hitParam_alg1_dr,
                                                       nsigm=self.son_min,
                                                       mask=self.combinedMask.astype(np.uint16))
        elif self.algorithm == 3:
            self.peaks = self.alg.peak_finder_v3(calib, rank=self.hitParam_alg3_rank, r0=self.hitParam_alg3_r0, dr=self.hitParam_alg3_dr)
        elif self.algorithm == 4:
            # v4 - aka iDroplet Finder - two-threshold peak-finding algorithm in restricted region
            #                            around pixel with maximal intensity.
            self.peaks = self.alg.peak_finder_v4(calib, thr_low=self.hitParam_alg4_thr_low, thr_high=self.hitParam_alg4_thr_high, \
                                   rank=self.hitParam_alg4_rank, r0=self.hitParam_alg4_r0, dr=self.hitParam_alg4_dr)
        self.numPeaksFound = self.peaks.shape[0]

        if self.numPeaksFound > 0:
            if facility == 'LCLS':
                cenX = self.iX[np.array(self.peaks[:, 0], dtype=np.int64), 
                               np.array(self.peaks[:, 1], dtype=np.int64), 
                               np.array(self.peaks[:, 2], dtype=np.int64)] + 0.5
                cenY = self.iY[np.array(self.peaks[:, 0], dtype=np.int64), 
                               np.array(self.peaks[:, 1], dtype=np.int64), 
                               np.array(self.peaks[:, 2], dtype=np.int64)] + 0.5
            elif facility == 'PAL':
                cenX = self.iX[np.array(self.peaks[:, 1], dtype=np.int64), np.array(self.peaks[:, 2], dtype=np.int64)] + 0.5
                cenY = self.iY[np.array(self.peaks[:, 1], dtype=np.int64), np.array(self.peaks[:, 2], dtype=np.int64)] + 0.5
            self.maxRes = getMaxRes(cenX, cenY, self.cx, self.cy)
        else:
            self.maxRes = 0

        if self.numPeaksFound >= 15:
            if self.powderHits is None:
                self.powderHits = calib
            else:
                self.powderHits = np.maximum(self.powderHits, calib)
        else:
            if self.powderMisses is None:
                self.powderMisses = calib
            else:
                self.powderMisses = np.maximum(self.powderMisses, calib)

        if self.powderHits is None: self.powderHits = np.zeros_like(calib)
        if self.powderMisses is None: self.powderMisses = np.zeros_like(calib)

def getMaxRes(posX, posY, centerX, centerY):
    maxRes = np.max(np.sqrt((posX - centerX) ** 2 + (posY - centerY) ** 2))
    return maxRes

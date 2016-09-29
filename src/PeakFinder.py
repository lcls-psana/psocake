import numpy as np
from ImgAlgos.PyAlgos import PyAlgos # peak finding
import myskbeam
import time
import psana
from pyimgalgos.RadialBkgd import RadialBkgd, polarization_factor
from pyimgalgos.MedianFilter import median_filter_ndarr

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class PeakFinder:
    def __init__(self,exp,run,detname,evt,detector,algorithm,hitParam_alg_npix_min,hitParam_alg_npix_max,
                 hitParam_alg_amax_thr,hitParam_alg_atot_thr,hitParam_alg_son_min,
                 streakMask_on,streakMask_sigma,streakMask_width,userMask_path,psanaMask_on,psanaMask_calib,
                 psanaMask_status,psanaMask_edges,psanaMask_central,psanaMask_unbond,psanaMask_unbondnrs,
                 medianFilterOn=0, medianRank=5, radialFilterOn=0, distance=0.0, minNumPeaks=15, maxNumPeaks=2048,
                 minResCutoff=-1, windows=None, **kwargs):
        self.exp = exp
        self.run = run
        self.detname = detname
        self.det = detector
        self.algorithm = algorithm
        self.maxRes = 0
        self.minNumPeaks = minNumPeaks
        self.maxNumPeaks = maxNumPeaks
        self.minRes = minResCutoff

        self.npix_min=hitParam_alg_npix_min
        self.npix_max=hitParam_alg_npix_max
        self.amax_thr=hitParam_alg_amax_thr
        self.atot_thr=hitParam_alg_atot_thr
        self.son_min=hitParam_alg_son_min

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

        # Make user mask
        if self.userMask_path is not None:
            self.userMask = np.load(self.userMask_path)

        # Make psana mask
        if self.psanaMask_on:
            self.psanaMask = detector.mask(evt, calib=self.psanaMask_calib, status=self.psanaMask_status,
                                           edges=self.psanaMask_edges, central=self.psanaMask_central,
                                           unbond=self.psanaMask_unbond, unbondnbrs=self.psanaMask_unbondnrs)

        # Combine userMask and psanaMask
        self.userPsanaMask = np.ones_like(self.det.calib(evt))
        if self.userMask is not None:
            self.userPsanaMask *= self.userMask
        if self.psanaMask is not None:
            self.userPsanaMask *= self.psanaMask

        # Powder of hits and misses
        self.powderHits = np.zeros_like(self.userPsanaMask)
        self.powderMisses = np.zeros_like(self.userPsanaMask)

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
        #elif algorithm == 3:
        #    self.hitParam_alg3_rank = kwargs["alg3_rank"]
        #    self.hitParam_alg3_r0 = int(kwargs["alg3_r0"])
        #    self.hitParam_alg3_dr = kwargs["alg3_dr"]
        #elif algorithm == 4:
        #    self.hitParam_alg4_thr_low = kwargs["alg4_thr_low"]
        #    self.hitParam_alg4_thr_high = kwargs["alg4_thr_high"]
        #    self.hitParam_alg4_rank = int(kwargs["alg4_rank"])
        #    self.hitParam_alg4_r0 = int(kwargs["alg4_r0"])
        #    self.hitParam_alg4_dr = kwargs["alg4_dr"]

        self.maxNumPeaks = 2048
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

    def setupExperiment(self):
        self.ds = psana.DataSource('exp=' + str(self.exp) + ':run=' + str(self.run) + ':idx')
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
            self.iy = np.tile(np.arange(self.userMask.shape[0]), [self.userMask.shape[1], 1])
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

    def getCheetahImg(self, calib):
        """Converts seg, row, col assuming (32,185,388)
           to cheetah 2-d table row and col (8*185, 4*388)
        """
        if 'cspad' in self.detname.lower():
            img = np.zeros((8 * 185, 4 * 388))
            counter = 0
            for quad in range(4):
                for seg in range(8):
                    img[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = calib[counter, :, :]
                    counter += 1
        elif 'rayonix' in self.detname.lower():
            # FIXME: check this is correct
            img = np.squeeze(calib)
            print "Rayonix shape: ", img.shape
        return img

    def findPeaks(self, calib, evt):
        # Apply background correction
        if self.medianFilterOn:
            calib -= median_filter_ndarr(calib, self.medianRank)

        if self.radialFilterOn:
            self.pf.shape = calib.shape  # FIXME: shape is 1d
            calib = self.rb.subtract_bkgd(calib * self.pf)
            calib.shape = self.userPsanaMask.shape  # FIXME: shape is 1d

        if self.streakMask_on: # make new streak mask
            self.streakMask = self.StreakMask.getStreakMaskCalib(evt)
        if self.streakMask is not None:
            self.combinedMask = self.userPsanaMask * self.streakMask
        else:
            self.combinedMask = self.userPsanaMask
        # set new mask
        self.alg.set_mask(self.combinedMask)
        # set algorithm specific parameters
        if self.algorithm == 1:
            # v1 - aka Droplet Finder - two-threshold peak-finding algorithm in restricted region
            #                           around pixel with maximal intensity.
            self.peaks = self.alg.peak_finder_v1(calib, thr_low=self.hitParam_alg1_thr_low, thr_high=self.hitParam_alg1_thr_high, \
                                   radius=self.hitParam_alg1_radius, dr=self.hitParam_alg1_dr)
        #elif self.algorithm == 3:
        #    self.peaks = self.alg.peak_finder_v3(calib, rank=self.hitParam_alg3_rank, r0=self.hitParam_alg3_r0, dr=self.hitParam_alg3_dr)
        #elif self.algorithm == 4:
        #    # v4 - aka iDroplet Finder - two-threshold peak-finding algorithm in restricted region
        #    #                            around pixel with maximal intensity.
        #    self.peaks = self.alg.peak_finder_v4(calib, thr_low=self.hitParam_alg4_thr_low, thr_high=self.hitParam_alg4_thr_high, \
        #                           rank=self.hitParam_alg4_rank, r0=self.hitParam_alg4_r0, dr=self.hitParam_alg4_dr)
        self.numPeaksFound = self.peaks.shape[0]

        if self.numPeaksFound > 0:
            cenX = self.iX[np.array(self.peaks[:, 0], dtype=np.int64), np.array(self.peaks[:, 1], dtype=np.int64), np.array(
                self.peaks[:, 2], dtype=np.int64)] + 0.5
            cenY = self.iY[np.array(self.peaks[:, 0], dtype=np.int64), np.array(self.peaks[:, 1], dtype=np.int64), np.array(
                self.peaks[:, 2], dtype=np.int64)] + 0.5
            self.maxRes = getMaxRes(cenX, cenY, self.cx, self.cy)
        else:
            self.maxRes = 0

        if self.numPeaksFound >= self.minNumPeaks:
            self.powderHits = np.maximum(self.powderHits, calib)
            self.cheetahImg = self.getCheetahImg(calib)
        else:
            self.powderMisses = np.maximum(self.powderMisses, calib)
            self.cheetahImg = 0

def getMaxRes(posX, posY, centerX, centerY):
    maxRes = np.max(np.sqrt((posX - centerX) ** 2 + (posY - centerY) ** 2))
    return maxRes

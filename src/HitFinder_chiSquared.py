import numpy as np
from ImgAlgos.PyAlgos import PyAlgos # peak finding
import myskbeam
import time

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class HitFinder_chiSquared:
    def __init__(self,exp,run,detname,evt,detector,pruneInterval,
                 streakMask_on,streakMask_sigma,streakMask_width,userMask_path,psanaMask_on,psanaMask_calib,
                 psanaMask_status,psanaMask_edges,psanaMask_central,psanaMask_unbond,psanaMask_unbondnrs,
                 **kwargs):
        self.exp = exp
        self.run = run
        self.detname = detname
        self.det = detector
        self.pruneInterval = pruneInterval

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

        self.StreakMask = myskbeam.StreakMask(self.det, evt, width=self.streakMask_width, sigma=self.streakMask_sigma)

        self.buffer_max_len = 120

        # Digest initial events
        self.digestInitialEvents()

    def digestInitialEvents(self):
        """
        Make a stack of calibrated images of length buffer_max_len
        """
        counter = 0
        for t in range(self.buffer_max_len):
            evt = self.run.event(t)
            print "Rank %d digested %d of %d frames" % (self.rank, counter, self.buffer_max_len)
            # Create masked calib
            calib = myDetList[detNum].calib_data(evt)
            if calib is None:
                continue
            calib *= myDetList[detNum].spiMask #self.myMask[detNum]
            self.bg_buffer[detNum].append(calib.copy())
            counter += 1

        print "Rank %d has %d total frames" % (self.rank, counter)

    def findHits(self, calib, evt):
        if self.streakMask_on: # make new streak mask
            self.streakMask = self.StreakMask.getStreakMaskCalib(evt)
        if self.streakMask is not None:
            self.combinedMask = self.userPsanaMask * self.streakMask
        else:
            self.combinedMask = self.userPsanaMask

        try:
            calib *= self.combinedMask
            self.nPixels = np.where(calib > self.litPixelThreshold)[0].shape[0]
        except:
            self.nPixels = 0
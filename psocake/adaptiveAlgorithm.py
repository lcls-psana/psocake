from psalgos.pypsalgos import PyAlgos  # replacement for: from ImgAlgos.PyAlgos import PyAlgos
import abstractAlgorithm

class adaptiveAlgorithm(abstractAlgorithm.abstractAlgorithm):
    def __init__(self):
        self.alg = PyAlgos(mask=None, pbits=0)

    def setParams(self):
        self.alg.set_peak_selection_pars(npix_min=
                   self.hitParam_alg1_npix_min, npix_max=
                   self.hitParam_alg1_npix_max, amax_thr=
                   self.hitParam_alg1_amax_thr, atot_thr=
                   self.hitParam_alg1_atot_thr, son_min=
                   self.hitParam_alg1_son_min)

    def initParams(self, **kwargs):
        self.hitParam_alg1_npix_min = kwargs["npix_min"]
        self.hitParam_alg1_npix_max = kwargs["npix_max"]
        self.hitParam_alg1_amax_thr = kwargs["amax_thr"]
        self.hitParam_alg1_atot_thr = kwargs["atot_thr"]
        self.hitParam_alg1_son_min = kwargs["son_min"]
        self.hitParam_alg1_rank = kwargs["rank"]
        self.hitParam_alg1_r0 = kwargs["r0"]
        self.hitParam_alg1_dr = kwargs["dr"]
        self.hitParam_alg1_nsigm = kwargs["nsigm"]
        self.setParams()

    def algorithm(self, nda, mask, **kwargs):
        self.initParams(**kwargs)
        self.peaks = self.alg.peak_finder_v3r3(nda,rank=
                           self.hitParam_alg1_rank, r0=
                           self.hitParam_alg1_r0, dr=
                           self.hitParam_alg1_dr, nsigm=
                           self.hitParam_alg1_nsigm, mask=
                           mask)
        return self.peaks

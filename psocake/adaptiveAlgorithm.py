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
        self.setParams()

    def algorithm(self, **kwargs):
        self.initParams(**kwargs)
        return self.alg

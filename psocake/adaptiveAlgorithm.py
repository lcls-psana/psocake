from psalgos.pypsalgos import PyAlgos  # replacement for: from ImgAlgos.PyAlgos import PyAlgos
import abstractAlgorithm
import json

class adaptiveAlgorithm(abstractAlgorithm.abstractAlgorithm):

    def __init__(self):
        """ Initialize instance of imported PyAlgos algorithm.
        """
        self.alg = PyAlgos(mask=None, pbits=0)
        self.setDefaultParams()

    def setParams(self):
        """ Use the algorithm's function, set_peak_selection_pars, 
        to set parameters.
        """
        self.alg.set_peak_selection_pars(npix_min=
                   self.alg1_npix_min, npix_max=
                   self.alg1_npix_max, amax_thr=
                   self.alg1_amax_thr, atot_thr=
                   self.alg1_atot_thr, son_min=
                   self.alg1_son_min)

    def initParams(self, **kwargs):
        """ Save the values of the parameters from kwargs.

        Arguments:
        **kwargs -- dictionary of parameters, either default or user inputted
        """
        self.alg1_npix_min = kwargs["npix_min"]
        self.alg1_npix_max = kwargs["npix_max"]
        self.alg1_amax_thr = kwargs["amax_thr"]
        self.alg1_atot_thr = kwargs["atot_thr"]
        self.alg1_son_min = kwargs["son_min"]
        self.alg1_rank = kwargs["rank"]
        self.alg1_r0 = kwargs["r0"]
        self.alg1_dr = kwargs["dr"]
        self.alg1_nsigm = kwargs["nsigm"]
        self.setParams()

    def algorithm(self, nda, mask, kw = None):
        """ Uses peak_finder_v3r3 (a.k.a. adaptive peak finder) to 
        find peaks on an image.

        Arguments:
        nda -- detector image
        mask -- detector mask
        kw -- dictionary or None, if None default parameters are used, 
          otherwise kw is used to initialize parameters
        """
        if kw == None:
            self.initParams(**self.default_params)
        else:
            self.initParams(**kw)
        self.peaks = self.alg.peak_finder_v3r3(nda,rank=
                           self.alg1_rank, r0=
                           self.alg1_r0, dr=
                           self.alg1_dr, nsigm=
                           self.alg1_nsigm, mask=
                           mask)
        return self.peaks

    def getDefaultParams(self):
        """ Return the default parameters in the form of a string, for Psocake to display
        """
        return json.dumps(self.default_params)

    def setDefaultParams(self):
        #The default parameters for the adaptive peak finding algorithm
        #self.default_params_str = "{\"npix_min\": 2,\"npix_max\":30,\"amax_thr\":300, \"atot_thr\":600,\"son_min\":10, \"rank\":3, \"r0\":3, \"dr\":2, \"nsigm\":5 }"
        self.default_params = {"npix_min": 2,"npix_max":30,"amax_thr":300, "atot_thr":600,"son_min":10, "rank":3, "r0":3, "dr":2, "nsigm":5 }

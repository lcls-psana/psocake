import torch
import os

class PeakFinderPeaknet:
    def __init__(self, exp, run, detname, detector, model_path=None, gpu=0, cutoff_eval=0.5,
                 print_every=10, upload_every=1, save_name=None, n_experiments=-1, n_per_run=-1, batch_size=5,
                 num_workers=0):
        print "Initializing parameters for PeakNet..."
        self.exp = exp
        self.run = run
        self.detname = detname
        self.det = detector

        if model_path is None:
            # default location of the model
            default_model = 'my_model'
            model_path = '/cds/home/a/axlevy/peaknet2020/peaknet/debug/' + default_model + '/model.pt'

        print "Loading model..."
        model = torch.load(model_path)

        if torch.cuda.is_available():
            print "Allocating gpu..."
            self.device = torch.device("cuda:{}".format(gpu))
        else:
            print "Failed to allocate gpu"

        self.model = model.to(self.device)

        self.params = {}
        self.params["cutoff_eval"] = cutoff_eval
        self.params["print_every"] = print_every
        self.params["upload_every"] = upload_every
        self.params["save_name"] = save_name
        self.params["n_experiments"] = n_experiments
        self.params["n_per_run"] = n_per_run
        self.params["batch_size"] = batch_size
        self.params["num_workers"] = num_workers

        # PUSH AND CHECK INITIALIZATION

    def findPeaks(self, calib, evt):
        print "Finding peaks with PeakNet..."

        self.calib = calib

        self.peaks = []

        return
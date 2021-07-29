import torch
import numpy as np

class PeakFinderPeaknet:
    def __init__(self, exp, run, detname, detector, batch_size, model_path=None, gpu=0, cutoff_eval=0.5, normalize=True):
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
        self.params["normalize"] = normalize

        self.maxRes = np.inf

        self.seen_events = 0
        self.batch = None
        self.batch_size = batch_size
        self.calibs_batch = []
        self.evt_batch = []
        self.peaks_batch = []

        self.model.eval()

    def _load_img(self, calib):
        img = calib
        img[img < 0] = 0

        if self.params["normalize"]:
            for i in range(img.shape[0]):
                img[i] = img[i] / np.max(img[i])

        h = img.shape[1]
        w = img.shape[2]

        img_tensor = torch.zeros(img.shape[0], h, w)
        img_tensor[:, 0:img.shape[1], 0:img.shape[2]] = torch.from_numpy(img)

        return img_tensor.view(1, -1, h, w)

    def add_to_batch(self, calib, evt):
        self.seen_events += 1
        self.calibs_batch.append(calib)
        self.evt_batch.append(evt)

        x = self._load_img(calib)
        x = x.to(self.device)

        if self.batch is None:
            self.batch = x
        else:
            self.batch = torch.cat((self.batch, x), 1)

    def batched_peak_finding(self, calib):
        h = self.batch.shape[2]
        w = self.batch.shape[3]
        with torch.no_grad():
            scores = self.model(self.batch).view(self.batch_size, -1, h, w)
            scores = torch.nn.Sigmoid()(scores).cpu().numpy()

        for i in range(self.batch_size):
            peaks = np.array(np.argwhere(scores[i] > self.params["cutoff_eval"])) # maybe invert row and cols?
            npeaks = peaks.shape[0]
            # Put zeros in all unknown parameters for now
            additional_zeros = np.zeros((npeaks, 14), dtype=int)
            peaks = np.concatenate((peaks, additional_zeros), axis=1)
            self.peaks_batch.append(peaks)

    def clean_batch(self):
        self.batch = None
        self.calibs_batch = []
        self.evt_batch = []
        self.peaks_batch = []
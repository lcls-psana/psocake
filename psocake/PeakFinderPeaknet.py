import torch
import numpy as np

class PeakFinderPeaknet:
    def __init__(self, exp, run, detname, detector, model_path=None, gpu=0, cutoff_eval=0.5,
                 print_every=10, normalize=True, upload_every=1, save_name=None, n_experiments=-1, n_per_run=-1,
                 batch_size=5, num_workers=0):
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
        # self.params["print_every"] = print_every
        self.params["normalize"] = normalize
        # self.params["upload_every"] = upload_every
        # self.params["save_name"] = save_name
        # self.params["n_experiments"] = n_experiments
        # self.params["n_per_run"] = n_per_run
        # self.params["batch_size"] = batch_size
        # self.params["num_workers"] = num_workers
        
        self.maxRes = np.inf

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

    def findPeaks(self, calib, evt):
        print "Finding peaks with PeakNet..."

        x = self._load_img(calib)
        x = x.to(self.device)

        with torch.no_grad():
            scores = self.model(x)
            scores = torch.nn.Sigmoid()(scores).cpu().numpy()
            self.peaks = np.array(np.argwhere(scores[:, 0] > self.params["cutoff_eval"]))

        npeaks = self.peaks.shape[0]
        print("Number of found peaks: " + str(npeaks))

        # Put zeros in all unknown parameters for now
        additional_zeros = np.zeros((npeaks, 14), dtype=int)
        self.peaks = np.concatenate((self.peaks, additional_zeros), axis=1)
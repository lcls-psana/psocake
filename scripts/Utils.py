import numpy as np
import pandas
from peaknet_utils import psanaImageLoader
import psana
from psalgos.pypsalgos import PyAlgos
from ImgAlgos.PyAlgos import PyAlgos as PA

def safeDataSource(exp, run):
    ds = psana.DataSource('exp=%s:run=%s:smd' % (exp, run))
    evt = None
    for _evt in ds.events():
        evt = _evt
        break
    if evt is not None:
        ds = psana.DataSource('exp=%s:run=%s:idx' % (exp, run))
        return ds
    return None

class psanaDataLoader(object):
    # TODO: refactor getting img
    def __init__(self, table, batchSize):
        self.table = table
        self.batchSize = batchSize
        self.numMacros = int(np.ceil(len(self.table)/self.batchSize))

    def __getitem__(self, n):
        assert(n < self.numMacros)
        endInd = min(len(self.table), (n+1)*self.batchSize)
        rows = self.table.loc[n*self.batchSize:endInd,:]
        imgs = psanaImageLoader(rows)
        labels = self.generateLabels(rows)
        return imgs, labels

    def getImg(self, ):
        print("Not implemented yet")

    def generateLabels(self, rows):
        # rows is a subset rows of a pytable containing exp, run, idx
        n = len(rows)
        labels = []
        for i, j in enumerate(rows.index):
            exp = str(rows.loc[j,"exp"])
            run = str(rows.loc[j,"run"])
            detname = str(rows.loc[j,"detector"])
            event_idx = int(rows.loc[j,"event"])
            label = self.getLabel(exp, run, detname, event_idx)
            labels.append(label)
            print("*%d %d: %s"%(i,j,label))
        print("@@@@@ labels:",labels)
        return labels

    def getLabel(self, exp, run, detname, event_idx): # FIXME: duplicate clientPeakFinder
        eventInfo = self.getDetectorInformation(exp, int(run), detname)
        d, hdr, fmt, numEvents, mask, times, env, run = eventInfo[:]

        npxmin = 2
        npxmax = 30
        amaxthr = 300
        atotthr = 600
        sonmin = 10
        alg = PyAlgos(mask=None, pbits=0)
        alg.set_peak_selection_pars(npix_min=npxmin, npix_max=npxmax, amax_thr=amaxthr, atot_thr=atotthr,
                                    son_min=sonmin)

        # Get Peak Info
        evt, nda, peaks, numPeaksFound = self.getPeaks(d, alg, hdr, fmt, mask, times, env, run, event_idx)
        labelsForThisEvent = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
        print("@@@ peaks: ",peaks)
        cls = 0
        height = 7
        width = 7
        for peak in peaks:
            seg, row, col, npix, amax, atot = peak[0:6]
            labelsForThisEvent[0] = np.append(labelsForThisEvent[0], np.array([cls]), axis=0)  # class
            labelsForThisEvent[1] = np.append(labelsForThisEvent[1], np.array([seg]), axis=0)  # seg
            labelsForThisEvent[2] = np.append(labelsForThisEvent[2], np.array([row]), axis=0)  # row
            labelsForThisEvent[3] = np.append(labelsForThisEvent[3], np.array([col]), axis=0)  # col
            labelsForThisEvent[4] = np.append(labelsForThisEvent[4], np.array([height]), axis=0)  # height
            labelsForThisEvent[5] = np.append(labelsForThisEvent[5], np.array([width]), axis=0)  # width
        return labelsForThisEvent

    def getDetectorInformation(self, exp, runnum, det): # FIXME: duplicate clientPeakFinder
        """ Returns the detector and the number of events for
        this run.

        Arguments:
        exp -- the experiment name
        runnum -- the run number for this experiment
        det -- the detector used for this experiment
        """
        ds = psana.DataSource('exp=%s:run=%d:idx' % (exp, runnum))
        d = psana.Detector(det)
        d.do_reshape_2d_to_3d(flag=True)
        hdr = '\nClass  Seg  Row  Col  Height  Width  Npix    Amptot'
        fmt = '%5d %4d %4d %4d  %6d %6d %5d  %8.1f'
        run = ds.runs().next()
        times = run.times()
        env = ds.env()
        numEvents = len(times)
        mask = d.mask(runnum, calib=True, status=True, edges=True, central=True, unbond=True, unbondnbrs=True)
        return [d, hdr, fmt, numEvents, mask, times, env, run]

    def getPeaks(self, d, alg, hdr, fmt, mask, times, env, run, j): # FIXME: duplicate clientPeakFinder
        """Finds peaks within an event, and returns the event information, peaks found, and hits found

        Arguments:
        d -- psana.Detector() of this experiment's detector
        alg -- the algorithm used to find peaks
        hdr -- Title row for printed chart of peaks found
        fmt -- Locations of peaks found for printed chart
        mask -- the detector mask
        times -- all the events for this run
        env -- ds.env()
        run -- ds.runs().next(), the run information
        j -- this event's number

        """
        evt = run.event(times[j])
        try:
            nda = d.calib(evt) * mask
        except TypeError:
            nda = d.calib(evt)
        if (nda is not None):
            peaks = alg.peak_finder_v3r3(nda, rank=3, r0=3, dr=2, nsigm=10)
            numPeaksFound = len(peaks)
            # alg = PA()
            # thr = 20
            return [evt, nda, peaks, numPeaksFound]
        else:
            return [None, None, None, None]


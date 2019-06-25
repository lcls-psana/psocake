import psana
import os
import numpy as np
import PeakFinderAnt as pf
import glob
from psalgos.pypsalgos import PyAlgos
from scipy.spatial import distance
import random
import re
import zmq
import time

def checkXtcSize(exp, runnum):
    minBytes = 100
    realpath = os.path.realpath('/reg/d/psdm/cxi/' + exp + '/xtc')
    if '/reg/data/ana13' in realpath:  # FIXME: ana013 is down temporarily
        return False
    runList = glob.glob('/reg/d/psdm/cxi/' + exp + '/xtc/*-r%04d*' % (runnum))
    idxList = glob.glob('/reg/d/psdm/cxi/' + exp + '/xtc/index/*-r%04d*' % (runnum))
    smdList = glob.glob('/reg/d/psdm/cxi/' + exp + '/xtc/smalldata/*-r%04d*' % (runnum))
    if runList and (len(runList) == len(idxList)) and (len(runList) == len(smdList)):
        for f in runList + idxList:
            if os.stat(f).st_size <= minBytes:  # bytes
                return False
        return True
    else:
        return False

def safeDataSource(exp, runnum):
    if checkXtcSize(exp, runnum):
        _ds = psana.DataSource('exp=%s:run=%s:smd' % (exp, runnum))
        evt = None
        for _evt in _ds.events():
            evt = _evt
            break
        if evt is not None:
            try:
                _ds = psana.DataSource('exp=%s:run=%s:idx' % (exp, runnum))
                run = _ds.runs().next()
                times = run.times()
                detname = getDetName(exp, runnum)
                det = psana.Detector(detname)
                evt = run.event(times[0])
                calib = det.calib(evt)
                if calib is not None:
                    ds = psana.DataSource('exp=%s:run=%s:idx' % (exp, runnum))
                    return ds
            except:
                return None
    return None

def isValidRun(filetype, name, files):
    """determines if an experiment, run has a corresponding idx file

    Arguments:
    filetype -- either cxi or mfx files allowed
    name -- experiment name
    files -- filename for this run number
    """
    if(filetype == "cxi"):
        extension = '/reg/d/psdm/cxi/' + name + '/xtc/index/' + files + '.idx'
        boolean = os.path.isfile(extension)
        boolean2 = (os.stat(extension).st_size != 0)
        return (boolean and boolean2)
    elif(filetype == "mfx"):
        extension = '/reg/d/psdm/mfx/' + name + '/xtc/index/' + files + '.idx'
        boolean = os.path.isfile(extension)
        boolean2 = (os.stat(extension).st_size != 0)
        return (boolean and boolean2)
    else:
        return False

def getDetName(exp, run):
    """ returns detector name.
    """
    detnames = psana.DetNames()
    for detname in detnames:
        if ("DscCsPad" in detname) or ("DsdCsPad" in detname) or ("DsaCsPad" in detname):
            return detname[1]
    return None

def randExpRunDet():
    while True:
        found, exp, runnum = randExpRun()

        if found:
            exists = safeDataSource(exp, runnum)

            if exists:
                det = getDetName(exp, runnum)

                if det is not None:
                    return exp, runnum, det

def randExpRun():
    """searches through the data files and reports a random experiment and run number
    """
    debugMode = False
    choice = None
    filetype = random.choice(["cxi"])
    myList = ['cxic0415']#'['cxil8416']#['cxi08216',]#['cxilr7616']# # FIXME: psana can not handle xtcs with no events
    myRuns = [85]#[219] #[7] #[99, 51] #[52]# # FIXME: psana can not handle xtcs with no events

    if debugMode:
        choice = random.choice(myList)
    else:
        choice = random.choice(os.listdir("/reg/d/psdm/%s" % filetype))

    if ("cxi" in choice):
        try:
            realpath = os.path.realpath('/reg/d/psdm/cxi/' + choice + '/xtc')
            if '/reg/data/ana01' in realpath:  # FIXME: ana01 is down temporarily
                return [False, 0, 0]
            runList = os.listdir(realpath)
            randomRun = random.choice(runList)

            if (".xtc.inprogress" in randomRun):
                return [False, 0, 0]
            elif (".xtc" in randomRun):
                if (isValidRun(filetype, choice, randomRun)):
                    num = re.findall("-r(\d+)-", randomRun)
                    if debugMode:
                        return [True, choice, random.choice(myRuns)]
                    else:
                        return [True, choice, int(num[0])]
                else:
                    return [False, 0, 0]
            else:
                return [False, 0, 0]
        except OSError:
            return [False, 0, 0]
    else:
        return [False, 0, 0]

class CrystalFindingAnt:

    def __init__(self, host):
        self.host = host
        self.outdir = '/reg/d/psdm/cxi/cxitut13/res/autosfx/output'
        self.goodLikelihood = .003
        #Minimum number of peaks to be found to calculate likelihood
        self.goodNumPeaks = 10
        #Minimum number of events to be found considered a good run
        self.minCrystals = 2

        self.context = zmq.Context()
        if self.host == "":
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind("tcp://*:5556")
        else:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect("tcp://%s:5556" % self.host)

    def calculate_likelihood(self, qPeaks):
        nPeaks = int(qPeaks.shape[1])
        selfD = distance.cdist(qPeaks.transpose(), qPeaks.transpose(), 'euclidean')
        sortedSelfD = np.sort(selfD)
        closestNeighborDist = sortedSelfD[:, 1]
        meanClosestNeighborDist = np.median(closestNeighborDist)
        closestPeaks = [None] * nPeaks
        coords = qPeaks.transpose()
        pairsFound = 0.
        for ii in range(nPeaks):
            index = np.where(selfD[ii, :] == closestNeighborDist[ii])
            closestPeaks[ii] = coords[list(index[0]), :].copy()
            p = coords[ii, :]
            flip = 2 * p - closestPeaks[ii]
            d = distance.cdist(coords, flip, 'euclidean')
            sigma = closestNeighborDist[ii] / 4.
            mu = 0.
            bins = d
            vals = np.exp(-(bins - mu) ** 2 / (2. * sigma ** 2))
            weight = np.sum(vals)
            pairsFound += weight

        pairsFound = pairsFound / 2.
        pairsFoundPerSpot = pairsFound / float(nPeaks)

        return [meanClosestNeighborDist, pairsFoundPerSpot]

    def likelihood(self, peaks):
        maxPeaks = 2048
        # Likelihood
        numPeaksFound = peaks.shape[0]
        if numPeaksFound >= self.goodNumPeaks and \
           numPeaksFound <= maxPeaks:
            cenX = self.iX[np.array(peaks[:, 0], dtype=np.int64),
                        np.array(peaks[:, 1], dtype=np.int64),
                        np.array(peaks[:, 2], dtype=np.int64)] + 0.5
            cenY = self.iY[np.array(peaks[:, 0], dtype=np.int64),
                        np.array(peaks[:, 1], dtype=np.int64),
                        np.array(peaks[:, 2], dtype=np.int64)] + 0.5

            x = cenX - self.ipx  # center[0]
            y = cenY - self.ipy  # center[1]
            pixSize = float(self.det.pixel_size(self.evt))
            self.detectorDistance = np.mean(self.det.coords_z(self.evt)) * 1e-6 # metres
            detdis = float(self.detectorDistance)
            z = detdis / pixSize * np.ones(x.shape)  # pixels

            ebeamDet = psana.Detector('EBeam')
            ebeam = ebeamDet.get(self.evt)
            try:
                photonEnergy = ebeam.ebeamPhotonEnergy()
            except:
                photonEnergy = 1

            wavelength = 12.407002 / float(photonEnergy)  # Angstrom
            norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            qPeaks = (np.array([x, y, z]) / norm - np.array([[0.], [0.], [1.]])) / wavelength
            [meanClosestNeighborDist, pairsFoundPerSpot] = self.calculate_likelihood(qPeaks)
        else:
            pairsFoundPerSpot = 0.0
        return pairsFoundPerSpot

    def respond2Clients(self):
        message = self.socket.recv()
        mode,exp,runnum,detname = message.split(",")
        runnum = int(runnum)
        if mode == "check":
            status = self.checkStatus(exp,runnum,detname)
        elif mode == "update":
            status = self.updateStatus(exp,runnum,detname)
        self.socket.send(str(status))

    def checkStatus(self, exp, runnum, detname):
        if self.host == "":
            outfile = os.path.join(self.outdir, exp + '.npz')
            try:
                npzfile = np.load(outfile)
                ind = np.where(npzfile['runs'] == runnum)[0]
                if len(ind) == 0:
                    return False
                else:
                    return True
            except:
                return False
        else:
            # ask master for status
            self.socket.send("check,"+exp+","+str(runnum)+","+detname)
            message = self.socket.recv()
            if message == "True":
                return True
            else:
                return False

    def updateStatus(self, exp, runnum, detname):
        if self.host == "":
            outfile = os.path.join(self.outdir, exp + '.npz')
            todo = os.path.join(self.outdir, 'todo.txt')
            try:
                npzfile = np.load(outfile)
                runs = npzfile['runs']
                ind = np.where(runs == runnum)[0]
                # Does the run exist?
                if len(ind) == 0:
                    print "Found one!"
                    # insert run
                    ind = np.where(runs < runnum)[0]
                    if len(ind) == 0:
                        runs = np.insert(runs, 0, runnum)
                    else:
                        runs = np.insert(runs, ind[-1]+1, runnum)
                    np.savez(outfile, runs=runs, detname=detname)
                else:
                    print "This exp:run exists"
            except:
                runs = np.array([runnum],)
                np.savez(outfile, runs=runs, detname=detname)
                print "Create new file"
            with open(todo, "a") as f:
                msg = exp + " " + str(runnum) + " " + detname + "\n"
                f.write(msg)
        else:
            # ask master to update
            self.socket.send("update," + exp + "," + str(runnum) + "," + detname)
            message = self.socket.recv()
            if message == "True":
                return True
            else:
                return False

    def run(self):
        """ Runs the peakfinder, adds values to the database, trains peaknet, and reports model to the master

        Arguments:
        alg -- the peakfinding algorithm
        kwargs -- peakfinding parameters, host and server name, client name
        """
        while True:
            if self.host == "":
                # respond to clients
                self.respond2Clients()
            else:
                # randomly choose experiment + run
                self.exp, self.runnum, self.detname = randExpRunDet()
                print "exp run det: ", self.exp, self.runnum, self.detname
                if not self.checkStatus(self.exp, self.runnum, self.detname):
                    self.ds = safeDataSource(self.exp, self.runnum)
                    self.run = self.ds.runs().next()
                    self.times = self.run.times()
                    self.det = psana.Detector(self.detname)
                    self.det.do_reshape_2d_to_3d(flag=True)
                    try:
                        self.iX = np.array(self.det.indexes_x(self.run), dtype=np.int64)
                        self.iY = np.array(self.det.indexes_y(self.run), dtype=np.int64)
                        self.ipx, self.ipy = self.det.point_indexes(self.run, pxy_um=(0, 0))
                        self.alg = PyAlgos()
                        self.alg.set_peak_selection_pars(npix_min=2, npix_max=30, amax_thr=300, atot_thr=600, son_min=10)
                        mask = self.det.mask(self.runnum, calib=True, status=True, edges=True, central=True, unbond=True, unbondnbrs=True)

                        samples = np.linspace(0, len(self.times), num=100, endpoint=False, retstep=False, dtype='int')
                        offset =  np.floor(np.random.uniform(0, len(self.times)-samples[-1])).astype('int')
                        mysamples = samples + offset
                        numCrystals = 0
                        for self.eventNum in mysamples:
                            self.evt = self.run.event(self.times[self.eventNum])
                            calib = self.det.calib(self.evt)
                            if calib is not None:
                                peaks = self.alg.peak_finder_v3r3(calib, rank=3, r0=3, dr=2, nsigm=10, mask=mask.astype(np.uint16))
                                if self.likelihood(peaks) >= self.goodLikelihood:
                                    numCrystals += 1
                                if numCrystals >= self.minCrystals:
                                    self.updateStatus(self.exp, self.runnum, self.detname)
                                    break
                    except:
                        print "Could not analyse this run"

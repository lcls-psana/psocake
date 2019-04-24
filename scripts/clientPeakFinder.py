import psana
import os
import numpy as np
from psalgos.pypsalgos import PyAlgos
from ImgAlgos.PyAlgos import PyAlgos as PA
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import time
import json
import base64
from psana import *
import random
from peaknet import Peaknet 
from crawler import Crawler
import clientAbstract
from clientSocket import clientSocket
from peakDatabase import PeakDatabase

from scipy import signal as sg
from skimage.measure import label, regionprops
from skimage import morphology

from peaknet_utils import json_parser, psanaRun, nEvents
import pandas
import Utils

import torch

#random.seed(9876)

class clientPeakFinder(clientAbstract.clientAbstract):

    def __init__(self):
        #Amount of events sent to PeakNet (For future, rayonix batchSize = 7 would be GPU memory efficient)
        self.batchSize = 6
        self.goodLikelihood = .003
        #Limit of events iterated through in a run
        self.eventLimit = self.batchSize * 10
        #Minimum number of peaks to be found to calculate likelihood
        self.goodNumPeaks = 10
        #Minimum number of events to be found before peak finding on 1000 events of a run
        self.minEvents = 1

        #If the last run was a good run:
        self.goodRun = False
        #If batch size reached in the middle of a good run, have the worker return to this run
        self.useLastRun = False
        self.det = None
        self.exp = None
        self.runnum = None
        self.eventNum = 0

        #Step 1: Both Queen and Clients make their own Peaknet instances.
        self.peaknet = Peaknet()
        print("Made a peaknet instance!")
        #Step 1a
        logdir = "/reg/d/psdm/cxi/cxitut13/res/autosfx/tensorboard"
        self.peaknet.set_writer(project_name=os.path.join(logdir,"test"))
        # FIXME: fetch weights from mongoDB: peaknet.updateModel(model)
        self.peaknet.loadCfg("/reg/neh/home/liponan/ai/pytorch-yolo2/cfg/newpeaksv10-asic.cfg")
        print("Loaded arbitrary weights!")
        #Step 2: Client loads arbitrary DN weights and connects to GPU
        print("Connecting to GPU...")
        self.peaknet.init_model()
        #self.peaknet.model = torch.load("/reg/d/psdm/cxi/cxic0415/res/liponan/antfarm_backup/api_demo_psana_model_000086880")
        self.peaknet.model.cuda()
        print("Connected!")

    # @Abstract method (this is the only required method for a plugin)
    def algorithm(self, **kwargs):
        """ Initialize the peakfinding algorithim with keyword 
        arguments given by the user, then run the peakfinding 
        algorithm on random sets of experiment runs

        Arguments:
        **kwargs -- a dictionary of arguments containing peakfinding parameters
        """
        npxmin = kwargs["npix_min"]
        npxmax = kwargs["npix_max"]
        amaxthr = kwargs["amax_thr"]
        atotthr = kwargs["atot_thr"]
        sonmin = kwargs["son_min"]
        alg = PyAlgos(mask = None, pbits = 0)
        alg.set_peak_selection_pars(npix_min=npxmin, npix_max=npxmax, amax_thr=amaxthr, atot_thr=atotthr, son_min=sonmin) #(npix_min=2, npix_max=30, amax_thr=300, atot_thr=600, son_min=10)
        self.run(alg, **kwargs)

    def createDictionary(self, exp, runnum, event, peaks, likelihood):
        """Create a dictionary that holds the important information of events with crystals

        Arguments:
        exp -- experiment name
        runnum -- run number
        event -- event number
        peaks -- number of peaks found
        labels -- location of peaks
        """
        post = {"Exp":exp,
                "RunNum":runnum,
                "Event":event,
                "Peaks":peaks,
                "Likelihood":likelihood}
        return post

    def bitwise_array(self, value):
        """ Convert a numpy array to a form that can be sent through json.
        
        Arguments:
        value -- a numpy array that will be converted.
        """
        if np.isscalar(value):
            return value
        val = np.asarray(value)
        return [base64.b64encode(val), val.shape, val.dtype.str]

    def calculate_likelihood(self, qPeaks):
        """ Calculate the likelihood that an event is a crystal
    
        Arguments:
        qPeaks -- 
        """
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


    def getPeaks(self, d, alg, hdr, fmt, mask, times, env, run, j):
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
            peaks = alg.peak_finder_v3r3(nda, rank=3, r0=3, dr=2, nsigm =10)
            numPeaksFound = len(peaks)
            return [evt, nda, peaks, numPeaksFound]
        else:
            return[None,None,None,None]

    def getLikelihood(self, d, evt, peaks, numPeaksFound):
        """ Returns the likeligood value for an event with 15 or more peaks
        
        Arguments:
        d -- psana.Detector() of this experiment's detector
        evt -- ds.env()
        peaks -- the peaks found for this event
        numPeaksFound -- number of peaks found for this event
        """
        if (numPeaksFound >= self.goodNumPeaks):
            ix = d.indexes_x(evt)
            iy = d.indexes_y(evt) 
            d.ipx, d.ipy = d.point_indexes(evt, pxy_um=(0, 0))
            d.iX = np.array(ix, dtype=np.int64)
            d.iY = np.array(iy, dtype=np.int64)
            cenX = d.iX[np.array(peaks[:, 0], dtype=np.int64),
                        np.array(peaks[:, 1], dtype=np.int64),
                        np.array(peaks[:, 2], dtype=np.int64)] + 0.5
            cenY = d.iY[np.array(peaks[:, 0], dtype=np.int64),
                        np.array(peaks[:, 1], dtype=np.int64),
                        np.array(peaks[:, 2], dtype=np.int64)] + 0.5
            x = cenX - d.ipx
            y = cenY - d.ipy
            pixSize = float(d.pixel_size(evt))
            detdis = np.mean(d.coords_z(evt)) * 1e-6 # metres
            z = detdis / pixSize * np.ones(x.shape)  # pixels
            #ebeam = ebeamDet.get(evt)
            #try:
            #    photonEnergy = ebeam.ebeamPhotonEnergy()
            #except:
            photonEnergy = 1
            wavelength = 12.407002 / float(photonEnergy)  # Angstrom	
            norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            qPeaks = (np.array([x, y, z]) / norm - np.array([[0.], [0.], [1.]])) / wavelength
            [meanClosestNeighborDist, pairsFoundPerSpot] = self.calculate_likelihood(qPeaks)
            return pairsFoundPerSpot
        else:
            return 0

    def getStreaks(self, det, times, run, j):
        """Finds peaks within an event, and returns the event information, peaks found, and hits found

        Arguments:
        det -- psana.Detector() of this experiment's detector
        times -- all the events for this run
        run -- ds.runs().next(), the run information
        j -- this event's number

        """
        evt = run.event(times[j])

        width = 300  # crop width
        sigma = 1
        smallObj = 15 # delete streaks if num pixels less than this
        calib = det.calib(evt)
        if calib is None:
            return [None, None, None]
        img = det.image(evt, calib)

        # Edge pixels
        edgePixels = np.zeros_like(calib)
        for i in range(edgePixels.shape[0]):
            edgePixels[i, 0, :] = 1
            edgePixels[i, -1, :] = 1
            edgePixels[i, :, 0] = 1
            edgePixels[i, :, -1] = 1
        imgEdges = det.image(evt, edgePixels)

        # Crop centre of image
        (ix, iy) = det.point_indexes(evt)
        halfWidth = int(width // 2)  # pixels
        imgCrop = img[ix - halfWidth:ix + halfWidth, iy - halfWidth:iy + halfWidth]
        imgEdges = imgEdges[ix - halfWidth:ix + halfWidth, iy - halfWidth:iy + halfWidth]
        myInd = np.where(imgEdges == 1)

        # Blur image
        imgBlur = sg.convolve(imgCrop, np.ones((2, 2)), mode='same')
        mean = imgBlur[imgBlur > 0].mean()
        std = imgBlur[imgBlur > 0].std()

        # Mask out pixels above 1 sigma
        mask = imgBlur > mean + sigma * std
        mask = mask.astype(int)
        signalOnEdge = mask * imgEdges
        mySigInd = np.where(signalOnEdge == 1)
        mask[myInd[0].ravel(), myInd[1].ravel()] = 1

        # Connected components
        myLabel = label(mask, neighbors=4, connectivity=1, background=0)
        # All pixels connected to edge pixels is masked out
        myMask = np.ones_like(mask)
        myParts = np.unique(myLabel[myInd])
        for i in myParts:
            myMask[np.where(myLabel == i)] = 0

        # Delete edges
        myMask[myInd] = 1
        myMask[mySigInd] = 0

        # Delete small objects
        myMask = morphology.remove_small_objects(np.invert(myMask.astype('bool')), smallObj)

        # Convert assembled to unassembled
        wholeMask = np.zeros_like(img)
        wholeMask[ix - halfWidth:ix + halfWidth, iy - halfWidth:iy + halfWidth] = myMask
        calibMask = det.ndarray_from_image(evt, wholeMask)
        streaks = []
        for i in range(calib.shape[0]):
            for j in regionprops(calibMask[i].astype('int')):
                xmin, ymin, xmax, ymax = j.bbox
                streaks.append([i, ymin, xmin, ymax - ymin, xmax - xmin])
        return [evt, calib, streaks]

    def evaluateExpRunDet(self, crawler, alg, peakDB, verbose):
        """ Finds a random experiment run, finds peaks, and determines likelihood of crystal event. If an event is
        likely to be a crystal, it will be used to train PeakNet. This function continues until the amount of 
        events found is equal to the batchSize.
       
        return the list of peaks in good events,  the list of corresponding images for the good events
        """
        # List labels from good events
        labels = [] 

        # List of images of good event
        imgs = [] 

        # Number of peaks found during this function's call
        totalNumPeaks = 0 

        # Number of events found during this function's call
        numGoodEvents = 0

        # Time when the client began working
        clientBeginTime = time.time()

        #How long the client has been running for
        runTime = 0

        while True:
            #Time when the client began working on this event, used in print statement only
            beginTimeForThisEvent = time.time()

            # Until the amount of good events found is equal to the batchSize, keep finding experiments to find peaks on
            if(len(labels) == self.batchSize):
                self.useLastRun = True
                if verbose > 2: print("### (outer) This run has some hits.")
                break

            #Keep working on the previous experiment/run
            #if((self.useLastRun) and (self.exp is not None)): #TODO: dont redo information functions
            #    pass
            #Use the crawler to fetch a random experiment+run+det
            #else:
            #    self.exp, self.runnum, self.det, self.eventNum = crawler.next(self.eventNum)
                #self.exp, self.runnum, self.det = ["cxif5315", 128, "DsaCsPad"] #A good run to use to quickly test if the client works
                #eventInfo = self.getDetectorInformation(self.exp, self.runnum, self.det)
                #self.d, self.hdr, self.fmt, self.mask, self.times, self.env, self.run, self.numEvents = eventInfo[:]
            #    if verbose > 0: print("\nExperiment: %s, Run: %s, Detector: %s, NumEvents: %d"%(self.exp, str(self.runnum), self.det, self.numEvents))
                #Initialize the number of good events/hits for this event
                # If less than 3 in the first 1000 events, this run is skipped
            print("#### here")
            self.exp, self.runnum, self.det, self.eventNum = crawler.next(self.eventNum, self.useLastRun)

            #Peak find for each event in an experiment+run
            
            while (self.eventNum < crawler.numEvents):
                print("#### there")
                if verbose > 0: print("eventNum: ", self.eventNum)

                # Until the amount of good events found is equal to the batchSize, keep finding experiments to find peaks on
                if(len(labels) == self.batchSize):
                    self.useLastRun = True
                    if verbose >= 0: print("### (inner) This run has some hits.")
                    break

                #If the amount of good events found is less than minEvents before the eventLimit, then 
                #stop and try peak finding on a new experiment+run
                if((self.eventNum >= self.eventLimit) and (crawler.numGoodEventsInThisRun < self.minEvents)):
                    self.useLastRun = False
                    if verbose >= 0: print("### This run is mostly misses. Skipping this run.")
                    break

                if((self.eventNum >= self.eventLimit) and crawler.numGoodEventsInThisRun/float(self.eventNum) < 0.01):
                    self.useLastRun = False
                    if verbose >= 0: print("### Hit rate is too low. Skipping this run: ", crawler.numGoodEventsInThisRun/float(self.eventNum))
                    break

                #Get Peak Info
                evt, nda, peaks, numPeaksFound = self.getPeaks(crawler.d, alg, crawler.hdr, crawler.fmt, crawler.mask,
                                                               crawler.times, crawler.env, crawler.run, self.eventNum)
                if verbose > 1: print("numPeaksFound: ", numPeaksFound)

                #Get Likelihood
                likelihood = self.getLikelihood(crawler.d, evt, peaks, numPeaksFound)
                if verbose > 1: print("likelihood: ", likelihood, self.goodLikelihood)
                if (likelihood >= self.goodLikelihood):
                    print("Crystal hit found: ", self.exp, self.runnum, self.eventNum, likelihood)
                else:
                    print("Miss: ", self.exp, self.runnum, self.eventNum, likelihood)
                # [[list of seg], [list of row], [list of col]], ... labelsForThisEvent[0:2][0] corresponds to one label
                labelsForThisEvent = [np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]

                #If this event is a good event, save the labels to train PeakNet
                if (likelihood >= self.goodLikelihood):
                    if verbose > 2: print crawler.hdr
                    cls = 0
                    height = 7
                    width = 7
                    for peak in peaks:
                        totalNumPeaks += 1 #number of peaks
                        seg,row,col,npix,amax,atot = peak[0:6]
                        labelsForThisEvent[0] = np.append(labelsForThisEvent[0], np.array([cls]), axis = 0) # class
                        labelsForThisEvent[1] = np.append(labelsForThisEvent[1], np.array([seg]), axis = 0) # seg
                        labelsForThisEvent[2] = np.append(labelsForThisEvent[2], np.array([row]), axis = 0) # row
                        labelsForThisEvent[3] = np.append(labelsForThisEvent[3], np.array([col]), axis=0) # col
                        labelsForThisEvent[4] = np.append(labelsForThisEvent[4], np.array([height]), axis=0) # height
                        labelsForThisEvent[5] = np.append(labelsForThisEvent[5], np.array([width]), axis=0) # width
                    if verbose > 2: print crawler.fmt % (cls, seg, row, col, height, width, npix, atot)

                    labels.append(labelsForThisEvent)

                    # Shape of imgs = (j <= (batchsize = 64), m = 32 tiles per nda, h = 185, w = 388)
                    imgs.append(nda)

                    #increase the skip condition value
                    crawler.numGoodEventsInThisRun += 1

                    #increase the database number of hits value
                    numGoodEvents += 1

                #create a dictionary to be posted on the database
                kwargs = self.createDictionary(self.exp, str(self.runnum), str(self.eventNum), numPeaksFound, likelihood) #, labelsForThisEvent)
                peakDB.addExpRunEventPeaks(**kwargs) #TODO: MongoDB doesnt like the numpy array

                self.eventNum += 1

            if(self.eventNum >= crawler.numEvents):
                 self.useLastRun = False

            #Time when the client ended working on this event, used in print statement only
            endTimeForThisEvent = time.time()
            if verbose > 1: print("This took %d seconds" % (endTimeForThisEvent-beginTimeForThisEvent))

        #add total peaks and hits value to database when finished
        #TODO: confirm that this updates the values and does not overwrite the values...
        peakDB.addPeaksAndHits(totalNumPeaks, numGoodEvents)
        #peakDB.printDatabase()

        # update runTime
        clientRunTime = time.time() - clientBeginTime

        if verbose > 0: print("Peaks", totalNumPeaks, "Events", numGoodEvents)
        if verbose > 1: print("Run time: ", clientRunTime)
        return labels, imgs

    def run(self, alg, **kwargs):
        """ Runs the peakfinder, adds values to the database, trains peaknet, and reports model to the master

        Arguments:
        alg -- the peakfinding algorithm
        kwargs -- peakfinding parameters, host and server name, client name
        """
        socket = clientSocket(**kwargs)
        peakDB = PeakDatabase(**kwargs) #create database to store good event info in
        verbose = kwargs["verbose"]

        # Crawler used to fetch a random experiment + run
        myCrawler = Crawler()
        counter = 0

        kk = 0
        jj = 0
        outdir = "/reg/d/psdm/cxi/cxic0415/res/liponan/antfarm_backup"

        while(True):
            # Randomly choose an experiment:run and look for crystal diffraction pattern and return with batchSize labels
            labels, imgs = self.evaluateExpRunDet(myCrawler, alg, peakDB, verbose)
            counter += 1

            imgs = np.array(imgs)

            #Step 3: Client tells queen it is ready
            socket.push(["Ready", counter])

            fname = os.path.join(outdir, kwargs["name"]+"_"+str(self.peaknet.model.seen) + ".pkl")
            if kk % 3 == 0:
                torch.save(self.peaknet.model, fname)
            kk += 1

            #Step 4: Client receives model from queen
            print("#### enter the dragon")
            val = socket.pull()
            print("#### flag: ", val)
            flag, model = val

            #Step 5: Client updateModel(model from queen)
            self.peaknet.updateModel(model, check=True)

            self.peaknet.model.cuda()

            fname = '/reg/neh/home/liponan/ai/peaknet4antfarm/val_and_test.json' # FIXME: don't hard code
            df = None

            print("**********************************: ", kwargs["isFirstWorker"], kwargs["isFirstWorker"]==1)

            if flag == 'train':
                # Step 6: Client trains its Peaknet instance
                numPanels = imgs[0].shape[0]
                mini_batch_size = numPanels * self.batchSize
                self.peaknet.train(imgs, labels, mini_batch_size=mini_batch_size, box_size=7, use_cuda=True,
                                   writer=self.peaknet.writer, verbose=True)
                if jj == 1:
                    self.peaknet.snapshot(imgs, labels, tag=kwargs["name"])
                    jj = 0
                jj += 1
                # Step 7: Client sends the new model to queen
                #print("##### Grad: ", self.peaknet.getGrad())
                socket.push(["Gradient", self.peaknet.getGrad(), mini_batch_size])

            print("@@@@@@@@@ isFirstWorker: ", kwargs["isFirstWorker"], kwargs["isFirstWorker"]==1)
            if kwargs["isFirstWorker"]==True:
                print("@@@@@ client running validation")
                if flag == 'validate':
                    # Read json
                    df = json_parser(fname, mode='validate', subset=False)
                    counter = 0
                elif flag == 'validateSubset':
                    # Read json
                    df = json_parser(fname, mode='validate', subset=True)

                if flag.startswith('validate'):
                    pdl = Utils.psanaDataLoader(df, self.batchSize)
                    for i in range(pdl.numMacros):
                        valImgs, valLabels = pdl[i]
                        numPanels = valImgs[0].shape[0]
                        # Step 6: Validation
                        self.peaknet.validate(valImgs, valLabels, mini_batch_size=numPanels * self.batchSize, box_size=7,
                                              use_cuda=True, writer=self.peaknet.writer, verbose=True)



            #Step 8: Queen does updateGradient(new model from client)
            #Step 9: Queen Optimizes
            #Step 10: Repeat Steps 3-10


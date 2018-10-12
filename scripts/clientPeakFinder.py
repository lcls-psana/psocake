import psana
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

class clientPeakFinder(clientAbstract.clientAbstract):

    #Intialize variables

    #Amount of events sent to PeakNet
    batchSize = 3
    #Calculated Likelihood that counts as a "good event"
    goodLikelihood = .03
    #Limit of events iterated through in a run
    eventLimit = 100
    #Minimum number of peaks to be found to calculate likelihood
    goodNumPeaks = 15
    #Minimum number of events to be found before peak finding on 1000 events of a run
    minEvents = 3

    #If the last run was a good run:
    goodRun = False
    #If batch size reached in the middle of a good run, have the worker return to this run
    useLastRun = False
    det = None
    exp = None
    runnum = None
    eventNum = None

    #Step 1: Both Queen and Clients make their own Peaknet instances. 
    peaknet = Peaknet()
    print("Made a peaknet instance!")
    # FIXME: fetch weights from mongoDB: peaknet.updateModel(model)
    peaknet.loadDNWeights()
    print("Loaded arbitrary weights!")
    #Step 2: Client loads arbitrary DN weights and connects to GPU
    print("Connecting to GPU...")
    peaknet.model.cuda()
    print("Connected!")

    # @Abstract method (this is the only required method for a plugin)
    def algorithm(self, **kwargs):
        """ Initialize the peakfinding algorithim with keyword 
        arguments given by the user, then run the peakfinding 
        algorithm on random sets of experiment runs

        Arguments:
        **kwargs -- a dictionary of arguments containing peakfinding parameters
        """
        print("In clientPeakFinder algorithm")
        npxmin = kwargs["npix_min"]
        npxmax = kwargs["npix_max"]
        amaxthr = kwargs["amax_thr"]
        atotthr = kwargs["atot_thr"]
        sonmin = kwargs["son_min"]
        alg = PyAlgos(mask = None, pbits = 0)
        alg.set_peak_selection_pars(npix_min=npxmin, npix_max=npxmax, amax_thr=amaxthr, atot_thr=atotthr, son_min=sonmin) #(npix_min=2, npix_max=30, amax_thr=300, atot_thr=600, son_min=10)
        self.run(alg, **kwargs)

    def createDictionary(self, exp, runnum, event, peaks, labels):
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
                "Labels":labels}
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

    def getDetectorInformation(self, exp, runnum, det):
        """ Returns the detector and the number of events for
        this run.
        
        Arguments:
        exp -- the experiment name
        runnum -- the run number for this experiment
        det -- the detector used for this experiment
        """
        ds = psana.DataSource('exp=%s:run=%d:idx'%(exp,runnum))
        d = psana.Detector(det)
        d.do_reshape_2d_to_3d(flag=True)
        hdr = '\nClass  Seg  Row  Col  Height  Width  Npix    Amptot'
        fmt = '%5d %4d %4d %4d  %6d %6d %5d  %8.1f'
        run = ds.runs().next()
        times = run.times()
        env = ds.env()
        numEvents = len(times)
        mask = d.mask(runnum,calib=True,status=True,edges=True,central=True,unbond=True,unbondnbrs=True)
        return [d, hdr, fmt, numEvents, mask, times, env, run]

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
            #alg = PA()
            #thr = 20
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
                print i, xmin, ymin, xmax - xmin, ymax - ymin
                streaks.append([i, ymin, xmin, ymax - ymin, xmax - xmin])
        return [evt, calib, streaks]

    def evaluateRun(self, alg, peakDB):
        """ Finds a random experiment run, finds peaks, and determines likelihood of crystal event. If an event is
        likely to be a crystal, it will be used to train PeakNet. This function continues until the amount of 
        events found is equal to the batchSize.
       
        return the list of peaks in good events,  the list of corresponding images for the good events,
        the total number of peaks found by this function, and the total number of hits found 
        """
        # Will be psana.Detector(det)
        d = 0 

        # Will be run.event(this event)
        evt = 0

        # List labels from good events
        labels = [] 

        # List of images of good event
        imgs = [] 

        # Number of peaks found during this function's call
        totalNumPeaks = 0 

        # Number of events found during this function's call
        numGoodEvents = 0

        # Crawler used to fetch a random experiment + run
        myCrawler = Crawler() 

        # Time when the client began working
        clientBeginTime = time.time() 

        #Time when the client finished working. Hasnt began yet so it is not set
        currentTime = clientBeginTime 

        #How long the client has been running for
        runTime = 0

        while True:
            #Time when the client began working on this event, used in print statement only
            beginTimeForThisEvent = time.time()

            # Until the amount of good events found is equal to the batchSize, keep finding experiments to find peaks on
            if(len(labels) >= self.batchSize):
                self.useLastRun = True
                break

            #Keep working on the previous experiment/run
            if((self.useLastRun) and (self.exp is not None)): #TODO: dont redo information functions
                exp = self.exp
                runnum = self.runnum
                det = self.det
            #Use the crawler to fetch a random experiment+run
            else:
                self.exp, self.runnum, self.det = myCrawler.returnOneRandomExpRunDet(self.goodRun)
                #self.exp, self.runnum, self.det = ["cxif5315", 128, "DsaCsPad"] #A good run to use to quickly test if the client works
                #self.exp, strrunnum, self.det = ["cxitut13", "0010", "DscCsPad"]  # A good run to use to quickly test if the client works
                self.eventNum = 0

            print("\nExperiment: %s, Run: %s, Detector: %s"%(self.exp, str(self.runnum), self.det))
            eventInfo = self.getDetectorInformation(self.exp, self.runnum, self.det)
            d, hdr, fmt, numEvents, mask, times, env, run = eventInfo[:]
            print("%d Events to find peaks on"%numEvents)

            #Initialize the number of good events/hits for this event
            # If less than 3 in the first 1000 events, this run is skipped
            numGoodEventsInThisRun = 0
            #Peak find for each event in an experiment+run
            while (self.eventNum < numEvents):

                print("eventNum: ", self.eventNum)

                # Until the amount of good events found is equal to the batchSize, keep finding experiments to find peaks on
                if(len(labels) >= self.batchSize):
                    self.useLastRun = True
                    print("1")
                    break

                #If the amount of good events found is less than minEvents before the eventLimit, then 
                #stop and try peak finding on a new experiment+run
                if((self.eventNum >= self.eventLimit) and (numGoodEventsInThisRun < self.minEvents)):
                    self.useLastRun = False
                    print("2")
                    break

                #Get jet streak
                evt, nda, streaks = self.getStreaks(d, times, run, self.eventNum)
                print("jet streak found: ", streaks)
                if nda is None:
                    print("3")
                    self.eventNum+=1
                    continue

                #Get Peak Info
                evt, nda, peaks, numPeaksFound = self.getPeaks(d, alg, hdr, fmt, mask, times, env, run, self.eventNum)
                print("numPeaksFound: ", numPeaksFound)

                #Get Likelihood
                likelihood = self.getLikelihood(d, evt, peaks, numPeaksFound)
                print("likelihood: ", likelihood, self.goodLikelihood)

                # [[list of seg], [list of row], [list of col]] ... labelsForThisEvent[0:2][0] corresponds to one label
                labelsForThisEvent = [np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])]

                # Add jet streak labels
                cls = 1
                for streak in streaks:
                    seg, row, col, height, width = streak
                    labelsForThisEvent[0] = np.append(labelsForThisEvent[0], np.array([cls]), axis=0)  # class
                    labelsForThisEvent[1] = np.append(labelsForThisEvent[1], np.array([seg]), axis=0)  # seg
                    labelsForThisEvent[2] = np.append(labelsForThisEvent[2], np.array([row]), axis=0)  # row
                    labelsForThisEvent[3] = np.append(labelsForThisEvent[3], np.array([col]), axis=0)  # col
                    labelsForThisEvent[4] = np.append(labelsForThisEvent[4], np.array([height]), axis=0)  # height
                    labelsForThisEvent[5] = np.append(labelsForThisEvent[5], np.array([width]), axis=0)  # width

                #If this event is a good event, save the labels to train PeakNet
                if (likelihood >= self.goodLikelihood):
                    print hdr
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
                    print fmt % (cls, seg, row, col, height, width, npix, atot)

                if len(labelsForThisEvent) > 0:
                    # labels[0:j] corresponds to each event in labels,
                    # for each event in labels there is a tuple:
                    # [[list of segments], [list of rows], [list of columns]]
                    # That is: labels =
                    # [ [[list of segments], [list of rows], [list of columns]] ,
                    #   [[list of segments], [list of rows], [list of columns]] ... ]
                    labels.append(np.array(labelsForThisEvent))

                    # Shape of imgs = (j <= (batchsize = 64), m = 32 tiles per nda, h = 185, w = 388)
                    imgs.append(nda)

                    #increase the skip condition value
                    numGoodEventsInThisRun += 1

                    #increase the database number of hits value
                    numGoodEvents += 1

                #create a dictionary to be posted on the database
                #kwargs = self.createDictionary(self.exp, strrunnum, str(self.eventNum+1), numPeaksFound, labelsForThisEvent)
                #peakDB.addExpRunEventPeaks(**kwargs) #TODO: MongoDB doesnt like the numpy array

                self.eventNum += 1
                print("Done ", self.eventNum, numEvents)

            if(self.eventNum >= numEvents):
                 self.useLastRun = False
            
            #update runTime
            currentTime = time.time()
            runTime = currentTime - clientBeginTime

            #Time when the client ended working on this event, used in print statement only
            endTimeForThisEvent = time.time()
            print("This took %d seconds" % (endTimeForThisEvent-beginTimeForThisEvent))

        #add total peaks and hits value to database when finished
        #TODO: confirm that this updates the values and does not overwrite the values...
        peakDB.addPeaksAndHits(totalNumPeaks, numGoodEvents)
        print("Peaks", totalNumPeaks, "Events", numGoodEvents)
        print(runTime)
        print("Labels: ", labels)
        return [labels, imgs, totalNumPeaks, numGoodEvents]

    def run(self, alg, **kwargs):
        """ Runs the peakfinder, adds values to the database, trains peaknet, and reports model to the master

        Arguments:
        alg -- the peakfinding algorithm
        kwargs -- peakfinding parameters, host and server name, client name
        """
        socket = clientSocket(**kwargs)
        peakDB = PeakDatabase(**kwargs) #create database to store good event info in
        while(True):
            # Randomly choose an experiment:run and look for crystal diffraction pattern and return with batchSize labels
            labels, imgs, totalNumPeaks, numGoodEvents = self.evaluateRun(alg, peakDB)

            #print(len(labels), "events in labels")
            #print(len(labels[0]), "lists in tuples of an event")
            #print(len(labels[0][0]), "values in first list in tuple")

            imgs = np.array(imgs)

            #Step 3: Client tells queen it is ready
            socket.push("Im Ready!")

            #Step 4: Client recieves model from queen 
            model = socket.pull()

            #Step 5: Client updateModel(model from queen)
            self.peaknet.updateModel(model)

            self.peaknet.model.cuda()

            #print("#### model.w before: ", next(self.peaknet.model.parameters())[-1,0,:,:])

            #Step 6: Client trains its Peaknet instance
            print("###### training: ", labels, len(labels), self.eventNum)
            import pickle
            outfile = "/reg/neh/home/yoon82/peaknetLabels.pkl"
            with open(outfile, 'wb') as outfd:
                pickle.dump(labels, outfd, protocol=pickle.HIGHEST_PROTOCOL)
            exit()
            self.peaknet.train(imgs, labels, batch_size=1, box_size = 7, use_cuda=True)

            #print("#### model.w after: ", next(self.peaknet.model.parameters())[-1,0,:,:])
            #print("#### model.grad: ", next(self.peaknet.model.parameters()).grad[-1,0,:,:])

            #Step 7: Client sends the new model to queen
            socket.push(self.peaknet.getGrad())

            #print("#### model.grad push: ", next(self.peaknet.model.parameters()).grad[-1,0,:,:])

            #Step 8: Queen does updateGradient(new model from client)
            #Step 9: Queen Optimizes
            #Step 10: Repeat Steps 3-10


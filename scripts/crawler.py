import random
import os
import re
import numpy as np
import psana
import json
import glob
import Utils

class Crawler:
    """A Crawler instance is used to search through the experimental 
    data and fetch a random, non-corrupt experiment run pair
    """
    def __init__(self):
        """When a Crawler is initialized, an empty list of the experiments and runs 
        the crawler has already reported is created, this will be used to keep track
        of experiments and runs the crawler has found, so that it does not re-report 
        these runs. The crawler also reads a file which reports to it which runs are 
        corrupt, so that it skips over these runs.
        """
        #random.seed(3)

        self.evt = None
        self.runnum = None
        self.det = None
        self.evt = 0

        self.d = None
        self.hdr = None
        self.fmt = None
        self.mask = None
        self.times = None
        self.env = None
        self.run= None
        self.numEvents = None

        self.numGoodEventsInThisRun = 0

        self.seenList = []
        self.badList = []
        self.readFile('listOfUnreadableFiles.txt')

    def next(self, prevEvt, useLastRun):
        # Return next exp,run,det,evt
        if not useLastRun: self.exp = None

        if self.exp is None:
            # pick a random experiment
            print("@@ fetching random exp run det")
            self.exp, self.runnum, self.det, self.evt = self.randExpRunDet()
            self.numGoodEventsInThisRun = 0
        else:
            # 1) continue current exp,run,det,evt
            # 2) move onto next run number (if exists)
            # 3) select random exp run
            if prevEvt+1 < self.numEvents:
                print("@@ fetching next set of events")
                self.evt = prevEvt+1
            else:
                print("@@ fetching next run")
                self.runnum = self.getNextRun()
                if self.runnum is None:
                    print("@@ fetching random exp run det")
                    self.exp, self.runnum, self.det, self.evt = self.randExpRunDet()
                self.evt = 0
                self.numGoodEventsInThisRun = 0
        print("@@ found: ", self.exp, self.runnum, self.det, self.evt, self.numEvents)
        return self.exp, self.runnum, self.det, self.evt

    def randExpRun(self):
        """searches through the data files and reports a random experiment and run number
        """
        #random.seed(1)
        debugMode = True
        choice = None
        filetype = random.choice(["cxi"])
        myList = ['cxic0415',]#['cxilr7616']# # FIXME: psana can not handle xtcs with no events
        myRuns = [95, 99] #[99, 51] #[52]# # FIXME: psana can not handle xtcs with no events

        if debugMode:
            choice = random.choice(myList)
        else:
            choice = random.choice(os.listdir("/reg/d/psdm/%s" % filetype))

        if ("cxi" in choice):
            try:
                realpath = os.path.realpath('/reg/d/psdm/cxi/' + choice + '/xtc')
                if '/reg/data/ana01' in realpath:  # FIXME: ana01 is down temporarily
                    return [False, 0, 0]
                self.runList = os.listdir(realpath)
                randomRun = random.choice(self.runList)
                if (".xtc.inprogress" in randomRun):
                    return [False, 0, 0]
                elif (".xtc" in randomRun):
                    if (self.isValidRun(filetype, choice, randomRun)):
                        num = re.findall("-r(\d+)-", randomRun)
                        if debugMode:
                            return [True, choice, random.choice(myRuns)]  # FIXME: psana can not handle xtcs with no events
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

    def checkXtc(self, exp, runnum):
        minBytes = 100
        realpath = os.path.realpath('/reg/d/psdm/cxi/' + exp + '/xtc')
        if '/reg/data/ana01' in realpath: # FIXME: ana01 is down temporarily
            return False
        runList = glob.glob('/reg/d/psdm/cxi/' + exp + '/xtc/*-r%04d*' % (runnum))
        idxList = glob.glob('/reg/d/psdm/cxi/' + exp + '/xtc/index/*-r%04d*' % (runnum))
        smdList = glob.glob('/reg/d/psdm/cxi/' + exp + '/xtc/smalldata/*-r%04d*' % (runnum))
        if runList and (len(runList) == len(idxList)) and (len(runList) == len(smdList)):
            for f in runList + idxList:
                if os.stat(f).st_size <= minBytes: # bytes
                    return False
            return True
        else:
            return False

    def randExpRunDet(self):
        while True:
            found, self.exp, self.runnum = self.randExpRun()
            print("@@ Try: ", self.exp, self.runnum)

            if found:
                exists = self.checkXtc(self.exp, self.runnum)
                print("@@ Exists: ", exists)

                if exists:
                    self.det = self.getDetName(self.exp, self.runnum)

                    if self.det is not None:
                        self.getDetectorInformation(self.exp, self.runnum, self.det)
                        self.evt = 0
                        return self.exp, self.runnum, self.det, self.evt

    def getDetName(self, exp, run):
        """ returns detector name.
        """
        if not((exp in self.badList) and (run in self.badList[exp])):
            ds = Utils.safeDataSource(exp, run)
            print("@@@@@ saveDataSource: ", ds)
            if ds is not None:
                print("@@@@@ ds: ", ds)
                detnames = psana.DetNames()
                print("@@@@@ detnames: ", detnames)
                for detname in detnames:

                    if ("DscCsPad" in detname) or ("DsdCsPad" in detname) or ("DsaCsPad" in detname):
                        return detname[1]
            return None

    def getDetectorInformation(self, exp, runnum, det):
        """ Returns the detector and the number of events for
        this run.

        Arguments:
        exp -- the experiment name
        runnum -- the run number for this experiment
        det -- the detector used for this experiment
        """
        ds = Utils.safeDataSource(exp, runnum)
        if ds is not None:
            self.d = psana.Detector(det)
            self.d.do_reshape_2d_to_3d(flag=True)
            self.hdr = '\nClass  Seg  Row  Col  Height  Width  Npix    Amptot'
            self.fmt = '%5d %4d %4d %4d  %6d %6d %5d  %8.1f'
            self.mask = self.d.mask(runnum, calib=True, status=True, edges=True, central=True, unbond=True, unbondnbrs=True)
            self.run = ds.runs().next()
            self.times = self.run.times()
            self.env = ds.env()
            self.numEvents = len(self.times)

    def getNextRun(self):
        # check xtc file exists
        # check idx file exists
        # otherwise return None
        nextRun = self.runnum+1
        exists = self.checkXtc(self.exp, nextRun)
        if exists:
            return nextRun
        else:
            return None



################################################################






    def addToList(self, name, run, det):
        """adds a run to the list of runs the crawler has seen
        
        Arguments:
        name -- experiment name
        run -- run number
        det -- detector name
        """
        self.seenList.append([name, run, det])

    def printList(self):
        """prints the list of exp,run,det the crawler has seen"""
        print(self.seenList)

    def inList(self, name, run, det):
        """checks to see if the exp,run,det the crawler just fetched
        has already been reported
        """
        if([name, run, det] in self.seenList):
            #print("Repeat!")
            return True
        else:
            return False

    def readFile(self, filename):
        """reads a file that contains a dictionary, which informs the crawler
        of which exp,run,det should not be reported

        Arguments:
        filename -- the filename/extension of the list of unreadable exp,runs
        """
        #Read a dictionary
        newlist = []
        with open(filename, 'r') as f:#'listOfUnreadableFiles.txt'
            newlist = json.load(f)
        self.badList = newlist
		
    def isValidRun(self, filetype, name, files):
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

    def detectorValidator(self):
        """ Validates that the experiment and run number found used a CsPad detector
        returns the experiment, run number, detector.
        """
        while True:
            found, name, run = self.randExpRun()
            if found:
                if ("cxi" in name):
                    if not((name in self.badList) and (run in self.badList[name])):
                        ds = Utils.safeDataSource(name, run)
                        if ds is not None:
                            detnames = psana.DetNames()
                            for detname in detnames:
                                if ("DscCsPad" in detname) or ("DsdCsPad" in detname) or ("DsaCsPad" in detname):
                                    return [True, name, run, detname[1]]

    def returnOneRandomExpRunDet(self, goodRun):
        """returns one single, random Experiment, Run Number, and Detector set.
        """
        if(not goodRun):
            loopCondition = True
            while loopCondition:
                boolean, name, run, det = self.detectorValidator()
                if(boolean):
                    if(not self.inList(name, run, det)):
                        self.addToList(name, run, det)
                        loopCondition = False
                        self.name =  name
                        self.runnum = run
                        self.det = det
                        return [name, int(run), det]
        else:
            return self.lastRunWasGood()

    def printSome(self,n):
        """prints n random Experiment, Run Number, and Detector sets.

        Arguments:
        n -- number of Experiment, Run Number, and Detector sets to be printed
        """
        for i in range(n):
            print(self.returnOneRandomExpRunDet())

    def lastRunWasGood(self):
        if(self.checkIfRunExists(int(self.runnum))):
            return [self.name, self.runnum, self.det]
        else:
            return self.returnOneRandomExpRunDet(False)

    def checkIfRunExists(self, runNum):
        boolean = False
        for files in self.runList:
            if (("%d"%(runNum+1)) in files):
                self.runnum = ("%d"%(runNum+1))
                boolean = True
                break
            else:
                continue
        return boolean

#myCrawler = Crawler()
#print(myCrawler.badList)
#myCrawler.printSome(10)
#print(myCrawler.returnOneRandomExpRunDet())
#myCrawler.printList()

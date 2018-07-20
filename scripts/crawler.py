import random
import os
import re
import numpy as np
from psana import *
import json

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
        self.seenList = []
        self.badList = []
        self.readFile('listOfUnreadableFiles.txt')

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

    def crawl(self):
        """searches through the data files and reports a random experiment and run number
        
        """
        filetype = random.choice(["cxi", "mfx"])
        choice = random.choice(os.listdir("/reg/d/psdm/%s"%filetype))
        #filetype = "cxi"
        #choice = random.choice(os.listdir("/reg/d/psdm/cxi"))
        #filetype = "mfx"
        #choice = random.choice(os.listdir("/reg/d/psdm/mfx"))
        if ("cxi" in choice):
            try:
                randomRun = random.choice(os.listdir('/reg/d/psdm/cxi/' + choice + '/xtc'))
                if(".xtc.inprogress" in randomRun):
                    return [False, 0, 0]
                elif(".xtc" in randomRun):
                    if(self.isValidRun(filetype, choice,randomRun)):
                        num = re.findall("-r(\d+)-", randomRun)
                        return [True, choice, num[0]]
                    else:
                        return [False, 0, 0]
                else:
                    return [False, 0, 0]
            except OSError:
                return [False, 0, 0]
        elif("mfx" in choice):
            try:
                randomRun = random.choice(os.listdir('/reg/d/psdm/mfx/' + choice + '/xtc'))
                if(".xtc.inprogress" in randomRun):
                    return [False, 0, 0]
                elif(".xtc" in randomRun):
                    if(self.isValidRun(filetype, choice,randomRun)):
                        num = re.findall("-r(\d+)-", randomRun)
                        return [True, choice, num[0]]
                    else:
                        return [False, 0, 0]
                else:
                    return [False, 0, 0]
            except OSError:
                return [False, 0, 0]
        else:
            return [False, 0, 0]

    def detectorValidator(self):
        """ Validates that the experiment and run number found used a CsPad detector
        returns the experiment, run number, detector.
        """
        loopCondition = True
        while loopCondition:
            boolean, name, run = self.crawl()
            if (boolean == True):
                if ("cxi" in name): 
                    if not((name in self.badList) and (run in self.badList[name])):
                        ds = DataSource('exp=%s:run=%s:idx'%(name,run))
                        detnames = DetNames()
                        for detname in detnames:
                            if ("DscCsPad" in detname) or ("DsdCsPad" in detname) or ("DsaCsPad" in detname):
                                loopCondition = False
                                return [True, name, run, detname[1]]
                        else:
                            return [False, 0, 0, 0]
                    else:
                        return[False, 0, 0, 0]
                elif("mfx" in name):
                    if not(name in self.badList):
                        ds = DataSource('exp=%s:run=%s:idx'%(name,run))
                        detnames = np.array(DetNames()).ravel()
                        if ("CsPad" in detnames):
                            loopCondition = False
                            return [True, name, run, "CsPad"]
                        else:
                            return [False, 0, 0, 0]
                    else:
                        return[False, 0, 0, 0]
                else:
                     return [False, 0, 0, 0]
            else:
                 return [False, 0, 0, 0]
            
    def returnOneRandomExpRunDet(self):
        """returns one single, random Experiment, Run Number, and Detector set.
        """
        loopCondition = True
        while loopCondition:
            boolean, name, run, det = self.detectorValidator()
            if(boolean):
                if(not self.inList(name, run, det)):
                    self.addToList(name, run, det)
                    loopCondition = False
                    return [name, run, det]

    def printSome(self,n):
        """prints n random Experiment, Run Number, and Detector sets.

        Arguments:
        n -- number of Experiment, Run Number, and Detector sets to be printed
        """
        for i in range(n):
            print(self.returnOneRandomExpRunDet())

#myCrawler = Crawler()
#print(myCrawler.badList)
#myCrawler.printSome(10)
#print(myCrawler.returnOneRandomExpRunDet())
#myCrawler.printList()

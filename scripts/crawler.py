import random
import os
import re
from psana import *
import json


class Crawler:

    def __init__(self):
        self.seenList = []
        self.badList = []
        self.readFile('listOfUnreadableFiles.txt')

    def addToList(self, name, run, det):
        self.seenList.append([name, run, det])

    def printList(self):
        print(self.seenList)

    def inList(self, name, run, det):
        if([name, run, det] in self.seenList):
            print("Repeat!")
            return True
        else:
            return False

    def readFile(self, filename):
        #Read a dictionary
        newlist = []
        with open(filename, 'r') as f:#'listOfUnreadableFiles.txt'
            newlist = json.load(f)
        self.badList = newlist
		
    def isValidRun(self, name, files):
        extension = '/reg/d/psdm/cxi/' + name + '/xtc/index/' + files + '.idx'
        boolean = os.path.isfile(extension)
        boolean2 = (os.stat(extension).st_size != 0)
        return (boolean and boolean2)

    def crawl(self):
        choice = random.choice(os.listdir("/reg/d/psdm/cxi"))
        if ("cxi" in choice):
            try:
                randomRun = random.choice(os.listdir('/reg/d/psdm/cxi/' + choice + '/xtc'))
                if(".xtc.inprogress" in randomRun):
                    return [False, 0, 0]
                elif(".xtc" in randomRun):
                    if(self.isValidRun(choice,randomRun)):
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
        loopCondition = True
        while loopCondition:
            boolean, name, run = self.crawl()
            if (boolean == True):
                if not((name in self.badList) and (run in self.badList[name])):
                    ds = DataSource('exp=%s:run=%s:idx'%(name,run))
                    detnames = DetNames()
                    detector = random.choice(DetNames())
                    if ("DscCsPad" in detector) or ("DsdCsPad" in detector) or ("DsaCsPad" in detector):
                        loopCondition = False
                        return [True, name, run, detector[1]]
                    else:
                        return [False, 0, 0, 0]
                else:
                    return[False, 0, 0, 0]
            else:
                 return [False, 0, 0, 0]
            
    def returnOneRandomExpRunDet(self):
        loopCondition = True
        while loopCondition:
            boolean, name, run, det = self.detectorValidator()
            if(boolean):
                if(not self.inList(name, run, det)):
                    self.addToList(name, run, det)
                    loopCondition = False
                    return [name, run, det]

    def printSome(self,n):
        for i in range(n):
            print(self.returnOneRandomExpRunDet())


    def main(self):
        for i in range(1000):
            boolean, name, run, det = self.detectorValidator()
            if(boolean):
                if(not self.inList(name, run, det)):
                    self.addToList(name, run, det)
                    print(i, name, run, det)


#myCrawler = Crawler()
#print(myCrawler.badList)
#myCrawler.printSome(10)
#print(myCrawler.returnOneRandomExpRunDet())
#myCrawler.printList()

import subprocess 
import os
import os.path
import re
import numpy as np
from psana import *
import time
import zmq
import base64
from pathlib import Path
import json


#################################################################
#Note: While this script is no longer used in the data pipeline,#
# it is still useful in that it can quickly parse through each  #
# experiment and run for mfx files, which is helpful for finding#
# corrupt files to blacklist.                                   #
#################################################################

def experimentListMaker():
    #First check the directory where each experiment is stored, and save that to a variable
    listOfExperiments = os.listdir('/reg/d/psdm/mfx')
    experimentNames = []
    #For each experiment there is a directory, save that directory's name to a list
    for experiment in listOfExperiments:
        if "mfx" in experiment:
            experimentNames.append(experiment)
    return experimentNames


def runListMaker(experimentNames):
    file_list = []
    count = 0
    #For each experiment's directory, there is a list of xtc files
    for name in experimentNames:
        #Some experiments are outdated, so the xtc files no longer exist. To avoid an error while looking
        #for xtc files that no longer exist, just ignore the directories that dont have xtc files.
        try:
            listOfXtcFiles = os.listdir('/reg/d/psdm/mfx/' + name + '/xtc')
        except OSError:
            continue
	#For every xtc file, save the run number, and pair that with its experiment in "file_list"
        for files in listOfXtcFiles:
            if ".xtc.inprogress" in files: #Some experiments have runs that are being copied. Ignore these.
                continue
            elif ".xtc" in files: #Check to make sure each run has a corresponding .idx file
                extension = '/reg/d/psdm/mfx/' + name + '/xtc/index/' + files + '.idx'
                boolean = os.path.isfile(extension)
                if (boolean and (os.stat(extension).st_size != 0)):
                    num = re.findall("-r(\d+)-", files)
                    file_list.append([name, num[0]])
                else:
                    count += 1
    #Now remove the redundant [experiment, run number] pairs
    reduce_file_list = set(map(tuple,file_list))
    reduce_file_list = map(list,reduce_file_list)
    print("%d missing pairs" %count)
    return reduce_file_list

def saveDictionary(dictionary):
    #Save this dictionary to a file
    with open('listOfUnreadableFiles.txt', 'w') as ofile:
        json.dump(dictionary, ofile)

def readFile(filename):
    #Read a dictionary
    newlist = []
    with open(filename, 'r') as f:#'listOfUnreadableFiles.txt'
        newlist = json.load(f)
    return newlist


def expRunDetListforCsPad(runList,newList):
    #For each pair of [experiment, run number] there are detectors. Here the list is limited to only 
    #contain experiments and run numbers that correspond to the DscCsPad, DsdCsPad, or DsaCsPad detectors
    csPadList = []
    count = 0
    n = len(runList) #n is the end value of the list
    m = 0 #m is the first value of the list
    totalNumberOfEvents = 0 #the sum of all the events in every experiment,run pair in the list
    for pairs in runList[m:n]: #[:n] only creates a list up to n 
        #If the pair is in the dictionary, move to the next pair
        if not (pairs[0] in newList):
            print(runList.index(pairs), pairs) #Prints the pair that is added to the list
            ds = DataSource('exp=%s:run=%s:idx'%(pairs[0],pairs[1])) #Fetches data from pair
            detnames = DetNames() #Fetches detector names	
            run = ds.runs().next() 
            for detname in DetNames(): 
                if ("CsPad" in detname):
                    #Add experiment, run, detector to list
                    csPadList.append([pairs[0],pairs[1],detname[1]]) 
                    times = run.times()
                    evn = ds.env()
                    numEvts = len(times) #number of events in a pair
                    totalNumberOfEvents += numEvts
        #count the number of pairs that are unreadable
        else:
            count += 1
    #Now sort the list alphanumerically according to the Experiment name, with ascending run number
    sortedCsPadList = sorted(csPadList, key = lambda x: x [1]) #First sort by run Number
    sortedCsPadList = sorted(sortedCsPadList, key = lambda x: x [0]) #Then sort by Name
    print("%d Events" %totalNumberOfEvents)
    print("%d omitted pairs" %count)
    return sortedCsPadList

#Save this dictionary to a file to be read by master.py
def saveList(thelist):
    with open('csPadList.txt', 'w') as ofile:
	json.dump(thelist, ofile)

saveList(expRunDetListforCsPad(runListMaker(experimentListMaker()), readFile('listOfUnreadableFiles.txt')))


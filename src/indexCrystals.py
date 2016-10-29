#!/usr/bin/python
import h5py, os, time, json
import argparse
import subprocess
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('expRun', nargs='?', default=None, help="Psana-style experiment/run string in the format (e.g. exp=cxi06216:run=22). This option trumps -e and -r options.")
parser.add_argument("-e","--exp", help="Experiment name (e.g. cxis0813 ). This option is ignored if expRun option is used.", default="", type=str)
parser.add_argument("-r","--run", help="Run number. This option is ignored if expRun option is used.",default=0, type=int)
parser.add_argument("-d","--det", help="Detector alias or DAQ name (e.g. DscCsPad or CxiDs1.0:Cspad.0), default=''",default="", type=str)
parser.add_argument("--geom", help="",default="", type=str)
parser.add_argument("--peakMethod", help="",default="", type=str)
parser.add_argument("--integrationRadius", help="",default="", type=str)
parser.add_argument("--pdb", help="",default="", type=str)
parser.add_argument("--indexingMethod", help="",default="", type=str)
parser.add_argument("--tolerance", help="",default="5,5,5,1.5", type=str)
parser.add_argument("--extra", help="",default="", type=str)
parser.add_argument("--minPeaks", help="",default=0, type=int)
parser.add_argument("--maxPeaks", help="",default=0, type=int)
parser.add_argument("--minRes", help="",default=0, type=int)
parser.add_argument("-o","--outDir", help="Use this directory for output instead.", default=None, type=str)
parser.add_argument("--sample", help="", default=None, type=str)
parser.add_argument("--tag", help="", default=None, type=str)
parser.add_argument("--queue", help="", default=None, type=str)
parser.add_argument("--chunkSize", help="", default=500, type=int)
parser.add_argument("--noe", help="", default=-1, type=int)
parser.add_argument("--instrument", help="", default=None, type=str)
parser.add_argument("--pixelSize", help="", default=0, type=float)
parser.add_argument("--coffset", help="", default=0, type=float)
parser.add_argument("--clenEpics", help="", default=None, type=str)
parser.add_argument("--logger", help="", default=False, type=bool)
parser.add_argument("--hitParam_threshold", help="", default=0, type=int)
parser.add_argument("--keepData", help="", default=False, type=str)
parser.add_argument("-v", help="verbosity level, default=0",default=0, type=int)
args = parser.parse_args()

def str2bool(v): return v.lower() in ("yes", "true", "t", "1")

# Init experiment parameters
if args.expRun is not None and ':run=' in args.expRun:
    experimentName = args.expRun.split('exp=')[-1].split(':')[0]
    runNumber = int(args.expRun.split('run=')[-1])
else:
    experimentName = args.exp
    runNumber = int(args.run)
detInfo = args.det
geom = args.geom
peakMethod = args.peakMethod
integrationRadius = args.integrationRadius
pdb = args.pdb
indexingMethod = args.indexingMethod
minPeaks = args.minPeaks
maxPeaks = args.maxPeaks
minRes = args.minRes
tolerance = args.tolerance
extra = args.extra
outDir = args.outDir
sample = args.sample
tag = args.tag
queue = args.queue
chunkSize = args.chunkSize
noe = args.noe
instrument = args.instrument
pixelSize = args.pixelSize
coffset = args.coffset
clenEpics = args.clenEpics
logger = args.logger
hitParam_threshold = args.hitParam_threshold
keepData = str2bool(args.keepData)

def checkJobExit(jobID):
    cmd = "bjobs -d | grep " + str(jobID)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    if "EXIT" in out:
        "*********** NODE FAILURE ************ ", jobID
        return 1
    else:
        return 0

def getMyChunkSize(numJobs, numWorkers, chunkSize, rank):
    """Returns number of events assigned to the slave calling this function."""
    print "numJobs, numWorkers: ", numJobs, numWorkers, chunkSize
    assert(numJobs >= numWorkers)
    allJobs = np.arange(numJobs)
    startInd = (np.arange(numWorkers)) * chunkSize
    endInd = (np.arange(numWorkers) + 1) * chunkSize
    endInd[-1] = numJobs
    myJobs = allJobs[startInd[rank]:endInd[rank]]
    return myJobs

def writeStatus(fname, d):
    json.dump(d, open(fname, 'w'))

def getIndexedPeaks():
    # Merge all stream files into one
    if tag is None:
        totalStream = runDir + "/" + experimentName + "_" + str(runNumber).zfill(4) + ".stream"
    else:
        totalStream = runDir + "/" + experimentName + "_" + str(runNumber).zfill(4) + "_" + tag + ".stream"
    with open(totalStream, 'w') as outfile:
        for fname in myStreamList:
            try:
                with open(fname) as infile:
                    outfile.write(infile.read())
            except:  # file may not exist yet
                pass

    # Add indexed peaks and remove images in hdf5
    try:
        f = h5py.File(peakFile, 'r')
        totalEvents = len(f['/entry_1/result_1/nPeaksAll'])
        hitEvents = f['/LCLS/eventNumber'].value
        f.close()
        # Add indexed peaks
        fstream = open(totalStream, 'r')
        content = fstream.readlines()
        fstream.close()
        indexedPeaks = -1 * np.ones((totalEvents,), dtype=int)
        numProcessed = 0
        for i, val in enumerate(content):
            if "Event: //" in val:
                _evt = int(val.split("Event: //")[-1].strip())
            if "indexed_by =" in val:
                _ind = val.split("indexed_by =")[-1].strip()
            if "fs/px" in val:
                startPeak = i
            if "End of peak list" in val:
                numPeaks = i - startPeak - 1
                numProcessed += 1
                if 'none' in _ind:
                    indexedPeaks[hitEvents[_evt]] = 0
                else:
                    indexedPeaks[hitEvents[_evt]] = numPeaks
        try:
            f = h5py.File(peakFile, 'r+')
            if '/entry_1/result_1/index' in f: del f['/entry_1/result_1/index']
            f['/entry_1/result_1/index'] = indexedPeaks
            f.close()
        except:
            pass
    except:
        indexedPeaks = None
        numProcessed = None
    return indexedPeaks, numProcessed

##############################################################################
runDir = outDir + "/r" + str(runNumber).zfill(4)
peakFile = runDir + '/' + experimentName + '_' + str(runNumber).zfill(4) + '.cxi'
indexingFile = runDir + '/.' + experimentName + '_' + str(runNumber).zfill(4) + '.txt'
fnameIndex = runDir+"/status_index.txt"

try:
    f = h5py.File(peakFile, 'r')
    hasData = '/entry_1/instrument_1/detector_1/data' in f and f['/status/findPeaks'].value == 'success'
    minPeaksUsed = f["entry_1/result_1/nPeaks"].attrs['minPeaks']
    maxPeaksUsed = f["entry_1/result_1/nPeaks"].attrs['maxPeaks']
    minResUsed = f["entry_1/result_1/nPeaks"].attrs['minRes']
    f.close()
except:
    print "Error while reading: ", peakFile
    print "Note that peak finding has to finish before launching indexing jobs"
    exit()

if hasData:
    # Update elog
    if logger == True:
        if args.v >= 1: print "Start indexing"
        try:
            d = {"message": "#StartIndexing"}
            writeStatus(fnameIndex, d)
        except:
            pass
    # Launch indexing
    try:
        f = h5py.File(peakFile, 'r')
        eventList = f['/LCLS/eventNumber'].value
        numEvents = len(eventList)
        f.close()
    except:
        print "Couldn't read file: ", peakFile
        exit()

    # Split into chunks for faster indexing
    numWorkers = int(np.ceil(numEvents*1./chunkSize))

    myLogList = []
    myJobList = []
    myStreamList = []
    for rank in range(numWorkers):
        myJobs = getMyChunkSize(numEvents, numWorkers, chunkSize, rank)
        if tag is None:
            jobName = experimentName + "_" + str(runNumber) + "_" + str(rank)
        else:
            jobName = experimentName + "_" + str(runNumber) + "_" + str(rank) + "_" + tag
        myList = runDir + "/temp_" + jobName + ".lst"
        myStream = runDir + "/temp_" + jobName + ".stream"
        myStreamList.append(myStream)

        # Write list
        with open(myList, "w") as text_file:
            for i, val in enumerate(myJobs):
                text_file.write("{} //{}\n".format(peakFile, val))

        cmd = "bsub -q " + queue + " -o " + runDir + "/.%J.log -J " + jobName + " -n 1 -x"
        cmd += " indexamajig -i " + myList + \
               " -j '`nproc`'" + \
               " -g " + geom + " --peaks=" + peakMethod + " --int-radius=" + integrationRadius + \
               " --indexing=" + indexingMethod + " -o " + myStream + \
               " --temp-dir=/tmp" + \
               " --tolerance=" + tolerance + \
               " --no-revalidate --profile"
        if pdb: cmd += " --pdb=" + pdb
        if extra: cmd += " " + extra
        print "Submitting batch job: ", cmd

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        jobID = out.split("<")[1].split(">")[0]
        myLog = runDir + "/." + jobID + ".log"
        myJobList.append(jobID)
        myLogList.append(myLog)
        print "bsub log filename: ", myLog

    myKeyString = "The output (if any) is above this job summary."
    mySuccessString = "Successfully completed."
    Done = 0
    haveFinished = np.zeros((numWorkers,))
    try:
        f = h5py.File(peakFile, 'r')
        hitEvents = f['/entry_1/result_1/nPeaksAll'].value
        numHits = len(np.where(hitEvents >= hitParam_threshold)[0])
        f.close()
    except:
        print "Couldn't read file: ", peakFile
        fname = runDir + '/status_peaks.txt'
        print "Try reading file: ", fname
        with open(fname) as infile:
            d = json.load(infile)
            numEvents = int(d['numHits'])

    while Done == 0:
        for i, myLog in enumerate(myLogList):
            if os.path.isfile(myLog):  # log file exists
                if haveFinished[i] == 0:  # job has not finished
                    p = subprocess.Popen(["grep", myKeyString, myLog], stdout=subprocess.PIPE)
                    output = p.communicate()[0]
                    p.stdout.close()
                    if myKeyString in output:  # job has completely finished
                        # check job was a success or a failure
                        p = subprocess.Popen(["grep", mySuccessString, myLog], stdout=subprocess.PIPE)
                        output = p.communicate()[0]
                        p.stdout.close()
                        if mySuccessString in output:  # success
                            print "successfully done indexing: ", runNumber, myLog
                            haveFinished[i] = 1
                            if len(np.where(abs(haveFinished) == 1)[0]) == numWorkers:
                                print "Done indexing"
                                Done = 1
                        else:  # failure
                            print "failed attempt", runNumber, myLog
                            haveFinished[i] = -1
                            if len(np.where(abs(haveFinished) == 1)[0]) == numWorkers:
                                print "Done indexing"
                                Done = -1
                    else:  # job is still going, update indexing rate
                        if args.v >= 1: print "indexing hasn't finished yet: ", runNumber, myJobList, haveFinished
                        indexedPeaks, numProcessed = getIndexedPeaks()

                        if indexedPeaks is not None:
                            numIndexedNow = len(np.where(indexedPeaks > 0)[0])
                            if numProcessed == 0:
                                indexRate = 0
                            else:
                                indexRate = numIndexedNow * 100. / numProcessed
                            fracDone = numProcessed * 100. / numHits

                            if args.v >= 1: print "Progress [runNumber, numIndexed, indexRate, fracDone]: ", runNumber, numIndexedNow, indexRate, fracDone
                            try:
                                d = {"numIndexed": numIndexedNow, "indexRate": indexRate, "fracDone": fracDone}
                                writeStatus(fnameIndex, d)
                            except:
                                print "Couldn't update status"
                                pass
                        else:
                            print "getIndexedPeaks returned None"
                        time.sleep(10)
            else:
                if args.v >= 1: print "no such file yet: ", runNumber, myLog
                nodeFailed = checkJobExit(myJobList[i])
                if nodeFailed == 1:
                    if args.v >= 0: print "indexing job node failure: ", myLog
                    haveFinished[i] = -1
                time.sleep(10)

    if abs(Done) == 1:
        indexedPeaks, numProcessed = getIndexedPeaks()
        if indexedPeaks is not None:
            numIndexedNow = len(np.where(indexedPeaks > 0)[0])
            if numProcessed == 0:
                indexRate = 0
            else:
                indexRate = numIndexedNow * 100. / numProcessed
            fracDone = numProcessed * 100. / numHits
            if args.v >= 1: print "Progress [runNumber, numIndexed, indexRate, fracDone]: ", runNumber, numIndexedNow, indexRate, fracDone

            try:
                d = {"numIndexed": numIndexedNow, "indexRate": indexRate, "fracDone": fracDone}
                writeStatus(fnameIndex, d)
            except:
                pass

        if args.v >= 1: print "Merging stream file: ", runNumber
        # Merge all stream files into one
        if tag is None:
            totalStream = runDir + "/" + experimentName + "_" + str(runNumber).zfill(4) + ".stream"
        else:
            totalStream = runDir + "/" + experimentName + "_" + str(runNumber).zfill(4) + "_" + tag + ".stream"
        with open(totalStream, 'w') as outfile:
            for fname in myStreamList:
                with open(fname) as infile:
                    outfile.write(infile.read())

        indexedPeaks, _ = getIndexedPeaks()
        numIndexed = len(np.where(indexedPeaks > 0)[0])

        # Write number of indexed
        try:
            f = h5py.File(peakFile, 'r+')
            if '/entry_1/result_1/index' in f: del f['/entry_1/result_1/index']
            indexedPeaks[np.where(indexedPeaks==-1)] = -2
            f['/entry_1/result_1/index'] = indexedPeaks
            # Remove large data
            if keepData == False:
                if '/entry_1/instrument_1/detector_1/data' in f:
                    del f['/entry_1/instrument_1/detector_1/data']
            f.close()
        except:
            if args.v >= 0: print "Couldn't modify hdf5 file: ", peakFile
            pass

        # Clean up temp files
        for fname in myStreamList:
            os.remove(fname)

print "Done indexCrystals: ", runNumber




#!/usr/bin/python
import h5py, os, time, json, sys
import argparse
import subprocess
import numpy as np
from psocake.utils import batchSubmit

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
parser.add_argument("--pkTag", help="", default='', type=str)
parser.add_argument("--tag", help="", default='', type=str)
parser.add_argument("--queue", help="", default=None, type=str)
parser.add_argument("--chunkSize", help="", default=500, type=int)
parser.add_argument("--cpu", help="", default=12, type=int)
parser.add_argument("--noe", help="", default=-1, type=int)
parser.add_argument("--instrument", help="", default=None, type=str)
parser.add_argument("--pixelSize", help="", default=0, type=float)
parser.add_argument("--coffset", help="", default=0, type=float)
parser.add_argument("--clenEpics", help="", default=None, type=str)
parser.add_argument("--logger", help="", default=False, type=bool)
parser.add_argument("--hitParam_threshold", help="", default=0, type=int)
parser.add_argument("--keepData", help="", default=False, type=str)
parser.add_argument("-v", help="verbosity level, default=0",default=0, type=int)
parser.add_argument("--likelihood", help="index hits with likelihood higher than this value", default=0, type=float)
parser.add_argument("--condition", help="logic condition", default='', type=str)
parser.add_argument("--batch", help="batch type: lsf or slurm",default="slurm", type=str)
args = parser.parse_args()

tic = time.time()

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
extra = args.extra.split("[")[-1].split("]")[0].replace(","," ")
outDir = args.outDir
sample = args.sample
tag = args.tag
queue = args.queue
chunkSize = args.chunkSize
cpu = args.cpu
noe = args.noe
instrument = args.instrument
pixelSize = args.pixelSize
coffset = args.coffset
clenEpics = args.clenEpics
logger = args.logger
hitParam_threshold = args.hitParam_threshold
keepData = str2bool(args.keepData)
batch = args.batch

runDir = outDir + "/r" + str(runNumber).zfill(4)
peakFile = runDir + '/' + experimentName + '_' + str(runNumber).zfill(4)
if args.pkTag: peakFile += '_'+args.pkTag
peakFile += '.cxi'

def checkJobExit(jobID):
    if args.batch == 'lsf':
        cmd = "bjobs -d | grep " + str(jobID)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        if "EXIT" in str(out):
            "*********** NODE FAILURE ************ ", jobID
            return 1
        else:
            return 0
    elif args.batch == 'slurm':
        cmd = "sacct --jobs="+str(jobID)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        out = out.decode('utf-8')
        for line in out.split('\n'):
            tok = line.split()
            if len(tok) > 0:
                # ['JobID', 'JobName', 'Partition', 'Account', 'AllocCPUS', 'State', 'ExitCode']
                if tok[0] == str(jobID):
                    if "CANCELLED" in tok[4] or "FAILED" in tok[4]:
                        return 1
        return 0

def getMyChunkSize(numJobs, numWorkers, chunkSize, rank):
    """Returns number of events assigned to the slave calling this function."""
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
    if not tag:
        totalStream = runDir + "/" + experimentName + "_" + str(runNumber).zfill(4) + ".stream"
    else:
        totalStream = runDir + "/" + experimentName + "_" + str(runNumber).zfill(4) + "_" + tag + ".stream"
    with open(totalStream, 'w') as outfile:
        for fname in myStreamList:
            #print("Reading: ", fname)
            try:
                with open(fname) as infile:
                    outfile.write(infile.read())
            except:  # file may not exist yet
                print("Couldn't open: ", fname)
                pass

    # Add indexed peaks and remove images in hdf5
    indexedPeaks = 0
    numProcessed = 0
    try:
        fstream = open(totalStream, 'r')
        content = fstream.readlines()
        fstream.close()

        for i, val in enumerate(content):
            if "indexed_by =" in val:
                _ind = val.split("indexed_by =")[-1].strip()
                if 'none' not in _ind:
                    indexedPeaks += 1
            if "End of peak list" in val:
                numProcessed += 1
    except:
        pass
    return indexedPeaks, numProcessed

def getpath(cxi_name):
    global search_name, hf
    if cxi_name.endswith(search_name):
        try: hf[cxi_name][()]
        except Exception: return None
        return cxi_name

def condition_check(hf, ival, iicondition):
    if not iicondition: return True
    return eval(iicondition)

def getPeakFileIndex(experimentName, runNumber, pkTag, eventSizes, currentInd):
    b = np.cumsum(eventSizes)
    peakfileInd = np.where((b - currentInd) > 0)[0][0]
    if peakfileInd > 0:
        pointer = currentInd - b[peakfileInd-1]
    else:
        pointer = currentInd
    pFile = runDir + '/' + experimentName + '_' + str(runNumber).zfill(4) + '_' + str(peakfileInd)
    if pkTag: pFile += '_'+pkTag
    pFile += '.cxi'

    return pFile, pointer

##############################################################################
# Get name of index file
fnameIndex = runDir+"/status_index"
if args.tag: fnameIndex += '_'+args.tag
fnameIndex += ".txt"

def findSize(runDir,experimentName,runNumber,pkTag):
    numSize = -1
    if pkTag:
        searchWord = pkTag+'.cxi'
    else:
        searchWord = '.cxi'

    for root, dirs, files in os.walk(runDir):
        for file in files:
            if str(file).startswith(experimentName) and str(file).endswith(searchWord):
                n = -1
                tok = str(file).split('_')
                if pkTag:
                    if experimentName in tok[0] and \
                       str(runNumber).zfill(4) in tok[1] and \
                       searchWord in tok[-1] and \
                       len(tok) == 4:
                        try:
                            n = int(tok[2])
                        except:
                            print("ignore: ", str(file))
                else:
                    if experimentName in tok[0] and \
                       str(runNumber).zfill(4) in tok[1] and \
                       searchWord in tok[-1] and \
                       len(tok) == 3:
                        try:
                            n = int(tok[2].split(searchWord)[0])
                        except:
                            print("ignore: ", str(file))
                if numSize == -1 and n > -1: 
                    numSize = n
                elif n > numSize:
                    numSize = n
    if numSize is not None:
        numSize += 1
    return numSize

numSize = findSize(runDir,experimentName,runNumber,args.pkTag)
print("Found {} cxi files".format(numSize))
if numSize is None: 
    print("Error: Could not find cxi files in: ", runDir)
    sys.exit()
numEventsArr = np.zeros((numSize,),dtype=int)
totalHits = 0

for ind in range(numSize):
    pFile = runDir + '/' + experimentName + '_' + str(runNumber).zfill(4) + '_' + str(ind)
    if args.pkTag: pFile += '_'+args.pkTag
    pFile += '.cxi'
    hf = h5py.File(pFile, 'r')
    icondition = args.condition
    iposition = [ipos for ipos, ichar in enumerate(icondition) if ichar == '#']
    istart = iposition[0::2]
    iend = iposition[1::2]
    assert (len(istart) == len(iend))
    iname = [icondition[(istart[i]+1):iend[i]] for i in range(len(istart))]
    ifullpath = []
    for idx in range(len(istart)): 
        search_name = iname[idx]
        assert(hf.visit(getpath))
        ifullpath.append("hf['"+hf.visit(getpath)+"'][ival]")
        icondition = icondition.replace('#'+iname[idx]+'#', ifullpath[-1])
    print('modified condition = ', icondition, '\n')
    try:
        f = h5py.File(pFile, 'r')
        if '/LCLS/eventNumber' in f:
            eventList = f['/LCLS/eventNumber'][()]
            numEvents = len(eventList)
        else:
            numEvents = 0
        totalHits += numEvents
        hasData = '/entry_1/instrument_1/detector_1/data' in f and 'success' in str(f['/status/findPeaks'][()])
        minPeaksUsed = f["entry_1/result_1/nPeaks"].attrs['minPeaks']
        maxPeaksUsed = f["entry_1/result_1/nPeaks"].attrs['maxPeaks']
        minResUsed = f["entry_1/result_1/nPeaks"].attrs['minRes']
        f.close()
    except:
        print("Error while reading: ", pFile)
        print("Note that peak finding has to finish before launching indexing jobs")
        sys.exit()

    if hasData:
        # Update elog
        if logger == True:
            if args.v >= 1: print("Start indexing")
            try:
                d = {"message": "#StartIndexing"}
                writeStatus(fnameIndex, d)
            except:
                pass
        # Launch indexing
        try:
            print("Reading images from: ", pFile)
            f = h5py.File(pFile, 'r')
            eventList = f['/LCLS/eventNumber'][()]
            if args.likelihood > 0:
                likelihood = f['/entry_1/result_1/likelihood'][()]
            numEvents = len(eventList)
            f.close()
        except:
            print("Couldn't read file: ", pFile)

        numEventsArr[ind] = numEvents
    else:
        print("No data found. Exiting.")
        sys.exit()

if totalHits == 0: 
    print("No events to process. Exiting.")
    sys.exit()

totalNumEvents = np.sum(numEventsArr)
# Split into chunks for faster indexing
numWorkers = int(np.ceil(totalNumEvents*1./chunkSize))
myLogList = []
myJobList = []
myStreamList = []
myLists = []
for rank in range(numWorkers):
    myJobs = getMyChunkSize(totalNumEvents, numWorkers, chunkSize, rank)
    if tag is '':
        jobName = str(runNumber) + "_" + str(rank)
    else:
        jobName = str(runNumber) + "_" + str(rank) + "_" + tag
    myList = runDir + "/temp_" + jobName + ".lst"
    myStream = runDir + "/temp_" + jobName + ".stream"
    myStreamList.append(myStream)
    myLists.append(myList)

    # Write list
    isat_event = []
    checkEnoughLikes = 0

    with open(myList, "w") as text_file:
        for i, val in enumerate(myJobs):
            pFile, val = getPeakFileIndex(experimentName,runNumber,args.pkTag,numEventsArr,val)
            if args.likelihood > 0:
                if likelihood[val] >= args.likelihood and condition_check(hf, val, icondition):
                    text_file.write("{} //{}\n".format(pFile, val))
                    checkEnoughLikes += 1
                    isat_event.append(val)
            else:
                if condition_check(hf, val, icondition):
                    text_file.write("{} //{}\n".format(pFile, val))
                    isat_event.append(val)
    isat_event = []

    # Submit job
    cmd = " indexamajig -i " + myList
    if args.likelihood > 0 and checkEnoughLikes > 0 and checkEnoughLikes <= 16:
        cmd += " -j 1"
    else:
        if "ffb" in args.queue or args.queue == "anaq": # limit cores for ffb
            cmd += " -j 1"
        else:
            cmd += " -j 1"#'`nproc`'"
    cmd += " -g " + geom + " --peaks=" + peakMethod + " --int-radius=" + integrationRadius + \
           " --indexing=" + indexingMethod + " -o " + myStream
    if batch == "lsf":
        cmd += " --temp-dir=/scratch"
    else:
        cmd += " --temp-dir=/tmp"
    cmd += " --tolerance=" + tolerance + \
           " --no-revalidate --multi --profile"
    if pdb: cmd += " --pdb=" + pdb
    if extra: cmd += " " + extra
    #if "slurm" in args.batch:
    #    slurmParams = {"--cpus-per-task": 3}#"--ntasks-per-node": 3}
    #    cmd = batchSubmit(cmd, queue, 1, runDir + "/%J.log", jobName, batch, slurmParams)
    #else:
    #    cmd = batchSubmit(cmd, queue, 1, runDir + "/%J.log", jobName, batch)
    cmd = batchSubmit(cmd, queue, 1, runDir + "/%J_"+jobName+".log", jobName, batch)
    print("Note: Indexing will use the mask saved in the cxi file (/entry_1/data_1/mask) for Bragg integration")
    print("Submitting job: ", cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    # Keep list
    if batch == "lsf":
        jobID = out.decode("utf-8").split("<")[1].split(">")[0]
    else:
        jobID = out.decode("utf-8").split("job ")[1].split("\n")[0]
    myLog = runDir + "/" + jobID + "_" + jobName + ".log"
    myJobList.append(jobID)
    myLogList.append(myLog)
    print("log filename: ", myLog)

##############################################################

if batch == "lsf":
    myKeyString = "The output (if any) is above this job summary."
    mySuccessString = "Successfully completed."
else: # slurm
    myKeyString = "Final: "
    mySuccessString = "Final: "
    myCancelString = "CANCELLED"
Done = 0
haveFinished = np.zeros((numWorkers,))
try:
    with h5py.File(peakFile, 'r') as f:
        hitEvents = f['/entry_1/result_1/nPeaksAll'][()]
        numHits = len(np.where(hitEvents >= hitParam_threshold)[0])
except:
    print("Couldn't read file: ", peakFile)
    fname = runDir + '/status_peaks'
    if args.pkTag: fname += '_' + args.pkTag
    fname += '.txt'
    print("Try reading file: ", fname)
    with open(fname) as infile:
        d = json.load(infile)
        numEvents = int(d['numHits'])
        
while Done == 0:
    for i, myLog in enumerate(myLogList):
        if os.path.isfile(myLog):  # log file exists
            if haveFinished[i] == 0:  # i^th job is still running
                # check if job has finished
                p = subprocess.Popen(["grep", "-E", myKeyString+"|"+myCancelString, myLog], stdout=subprocess.PIPE)
                output = p.communicate()[0]
                p.stdout.close()
                if myKeyString in str(output):  # job has completely finished
                    # check job was a success or a failure
                    p = subprocess.Popen(["grep", mySuccessString, myLog], stdout=subprocess.PIPE)
                    output = p.communicate()[0]
                    p.stdout.close()
                    if mySuccessString in str(output):  # success
                        print("successfully done indexing: ", runNumber, myLog)
                        haveFinished[i] = 1
                        if len(np.where(abs(haveFinished) == 1)[0]) == numWorkers:
                            print("Done indexing")
                            Done = 1
                    else:  # failure
                        print("failed attempt", runNumber, myLog)
                        haveFinished[i] = -1
                        if len(np.where(abs(haveFinished) == 1)[0]) == numWorkers:
                            print("Done indexing")
                            Done = -1
                elif myCancelString in str(output): # job has been cancelled
                    print("cancelled job", runNumber, myLog)
                    haveFinished[i] = -1
                    if len(np.where(abs(haveFinished) == 1)[0]) == numWorkers:
                        print("Done indexing")
                        Done = -1
                else: # job is still going, update indexing rate
                    if args.v >= 1: print("indexing hasn't finished yet: ", runNumber, myJobList, haveFinished)
                    indexedPeaks = None#, numProcessed = getIndexedPeaks()

                    if indexedPeaks is not None:
                        numIndexedNow = indexedPeaks #len(np.where(indexedPeaks > 0)[0])
                        if numProcessed == 0:
                            indexRate = 0
                        else:
                            indexRate = numIndexedNow * 100. / numProcessed
                        if numHits > 0:
                            fracDone = numProcessed * 100. / numHits
                        else:
                            fracDone = 0

                        if args.v >= 1: print("Progress [runNumber, numIndexed, indexRate, fracDone]: ", runNumber, numIndexedNow, indexRate, fracDone)
                        try:
                            d = {"numIndexed": numIndexedNow, "indexRate": indexRate, "fracDone": fracDone}
                            writeStatus(fnameIndex, d)
                        except:
                            print("Couldn't update status")
                            pass
                    else:
                        pass #print "getIndexedPeaks returned None"
                    time.sleep(10)
        else: # log file does not exist
            if args.v >= 1: print("no such file yet: ", runNumber, myLog)
            jobKilled = checkJobExit(myJobList[i])
            if jobKilled == 1:
                if args.v >= 0: print("indexing job failure: ", myLog)
                haveFinished[i] = -1
                if args.v >= 0: print("Error: exit indexing crystals")
                sys.exit()
            time.sleep(10)

    if abs(Done) == 1:
        indexedPeaks, numProcessed = getIndexedPeaks()
        if indexedPeaks is not None:
            numIndexedNow = indexedPeaks #len(np.where(indexedPeaks > 0)[0])
            if numProcessed == 0:
                indexRate = 0
            else:
                indexRate = numIndexedNow * 100. / numProcessed
            fracDone = numProcessed * 100. / numHits
            if args.v >= 1: print("Progress [runNumber, numIndexed, indexRate, fracDone]: ", runNumber, numIndexedNow, indexRate, fracDone)

            try:
                d = {"numIndexed": numIndexedNow, "indexRate": indexRate, "fracDone": fracDone}
                writeStatus(fnameIndex, d)
            except:
                pass

        if args.v >= 1: print("Merging stream file: ", runNumber)
        numIndexed, numProcessed = getIndexedPeaks()
        if args.v >= 1: print("Status update: ", runNumber, numIndexed, numProcessed)
        if numProcessed == 0:
            indexRate = 0
        else:
            indexRate = numIndexed * 100. / numProcessed

        try:
            d = {"numIndexed": numIndexed, "indexRate": indexRate, "fracDone": 100.0}
            writeStatus(fnameIndex, d)
        except:
            print("Couldn't update status")
            pass

        # Clean up temp files
        if args.v >= 1: print("Cleaning up temp files: ", runNumber)
        for fname in myStreamList:
            try:
                os.remove(fname)
            except:
                print("Couldn't remove {}".format(fname))
        for fname in myLists:
            try:
                os.remove(fname)
            except:
                print("Couldn't remove {}".format(fname))
hf.close()
print("Done indexing run (time elapsed): ", runNumber, time.time()-tic)


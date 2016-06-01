import numpy as np
import psanaWhisperer as ps
from pyqtgraph.Qt import QtCore
import time
import subprocess
import os
import pandas as pd
import h5py

class CrystalIndexing(object):
    def __init__(self, parent = None):
        self.parent = parent

        self.index_grp = 'Crystal indexing'
        self.index_on_str = 'Indexing on'
        self.index_geom_str = 'CrystFEL geometry'
        self.index_peakMethod_str = 'Peak method'
        self.index_intRadius_str = 'Integration radii'
        self.index_pdb_str = 'PDB'
        self.index_method_str = 'Indexing method'

        self.launch_grp = 'Batch'
        self.outDir_str = 'Output directory'
        self.runs_str = 'Runs(s)'
        self.queue_str = 'queue'
        self.cpu_str = 'CPUs'
        self.noe_str = 'Number of events to process'
        (self.psanaq_str,self.psnehq_str,self.psfehq_str,self.psnehprioq_str,self.psfehprioq_str,self.psnehhiprioq_str,self.psfehhiprioq_str) = \
            ('psanaq','psnehq','psfehq','psnehprioq','psfehprioq','psnehhiprioq','psfehhiprioq')

        self.outDir = self.parent.psocakeDir
        self.runs = ''
        self.queue = self.psanaq_str
        self.cpus = 32
        self.noe = 0

        self.indexingOn = False
        self.geom = '.temp.geom'
        self.peakMethod = 'cxi'
        self.intRadius = '2,3,4'
        self.pdb = 'lys.cell'
        self.indexingMethod = 'dirax-axes-latt'

        #######################
        # Mandatory parameter #
        #######################
        self.params = [
            {'name': self.index_grp, 'type': 'group', 'children': [
                {'name': self.index_on_str, 'type': 'bool', 'value': self.indexingOn, 'tip': "Turn on indexing"},
                {'name': self.index_geom_str, 'type': 'str', 'value': self.geom, 'tip': "Turn on indexing"},
                #{'name': self.index_peakMethod_str, 'type': 'str', 'value': self.peakMethod, 'tip': "Turn on indexing"},
                {'name': self.index_intRadius_str, 'type': 'str', 'value': self.intRadius, 'tip': "Turn on indexing"},
                {'name': self.index_pdb_str, 'type': 'str', 'value': self.pdb, 'tip': "Turn on indexing"},
                {'name': self.index_method_str, 'type': 'str', 'value': self.indexingMethod, 'tip': "Turn on indexing"},
            ]},
            {'name': self.launch_grp, 'type': 'group', 'children': [
                {'name': self.outDir_str, 'type': 'str', 'value': self.outDir},
                {'name': self.runs_str, 'type': 'str', 'value': self.runs},
                {'name': self.queue_str, 'type': 'list', 'values': {self.psfehhiprioq_str: self.psfehhiprioq_str,
                                                               self.psnehhiprioq_str: self.psnehhiprioq_str,
                                                               self.psfehprioq_str: self.psfehprioq_str,
                                                               self.psnehprioq_str: self.psnehprioq_str,
                                                               self.psfehq_str: self.psfehq_str,
                                                               self.psnehq_str: self.psnehq_str,
                                                               self.psanaq_str: self.psanaq_str},
                 'value': self.queue, 'tip': "Choose queue"},
                {'name': self.cpu_str, 'type': 'int', 'value': self.cpus},
                {'name': self.noe_str, 'type': 'int', 'value': self.noe, 'tip': "number of events to process, default=0 means process all events"},
            ]},
        ]

    ##############################
    # Mandatory parameter update #
    ##############################
    def paramUpdate(self, path, change, data):
        if path[1] == self.index_on_str:
            self.updateIndexStatus(data)
        elif path[1] == self.index_geom_str:
            self.updateGeom(data)
        elif path[1] == self.index_peakMethod_str:
            self.updatePeakMethod(data)
        elif path[1] == self.index_intRadius_str:
            self.updateIntegrationRadius(data)
        elif path[1] == self.index_pdb_str:
            self.updatePDB(data)
        elif path[1] == self.index_method_str:
            self.updateIndexingMethod(data)
        # launch grp
        elif path[1] == self.outDir_str:
            self.updateOutputDir(data)
        elif path[1] == self.runs_str:
            self.updateRuns(data)
        elif path[1] == self.queue_str:
            self.updateQueue(data)
        elif path[1] == self.cpu_str:
            self.updateCpus(data)
        elif path[1] == self.noe_str:
            self.updateNoe(data)

    def updateIndexStatus(self, data):
        self.indexingOn = data
        print "indexing on: ", self.indexingOn
        if self.indexingOn:
            self.updateIndex()
        print "Done updateIndexStatus"

    def updateGeom(self, data):
        self.geom = data
        if self.indexingOn:
            self.updateIndex()

    def updatePeakMethod(self, data):
        self.peakMethod = data
        if self.indexingOn:
            self.updateIndex()

    def updateIntegrationRadius(self, data):
        self.intRadius = data
        if self.indexingOn:
            self.updateIndex()

    def updatePDB(self, data):
        self.pdb = data
        if self.indexingOn:
            self.updateIndex()

    def updateIndexingMethod(self, data):
        self.indexingMethod = data
        if self.indexingOn:
            self.updateIndex()

    def updateIndex(self):
        if self.indexingOn:
            self.indexer = IndexHandler(parent=self.parent)
            self.indexer.computeIndex(self.parent.experimentName, self.parent.runNumber, self.parent.detInfo,
                                      self.parent.eventNumber, self.geom, self.peakMethod, self.intRadius, self.pdb, self.indexingMethod)
        print "Done updateIndex"

    def updateOutputDir(self, data):
        self.outDir = data
        print "Done updateOutputDir ", self.outDir

    def updateRuns(self, data):
        self.runs = data
        print "Done updateRuns"

    def updateQueue(self, data):
        self.queue = data
        print "Done updateQueue"

    def updateCpus(self, data):
        self.cpus = data
        print "Done updateCpu"

    def updateNoe(self, data):
        self.noe = data
        print "Done updateNoe"

    def launchIndexing(self, requestRun=None):
        self.batchIndexer = IndexHandler(parent=self.parent)
        if requestRun is None:
            self.batchIndexer.computeIndex(self.parent.experimentName, self.parent.runNumber, self.parent.detInfo,
                                  self.parent.eventNumber, self.geom, self.peakMethod, self.intRadius, self.pdb,
                                       self.indexingMethod, self.outDir, self.runs, self.queue, self.cpus, self.noe)
        else:
            self.batchIndexer.computeIndex(self.parent.experimentName, requestRun, self.parent.detInfo,
                                  self.parent.eventNumber, self.geom, self.peakMethod, self.intRadius, self.pdb,
                                       self.indexingMethod, self.outDir, self.runs, self.queue, self.cpus, self.noe)
        print "outDir: ", self.outDir
        print "Done updateIndex"

class IndexHandler(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        print "WORKER!!!!!!!!!!"
        self.parent = parent
        self.experimentName = None
        self.runNumber = None
        self.detInfo = None
        self.eventNumber = None
        self.geom = None
        self.peakMethod = None
        self.intRadius = None
        self.pdb = None
        self.indexingMethod = None
        self.unitCell = None
        # batch
        self.outDir = None
        self.runs = None
        self.queue = None
        self.cpus = None
        self.noe = None

    def __del__(self):
        print "del IndexHandler #$!@#$!#"
        self.exiting = True
        self.wait()

    def computeIndex(self, experimentName, runNumber, detInfo, eventNumber, geom, peakMethod, intRadius, pdb, indexingMethod,
                     outDir=None, runs=None, queue=None, cpus=None, noe=None):
        self.experimentName = experimentName
        self.runNumber = runNumber
        self.detInfo = detInfo
        self.eventNumber = eventNumber
        self.geom = geom
        self.peakMethod = peakMethod
        self.intRadius = intRadius
        self.pdb = pdb
        self.indexingMethod = indexingMethod
        # batch
        self.outDir = outDir
        self.runs = runs
        self.queue = queue
        self.cpus = cpus
        self.noe = noe
        if self.geom is not '':
            self.start()

    def getMyUnfairShare(self, numJobs, numWorkers, rank):
        """Returns number of events assigned to the slave calling this function."""
        assert(numJobs >= numWorkers)
        allJobs = np.arange(numJobs)
        jobChunks = np.array_split(allJobs,numWorkers)
        myChunk = jobChunks[rank]
        myJobs = allJobs[myChunk[0]:myChunk[-1]+1]
        return myJobs

    def run(self):
        if self.outDir is None: # interactive indexing
            print "Running indexing!!!!!!!!!!!!"
            # Running indexing ...
            self.parent.numIndexedPeaksFound = 0
            self.parent.indexedPeaks = None
            self.parent.clearIndexedPeaks()

            # Write list
            with open(self.parent.hiddenCrystfelList, "w") as text_file:
                text_file.write("{} //0".format(self.parent.hiddenCXI))

            # FIXME: convert psana geom to crystfel geom
            cmd = "indexamajig -j 1 -i "+self.parent.hiddenCrystfelList+" -g "+self.geom+" --peaks="+self.peakMethod+\
                  " --int-radius="+self.intRadius+" --indexing="+self.indexingMethod+" -o "+self.parent.hiddenCrystfelStream
            if self.pdb is not '':
                cmd += " --pdb="+self.pdb

            print "cmd: ", cmd
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            #print "out: ", out
            #print "err: ", err

            mySuccessString = "1 had crystals"
            # Read CrystFEL CSPAD geometry in stream
            if mySuccessString in err: # success
                print "Indexing successful"
                #print "Munging geometry file"
                f = open(self.parent.hiddenCrystfelStream)
                content = f.readlines()
                for i, val in enumerate(content):
                    if '----- Begin geometry file -----' in val:
                        startLine = i
                    elif   '----- End geometry file -----' in val:
                        endLine = i
                        break
                geom = content[startLine:endLine]
                numLines = endLine-startLine
                # Remove comments
                for i in np.arange(numLines-1,-1,-1): # Start from bottom
                    if ';' in geom[i].lstrip(' ')[0]: geom.pop(i)

                #print "### Geometry file: "
                #print geom

                numQuads = 4
                numAsics = 16
                columns=['min_fs','min_ss','max_fs','max_ss','res','fs','ss','corner_x','corner_y']
                columnsScan=['fsx','fsy','ssx','ssy']
                indexScan=[]
                for i in np.arange(numQuads):
                    for j in np.arange(numAsics):
                        indexScan.append('q'+str(i)+'a'+str(j))

                dfGeom = pd.DataFrame(np.empty((numQuads*numAsics,len(columns))), index=indexScan, columns=columns)
                dfScan = pd.DataFrame(np.empty((numQuads*numAsics,len(columnsScan))), index=indexScan, columns=columnsScan)
                counter = 0
                for i in np.arange(numQuads):
                    for j in np.arange(numAsics):
                        myAsic = indexScan[counter]
                        for k in columns:
                            myLine = [s for s in geom if myAsic+'/'+k in s]
                            myVal = myLine[-1].split('=')[-1].rstrip().lstrip()
                            if k == 'fs' or k == 'ss':
                                dfGeom.loc[myAsic,k] = myVal
                            else:
                                dfGeom.loc[myAsic,k] = float(myVal)
                            if k == 'fs':
                                fsx = float(myVal.split('x')[0])
                                fsy = float(myVal.split('x')[-1].split('y')[0])
                                dfScan.loc[myAsic,'fsx'] = fsx
                                dfScan.loc[myAsic,'fsy'] = fsy
                            elif k == 'ss':
                                ssx = float(myVal.split('x')[0])
                                ssy = float(myVal.split('x')[-1].split('y')[0])
                                dfScan.loc[myAsic,'ssx'] = ssx
                                dfScan.loc[myAsic,'ssy'] = ssy
                        counter += 1
                #print "#### GEOM: "
                #print dfGeom
                #print "#### SCAN: "
                #print dfScan
                f.close()
            else:
                print "Indexing failed"
                self.parent.drawIndexedPeaks()

            # Read CrystFEL indexed peaks
            if mySuccessString in err: # success
                f = open(self.parent.hiddenCrystfelStream)
                content = f.readlines()
                for i, val in enumerate(content):
                    if   'num_peaks =' in val:
                        numPeaks = int(val.split('=')[-1])
                    elif   'fs/px   ss/px (1/d)/nm^-1   Intensity  Panel' in val:
                        startLine = i+1
                        endLine = startLine+numPeaks
                    elif 'Cell parameters' in val:
                        (_,_,a,b,c,_,al,be,ga,_) = val.split()
                        self.unitCell = (a,b,c,al,be,ga)

                #print "### Peaks: "
                #print content[startLine:endLine]

                columns=['fs','ss','res','intensity','asic']
                df = pd.DataFrame(np.empty((numPeaks,len(columns))), columns=columns)
                for i in np.arange(numPeaks):
                    contentLine = startLine+i
                    df['fs'][i] = float(content[contentLine][0:7])
                    df['ss'][i] = float(content[contentLine][7:15])
                    df['res'][i] = float(content[contentLine][15:26])
                    df['intensity'][i] = float(content[contentLine][26:38])
                    df['asic'][i] = str(content[contentLine][38:-1])
                #print "### Stream"
                #print df
                f.close()

                # Convert to CrystFEL coordinates
                columnsPeaks=['x','y','psocakeX','psocakeY']
                dfPeaks = pd.DataFrame(np.empty((numPeaks,len(columnsPeaks))), columns=columnsPeaks)
                for i in np.arange(numPeaks):
                    myAsic = df['asic'][i].strip()
                    x = (df['fs'][i] - dfGeom.loc[myAsic,'min_fs'])*dfScan.loc[myAsic,'fsx'] + (df['ss'][i] - dfGeom.loc[myAsic,'min_ss'])*dfScan.loc[myAsic,'ssx']
                    x += dfGeom.loc[myAsic,'corner_x']
                    y = (df['fs'][i] - dfGeom.loc[myAsic,'min_fs'])*dfScan.loc[myAsic,'fsy'] + (df['ss'][i] - dfGeom.loc[myAsic,'min_ss'])*dfScan.loc[myAsic,'ssy']
                    y += dfGeom.loc[myAsic,'corner_y']
                    dfPeaks['x'][i]=x
                    dfPeaks['y'][i]=y

                # Convert to psocake coordinates
                for i in np.arange(numPeaks):
                    dfPeaks['psocakeX'][i] = self.parent.cx - dfPeaks['x'][i]
                    dfPeaks['psocakeY'][i] = self.parent.cy + dfPeaks['y'][i]

                if self.parent.showIndexedPeaks and self.eventNumber == self.parent.eventNumber:
                    self.parent.numIndexedPeaksFound = numPeaks
                    self.parent.indexedPeaks = dfPeaks[['psocakeX','psocakeY']].as_matrix()
                    self.parent.drawIndexedPeaks(self.unitCell)
        else: # batch indexing
            # Open hdf5
            peakFile = self.outDir+'/'+self.experimentName+'_'+str(self.runNumber).zfill(4)+'.cxi'
            f = h5py.File(peakFile,'r')
            #eventList = f['/LCLS/eventNumber'].value
            hasData = '/entry_1/instrument_1/detector_1/data' in f
            #numEvents = len(eventList)
            f.close()
            print "peakFile: ", peakFile
            #print "eventList: ", eventList

            if hasData is False:
                # Run xtc2cxidbMPI
                print "$$$ self.parent.det.instrument(): ", self.parent.det.instrument()
                cmd = "bsub -q "+self.queue+" -a mympi -n "+str(self.cpus)+" -o .%J.log python xtc2cxidbMPI.py" \
                      " -e "+self.experimentName+" -d "+self.detInfo+" -i "+self.outDir+" --sample lysozyme" \
                      " --instrument "+self.parent.det.instrument()+" --pixelSize "+str(self.parent.pixelSize)+ \
                      " --coffset "+str(self.parent.coffset)+" --clen "+self.parent.clenEpics+" --run "+str(self.runNumber)+\
                      " --condition /entry_1/result_1/nPeaksAll,ge,15"
                print "Submitting batch job: ", cmd
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                out, err = process.communicate()
                jobid = out.split("<")[1].split(">")[0]
                myLog = "."+jobid+".log"
                print "*******************"
                print "bsub log filename: ", myLog
                myKeyString = "The output (if any) is above this job summary."
                mySuccessString = "Successfully completed."
                notDone = 1
                while notDone:
                    if os.path.isfile(myLog):
                        p = subprocess.Popen(["grep", myKeyString, myLog],stdout=subprocess.PIPE)
                        output = p.communicate()[0]
                        p.stdout.close()
                        if myKeyString in output: # job has finished
                            # check job was a success or a failure
                            p = subprocess.Popen(["grep", mySuccessString, myLog], stdout=subprocess.PIPE)
                            output = p.communicate()[0]
                            p.stdout.close()
                            if mySuccessString in output: # success
                                print "successfully done: ", self.runNumber
                                hasData = True
                            else:
                                print "failed attempt"
                            notDone = 0
                        else:
                            print "cxidb job hasn't finished yet: ", myLog
                            time.sleep(10)
                    else:
                        print "no such file yet"
                        time.sleep(10)

            if hasData:
                f = h5py.File(peakFile,'r')
                eventList = f['/LCLS/eventNumber'].value
                numEvents = len(eventList)
                f.close()
                # Split into chunks for faster indexing
                numWorkers = 4
                myLogList = []
                myStreamList = []
                for rank in range(numWorkers):
                    myJobs = self.getMyUnfairShare(numEvents,numWorkers,rank)

                    myList = "temp_"+self.experimentName+"_"+str(self.runNumber)+"_"+str(rank)+".lst"
                    myStream = "temp_"+self.experimentName+"_"+str(self.runNumber)+"_"+str(rank)+".stream"
                    myStreamList.append(myStream)

                    # Write list
                    with open(myList, "w") as text_file:
                        for i,val in enumerate(myJobs):
                            text_file.write("{} //{}\n".format(peakFile,val))

                    cmd = "bsub -q "+self.queue+" -a mympi -n 1 -o .%J.log indexamajig -j "+str(self.cpus)+" -i "+myList+\
                          " -g "+self.geom+" --peaks="+self.peakMethod+" --int-radius="+self.intRadius+\
                          " --indexing="+self.indexingMethod+" -o "+myStream
                    if self.pdb is not '':
                        cmd += " --pdb="+self.pdb
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                    out, err = process.communicate()
                    jobid = out.split("<")[1].split(">")[0]
                    myLog = "."+jobid+".log"
                    myLogList.append(myLog)
                    print "*******************"
                    print "bsub log filename: ", myLog

                myKeyString = "The output (if any) is above this job summary."
                mySuccessString = "Successfully completed."
                Done = 0
                haveFinished = np.zeros((numWorkers,))
                while Done == 0:
                    for i,myLog in enumerate(myLogList):
                        if os.path.isfile(myLog):
                            if haveFinished[i] == 0:
                                p = subprocess.Popen(["grep", myKeyString, myLog],stdout=subprocess.PIPE)
                                output = p.communicate()[0]
                                p.stdout.close()
                                if myKeyString in output: # job has finished
                                    # check job was a success or a failure
                                    p = subprocess.Popen(["grep", mySuccessString, myLog], stdout=subprocess.PIPE)
                                    output = p.communicate()[0]
                                    p.stdout.close()
                                    if mySuccessString in output: # success
                                        print "successfully done: ", myLog
                                        haveFinished[i] = 1
                                        if len(np.where(haveFinished==1)[0]) == numWorkers:
                                            Done = 1
                                    else:
                                        print "failed attempt"
                                        haveFinished[i] = -1
                                        Done = -1
                                else:
                                    print "indexing job hasn't finished yet: ", self.runNumber
                                    time.sleep(10)
                        else:
                            print "no such file yet: ", myLog
                            time.sleep(10)

                if Done == 1:
                    # Merge all stream files into one
                    totalStream = self.outDir+"/"+self.experimentName+"_"+str(self.runNumber)+".stream"
                    with open(totalStream, 'w') as outfile:
                        for fname in myStreamList:
                            with open(fname) as infile:
                                outfile.write(infile.read())

                    # Remove images in hdf5
                    f = h5py.File(peakFile,'r+')
                    if '/entry_1/instrument_1/detector_1/data' in f:
                        del f['/entry_1/instrument_1/detector_1/data']
                    f.close()

                    # Clean up temp files
                    for fname in myStreamList:
                        os.remove(fname)





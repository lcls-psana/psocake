# Operates in two modes; interactive mode and batch mode
# Interactive mode: temporary list, cxi, geom files are written per update
# Batch mode: Creates a CXIDB file containing hits, turbo index the file, save single stream and delete CXIDB
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import subprocess
import pandas as pd
import h5py
import pyqtgraph as pg
from pyqtgraph.dockarea import *
from pyqtgraph.parametertree import Parameter, ParameterTree
import LaunchIndexer
from PSCalib.CalibFileFinder import deploy_calib_file

class CrystalIndexing(object):
    def __init__(self, parent = None):
        self.parent = parent

        ## Dock 14: Indexing
        self.d14 = Dock("Indexing", size=(1, 1))
        self.w21 = ParameterTree()
        self.w21.setWindowTitle('Indexing')
        self.d14.addWidget(self.w21)
        self.w22 = pg.LayoutWidget()
        self.launchIndexBtn = QtGui.QPushButton('Launch indexing')
        self.w22.addWidget(self.launchIndexBtn, row=0, col=0)
        self.synchBtn = QtGui.QPushButton('Deploy CrystFEL geometry')
        self.w22.addWidget(self.synchBtn, row=1, col=0)
        self.d14.addWidget(self.w22)


        self.index_grp = 'Crystal indexing'
        self.index_on_str = 'Indexing on'
        self.index_geom_str = 'CrystFEL geometry'
        self.index_peakMethod_str = 'Peak method'
        self.index_intRadius_str = 'Integration radii'
        self.index_pdb_str = 'PDB'
        self.index_method_str = 'Indexing method'
        #self.index_minPeaks_str = 'Minimum number of peaks'
        #self.index_maxPeaks_str = 'Maximum number of peaks'
        #self.index_minRes_str = 'Minimum resolution (pixels)'
        self.index_tolerance_str = 'Tolerance'
        self.index_extra_str = 'Extra CrystFEL parameters'

        self.launch_grp = 'Batch'
        self.outDir_str = 'Output directory'
        self.runs_str = 'Runs(s)'
        self.sample_str = 'Sample name'
        self.tag_str = 'Tag'
        self.queue_str = 'Queue'
        self.chunkSize_str = 'Chunk size'
        self.keepData_str = 'Keep CXI images'
        self.noe_str = 'Number of events to process'
        (self.psanaq_str,self.psnehq_str,self.psfehq_str,self.psnehprioq_str,self.psfehprioq_str,self.psnehhiprioq_str,self.psfehhiprioq_str,self.psdebugq_str) = \
            ('psanaq','psnehq','psfehq','psnehprioq','psfehprioq','psnehhiprioq','psfehhiprioq','psdebugq')

        self.outDir = self.parent.psocakeDir
        self.outDir_overridden = False
        self.runs = ''
        self.sample = 'crystal'
        self.tag = ''
        self.queue = self.psanaq_str
        self.chunkSize = 500
        self.noe = -1

        # Indexing
        self.showIndexedPeaks = False
        self.indexedPeaks = None
        self.hiddenCXI = '.temp.cxi'
        self.hiddenCrystfelStream = '.temp.stream'
        self.hiddenCrystfelList = '.temp.lst'

        self.indexingOn = False
        self.numIndexedPeaksFound = 0
        self.geom = '.temp.geom'
        self.peakMethod = 'cxi'
        self.intRadius = '2,3,4'
        self.pdb = ''
        self.indexingMethod = 'mosflm-noretry,dirax'
        #self.minPeaks = 15
        #self.maxPeaks = 2048
        #self.minRes = 0
        self.tolerance = '5,5,5,1.5'
        self.extra = ''
        self.keepData = True

        #######################
        # Mandatory parameter #
        #######################
        self.params = [
            {'name': self.index_grp, 'type': 'group', 'children': [
                {'name': self.index_on_str, 'type': 'bool', 'value': self.indexingOn, 'tip': "Turn on indexing"},
                {'name': self.index_geom_str, 'type': 'str', 'value': self.geom, 'tip': "CrystFEL geometry file"},
                #{'name': self.index_peakMethod_str, 'type': 'str', 'value': self.peakMethod, 'tip': "Turn on indexing"},
                {'name': self.index_intRadius_str, 'type': 'str', 'value': self.intRadius, 'tip': "Integration radii"},
                {'name': self.index_pdb_str, 'type': 'str', 'value': self.pdb, 'tip': "(Optional) CrystFEL unitcell file"},
                {'name': self.index_method_str, 'type': 'str', 'value': self.indexingMethod, 'tip': "comma separated indexing methods"},
                {'name': self.index_tolerance_str, 'type': 'str', 'value': self.tolerance,
                 'tip': "Indexing tolerance, default: 5,5,5,1.5"},
                {'name': self.index_extra_str, 'type': 'str', 'value': self.extra,
                 'tip': "Other indexing parameters"},
                #{'name': self.index_minPeaks_str, 'type': 'int', 'value': self.minPeaks,
                # 'tip': "Index only if there are more Bragg peaks found"},
                #{'name': self.index_maxPeaks_str, 'type': 'int', 'value': self.maxPeaks,
                # 'tip': "Index only if there are less Bragg peaks found"},
                #{'name': self.index_minRes_str, 'type': 'int', 'value': self.minRes,
                # 'tip': "Index only if Bragg peak resolution is at least this"},
            ]},
            {'name': self.launch_grp, 'type': 'group', 'children': [
                {'name': self.outDir_str, 'type': 'str', 'value': self.outDir},
                {'name': self.runs_str, 'type': 'str', 'value': self.runs, 'tip': "comma separated or use colon for a range, e.g. 1,3,5:7 = runs 1,3,5,6,7"},
                {'name': self.sample_str, 'type': 'str', 'value': self.sample, 'tip': "name of the sample saved in the cxidb file, e.g. lysozyme"},
                {'name': self.tag_str, 'type': 'str', 'value': self.tag, 'tip': "attach tag to stream, e.g. cxitut13_0010_tag.stream"},
                {'name': self.queue_str, 'type': 'list', 'values': {self.psfehhiprioq_str: self.psfehhiprioq_str,
                                                               self.psnehhiprioq_str: self.psnehhiprioq_str,
                                                               self.psfehprioq_str: self.psfehprioq_str,
                                                               self.psnehprioq_str: self.psnehprioq_str,
                                                               self.psfehq_str: self.psfehq_str,
                                                               self.psnehq_str: self.psnehq_str,
                                                               self.psanaq_str: self.psanaq_str,
                                                               self.psdebugq_str: self.psdebugq_str},
                 'value': self.queue, 'tip': "Choose queue"},
                {'name': self.chunkSize_str, 'type': 'int', 'value': self.chunkSize, 'tip': "number of patterns to process per worker"},
                {'name': self.keepData_str, 'type': 'bool', 'value': self.keepData, 'tip': "Do not delete cxidb images in cxi file"},
            ]},
        ]

        self.p9 = Parameter.create(name='paramsCrystalIndexing', type='group', \
                                   children=self.params, expanded=True)
        self.w21.setParameters(self.p9, showTop=False)
        self.p9.sigTreeStateChanged.connect(self.change)

        self.parent.connect(self.launchIndexBtn, QtCore.SIGNAL("clicked()"), self.indexPeaks)
        self.parent.connect(self.synchBtn, QtCore.SIGNAL("clicked()"), self.syncGeom)

    # Launch indexing
    def indexPeaks(self):
        self.parent.thread.append(LaunchIndexer.LaunchIndexer(self.parent))  # send parent parameters with self
        self.parent.thread[self.parent.threadCounter].launch(self.parent.experimentName, self.parent.detInfo)
        self.parent.threadCounter += 1

    # Update psana geometry
    def syncGeom(self):
        with pg.BusyCursor():
            print "#################################################"
            print "Updating psana geometry with CrystFEL geometry"
            print "#################################################"
            self.parent.geom.findPsanaGeometry()
            psanaGeom = self.parent.psocakeRunDir + "/.temp.data"
            if self.parent.args.localCalib:
                cmd = ["crystfel2psana",
                       "-e", self.parent.experimentName,
                       "-r", str(self.parent.runNumber),
                       "-d", str(self.parent.det.name),
                       "--rootDir", '.',
                       "-c", self.geom,
                       "-p", psanaGeom]
            else:
                cmd = ["crystfel2psana",
                       "-e", self.parent.experimentName,
                       "-r", str(self.parent.runNumber),
                       "-d", str(self.parent.det.name),
                       "--rootDir", self.parent.rootDir,
                       "-c", self.geom,
                       "-p", psanaGeom]
            if self.parent.args.v >= 1: print "cmd: ", cmd
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            output = p.communicate()[0]
            p.stdout.close()
            # Reload new psana geometry
            cmts = {'exp': self.parent.experimentName, 'app': 'psocake', 'comment': 'converted from crystfel geometry'}
            if self.parent.args.localCalib:
                calibDir = './calib'
            elif self.parent.args.outDir is None:
                calibDir = self.parent.rootDir + '/calib'
            else:
                calibDir = '/reg/d/psdm/' + self.parent.experimentName[:3] + '/' + self.parent.experimentName + '/calib'
            deploy_calib_file(cdir=calibDir, src=str(self.parent.det.name), type='geometry',
                              run_start=self.parent.runNumber, run_end=None, ifname=psanaGeom, dcmts=cmts, pbits=0)
            self.parent.exp.setupExperiment()
            self.parent.img.getDetImage(self.parent.eventNumber)
            self.parent.geom.updateRings()
            self.parent.index.updateIndex()
            self.parent.geom.drawCentre()

    # If anything changes in the parameter tree, print a message
    def change(self, panel, changes):
        for param, change, data in changes:
            path = panel.childPath(param)
            if self.parent.args.v >= 1:
                print('  path: %s' % path)
                print('  change:    %s' % change)
                print('  data:      %s' % str(data))
                print('  ----------')
            self.paramUpdate(path, change, data)

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
        #elif path[1] == self.index_minPeaks_str:
        #    self.updateMinPeaks(data)
        #elif path[1] == self.index_maxPeaks_str:
        #    self.updateMaxPeaks(data)
        #elif path[1] == self.index_minRes_str:
        #    self.updateMinRes(data)
        elif path[1] == self.index_tolerance_str:
            self.updateTolerance(data)
        elif path[1] == self.index_extra_str:
            self.updateExtra(data)
        # launch grp
        elif path[1] == self.outDir_str:
            self.updateOutputDir(data)
        elif path[1] == self.runs_str:
            self.updateRuns(data)
        elif path[1] == self.sample_str:
            self.updateSample(data)
        elif path[1] == self.tag_str:
            self.updateTag(data)
        elif path[1] == self.queue_str:
            self.updateQueue(data)
        elif path[1] == self.chunkSize_str:
            self.updateChunkSize(data)
        elif path[1] == self.noe_str:
            self.updateNoe(data)
        elif path[1] == self.keepData_str:
            self.keepData = data

    def updateIndexStatus(self, data):
        self.indexingOn = data
        self.showIndexedPeaks = data
        self.updateIndex()

    def updateGeom(self, data):
        self.geom = data
        self.updateIndex()

    def updatePeakMethod(self, data):
        self.peakMethod = data
        if self.indexingOn:
            self.updateIndex()

    def updateIntegrationRadius(self, data):
        self.intRadius = data
        self.updateIndex()

    def updatePDB(self, data):
        self.pdb = data
        self.updateIndex()

    def updateIndexingMethod(self, data):
        self.indexingMethod = data
        self.updateIndex()

    #def updateMinPeaks(self, data):
    #    self.minPeaks = data
    #    self.updateIndex()

    #def updateMaxPeaks(self, data):
    #    self.maxPeaks = data
    #    self.updateIndex()

    #def updateMinRes(self, data):
    #    self.minRes = data
    #    self.updateIndex()

    def updateTolerance(self, data):
        self.tolerance = data
        self.updateIndex()

    def updateExtra(self, data):
        self.extra = data
        self.updateIndex()

    def updateIndex(self):
        if self.indexingOn:
            self.indexer = IndexHandler(parent=self.parent)
            self.indexer.computeIndex(self.parent.experimentName, self.parent.runNumber, self.parent.detInfo,
                                      self.parent.eventNumber, self.geom, self.peakMethod, self.intRadius, self.pdb,
                                      self.indexingMethod, self.parent.pk.minPeaks, self.parent.pk.maxPeaks, self.parent.pk.minRes,
                                      self.tolerance, self.extra, self.outDir, queue=None)

    def updateOutputDir(self, data):
        self.outDir = data
        self.outDir_overridden = True

    def updateRuns(self, data):
        self.runs = data

    def updateSample(self, data):
        self.sample = data

    def updateTag(self, data):
        self.tag = data

    def updateQueue(self, data):
        self.queue = data

    def updateChunkSize(self, data):
        self.chunkSize = data

    def updateNoe(self, data):
        self.noe = data

    def clearIndexedPeaks(self):
        self.parent.img.w1.getView().removeItem(self.parent.img.abc_text)
        self.parent.img.indexedPeak_feature.setData([], [], pxMode=False)
        if self.parent.args.v >= 1: print "Done clearIndexedPeaks"

    def displayWaiting(self):
        if self.showIndexedPeaks:
            if self.numIndexedPeaksFound == 0:  # indexing proceeding
                xMargin = 5  # pixels
                maxX = np.max(self.parent.det.indexes_x(self.parent.evt)) + xMargin
                maxY = np.max(self.parent.det.indexes_y(self.parent.evt))
                # Draw a big X
                cenX = np.array((self.parent.cx,)) + 0.5
                cenY = np.array((self.parent.cy,)) + 0.5
                diameter = 256  # self.peakRadius*2+1
                self.parent.img.indexedPeak_feature.setData(cenX, cenY, symbol='t', \
                                                            size=diameter, brush=(255, 255, 255, 0), \
                                                            pen=pg.mkPen({'color': "#FF00FF", 'width': 3}),
                                                            pxMode=False)
                self.parent.img.abc_text = pg.TextItem(html='', anchor=(0, 0))
                self.parent.img.w1.getView().addItem(self.parent.img.abc_text)
                self.parent.img.abc_text.setPos(maxX, maxY)

    def drawIndexedPeaks(self, latticeType=None, centering=None, numSaturatedPeaks=None, unitCell=None):
        self.clearIndexedPeaks()
        if self.showIndexedPeaks:
            if self.indexedPeaks is not None and self.numIndexedPeaksFound > 0: # indexing succeeded
                cenX = self.indexedPeaks[:,0]+0.5
                cenY = self.indexedPeaks[:,1]+0.5
                cenX = np.concatenate((cenX,cenX,cenX))
                cenY = np.concatenate((cenY,cenY,cenY))
                diameter = np.ones_like(cenX)
                diameter[0:self.numIndexedPeaksFound] = float(self.intRadius.split(',')[0])*2
                diameter[self.numIndexedPeaksFound:2*self.numIndexedPeaksFound] = float(self.intRadius.split(',')[1])*2
                diameter[2*self.numIndexedPeaksFound:3*self.numIndexedPeaksFound] = float(self.intRadius.split(',')[2])*2
                self.parent.img.indexedPeak_feature.setData(cenX, cenY, symbol='o', \
                                          size=diameter, brush=(255,255,255,0), \
                                          pen=pg.mkPen({'color': "#FF00FF", 'width': 1.5}), pxMode=False)

                # Write unit cell parameters
                if unitCell is not None:
                    xMargin = 5
                    yMargin = 400
                    maxX   = np.max(self.parent.det.indexes_x(self.parent.evt)) + xMargin
                    maxY   = np.max(self.parent.det.indexes_y(self.parent.evt)) - yMargin
                    myMessage = '<div style="text-align: center"><span style="color: #FF00FF; font-size: 12pt;">saturated='+\
                                str(numSaturatedPeaks) + '<br>lattice='+\
                                latticeType +'<br>centering=' + centering + '<br>a='+\
                                str(round(float(unitCell[0])*10,2))+'A <br>b='+str(round(float(unitCell[1])*10,2))+'A <br>c='+\
                                str(round(float(unitCell[2])*10,2))+'A <br>&alpha;='+str(round(float(unitCell[3]),2))+\
                                '&deg; <br>&beta;='+str(round(float(unitCell[4]),2))+'&deg; <br>&gamma;='+\
                                str(round(float(unitCell[5]),2))+'&deg; <br></span></div>'

                    self.parent.img.abc_text = pg.TextItem(html=myMessage, anchor=(0,0))
                    self.parent.img.w1.getView().addItem(self.parent.img.abc_text)
                    self.parent.img.abc_text.setPos(maxX, maxY)
            else: # Failed indexing
                xMargin = 5 # pixels
                maxX   = np.max(self.parent.det.indexes_x(self.parent.evt))+xMargin
                maxY   = np.max(self.parent.det.indexes_y(self.parent.evt))
                # Draw a big X
                cenX = np.array((self.parent.cx,))+0.5
                cenY = np.array((self.parent.cy,))+0.5
                diameter = 256 #self.peakRadius*2+1
                self.parent.img.indexedPeak_feature.setData(cenX, cenY, symbol='x', \
                                          size=diameter, brush=(255,255,255,0), \
                                          pen=pg.mkPen({'color': "#FF00FF", 'width': 3}), pxMode=False)
                self.parent.img.abc_text = pg.TextItem(html='', anchor=(0,0))
                self.parent.img.w1.getView().addItem(self.parent.img.abc_text)
                self.parent.img.abc_text.setPos(maxX,maxY)
        else:
            self.parent.img.indexedPeak_feature.setData([], [], pxMode=False)
        if self.parent.args.v >= 1: print "Done updatePeaks"

    # This function probably doesn't get called
    def launchIndexing(self, requestRun=None):
        self.batchIndexer = IndexHandler(parent=self.parent)
        if requestRun is None:
            self.batchIndexer.computeIndex(self.parent.experimentName, self.parent.runNumber, self.parent.detInfo,
                                  self.parent.eventNumber, self.geom, self.peakMethod, self.intRadius, self.pdb,
                                       self.indexingMethod, self.parent.pk.minPeaks, self.parent.pk.maxPeaks, self.parent.pk.minRes,
                                           self.tolerance, self.extra, self.outDir, self.runs, self.sample, self.tag, self.queue, self.chunkSize, self.noe)
        else:
            self.batchIndexer.computeIndex(self.parent.experimentName, requestRun, self.parent.detInfo,
                                  self.parent.eventNumber, self.geom, self.peakMethod, self.intRadius, self.pdb,
                                       self.indexingMethod, self.parent.pk.minPeaks, self.parent.pk.maxPeaks, self.parent.pk.minRes,
                                           self.tolerance, self.extra, self.outDir, self.runs, self.sample, self.tag, self.queue, self.chunkSize, self.noe)
        if self.parent.args.v >= 1: print "Done updateIndex"

class IndexHandler(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
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
        self.latticeType = None
        self.centering = None
        self.numSaturatedPeaks = None
        self.unitCell = None
        self.minPeaks = None
        self.maxPeaks = None
        self.minRes = None
        # batch
        self.outDir = None
        self.runs = None
        self.sample = None
        self.tag = None
        self.queue = None
        self.chunkSize = None
        self.noe = None

    def __del__(self):
        if self.parent.args.v >= 1: print "del IndexHandler"
        self.exiting = True
        self.wait()

    def computeIndex(self, experimentName, runNumber, detInfo, eventNumber, geom, peakMethod, intRadius, pdb, indexingMethod,
                     minPeaks, maxPeaks, minRes, tolerance, extra, outDir=None, runs=None, sample=None, tag=None, queue=None,
                     chunkSize=None, noe=None):
        self.experimentName = experimentName
        self.runNumber = runNumber
        self.detInfo = detInfo
        self.eventNumber = eventNumber
        self.geom = geom
        self.peakMethod = peakMethod
        self.intRadius = intRadius
        self.pdb = pdb
        self.indexingMethod = indexingMethod
        self.minPeaks = minPeaks
        self.maxPeaks = maxPeaks
        self.minRes = minRes
        self.tolerance = tolerance
        self.extra = extra
        # batch
        self.outDir = outDir
        self.runs = runs
        self.sample = sample
        self.tag = tag
        self.queue = queue
        self.chunkSize = chunkSize
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

    def getIndexedPeaks(self):
        # Merge all stream files into one
        totalStream = self.outDir+"/r"+str(self.runNumber).zfill(4)+"/"+self.experimentName+"_"+str(self.runNumber).zfill(4)+".stream"
        with open(totalStream, 'w') as outfile:
            for fname in self.myStreamList:
                try:
                    with open(fname) as infile:
                        outfile.write(infile.read())
                except: # file may not exist yet
                    continue

        # Add indexed peaks and remove images in hdf5
        f = h5py.File(self.peakFile,'r')
        totalEvents = len(f['/entry_1/result_1/nPeaksAll'])
        hitEvents = f['/LCLS/eventNumber'].value
        f.close()
        # Add indexed peaks
        fstream = open(totalStream,'r')
        content=fstream.readlines()
        fstream.close()
        indexedPeaks = np.zeros((totalEvents,),dtype=int)
        numProcessed = 0
        for i,val in enumerate(content):
            if "Event: //" in val:
                _evt = int(val.split("Event: //")[-1].strip())
            if "indexed_by =" in val:
                _ind = val.split("indexed_by =")[-1].strip()
            if "num_peaks =" in val:
                _num = val.split("num_peaks =")[-1].strip()
                numProcessed += 1
                if 'none' in _ind:
                    continue
                else:
                    indexedPeaks[hitEvents[_evt]] = _num
        return indexedPeaks, numProcessed

    def checkJobExit(self, jobID):
        cmd = "bjobs -d | grep "+str(jobID)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        if "EXIT" in out:
            "*********** NODE FAILURE ************ ", jobID
            return 1
        else:
            return 0

    def run(self):
        if self.queue is None: # interactive indexing
            # Check if requirements are met for indexing
            if self.parent.pk.numPeaksFound >= self.minPeaks and \
                self.parent.pk.numPeaksFound <= self.maxPeaks and \
                self.parent.pk.peaksMaxRes >= self.minRes:
                print "OK, I'll index this pattern now"

                if self.parent.args.v >= 1: print "Running indexing!!!!!!!!!!!!"
                # Running indexing ...
                self.parent.index.numIndexedPeaksFound = 0
                self.parent.index.indexedPeaks = None
                self.parent.index.clearIndexedPeaks()
                self.parent.index.displayWaiting()

                # Write list
                with open(self.parent.index.hiddenCrystfelList, "w") as text_file:
                    text_file.write("{} //0".format(self.parent.index.hiddenCXI))

                # FIXME: convert psana geom to crystfel geom
                cmd = "indexamajig -j 1 -i " + self.parent.index.hiddenCrystfelList + " -g " + self.geom + " --peaks=" + self.peakMethod + \
                      " --int-radius=" + self.intRadius + " --indexing=" + self.indexingMethod + \
                      " -o " + self.parent.index.hiddenCrystfelStream + " --temp-dir=" + self.outDir + "/r" + str(
                      self.runNumber).zfill(4) + " --tolerance=" + str(self.tolerance)
                if self.pdb: cmd += " --pdb=" + self.pdb
                if self.extra: cmd += " " + self.extra

                print "cmd: ", cmd
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                out, err = process.communicate()

                mySuccessString = "1 had crystals"
                # Read CrystFEL CSPAD geometry in stream
                if mySuccessString in err:  # success
                    if self.parent.args.v >= 1: print "Indexing successful"
                    # print "Munging geometry file"
                    f = open(self.parent.index.hiddenCrystfelStream)
                    content = f.readlines()
                    for i, val in enumerate(content):
                        if '----- Begin geometry file -----' in val:
                            startLine = i
                        elif '----- End geometry file -----' in val:
                            endLine = i
                            break
                    geom = content[startLine:endLine]
                    numLines = endLine - startLine
                    # Remove comments
                    for i in np.arange(numLines - 1, -1, -1):  # Start from bottom
                        if ';' in geom[i].lstrip(' ')[0]: geom.pop(i)

                    numQuads = 4
                    numAsics = 16
                    columns = ['min_fs', 'min_ss', 'max_fs', 'max_ss', 'res', 'fs', 'ss', 'corner_x', 'corner_y']
                    columnsScan = ['fsx', 'fsy', 'ssx', 'ssy']
                    indexScan = []
                    for i in np.arange(numQuads):
                        for j in np.arange(numAsics):
                            indexScan.append('q' + str(i) + 'a' + str(j))

                    dfGeom = pd.DataFrame(np.empty((numQuads * numAsics, len(columns))), index=indexScan,
                                          columns=columns)
                    dfScan = pd.DataFrame(np.empty((numQuads * numAsics, len(columnsScan))), index=indexScan,
                                          columns=columnsScan)
                    counter = 0
                    for i in np.arange(numQuads):
                        for j in np.arange(numAsics):
                            myAsic = indexScan[counter]
                            for k in columns:
                                myLine = [s for s in geom if myAsic + '/' + k in s]
                                if myLine:  # sometimes elements in columns can be missing
                                    myVal = myLine[-1].split('=')[-1].rstrip().lstrip()
                                    if k == 'fs' or k == 'ss':
                                        dfGeom.loc[myAsic, k] = myVal
                                    else:
                                        dfGeom.loc[myAsic, k] = float(myVal)
                                    if k == 'fs':
                                        fsx = float(myVal.split('x')[0])
                                        fsy = float(myVal.split('x')[-1].split('y')[0])
                                        dfScan.loc[myAsic, 'fsx'] = fsx
                                        dfScan.loc[myAsic, 'fsy'] = fsy
                                    elif k == 'ss':
                                        ssx = float(myVal.split('x')[0])
                                        ssy = float(myVal.split('x')[-1].split('y')[0])
                                        dfScan.loc[myAsic, 'ssx'] = ssx
                                        dfScan.loc[myAsic, 'ssy'] = ssy
                                else:
                                    if self.parent.args.v >= 1: print myAsic + '/' + k + " doesn't exist"
                            counter += 1
                    f.close()
                else:
                    if self.parent.args.v >= 1: print "Indexing failed"
                    self.parent.index.drawIndexedPeaks()

                # Read CrystFEL indexed peaks
                if mySuccessString in err:  # success
                    f = open(self.parent.index.hiddenCrystfelStream)
                    content = f.readlines()
                    for i, val in enumerate(content):
                        if 'End of peak list' in val:
                            endLine = i-1
                        elif 'num_saturated_peaks =' in val:
                            self.numSaturatedPeaks = int(val.split('=')[-1])
                        elif 'lattice_type =' in val:
                            self.latticeType = val.split('=')[-1]
                        elif 'centering =' in val:
                            self.centering = val.split('=')[-1]
                        elif 'fs/px   ss/px (1/d)/nm^-1   Intensity  Panel' in val:
                            startLine = i + 1
                        elif 'Cell parameters' in val:
                            (_, _, a, b, c, _, al, be, ga, _) = val.split()
                            self.unitCell = (a, b, c, al, be, ga)
                        elif 'diffraction_resolution_limit =' in val:
                            (_, _, _, _, _, resLim, _) = val.split() # Angstrom
                        elif 'End of reflections' in val:
                            endReflectionLine = i-1
                        elif '   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel' in val:
                            startReflectionLine = i+1

                    numPeaks = endLine-startLine
                    numReflections = endReflectionLine-startReflectionLine

                    #print "numReflections: ", numPeaks, numReflections

                    columns = ['fs', 'ss', 'res', 'intensity', 'asic']
                    df = pd.DataFrame(np.empty((numPeaks, len(columns))), columns=columns)
                    for i in np.arange(numPeaks):
                        contentLine = startLine + i
                        df['fs'][i] = float(content[contentLine][0:7])
                        df['ss'][i] = float(content[contentLine][7:15])
                        df['res'][i] = float(content[contentLine][15:26])
                        df['intensity'][i] = float(content[contentLine][26:38])
                        df['asic'][i] = str(content[contentLine][38:-1])

                    if numReflections > 0:
                        columns = ['h', 'k', 'l', 'I', 'sigma', 'peak', 'background', 'fs', 'ss', 'panel']
                        dfRefl = pd.DataFrame(np.empty((numReflections, len(columns))), columns=columns)
                        for i in np.arange(numReflections):
                            contentLine = startReflectionLine + i
                            dfRefl['h'][i] = float(content[contentLine][0:4])
                            dfRefl['k'][i] = float(content[contentLine][4:9])
                            dfRefl['l'][i] = float(content[contentLine][9:14])
                            dfRefl['I'][i] = float(content[contentLine][14:25])
                            dfRefl['sigma'][i] = str(content[contentLine][25:36])
                            dfRefl['peak'][i] = float(content[contentLine][36:47])
                            dfRefl['background'][i] = float(content[contentLine][47:58])
                            dfRefl['fs'][i] = float(content[contentLine][58:65])
                            dfRefl['ss'][i] = float(content[contentLine][65:72])
                            dfRefl['panel'][i] = str(content[contentLine][72:-1])
                    f.close()

                    # #Convert to CrystFEL coordinates
                    # columnsPeaks = ['x', 'y', 'psocakeX', 'psocakeY']
                    # dfPeaks = pd.DataFrame(np.empty((numPeaks, len(columnsPeaks))), columns=columnsPeaks)
                    # for i in np.arange(numPeaks):
                    #    myAsic = df['asic'][i].strip()
                    #    x = (df['fs'][i] - dfGeom.loc[myAsic, 'min_fs']) * dfScan.loc[myAsic, 'fsx'] + (df['ss'][i] -
                    #                                                                                    dfGeom.loc[
                    #                                                                                        myAsic, 'min_ss']) * \
                    #                                                                                   dfScan.loc[
                    #                                                                                       myAsic, 'ssx']
                    #    x += dfGeom.loc[myAsic, 'corner_x']
                    #    y = (df['fs'][i] - dfGeom.loc[myAsic, 'min_fs']) * dfScan.loc[myAsic, 'fsy'] + (df['ss'][i] -
                    #                                                                                    dfGeom.loc[
                    #                                                                                        myAsic, 'min_ss']) * \
                    #                                                                                   dfScan.loc[
                    #                                                                                       myAsic, 'ssy']
                    #    y += dfGeom.loc[myAsic, 'corner_y']
                    #    dfPeaks['x'][i] = x
                    #    dfPeaks['y'][i] = y
                    #
                    # # Convert to psocake coordinates
                    # for i in np.arange(numPeaks): # numPeaks
                    #     dfPeaks['psocakeX'][i] = self.parent.cx - dfPeaks['x'][i]
                    #     dfPeaks['psocakeY'][i] = self.parent.cy + dfPeaks['y'][i]
                    #
                    # if self.parent.index.showIndexedPeaks and self.eventNumber == self.parent.eventNumber:
                    #     self.parent.index.numIndexedPeaksFound = numPeaks
                    #     self.parent.index.indexedPeaks = dfPeaks[['psocakeX', 'psocakeY']].as_matrix()
                    #     self.parent.index.drawIndexedPeaks(self.unitCell)

                    # Convert predicted spots to CrystFEL coordinates
                    columnsPeaks = ['x', 'y', 'psocakeX', 'psocakeY']
                    dfPeaks = pd.DataFrame(np.empty((numReflections, len(columnsPeaks))), columns=columnsPeaks)
                    for i in np.arange(numReflections):
                        myAsic = dfRefl['panel'][i].strip()
                        x = (dfRefl['fs'][i] - dfGeom.loc[myAsic, 'min_fs']) * dfScan.loc[myAsic, 'fsx'] + (dfRefl['ss'][i] -
                                                                                                        dfGeom.loc[
                                                                                                            myAsic, 'min_ss']) * \
                                                                                                       dfScan.loc[
                                                                                                           myAsic, 'ssx']
                        x += dfGeom.loc[myAsic, 'corner_x']
                        y = (dfRefl['fs'][i] - dfGeom.loc[myAsic, 'min_fs']) * dfScan.loc[myAsic, 'fsy'] + (dfRefl['ss'][i] -
                                                                                                        dfGeom.loc[
                                                                                                            myAsic, 'min_ss']) * \
                                                                                                       dfScan.loc[
                                                                                                           myAsic, 'ssy']
                        y += dfGeom.loc[myAsic, 'corner_y']
                        dfPeaks['x'][i] = x
                        dfPeaks['y'][i] = y

                    # Convert to psocake coordinates
                    for i in np.arange(numReflections):
                        dfPeaks['psocakeX'][i] = self.parent.cx - dfPeaks['x'][i]
                        dfPeaks['psocakeY'][i] = self.parent.cy + dfPeaks['y'][i]

                    if self.parent.index.showIndexedPeaks and self.eventNumber == self.parent.eventNumber:
                        self.parent.index.numIndexedPeaksFound = numReflections
                        self.parent.index.indexedPeaks = dfPeaks[['psocakeX', 'psocakeY']].as_matrix()
                        self.parent.index.drawIndexedPeaks(self.latticeType, self.centering, self.numSaturatedPeaks, self.unitCell)
            else:
                print "Indexing requirement not met."
                if self.parent.pk.numPeaksFound < self.minPeaks: print "Decrease minimum number of peaks"
                if self.parent.pk.numPeaksFound > self.maxPeaks: print "Increase maximum number of peaks"
                if self.parent.pk.peaksMaxRes < self.minRes: print "Decrease minimum resolution"




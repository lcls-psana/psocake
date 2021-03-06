# Operates in two modes; interactive mode and batch mode
# Interactive mode: temporary list, cxi, geom files are written per update
# Batch mode: Creates a CXIDB file containing hits, turbo index the file, save single stream and delete CXIDB
# TODO: special display for systematic absences given spacegroup (to check peak finding parameters produce zero)
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import subprocess
import h5py
import pyqtgraph as pg
from pyqtgraph.dockarea import *
from pyqtgraph.parametertree import Parameter, ParameterTree
import LaunchIndexer
import os
import pandas as pd
from utils import highlight
try:
    from PyQt5.QtWidgets import *
    using_pyqt4 = False
except ImportError:
    using_pyqt4 = True
    pass
from PSCalib.CalibFileFinder import deploy_calib_file

class CrystalIndexing(object):
    def __init__(self, parent = None):
        self.parent = parent

        ## Dock: Indexing
        self.dock = Dock("Indexing", size=(1, 1))
        self.win = ParameterTree()
        self.win.setWindowTitle('Indexing')
        self.dock.addWidget(self.win)
        self.winL = pg.LayoutWidget()
        self.launchIndexBtn = QtGui.QPushButton('Launch indexing')
        self.winL.addWidget(self.launchIndexBtn, row=0, col=0)
        self.synchBtn = QtGui.QPushButton('Deploy CrystFEL geometry')
        self.winL.addWidget(self.synchBtn, row=1, col=0)
        self.dock.addWidget(self.winL)

        self.index_grp = 'Crystal indexing'
        self.index_on_str = 'Indexing on'
        self.index_geom_str = 'CrystFEL geometry'
        self.index_peakMethod_str = 'Peak method'
        self.index_intRadius_str = 'Integration radii'
        self.index_pdb_str = 'Unitcell'
        self.index_method_str = 'Indexing method'
        self.index_tolerance_str = 'Tolerance'
        self.index_extra_str = 'Extra CrystFEL parameters'
        self.index_condition_str = 'Index condition'

        self.launch_grp = 'Batch'
        self.outDir_str = 'Output directory'
        self.runs_str = 'Runs(s)'
        self.sample_str = 'Sample name'
        self.tag_str = '.stream tag'
        self.queue_str = 'Queue'
        self.chunkSize_str = 'Chunk size'
        self.cpu_str = 'CPUs'
        self.keepData_str = 'Keep CXI images'
        self.noe_str = 'Number of events to process'
        (self.psanaq_str,self.psnehq_str,self.psfehq_str,self.psnehprioq_str,self.psfehprioq_str,self.psnehhiprioq_str,self.psfehhiprioq_str,self.psdebugq_str) = \
            ('psanaq','psnehq','psfehq','psnehprioq','psfehprioq','psnehhiprioq','psfehhiprioq','psdebugq')
        self.noQueue_str = 'N/A'

        self.outDir = self.parent.psocakeDir
        self.outDir_overridden = False
        self.runs = ''
        self.sample = 'crystal'
        self.tag = ''
        self.queue = self.psanaq_str
        self.chunkSize = 500
        self.cpu = 12
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
        self.intRadius = '4,5,6'
        self.pdb = ''
        self.indexingMethod = 'mosflm'
        self.tolerance = '5,5,5,1.5'
        self.extra = ''
        self.condition = ''
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
                 'tip': "Other indexing parameters, comma separated (e.g. --multi,--no-check-peaks)"},
                {'name': self.index_condition_str, 'type': 'str', 'value': self.condition,
                 'tip': "indexing condition e.g. 41 in #evr1# and #eventNumber# > 3"},
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
        self.win.setParameters(self.p9, showTop=False)
        self.p9.sigTreeStateChanged.connect(self.change)

        if using_pyqt4:
            self.parent.connect(self.launchIndexBtn, QtCore.SIGNAL("clicked()"), self.indexPeaks)
            self.parent.connect(self.synchBtn, QtCore.SIGNAL("clicked()"), self.syncGeom)
        else:
            self.launchIndexBtn.clicked.connect(self.indexPeaks)
            self.synchBtn.clicked.connect(self.syncGeom)

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
                       "-p", psanaGeom,
                       "-z", str(self.parent.clen)]
            else:
                cmd = ["crystfel2psana",
                       "-e", self.parent.experimentName,
                       "-r", str(self.parent.runNumber),
                       "-d", str(self.parent.det.name),
                       "--rootDir", self.parent.rootDir,
                       "-c", self.geom,
                       "-p", psanaGeom,
                       "-z", str(self.parent.clen)]
            if self.parent.args.v >= 0: print "cmd: ", cmd
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
                calibDir = self.parent.dir + '/' + self.parent.experimentName[:3] + '/' + self.parent.experimentName + '/calib'
            deploy_calib_file(cdir=calibDir, src=str(self.parent.det.name), type='geometry',
                              run_start=self.parent.runNumber, run_end=None, ifname=psanaGeom, dcmts=cmts, pbits=0)
            self.parent.exp.setupExperiment()
            self.parent.img.getDetImage(self.parent.eventNumber)
            self.parent.geom.updateRings()
            self.parent.index.updateIndex()
            self.parent.geom.drawCentre()
            # Show mask
            self.parent.mk.updatePsanaMaskOn()

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
        elif path[1] == self.index_tolerance_str:
            self.updateTolerance(data)
        elif path[1] == self.index_extra_str:
            self.updateExtra(data)
        elif path[1] == self.index_condition_str:
            self.updateCondition(data)
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
        elif path[1] == self.cpu_str:
            self.updateCpu(data)
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

    def updateTolerance(self, data):
        self.tolerance = data
        self.updateIndex()

    def updateExtra(self, data):
        self.extra = data.replace(" ","")
        self.updateIndex()

    def updateCondition(self, data):
        self.condition = data
        self.updateIndex()

    def updateIndex(self):
        if self.indexingOn:
            if self.parent.pk.peaks is None:
                self.parent.index.clearIndexedPeaks()
            else:
                self.indexer = IndexHandler(parent=self.parent)
                self.indexer.computeIndex(self.parent.experimentName, self.parent.runNumber, self.parent.detInfo,
                                          self.parent.eventNumber, self.geom, self.peakMethod, self.intRadius, self.pdb,
                                          self.indexingMethod, self.parent.pk.minPeaks, self.parent.pk.maxPeaks, self.parent.pk.minRes,
                                          self.tolerance, self.extra, self.outDir, queue=None)
        else:
            # do not display predicted spots
            self.parent.index.clearIndexedPeaks()

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

    def updateCpu(self, data):
        self.cpu = data

    def updateNoe(self, data):
        self.noe = data

    def clearIndexedPeaks(self):
        self.parent.img.win.getView().removeItem(self.parent.img.abc_text)
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
                self.parent.img.indexedPeak_feature.setData(cenY, cenX, symbol='t', \
                                                            size=diameter, brush=(255, 255, 255, 0), \
                                                            pen=pg.mkPen({'color': "#FF00FF", 'width': 3}),
                                                            pxMode=False)
                self.parent.img.abc_text = pg.TextItem(html='', anchor=(0, 0))
                self.parent.img.win.getView().addItem(self.parent.img.abc_text)
                self.parent.img.abc_text.setPos(maxY, maxX)

    def drawIndexedPeaks(self, latticeType=None, centering=None, unitCell=None):
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
                self.parent.img.indexedPeak_feature.setData(cenY, cenX, symbol='o', \
                                          size=diameter, brush=(255,255,255,0), \
                                          pen=pg.mkPen({'color': "#FF00FF", 'width': 1.5}), pxMode=False)

                # Write unit cell parameters
                if unitCell is not None:
                    xMargin = 0#5
                    yMargin = self.parent.data.shape[1]#400
                    maxX   = np.max(self.parent.det.indexes_x(self.parent.evt)) + xMargin
                    maxY   = np.max(self.parent.det.indexes_y(self.parent.evt)) - yMargin
                    myMessage = '<div style="text-align: center"><span style="color: #FF00FF; font-size: 12pt;">lattice='+\
                               str(latticeType) +'<br>centering=' + str(centering) + '<br></span></div>'
                    self.parent.img.abc_text = pg.TextItem(html=myMessage, anchor=(0,0))
                    self.parent.img.win.getView().addItem(self.parent.img.abc_text)
                    self.parent.img.abc_text.setPos(maxY, maxX)
            else: # Failed indexing
                # Draw a big X
                cenX = np.array((self.parent.cx,))+0.5
                cenY = np.array((self.parent.cy,))+0.5
                diameter = 256 #self.peakRadius*2+1
                self.parent.img.indexedPeak_feature.setData(cenY, cenX, symbol='x', \
                                          size=diameter, brush=(255,255,255,0), \
                                          pen=pg.mkPen({'color': "#FF00FF", 'width': 3}), pxMode=False)
                self.parent.img.abc_text = pg.TextItem(html='', anchor=(0,0))
                self.parent.img.win.getView().addItem(self.parent.img.abc_text)
                self.parent.img.abc_text.setPos(0,0)
        else:
            self.parent.img.indexedPeak_feature.setData([], [], pxMode=False)
            self.parent.img.abc_text = pg.TextItem(html='', anchor=(0,0))
            self.parent.img.win.getView().addItem(self.parent.img.abc_text)
            self.parent.img.abc_text.setPos(0,0)
        if self.parent.args.v >= 1: print "Done drawIndexedPeaks"

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
        self.uniqueAxis = None
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
        self.cpu = None
        self.noe = None

    def __del__(self):
        if self.parent.args.v >= 1: print "del IndexHandler"
        self.exiting = True
        self.wait()

    def computeIndex(self, experimentName, runNumber, detInfo, eventNumber, geom, peakMethod, intRadius, pdb, indexingMethod,
                     minPeaks, maxPeaks, minRes, tolerance, extra, outDir=None, runs=None, sample=None, tag=None, queue=None,
                     chunkSize=None, cpu=None, noe=None):
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
        self.cpu = cpu
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

                # Write list of files to index
                with open(self.parent.index.hiddenCrystfelList, "w") as text_file:
                    text_file.write("{} //0".format(self.parent.index.hiddenCXI))

                # Generate a static mask of bad pixels for indexing
                self.parent.mk.saveCheetahStaticMask()

                # Check indexamajig version
                cmd = 'indexamajig --version'
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                out, err = process.communicate()
                (major,minor,patch) = out.split('\n')[0].split(': ')[-1].split('.')
                if int(major) == 0 and int(minor) < 8:
                    mySuccessString = "1 had crystals" # changed from crystfel v0.8
                else:
                    mySuccessString = "1 crystals"

                # FIXME: convert psana geom to crystfel geom
                cmd = "indexamajig -j 1 -i " + self.parent.index.hiddenCrystfelList + " -g " + self.geom + " --peaks=" + self.peakMethod + \
                      " --int-radius=" + self.intRadius + " --indexing=" + self.indexingMethod + \
                      " -o " + self.parent.index.hiddenCrystfelStream + " --temp-dir=" + self.outDir + "/r" + str(
                      self.runNumber).zfill(4) + " --tolerance=" + str(self.tolerance)
                if self.pdb: cmd += " --pdb=" + self.pdb
                if self.extra:
                    _extra = self.extra.replace(",", " ")
                    cmd += " " + _extra

                print "cmd: ", cmd
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                out, err = process.communicate()
                if 'command not found' in err:
                    print "######################################################################"
                    print highlight("FATAL ERROR: I can't find indexamajig on this machine. Refer to:      ",'r',1)
                    print highlight("https://confluence.slac.stanford.edu/display/PSDM/Psocake+SFX+tutorial",'r',1)
                    print "######################################################################"

                # Read CrystFEL geometry in stream
                if mySuccessString in err:  # success
                    if self.parent.args.v >= 1: print "Indexing successful!"
                    # Munging geometry file
                    f = open(self.parent.index.hiddenCrystfelStream)
                    content = f.readlines()
                    try:
                        for i, val in enumerate(content):
                            if '----- Begin geometry file -----' in val:
                                startLine = i
                            elif '----- End geometry file -----' in val:
                                endLine = i
                                break
                        geom = content[startLine:endLine]
                        numLines = endLine - startLine
                    except:
                        geom = content[0] # This shouldn't happen
                        numLines = 0
                    # Remove comments
                    for i in np.arange(numLines - 1, -1, -1):  # Start from bottom
                        if ';' in geom[i].lstrip(' ')[0]: geom.pop(i)

                    columns = ['min_fs', 'min_ss', 'max_fs', 'max_ss', 'res', 'fs', 'ss', 'corner_x', 'corner_y']
                    columnsScan = ['fsx', 'fsy', 'ssx', 'ssy']
                    indexScan = []
                    if 'cspad' in self.parent.detInfo.lower():
                        numQuads = 4
                        numAsics = 16
                        for i in np.arange(numQuads):
                            for j in np.arange(numAsics):
                                indexScan.append('q' + str(i) + 'a' + str(j))
                    elif 'rayonix' in self.parent.detInfo.lower():
                        numQuads = 1
                        numAsics = 1
                        for i in np.arange(numQuads):
                            for j in np.arange(numAsics):
                                indexScan.append('p' + str(i) + 'a' + str(j))
                    elif 'epix10k' in self.parent.detInfo.lower() and '2m' in self.parent.detInfo.lower():
                        numQuads = 16
                        numAsics = 4
                        for i in np.arange(numQuads):
                            for j in np.arange(numAsics):
                                indexScan.append('p' + str(i) + 'a' + str(j))
                    elif 'jungfrau4m' in self.parent.detInfo.lower():
                        numQuads = 8
                        numAsics = 8
                        for i in np.arange(numQuads):
                            for j in np.arange(numAsics):
                                indexScan.append('p' + str(i) + 'a' + str(j))

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
                        elif 'indexed_by =' in val:
                            self.indexingAlg = val.split('=')[-1]
                        elif 'num_saturated_reflections =' in val:
                            self.numSaturatedPeaks = int(val.split('=')[-1])
                        elif 'lattice_type =' in val:
                            self.latticeType = val.split('=')[-1]
                        elif 'centering =' in val:
                            self.centering = val.split('=')[-1]
                        elif 'unique_axis =' in val:
                            self.uniqueAxis = val.split('=')[-1]
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

                    columns = ['fs', 'ss', 'res', 'intensity', 'asic']
                    peaks = content[startLine:endLine + 1]
                    myPeaks = []
                    for line in peaks:
                        myPeaks.append(line.split())
                    df = pd.DataFrame(myPeaks, columns=columns, dtype=float)
                    if numReflections > 0:
                        columns = ['h', 'k', 'l', 'I', 'sigma', 'peak', 'background', 'fs', 'ss', 'panel']
                        reflections = content[startReflectionLine:endReflectionLine + 1]
                        myReflections = []
                        for line in reflections:
                            myReflections.append(line.split())
                        dfRefl = pd.DataFrame(myReflections, columns=columns, dtype=float)
                    f.close()

                    # Convert predicted spots to CrystFEL coordinates
                    columnsPeaks = ['x', 'y', 'psocakeX', 'psocakeY']
                    dfPeaks = pd.DataFrame(np.empty((numReflections, len(columnsPeaks))), columns=columnsPeaks)
                    for i in np.arange(numReflections):
                        myAsic = dfRefl['panel'][i].strip()
                        x = (dfRefl['fs'][i] - dfGeom.loc[myAsic, 'min_fs']) * dfScan.loc[myAsic, 'fsx'] + \
                            (dfRefl['ss'][i] - dfGeom.loc[myAsic, 'min_ss']) * dfScan.loc[myAsic, 'ssx']
                        x += dfGeom.loc[myAsic, 'corner_x']
                        y = (dfRefl['fs'][i] - dfGeom.loc[myAsic, 'min_fs']) * dfScan.loc[myAsic, 'fsy'] + \
                            (dfRefl['ss'][i] - dfGeom.loc[myAsic, 'min_ss']) * dfScan.loc[myAsic, 'ssy']
                        y += dfGeom.loc[myAsic, 'corner_y']
                        dfPeaks['x'][i] = x
                        dfPeaks['y'][i] = y
                    # Convert to psocake coordinates
                    for i in np.arange(numReflections):
                        dfPeaks['psocakeX'][i] = self.parent.cy - dfPeaks['x'][i]
                        dfPeaks['psocakeY'][i] = self.parent.cx + dfPeaks['y'][i]

                    if self.parent.index.showIndexedPeaks and self.eventNumber == self.parent.eventNumber:
                        if self.parent.mouse.movie is None: # display gif
                            self.parent.mouse.tm.start(3000)  # ms
                        self.parent.index.numIndexedPeaksFound = numReflections
                        self.parent.index.indexedPeaks = dfPeaks[['psocakeX', 'psocakeY']].as_matrix()
                        self.parent.index.drawIndexedPeaks(self.latticeType, self.centering, self.unitCell) #FIXME: segfaults when indexing twice
                        try:
                            print "Indexed_by = ", str(self.indexingAlg.strip())
                            print "####################"
                            print "lattice_type = ", str(self.latticeType.strip())
                            print "centering = ", str(self.centering.strip())
                            print "unique_axis = ", str(self.uniqueAxis.strip())
                            print "a = ", str(round(float(self.unitCell[0])*10, 2)), " A"
                            print "b = ", str(round(float(self.unitCell[1])*10, 2)), " A"
                            print "c = ", str(round(float(self.unitCell[2])*10, 2)), " A"
                            print "al = ", str(round(float(self.unitCell[3]), 2)), " deg"
                            print "be = ", str(round(float(self.unitCell[4]), 2)), " deg"
                            print "ga = ", str(round(float(self.unitCell[5]), 2)), " deg"
                            print "####################"
                        except:
                            print "Could not print unit cell"
            else:
                print "Indexing requirement not met."
                if self.parent.pk.numPeaksFound < self.minPeaks: print "Decrease minimum number of peaks"
                if self.parent.pk.numPeaksFound > self.maxPeaks: print "Increase maximum number of peaks"
                if self.parent.pk.peaksMaxRes < self.minRes: print "Decrease minimum resolution"



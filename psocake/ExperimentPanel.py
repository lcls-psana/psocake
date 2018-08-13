import numpy as np
import subprocess
import os
import glob
try:
    logbook_present = True
    from LogBook.runtables import RunTables
except ImportError:
    logbook_present = False
if logbook_present:
    import LogbookCrawler
from pyqtgraph.dockarea import *
from pyqtgraph.parametertree import Parameter, ParameterTree
import time

if 'LCLS' in os.environ['PSOCAKE_FACILITY'].upper():
    import psana
    import Detector.PyDetector
elif 'PAL' in os.environ['PSOCAKE_FACILITY'].upper():
    pass

class ExperimentInfo(object):
    def __init__(self, parent = None):
        self.parent = parent

        #############################
        ## Dock: parameter
        #############################
        self.dock = Dock("Experiment Parameters", size=(1, 1))
        self.win = ParameterTree()
        self.win.setWindowTitle('Parameters')
        self.dock.addWidget(self.win)

        self.exp_grp = 'Experiment information'
        self.exp_name_str = 'Experiment Name'
        self.exp_run_str = 'Run Number'
        self.exp_det_str = 'DetInfo'
        self.exp_evt_str = 'Event Number'
        self.exp_eventID_str = 'EventID'
        self.exp_second_str = 'Seconds'
        self.exp_nanosecond_str = 'Nanoseconds'
        self.exp_fiducial_str = 'Fiducial'
        self.exp_numEvents_str = 'Total Events'
        self.exp_detInfo_str = 'Detector ID'

        self.eventSeconds = 0
        self.eventNanoseconds = 0
        self.eventFiducial = 0
        self.eventTotal = 0
        self.run = None
        self.times = None

        self.disp_grp = 'Display'
        self.disp_log_str = 'Logscale'
        self.disp_aduPerPhoton_str = 'ADUs per Photon'
        self.disp_image_str = 'Image properties'
        self.disp_adu_str = 'gain corrected ADU'
        self.disp_gain_str = 'gain'
        self.disp_gainMask_str = 'gain_mask'
        self.disp_coordx_str = 'coord_x'
        self.disp_coordy_str = 'coord_y'
        self.disp_quad_str = 'quad number'
        self.disp_seg_str = 'seg number'
        self.disp_row_str = 'row number'
        self.disp_col_str = 'col number'
        self.disp_raw_str = 'raw ADU'
        self.disp_pedestalCorrected_str = 'pedestal corrected ADU'
        self.disp_commonModeCorrected_str = 'common mode corrected ADU'
        self.disp_photons_str = 'photon counts'
        self.disp_rms_str = 'pixel rms'
        self.disp_status_str = 'pixel status'
        self.disp_pedestal_str = 'pedestal'
        self.disp_commonMode_str = 'common mode'
        self.disp_friedel_str = 'Apply Friedel symmetry'
        self.disp_commonModeOverride_str = 'Common mode (override)'
        self.disp_overrideCommonMode_str = 'Apply common mode (override)'
        self.disp_commonModeParam0_str = 'parameters 0'
        self.disp_commonModeParam1_str = 'parameters 1'
        self.disp_commonModeParam2_str = 'parameters 2'
        self.disp_commonModeParam3_str = 'parameters 3'
        self.disp_medianCorrection_str = 'median background corrected ADU'
        self.disp_radialCorrection_str = 'radial background corrected ADU'
        self.disp_medianFilterRank_str = 'median filter rank'

        self.logscaleOn = False
        self.aduPerPhoton = 1.
        self.medianFilterRank = 5

        # image properties
        self.disp_medianCorrection = 19
        self.disp_radialCorrection = 18
        self.disp_gainMask = 17
        self.disp_coordy= 16
        self.disp_coordx= 15
        self.disp_col= 14
        self.disp_row= 13
        self.disp_seg= 12
        self.disp_quad= 11
        self.disp_gain= 10
        self.disp_commonMode= 9
        self.disp_rms= 8
        self.disp_status= 7
        self.disp_pedestal= 6
        self.disp_photons= 5
        self.disp_raw= 4
        self.disp_pedestalCorrected= 3
        self.disp_commonModeCorrected= 2
        self.disp_adu= 1

        self.image_property = self.disp_adu

        self.applyFriedel = False

        self.applyCommonMode = False
        self.commonModeParams = np.array([0,0,0,0])
        self.commonMode = np.array([0, 0, 0, 0])
        self.firstSetupExperiment = True

        # e-log
        self.logger = False
        self.loggerFile = None
        self.crawlerRunning = False
        self.rt = None
        self.table = None

        self.params = [
            {'name': self.exp_grp, 'type': 'group', 'children': [
                {'name': self.exp_name_str, 'type': 'str', 'value': self.parent.experimentName,
                 'tip': "Experiment name, .e.g. cxic0415"},
                {'name': self.exp_run_str, 'type': 'int', 'value': self.parent.runNumber, 'tip': "Run number, e.g. 15"},
                {'name': self.exp_detInfo_str, 'type': 'str', 'value': self.parent.detInfo,
                 'tip': "Detector ID. Look at the terminal for available area detectors, e.g. DscCsPad"},
                {'name': self.exp_evt_str, 'type': 'int', 'value': self.parent.eventNumber, 'decimals': 7, 'tip': "Event number, first event is 0",
                 'children': [
                     # {'name': exp_eventID_str, 'type': 'str', 'value': self.eventID},#, 'readonly': False},
                     {'name': self.exp_second_str, 'type': 'str', 'value': self.eventSeconds, 'readonly': True},
                     {'name': self.exp_nanosecond_str, 'type': 'str', 'value': self.eventNanoseconds, 'readonly': True},
                     {'name': self.exp_fiducial_str, 'type': 'str', 'value': self.eventFiducial, 'readonly': True},
                     {'name': self.exp_numEvents_str, 'type': 'str', 'value': self.eventTotal, 'readonly': True},
                 ]},
            ]},
            {'name': self.disp_grp, 'type': 'group', 'children': [
                {'name': self.disp_log_str, 'type': 'bool', 'value': self.logscaleOn, 'tip': "Display in log10"},
                {'name': self.disp_aduPerPhoton_str, 'type': 'float', 'value': self.aduPerPhoton, 'tip': "ADUs per photon is used for photon conversion"},
                {'name': self.disp_medianFilterRank_str, 'type': 'int', 'value': self.medianFilterRank, 'tip': "Window size for median filter"},
                {'name': self.disp_image_str, 'type': 'list', 'values': {self.disp_medianCorrection_str: self.disp_medianCorrection,
                                                                         self.disp_radialCorrection_str: self.disp_radialCorrection,
                                                                         self.disp_gainMask_str: self.disp_gainMask,
                                                                         self.disp_coordy_str: self.disp_coordy,
                                                                         self.disp_coordx_str: self.disp_coordx,
                                                                         self.disp_col_str: self.disp_col,
                                                                         self.disp_row_str: self.disp_row,
                                                                         self.disp_seg_str: self.disp_seg,
                                                                         self.disp_quad_str: self.disp_quad,
                                                                         self.disp_gain_str: self.disp_gain,
                                                                         self.disp_commonMode_str: self.disp_commonMode,
                                                                         self.disp_rms_str: self.disp_rms,
                                                                         self.disp_status_str: self.disp_status,
                                                                         self.disp_pedestal_str: self.disp_pedestal,
                                                                         self.disp_photons_str: self.disp_photons,
                                                                         self.disp_raw_str: self.disp_raw,
                                                                         self.disp_pedestalCorrected_str: self.disp_pedestalCorrected,
                                                                         self.disp_commonModeCorrected_str: self.disp_commonModeCorrected,
                                                                         self.disp_adu_str: self.disp_adu},
                 'value': self.image_property, 'tip': "Choose image property to display"},
                {'name': self.disp_friedel_str, 'type': 'bool', 'value': self.applyFriedel,
                 'tip': "Click to apply Friedel symmetry to the detector image."},
                {'name': self.disp_commonModeOverride_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "",
                 'readonly': True, 'children': [
                    {'name': self.disp_overrideCommonMode_str, 'type': 'bool', 'value': self.applyCommonMode,
                     'tip': "Click to play around with common mode settings.\n This does not change your deployed calib file."},
                    {'name': self.disp_commonModeParam0_str, 'type': 'int', 'value': self.commonModeParams[0]},
                    {'name': self.disp_commonModeParam1_str, 'type': 'int', 'value': self.commonModeParams[1]},
                    {'name': self.disp_commonModeParam2_str, 'type': 'int', 'value': self.commonModeParams[2]},
                    {'name': self.disp_commonModeParam3_str, 'type': 'int', 'value': self.commonModeParams[3]},
                ]},
            ]},
        ]

        self.p = Parameter.create(name='params', type='group', children=self.params, expanded=True)
        self.win.setParameters(self.p, showTop=False)
        self.p.sigTreeStateChanged.connect(self.change)

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
        if path[0] == self.exp_grp:
            if path[1] == self.exp_name_str:
                self.updateExpName(data)
                if self.parent.pk.showPeaks: self.parent.pk.updateClassification()
            elif path[1] == self.exp_run_str:
                self.updateRunNumber(data)
                if self.parent.pk.showPeaks: self.parent.pk.updateClassification()
                self.parent.small.reloadQuantifier()
            elif path[1] == self.exp_detInfo_str:
                self.updateDetInfo(data)
                if self.parent.pk.showPeaks: self.parent.pk.updateClassification()
            elif path[1] == self.exp_evt_str and len(path) == 2 and change is 'value':
                self.updateEventNumber(data)
                if self.parent.pk.showPeaks: self.parent.pk.updateClassification()
        elif path[0] == self.disp_grp:
            if path[1] == self.disp_log_str:
                self.updateLogscale(data)
            elif path[1] == self.disp_aduPerPhoton_str:
                self.updateAduPerPhoton(data)
                if self.parent.pk.showPeaks: self.parent.pk.updateClassification()
            elif path[1] == self.disp_medianFilterRank_str:
                self.updateMedianFilter(data)
                if self.parent.pk.showPeaks: self.parent.pk.updateClassification()
            elif path[1] == self.disp_image_str:
                self.updateImageProperty(data)
                if self.parent.pk.showPeaks: self.parent.pk.updateClassification()
            elif path[1] == self.disp_friedel_str:
                self.updateFriedel(data)
            elif path[2] == self.disp_commonModeParam0_str:
                self.updateCommonModeParam(data, 0)
            elif path[2] == self.disp_commonModeParam1_str:
                self.updateCommonModeParam(data, 1)
            elif path[2] == self.disp_commonModeParam2_str:
                self.updateCommonModeParam(data, 2)
            elif path[2] == self.disp_commonModeParam3_str:
                self.updateCommonModeParam(data, 3)
            elif path[2] == self.disp_overrideCommonMode_str:
                self.updateCommonMode(data)
                if self.parent.pk.showPeaks: self.parent.pk.updateClassification()

    ###################################
    ###### Experiment Parameters ######
    ###################################
    def resetVariables(self):
        self.secList = None
        self.nsecList = None
        self.fidList = None

    def setupRunTable(self):
        if logbook_present:
            self.rt = RunTables(**{'web-service-url': 'https://pswww.slac.stanford.edu/ws-kerb'})
            try:
                self.table = self.rt.findUserTable(exper_name=self.parent.experimentName, table_name='Run summary')
            except:
                self.table = None

    def updateExpName(self, data):
        self.parent.experimentName = data
        self.parent.hasExperimentName = True
        self.parent.detInfoList = None
        self.resetVariables()

        if self.parent.doneInit and self.hasExpRunDetInfo():
            # Setup elog
            self.setupRunTable()
            self.getDatasource()
            self.setupRunDir()
            self.setupTotalEvents()
            self.printDetectorNames()
            # Update paths in all the panels
            self.updatePanels()
            self.setupPsocake()
            # Update hidden CrystFEL files
            self.updateHiddenCrystfelFiles(self.parent.facility)
            # Optionally use local calib directory
            self.setupLocalCalib()
            # Launch e-log crawler
            self.setupCrawler()
            # reset masks when run number is changed
            self.parent.mk.resetMasks()
            self.resetVariables()
            self.parent.pk.userUpdate = None

            self.setupDetGeom()
            if self.parent.facility == self.parent.facilityLCLS:
                self.parent.cx, self.parent.cy = self.parent.det.point_indexes(self.parent.evt, pxy_um=(0, 0))

            # update image
            self.getEventAndDisplay()
            # Indicate centre of detector
            self.parent.geom.drawCentre()

        #self.parent.detInfoList = None

        #self.resetVariables()
        # Setup elog
        #self.setupRunTable()

        #self.setupExperiment()
        if self.hasExpRunDetInfo(): print "starting setup"
    
        #self.parent.img.updateImage()
        if self.parent.args.v >= 1: print "Done updateExperimentName:", self.parent.experimentName
    
    def updateRunNumber(self, data):
        if data == 0:
            self.parent.runNumber = data
            self.parent.hasRunNumber = False
        else:
            self.parent.runNumber = data
            self.parent.hasRunNumber = True

            if self.parent.doneInit and self.hasExpRunInfo():
                self.getDatasource()




                self.printDetectorNames()

                if self.hasExpRunDetInfo():
                    # Setup elog
                    self.setupRunTable()
                    self.setupRunDir()
                    self.setupTotalEvents()

                    # Update paths in all the panels
                    self.updatePanels()
                    self.setupPsocake()
                    # Update hidden CrystFEL files
                    self.updateHiddenCrystfelFiles(self.parent.facility)
                    # Optionally use local calib directory
                    self.setupLocalCalib()
                    # Launch e-log crawler
                    self.setupCrawler()
                    # reset masks when run number is changed
                    self.parent.mk.resetMasks()
                    self.resetVariables()
                    self.parent.pk.userUpdate = None

                    self.setupDetGeom()
                    if self.parent.facility == self.parent.facilityLCLS:
                        self.parent.cx, self.parent.cy = self.parent.det.point_indexes(self.parent.evt, pxy_um=(0, 0))

                    # update image
                    self.getEventAndDisplay()
                    # Indicate centre of detector
                    self.parent.geom.drawCentre()

        if self.parent.args.v >= 1: print "Done updateRunNumber: ", self.parent.runNumber

    def updateDetInfo(self, data):
        if data == '':
            self.parent.detInfo = data
            self.parent.hasDetInfo = False
        else:
            self.parent.detInfo = data
            self.parent.hasDetInfo = True

        if self.parent.doneInit and self.hasExpRunDetInfo():
            # Setup elog
            self.setupRunTable()
            self.getDatasource()
            self.setupRunDir()
            self.setupTotalEvents()
            self.printDetectorNames()
            # Update paths in all the panels
            self.updatePanels()
            self.setupPsocake()
            # Update hidden CrystFEL files
            self.updateHiddenCrystfelFiles(self.parent.facility)
            # Optionally use local calib directory
            self.setupLocalCalib()
            # Launch e-log crawler
            self.setupCrawler()
            # reset masks when run number is changed
            self.parent.mk.resetMasks()
            self.resetVariables()
            self.parent.pk.userUpdate = None

            self.setupDetGeom()
            if self.parent.facility == self.parent.facilityLCLS:
                self.parent.cx, self.parent.cy = self.parent.det.point_indexes(self.parent.evt, pxy_um=(0, 0))

            # update image
            self.getEventAndDisplay()
            # Indicate centre of detector
            self.parent.geom.drawCentre()

        #if self.parent.doneInit:
        #    self.setupDetGeom()
        #    self.parent.img.updateImage()
        #    self.parent.geom.drawCentre()

        if self.parent.args.v >= 1: print "Done updateDetInfo: ", self.parent.detInfo

    def findEventFromTimestamp(self, secList, nsecList, fidList, sec, nsec, fid):
        eventNumber = (np.where(secList == sec)[0] & np.where(nsecList == nsec)[0] & np.where(fidList == fid)[0])[0]
        return eventNumber

    def convertTimestamp64(self, t):
        _sec = int(t) >> 32
        _nsec = int(t) & 0xFFFFFFFF
        return _sec, _nsec

    def convertSecNanosec(self, sec, nsec):
        _timestamp64 = int(sec << 32 | nsec)
        return _timestamp64

    def getEvt(self, evtNumber):
        if self.hasExpRunDetInfo():
            if self.parent.hasRunNumber:
                _evt = self.run.event(self.times[evtNumber])
                return _evt
            else:
                return None
        return None

    def getEventID(self, evt):
        if evt is not None:
            _evtid = evt.get(psana.EventId)
            _seconds = _evtid.time()[0]
            _nanoseconds = _evtid.time()[1]
            _fiducials = _evtid.fiducials()
            return _seconds, _nanoseconds, _fiducials

    def getEventAndDisplay(self):
        if self.parent.facility == self.parent.facilityLCLS:
           # update timestamps and fiducial
           self.parent.evt = self.getEvt(self.parent.eventNumber)
           if self.parent.evt is not None:
               sec, nanosec, fid = self.getEventID(self.parent.evt)
               self.eventSeconds = str(sec)
               self.eventNanoseconds = str(nanosec)
               self.eventFiducial = str(fid)
               self.updateEventID(self.eventSeconds, self.eventNanoseconds, self.eventFiducial)
               self.p.param(self.exp_grp, self.exp_evt_str).setValue(self.parent.eventNumber)
               self.parent.img.updateImage()
        elif self.parent.facility == self.parent.facilityPAL:
           self.p.param(self.exp_grp, self.exp_evt_str).setValue(self.parent.eventNumber)
           self.parent.img.updateImage()

    def updateEventNumber(self, data):
        self.parent.eventNumber = data
        self.parent.labeling.labels = None
        self.parent.pk.peaks = None
        self.parent.labeling.numLabelsFound = 0
        self.parent.pk.numPeaksFound = 0
        self.parent.pk.peaksMaxRes = 0

        if self.parent.eventNumber >= self.eventTotal and self.eventTotal > 0:
            self.parent.eventNumber = self.eventTotal - 1

        if self.parent.doneInit and self.hasExpRunDetInfo():
            self.getEventAndDisplay()

        self.parent.labeling.actionEventChange()

        if self.parent.index.showIndexedPeaks: self.parent.index.updateIndex()

        #if self.hasExpRunDetInfo():
        #    if self.parent.facility == self.parent.facilityLCLS:
        #        # update timestamps and fiducial
        #        self.parent.evt = self.getEvt(self.parent.eventNumber)
        #        if self.parent.evt is not None:
        #            sec, nanosec, fid = self.getEventID(self.parent.evt)
        #            self.eventSeconds = str(sec)
        #            self.eventNanoseconds = str(nanosec)
        #            self.eventFiducial = str(fid)
        #            self.updateEventID(self.eventSeconds, self.eventNanoseconds, self.eventFiducial)
        #            self.p.param(self.exp_grp, self.exp_evt_str).setValue(self.parent.eventNumber)
        #            self.parent.img.updateImage()
        #    elif self.parent.facility == self.parent.facilityPAL:
        #        self.p.param(self.exp_grp, self.exp_evt_str).setValue(self.parent.eventNumber)
        #        self.parent.img.updateImage()

        # update labels
        #if self.parent.args.mode == "all": if self.parent.evtLabels is not None: self.parent.evtLabels.refresh()

        if self.parent.args.v >= 1: print "Done updateEventNumber: ", self.parent.eventNumber

    def hasExpRunInfo(self):
        if self.parent.hasExperimentName and self.parent.hasRunNumber:
            # Check such a run exists
            if self.parent.facility == self.parent.facilityLCLS:
                _numData = len(glob.glob(self.parent.dir + '/' + self.parent.experimentName[:3] + '/' + \
                                         self.parent.experimentName + '/xtc/*-r' + str(self.parent.runNumber).zfill(4) + '-*.xtc'))
            elif self.parent.facility == self.parent.facilityPAL:
                _numData = self.getNumberOfEvents(self.parent.facilityPAL)
                if self.parent.args.v >= 1: print "hasExpRunInfo::_numData: ", _numData
            if _numData > 0:
                return True
            else:
                # reset run number
                if self.parent.runNumber > 0:
                    print "WARNING: No such run exists in ", self.parent.experimentName
                    self.parent.runNumber = 0
                    self.updateRunNumber(self.parent.runNumber)
                    self.p.param(self.exp_grp, self.exp_run_str).setValue(self.parent.runNumber)
                    return False
        return False
     
    def hasExpRunDetInfo(self):
        if self.parent.args.v >= 1: print "exp,run,det: ", self.parent.hasExperimentName, self.parent.hasRunNumber, self.parent.hasDetInfo
        if self.parent.hasExperimentName and self.parent.hasRunNumber and self.parent.hasDetInfo:
            if self.parent.args.v >= 1: print "hasExpRunDetInfo: True ", self.parent.runNumber
            return True
        else:
            if self.parent.args.v >= 1: print "hasExpRunDetInfo: False ", self.parent.runNumber
            return False

    def setupPsocake(self):
        self.loggerFile = self.parent.elogDir + '/' + self.parent.experimentName + '.data'
        if os.path.exists(self.parent.elogDir) is False:
            try:
                os.makedirs(self.parent.elogDir, 0774)
                # setup permissions
                process = subprocess.Popen('chgrp -R ' + self.parent.experimentName + ' ' + self.parent.elogDir, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, shell=True)
                out, err = process.communicate()
                process = subprocess.Popen('chmod -R u+rwx,g+rws,o+rx ' + self.parent.elogDir, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, shell=True)
                out, err = process.communicate()

                # create logger
                with open(self.loggerFile, "w") as myfile:
                    if self.parent.args.outDir is None:
                        myfile.write(self.parent.username)
                    else:
                        myfile.write("NOONE")
            except:
                print "No write access: ", self.parent.elogDir
        else:
            # check if I'm a logger
            if os.path.exists(self.loggerFile):
                with open(self.loggerFile, "r") as myfile:
                    content = myfile.readlines()
                    try:
                        if content[0].strip() == self.parent.username:
                            if logbook_present and self.table is not None:
                                self.logger = True
                            if logbook_present and self.parent.args.v >= 1: print "I'm an elogger"
                        else:
                            self.logger = False
                            if self.parent.args.v >= 1: print "I'm not an elogger"
                    except:
                        self.logger = False
            else:
                self.logger = False
        # Make run folder
        try:
            if os.path.exists(self.parent.psocakeRunDir) is False:
                if self.parent.args.outDir is None:
                    os.makedirs(self.parent.psocakeRunDir, 0774)
                else: # don't let groups and others access to this folder
                    os.makedirs(self.parent.psocakeRunDir, 0700)
                    process = subprocess.Popen('chmod -R ' + self.parent.psocakeRunDir,
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE, shell=True)
                    out, err = process.communicate()
            self.parent.writeAccess = True
        except:
            print "No write access: ", self.parent.psocakeRunDir
            self.parent.writeAccess = False

    # Launch crawler
    crawlerThread = []
    crawlerThreadCounter = 0
    
    def launchCrawler(self):
        self.crawlerThread.append(LogbookCrawler.LogbookCrawler(self.parent))  # send parent parameters with self
        self.crawlerThread[self.crawlerThreadCounter].updateLogbook(self.parent.experimentName, self.parent.psocakeDir)
        self.crawlerThreadCounter += 1

    def getDetectorAlias(self, srcOrAlias):
        for i in self.parent.detInfoList:
            src, alias, _ = i
            if srcOrAlias.lower() == src.lower() or srcOrAlias.lower() == alias.lower():
                return alias

    def updateHiddenCrystfelFiles(self, arg):
        if arg == self.parent.facilityLCLS:
            if 'cspad' in self.parent.detInfo.lower() or \
               ('rayonix' in self.parent.detInfo.lower() and 'mfx' in self.parent.experimentName) or \
               ('rayonix' in self.parent.detInfo.lower() and 'xpp' in self.parent.experimentName):
                self.parent.index.hiddenCXI = self.parent.psocakeRunDir + '/.temp.cxi'
                self.parent.index.hiddenCrystfelStream = self.parent.psocakeRunDir + '/.temp.stream'
                self.parent.index.hiddenCrystfelList = self.parent.psocakeRunDir + '/.temp.lst'
            else:
                if self.parent.args.v >= 1: print "updateHiddenCrystfelFiles not implemented"
                self.parent.index.hiddenCXI = None
        elif arg == self.parent.facilityPAL:
            self.parent.index.hiddenCXI = self.parent.psocakeRunDir + '/.temp.cxi'
            self.parent.index.hiddenCrystfelStream = self.parent.psocakeRunDir + '/.temp.stream'
            self.parent.index.hiddenCrystfelList = self.parent.psocakeRunDir + '/.temp.lst'

    def findGeometry(self):
        geomFiles = glob.glob(self.parent.rootDir + '/calib/' + self.parent.detInfo + '/geometry/*.geom')
        geomRun = 0
        minDiff = 0
        for f in geomFiles:
            startRun = int(f.split('/')[-1].split('-')[0])
            diff = self.parent.runNumber - startRun
            if diff < minDiff and diff >= 0:
                minDiff = diff
                geomRun = startRun
        return str(geomRun) + '-end.geom'

    def readCrystfelGeometry(self, arg):
        if arg == self.parent.facilityPAL:
            _geomFile = self.parent.geom.getClosestGeom()
            with open(_geomFile, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if 'clen' in line:
                    if self.parent.detectorDistance == 0.0: # don't read from crystfel geometry if we have a detector distance
                        self.parent.detectorDistance = float(line.split('=')[-1])
                elif 'photon_energy' in line:
                    self.parent.photonEnergy = float(line.split('=')[-1])
                elif 'p0/res' in line:
                    self.parent.pixelSize = 1./float(line.split('=')[-1])
                elif 'p0/corner_x' in line:
                    self.parent.cy = -1 * float(line.split('=')[-1])
                elif 'p0/corner_y' in line:
                    self.parent.cx = float(line.split('=')[-1])

    def updateDetectorDistance(self, arg):
        if arg == self.parent.facilityLCLS:
            if 'cspad' in self.parent.detInfo.lower() and 'cxi' in self.parent.experimentName:
                if self.parent.detectorDistance < 0.01:
                    self.parent.detectorDistance = np.mean(self.parent.det.coords_z(self.parent.evt)) * 1e-6 # metres
                    self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_detectorDistance_str).setValue(self.parent.detectorDistance*1e3) # mm
                self.parent.coffset = self.parent.detectorDistance - self.parent.clen
                self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_clen_str).setValue(self.parent.clen)
            elif ('rayonix' in self.parent.detInfo.lower() and 'mfx' in self.parent.experimentName) or \
                 ('cspad' in self.parent.detInfo.lower() and 'mfx' in self.parent.experimentName):
                if self.parent.detectorDistance < 0.01:
                    try:
                        self.parent.detectorDistance = np.mean(self.parent.det.coords_z(self.parent.evt)) * 1e-6 # metres
                        self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_detectorDistance_str).setValue(self.parent.detectorDistance*1e3) # mm
                    except:
                        self.parent.detectorDistance = 0
                        print "######################################################"
                        print "Detector distance not found. Check your geometry file."
                        print "######################################################"
                self.parent.coffset = self.parent.detectorDistance - self.parent.clen
                self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_clen_str).setValue(self.parent.clen)
            elif 'rayonix' in self.parent.detInfo.lower() and 'xpp' in self.parent.experimentName:
                if self.parent.detectorDistance < 0.01:
                    self.parent.detectorDistance = np.mean(self.parent.det.coords_z(self.parent.evt)) * 1e-6 # metres
                    self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_detectorDistance_str).setValue(self.parent.detectorDistance*1e3) # mm
                self.parent.coffset = self.parent.detectorDistance - self.parent.clen
                self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_clen_str).setValue(
                    self.parent.clen)
            elif 'camp.0:pnccd.0' in self.parent.detInfo.lower() or 'pnccdfront' in self.parent.detInfo.lower():
                if self.parent.detectorDistance < 0.01:
                    self.parent.detectorDistance = np.mean(self.parent.det.coords_z(self.parent.evt)) * 1e-6 # metres
                    self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_detectorDistance_str).setValue(self.parent.detectorDistance*1e3) # mm
                self.parent.coffset = self.parent.detectorDistance - self.parent.clen
                self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_clen_str).setValue(
                    self.parent.clen)
            elif 'camp.0:pnccd.1' in self.parent.detInfo.lower() or 'pnccdback' in self.parent.detInfo.lower():
                if self.parent.detectorDistance < 0.01:
                    self.parent.detectorDistance = np.mean(self.parent.det.coords_z(self.parent.evt)) * 1e-6 # metres
                    self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_detectorDistance_str).setValue(self.parent.detectorDistance*1e3) # mm
                self.parent.coffset = self.parent.detectorDistance - self.parent.clen
                self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_clen_str).setValue(
                    self.parent.clen)
            else:
                print "#############################################################"
                print "updateDetectorDistance: not implemented for this detector yet"
                print "#############################################################"
        elif arg == self.parent.facilityPAL:
            self.parent.coffset = self.parent.detectorDistance - self.parent.clen
            self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_detectorDistance_str).setValue(
                self.parent.detectorDistance * 1e3)  # mm
            self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_clen_str).setValue(self.parent.clen)
        if self.parent.args.v >= 1: print "updateDetectorDistance::detectorDistance (m), self.clen (m), self.coffset (m): ", \
            self.parent.detectorDistance, self.parent.clen, self.parent.coffset

    def updatePixelSize(self, arg):
        if arg == self.parent.facilityLCLS:
            if 'cspad' in self.parent.detInfo.lower():  # TODO: increase pixel size list: epix, rayonix
                self.parent.pixelSize = 110e-6  # metres
            elif 'pnccd' in self.parent.detInfo.lower():
                self.parent.pixelSize = 75e-6  # metres
            elif 'rayonix' in self.parent.detInfo.lower():
                self.parent.pixelSize = 89e-6  # metres
            # Update geometry panel
            self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_pixelSize_str).setValue(
                    self.parent.pixelSize)  # pixel size
        elif arg == self.parent.facilityPAL:
            # Update geometry panel
            self.parent.geom.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_pixelSize_str).setValue(
                    self.parent.pixelSize)  # pixel size

    def updatePhotonEnergy(self, arg):
        if arg == self.parent.facilityLCLS:
            self.parent.ebeam = self.parent.evt.get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
            self.parent.photonEnergy = self.parent.epics.value('SIOC:SYS0:ML00:AO541')
            if self.parent.photonEnergy is None and self.parent.ebeam:
                self.parent.photonEnergy = self.parent.ebeam.ebeamPhotonEnergy()
            else:
                try:
                    wavelength = self.parent.epics.value('SIOC:SYS0:ML00:AO192')  # nanometre
                    h = 6.626070e-34  # J.m
                    c = 2.99792458e8  # m/s
                    joulesPerEv = 1.602176621e-19  # J/eV
                    self.parent.photonEnergy = (h / joulesPerEv * c) / (wavelength * 1e-9)
                except:
                    self.parent.photonEnergy = 0.0
            self.parent.geom.p1.param(self.parent.geom.geom_grp,
                                 self.parent.geom.geom_photonEnergy_str).setValue(self.parent.photonEnergy)
        elif arg == self.parent.facilityPAL:
            self.parent.geom.p1.param(self.parent.geom.geom_grp,
                                      self.parent.geom.geom_photonEnergy_str).setValue(self.parent.photonEnergy)

    def setClen(self):
        if self.parent.facility == self.parent.facilityLCLS:
            if 'cspad2x2' in self.parent.detInfo.lower():
                self.parent.clen = 0.0 # the cspad2x2 detector is not motorized
            elif 'cspad' in self.parent.detInfo.lower() and 'cxi' in self.parent.experimentName:
                try:
                    self.parent.clenEpics = str(self.parent.detAlias) + '_z'
                    self.readEpicsClen()
                except:
                    if 'ds1' in self.parent.detInfo.lower():
                        self.parent.clenEpics = str('CXI:DS1:MMS:06.RBV')
                        self.readEpicsClen()
                    elif 'ds2' in self.parent.detInfo.lower():
                        self.parent.clenEpics = str('CXI:DS2:MMS:06.RBV')
                        self.readEpicsClen()

                    else:
                        print "Couldn't handle detector clen. Try using the full detector name."
                        exit()
            elif ('rayonix' in self.parent.detInfo.lower() and 'mfx' in self.parent.experimentName) or \
                 ('cspad' in self.parent.detInfo.lower() and 'mfx' in self.parent.experimentName):
                self.parent.clenEpics =  str('MFX:DET:MMS:04.RBV') #'Rayonix_z'
                try:
                    self.readEpicsClen()
                except:
                    print "ERROR: No such epics variable, ", self.parent.clenEpics
                    print "ERROR: setting clen to 0.0 metre"
                    self.parent.clen = 0.0 # metres
            elif 'rayonix' in self.parent.detInfo.lower() and 'xpp' in self.parent.experimentName:
                self.parent.clenEpics = 'XPP:ROB:POS:Z'
                try:
                    self.readEpicsClen()
                except:
                    print "ERROR: No such epics variable, ", self.parent.clenEpics
                    print "ERROR: setting clen to 0.0 metre"
                    self.parent.clen = 0.0  # metres
            elif 'camp.0:pnccd.0' in self.parent.detInfo.lower() or 'pnccdfront' in self.parent.detInfo.lower():
                self.parent.clenEpics = 'AMO:LMP:MMS:10.RBV' # There's no alias
                try:
                    self.readEpicsClen()
                except:
                    print "ERROR: No such epics variable, ", self.parent.clenEpics
                    print "ERROR: setting clen to 0.0 metre"
                    self.parent.clen = 0.0 # metres
            elif 'camp.0:pnccd.1' in self.parent.detInfo.lower() or 'pnccdback' in self.parent.detInfo.lower():
                self.parent.clen = 0.0 # the back pnccd is not motorized at AMO
            else:
                print "#########################################"
                print "Not implemented yet clen: ", self.parent.detInfo
                print "#########################################"
        elif self.parent.facility == self.parent.facilityPAL:
            self.parent.clen = self.parent.detectorDistance # metres

    def readEpicsClen(self):
        numEvt = len(self.times)
        if numEvt > 120: numEvt = 120
        for i in range(numEvt):  # assume at least 1 second run time
            evt = self.run.event(self.times[i])
            self.parent.clen = self.parent.epics.value(self.parent.clenEpics) / 1000. # metres
            if i == 0:
                _temp = self.parent.clen
            elif i > 0:
                if abs(_temp - self.parent.clen) >= 0.01:
                    break
                _temp = self.parent.clen
        if self.parent.args.v >= 1: print "Best guess at clen: ", self.parent.clen

    def setupRunDir(self):
        # Set up psocake directory in scratch
        if self.parent.args.outDir is None:
            self.parent.rootDir = self.parent.dir + '/' + self.parent.experimentName[:3] + '/' + self.parent.experimentName
            self.parent.elogDir = self.parent.rootDir + '/scratch/psocake'
            self.parent.psocakeDir = self.parent.rootDir + '/scratch/' + self.parent.username + '/psocake'
        else:
            self.parent.rootDir = self.parent.args.outDir
            self.parent.elogDir = self.parent.rootDir + '/psocake'
            self.parent.psocakeDir = self.parent.rootDir + '/' + self.parent.experimentName + '/' + self.parent.username + '/psocake'
        self.parent.psocakeRunDir = self.parent.psocakeDir + '/r' + str(self.parent.runNumber).zfill(4)

    def updatePanels(self):
        # Update peak finder outdir and run number
        self.parent.pk.p3.param(self.parent.pk.hitParam_grp, self.parent.pk.hitParam_outDir_str).setValue(
            self.parent.psocakeDir)
        self.parent.pk.p3.param(self.parent.pk.hitParam_grp, self.parent.pk.hitParam_runs_str).setValue(
            self.parent.runNumber)
        # Update powder outdir and run number
        self.parent.mk.p6.param(self.parent.mk.powder_grp, self.parent.mk.powder_outDir_str).setValue(
            self.parent.psocakeDir)
        self.parent.mk.p6.param(self.parent.mk.powder_grp, self.parent.mk.powder_runs_str).setValue(
            self.parent.runNumber)
        # Update hit finding outdir, run number
        self.parent.hf.p8.param(self.parent.hf.spiParam_grp, self.parent.hf.spiParam_outDir_str).setValue(
            self.parent.psocakeDir)
        self.parent.hf.p8.param(self.parent.hf.spiParam_grp, self.parent.hf.spiParam_runs_str).setValue(
            self.parent.runNumber)
        # Update indexing outdir, run number
        self.parent.index.p9.param(self.parent.index.index_grp, self.parent.index.index_geom_str).setValue(
            self.parent.psocakeRunDir+'/.temp.geom')
        self.parent.index.p9.param(self.parent.index.launch_grp, self.parent.index.outDir_str).setValue(
            self.parent.psocakeDir)
        self.parent.index.p9.param(self.parent.index.launch_grp, self.parent.index.runs_str).setValue(
            self.parent.runNumber)
        # Update quantifier filename
        fname = self.parent.psocakeRunDir + '/' + self.parent.experimentName + '_' + str(self.parent.runNumber).zfill(4) + '.cxi'
        if self.parent.args.mode == 'sfx':
            dsetname = '/entry_1/result_1/nPeaksAll'
        elif self.parent.args.mode == 'spi':
            dsetname = '/entry_1/result_1/nHitsAll'
        else:
            dsetname = '/entry_1/result_1/'
        self.parent.small.pSmall.param(self.parent.small.quantifier_grp,
                                       self.parent.small.quantifier_filename_str).setValue(fname)
        self.parent.small.pSmall.param(self.parent.small.quantifier_grp,
                                       self.parent.small.quantifier_dataset_str).setValue(dsetname)

    def setupLocalCalib(self):
        if self.parent.facility == self.parent.facilityLCLS:
            if self.parent.args.localCalib:
                if os.path.exists('calib'):
                    psana.setOption('psana.calib-dir', './calib')
                    if self.parent.args.v >= 1: print "Using local calib directory"
                else:
                    print "./calib directory does not exist in the present working directory"

    def getDatasource(self):
        if self.parent.facility == self.parent.facilityLCLS:
            try:
                access = 'exp=' + str(self.parent.experimentName) + ':run=' + str(self.parent.runNumber) + ':idx'
                if 'ffb' in self.parent.access: access += ':dir=/reg/d/ffb/' + self.parent.experimentName[:3] + \
                                                          '/' + self.parent.experimentName + '/xtc'
                self.ds = psana.DataSource(access)
                self.run = self.ds.runs().next()
                self.times = self.run.times()
                self.env = self.ds.env()
            except:
                print "############# No such datasource exists ###############"

    def setupTotalEvents(self):
        if self.parent.facility == self.parent.facilityLCLS:
            self.eventTotal = len(self.times)
            self.parent.labeling.initEventsToDo()
            self.parent.stack.spinBox.setMaximum(self.eventTotal - self.parent.stack.stackSize)
            self.p.param(self.exp_grp, self.exp_evt_str).setLimits((0, self.eventTotal - 1))
            self.p.param(self.exp_grp, self.exp_evt_str, self.exp_numEvents_str).setValue(self.eventTotal)
        elif self.parent.facility == self.parent.facilityPAL:
            self.eventTotal = self.getNumberOfEvents(self.parent.facilityPAL)
            self.parent.stack.spinBox.setMaximum(self.eventTotal - self.parent.stack.stackSize)
            self.p.param(self.exp_grp, self.exp_evt_str).setLimits((0, self.eventTotal - 1))
            self.p.param(self.exp_grp, self.exp_evt_str, self.exp_numEvents_str).setValue(self.eventTotal)

    def printDetectorNames(self):
        # Print list of available detectors
        if self.parent.detInfoList is None:
            if self.parent.facility == self.parent.facilityLCLS:
                self.parent.evt = self.run.event(self.times[-1])
                myAreaDetectors = []
                self.parent.detnames = psana.DetNames()
                for k in self.parent.detnames:
                    try:
                        if Detector.PyDetector.dettype(str(k[0]), self.env) == Detector.AreaDetector.AreaDetector:
                            myAreaDetectors.append(k)
                    except ValueError:
                        continue
                self.parent.detInfoList = list(set(myAreaDetectors))
                print "#######################################"
                print "# Available area detectors: "
                for k in self.parent.detInfoList:
                    print "#", k
                print "#######################################"
            elif self.parent.facility == self.parent.facilityPAL:
                self.parent.detInfoList = ['MX225-HS']  # PAL does not provide detectors available, so hard-code
                print "#######################################"
                print "# Available area detectors: "
                for k in self.parent.detInfoList:
                    print "#", k
                print "#######################################"

    def setupCrawler(self):
        if self.parent.facility == self.parent.facilityLCLS:
            if self.logger and self.crawlerRunning == False:
                if self.parent.args.v >= 1: print "Launching crawler"
                self.launchCrawler()
                self.crawlerRunning = True

    def setupDetGeom(self):
        if self.parent.facility == self.parent.facilityLCLS:
            self.parent.det = psana.Detector(str(self.parent.detInfo), self.env)
            self.parent.det.do_reshape_2d_to_3d(flag=True)
            self.parent.detAlias = self.getDetectorAlias(str(self.parent.detInfo))
            self.parent.epics = self.ds.env().epicsStore()
            self.setClen()

            # Some detectors do not read out at 120 Hz. So need to loop over events to guarantee a valid detector image.
            if self.parent.evt is None:
                self.parent.evt = self.run.event(self.times[0])
            self.detGuaranteed = self.parent.det.calib(self.parent.evt)
            if self.detGuaranteed is None:  # image isn't present for this event
                print "No image in this event. Searching for an event..."
                for i in np.arange(len(self.times)):
                    evt = self.run.event(self.times[i])
                    self.detGuaranteed = self.parent.det.calib(evt)
                    if self.detGuaranteed is not None:
                        print "Found an event with image: ", i
                        break

            # detector distance
            self.updateDetectorDistance(self.parent.facility)
            # pixel size
            self.updatePixelSize(self.parent.facility)
            # photon energy
            self.updatePhotonEnergy(self.parent.facility)

            # Setup pixel indices
            if self.detGuaranteed is not None:
                self.parent.pixelInd = np.reshape(np.arange(self.detGuaranteed.size) + 1, self.detGuaranteed.shape)
                self.parent.pixelIndAssem = self.parent.img.getAssembledImage(self.parent.facilityLCLS, self.parent.pixelInd)
                self.parent.pixelIndAssem -= 1  # First pixel is 0
                # Get detector shape
                self.detGuaranteedData = self.parent.det.image(self.parent.evt, self.detGuaranteed)

            # Write a temporary geom file
            self.parent.geom.deployCrystfelGeometry(self.parent.facility)
            self.parent.geom.writeCrystfelGeom(self.parent.facility)

            self.parent.img.setupRadialBackground()
            self.parent.img.updatePolarizationFactor()

        elif self.parent.facility == self.parent.facilityPAL:
            # read geometry
            self.readCrystfelGeometry(self.parent.facility)
            # detector distance
            self.updateDetectorDistance(self.parent.facility)
            # pixel size
            self.updatePixelSize(self.parent.facility)
            # photon energy
            self.updatePhotonEnergy(self.parent.facility)
            self.detGuaranteed, self.detGuaranteedData = self.parent.img.getDetImage(0)
            # Setup pixel indices
            if self.detGuaranteed is not None:
                self.parent.pixelInd = np.reshape(np.arange(self.detGuaranteed.size) + 1, self.detGuaranteed.shape)
                self.parent.pixelIndAssem = self.parent.img.getAssembledImage(self.parent.facility,
                                                                              self.parent.pixelInd)
                self.parent.pixelIndAssem -= 1  # First pixel is 0

            # Write a temporary geom file
            # self.parent.geom.deployCrystfelGeometry(self.parent.facility)
            self.parent.geom.writeCrystfelGeom(self.parent.facility)

            self.parent.img.setupRadialBackground()
            self.parent.img.updatePolarizationFactor()

    def setupExperiment(self):
        if self.parent.args.v >= 1: print "Doing setupExperiment"

        if self.hasExpRunInfo() and not self.hasExpRunDetInfo():
            # Setup elog
            self.setupRunTable()
            self.getDatasource()
            self.setupRunDir()
            self.setupTotalEvents()
            self.printDetectorNames()
            # Update paths in all the panels
            self.exp.updatePanels()
            # setup psocake-relate paths
            self.setupPsocake()
            # Update hidden CrystFEL files
            self.updateHiddenCrystfelFiles(self.parent.facility)
            # Optionally use local calib directory
            self.setupLocalCalib()
            # Launch e-log crawler
            self.setupCrawler()
        if self.hasExpRunDetInfo():
            # Set up detector and geometry
            self.setupDetGeom()
            if self.parent.facility == self.parent.facilityLCLS:
                self.parent.cx, self.parent.cy = self.parent.det.point_indexes(self.parent.evt, pxy_um=(0, 0))

            # update image
            self.getEventAndDisplay()
            # Indicate centre of detector
            self.parent.geom.drawCentre()

        if self.parent.args.v >= 1: print "Done setupExperiment"

    def getNumberOfEvents(self, facility):
        try:
            if facility == self.parent.facilityPAL:
                _path = self.parent.dir + '/' + self.parent.experimentName[:3] + '/' + self.parent.experimentName + \
                        '/data/r' + str(self.parent.runNumber).zfill(4) + '/*.h5'
                _data = glob.glob(_path)
            if self.parent.args.v >= 1: print "Done getNumberOfEvents: ", _path, len(_data)
            return len(_data)
        except:
            return 0

    def updateLogscale(self, data):
        self.logscaleOn = data
        if self.hasExpRunDetInfo():
            self.parent.firstUpdate = True  # clicking logscale resets plot colorscale
            self.parent.img.updateImage()
        if self.parent.args.v >= 1: print "Done updateLogscale: ", self.logscaleOn

    def updateAduPerPhoton(self, data):
        self.aduPerPhoton = data
        if self.hasExpRunDetInfo() is True and self.image_property == self.disp_photons:
            #self.parent.firstUpdate = True  # clicking logscale resets plot colorscale
            self.parent.img.updateImage()
        if self.parent.args.v >= 1: print "Done updateAduPerPhoton: ", self.aduPerPhoton

    def updateMedianFilter(self, data):
        self.medianFilterRank = data
        if self.hasExpRunDetInfo() is True and self.image_property == self.disp_medianCorrection:
            # self.parent.firstUpdate = True  # clicking logscale resets plot colorscale
            self.parent.img.updateImage()
        if self.parent.args.v >= 1: print "Done updateMedianFilter: ", self.medianFilterRank

    def updateImageProperty(self, data):
        self.image_property = data
        self.parent.img.updateImage()
        if self.parent.args.v >= 1: print "Done updateImageProperty: ", self.image_property

    def updateFriedel(self, data):
        self.applyFriedel = data
        self.parent.img.updateImage()
        if self.parent.args.v >= 1: print "Done updateFriedel: ", self.applyFriedel
    
    def updateCommonModeParam(self, data, ind):
        self.commonModeParams[ind] = data
        self.updateCommonMode(self.applyCommonMode)
        if self.parent.args.v >= 1: print "Done updateCommonModeParam: ", self.commonModeParams

    def updateCommonMode(self, data):
        self.applyCommonMode = data
        if self.applyCommonMode:
            self.commonMode = self.checkCommonMode(self.commonModeParams)
        if self.hasExpRunDetInfo():
            if self.parent.args.v >= 1: print "%%% Redraw image with new common mode: ", self.commonMode
            self.setupExperiment()
            self.parent.img.updateImage()
        if self.parent.args.v >= 1: print "Done updateCommonMode: ", self.commonMode

    def checkCommonMode(self, _commonMode):
        # TODO: cspad2x2 can only use algorithms 1 and 5
        _alg = int(_commonMode[0])
        if _alg >= 1 and _alg <= 4:
            _param1 = int(_commonMode[1])
            _param2 = int(_commonMode[2])
            _param3 = int(_commonMode[3])
            return (_alg,_param1,_param2,_param3)
        elif _alg == 5:
            _param1 = int(_commonMode[1])
            return (_alg,_param1)
        else:
            print "Undefined common mode algorithm"
            return None

    def updateEventID(self, sec, nanosec, fid):
        if self.parent.args.v >= 1: print "eventID: ", sec, nanosec, fid
        self.p.param(self.exp_grp, self.exp_evt_str, self.exp_second_str).setValue(self.eventSeconds)
        self.p.param(self.exp_grp, self.exp_evt_str, self.exp_nanosecond_str).setValue(self.eventNanoseconds)
        self.p.param(self.exp_grp, self.exp_evt_str, self.exp_fiducial_str).setValue(self.eventFiducial)

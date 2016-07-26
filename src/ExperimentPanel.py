import numpy as np
from pyqtgraph.Qt import QtCore
import subprocess
import pandas as pd
import h5py, os
import psana
import PSCalib.GlobalUtils as gu
from LogBook.runtables import RunTables
import LogbookCrawler
import Detector.PyDetector

class ExperimentInfo(object):
    def __init__(self, parent = None):
        self.parent = parent

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
        #self.disp_aduThresh_str = 'ADU threshold'
        self.disp_commonModeOverride_str = 'Common mode (override)'
        self.disp_overrideCommonMode_str = 'Apply common mode (override)'
        self.disp_commonModeParam0_str = 'parameters 0'
        self.disp_commonModeParam1_str = 'parameters 1'
        self.disp_commonModeParam2_str = 'parameters 2'
        self.disp_commonModeParam3_str = 'parameters 3'

        self.logscaleOn = False
        self.image_property = 1
        #self.aduThresh = -100.
        self.applyCommonMode = False
        self.commonModeParams = np.array([0,0,0,0])

        # e-log
        self.logger = False
        self.loggerFile = None
        self.crawlerRunning = False
        self.username = None
        self.rt = None
        self.table = None

        self.params = [
            {'name': self.exp_grp, 'type': 'group', 'children': [
                {'name': self.exp_name_str, 'type': 'str', 'value': self.parent.experimentName,
                 'tip': "Experiment name, .e.g. cxic0415"},
                {'name': self.exp_run_str, 'type': 'int', 'value': self.parent.runNumber, 'tip': "Run number, e.g. 15"},
                {'name': self.exp_detInfo_str, 'type': 'str', 'value': self.parent.detInfo,
                 'tip': "Detector ID. Look at the terminal for available area detectors, e.g. DscCsPad"},
                {'name': self.exp_evt_str, 'type': 'int', 'value': self.parent.eventNumber, 'tip': "Event number, first event is 0",
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
                {'name': self.disp_image_str, 'type': 'list', 'values': {self.disp_gainMask_str: 17,
                                                                         self.disp_coordy_str: 16,
                                                                         self.disp_coordx_str: 15,
                                                                         self.disp_col_str: 14,
                                                                         self.disp_row_str: 13,
                                                                         self.disp_seg_str: 12,
                                                                         self.disp_quad_str: 11,
                                                                         self.disp_gain_str: 10,
                                                                         self.disp_commonMode_str: 9,
                                                                         self.disp_rms_str: 8,
                                                                         self.disp_status_str: 7,
                                                                         self.disp_pedestal_str: 6,
                                                                         self.disp_photons_str: 5,
                                                                         self.disp_raw_str: 4,
                                                                         self.disp_pedestalCorrected_str: 3,
                                                                         self.disp_commonModeCorrected_str: 2,
                                                                         self.disp_adu_str: 1},
                 'value': self.image_property, 'tip': "Choose image property to display"},
                #{'name': self.disp_aduThresh_str, 'type': 'float', 'value': self.aduThresh,
                # 'tip': "Only display ADUs above this threshold"},
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

    ##############################
    # Mandatory parameter update #
    ##############################
    def paramUpdate(self, path, change, data):
        if path[0] == self.exp_grp:
            if path[1] == self.exp_name_str:
                self.updateExpName(data)
                if self.parent.pk.showPeaks:
                    self.parent.pk.updateClassification()
            elif path[1] == self.exp_run_str:
                self.updateRunNumber(data)
                if self.parent.pk.showPeaks:
                    self.parent.pk.updateClassification()
            elif path[1] == self.exp_detInfo_str:
                self.updateDetInfo(data)
                if self.parent.pk.showPeaks:
                    self.parent.pk.updateClassification()
            elif path[1] == self.exp_evt_str and len(path) == 2 and change is 'value':
                self.updateEventNumber(data)
                if self.parent.pk.showPeaks:
                    self.parent.pk.updateClassification()
        elif path[0] == self.disp_grp:
            if path[1] == self.disp_log_str:
                self.updateLogscale(data)
            elif path[1] == self.disp_image_str:
                self.updateImageProperty(data)
            #elif path[1] == self.disp_aduThresh_str:
            #    self.updateAduThreshold(data)
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

    ###################################
    ###### Experiment Parameters ######
    ###################################
    
    def resetVariables(self):
        self.secList = None
        self.nsecList = None
        self.fidList = None
        
    def updateExpName(self, data):
        self.experimentName = data
        self.hasExperimentName = True
        self.parent.detInfoList = None
        self.resetVariables()
    
        # Setup elog
        self.rt = RunTables(**{'web-service-url': 'https://pswww.slac.stanford.edu/ws-kerb'})
        try:
            self.table = self.rt.findUserTable(exper_name=self.parent.experimentName, table_name='Run summary')
        except:
            print "Ooops. You need a kerberos ticket. Type: kinit"
            exit()
    
        self.setupExperiment()
    
        self.parent.updateImage()
        if self.parent.args.v >= 1: print "Done updateExperimentName:", self.parent.experimentName
    
    def updateRunNumber(self, data):
        if data == 0:
            self.parent.runNumber = data
            self.hasRunNumber = False
        else:
            self.parent.runNumber = data
            self.hasRunNumber = True
            self.parent.detInfoList = None
            self.setupExperiment()
            self.parent.mk.resetMasks()
            self.resetVariables()
            self.parent.updateImage()
        if self.parent.args.v >= 1: print "Done updateRunNumber: ", self.parent.runNumber

    def updateDetInfo(self, data):
        if self.parent.hasDetInfo is False or self.parent.detInfo is not data:
            self.parent.mk.resetMasks()
            self.parent.calib = None
            self.parent.data = None
            self.parent.firstUpdate = True
    
        self.parent.detInfo = data
        if data == 'DscCsPad' or data == 'DsdCsPad' or data == 'DsaCsPad':
            self.parent.isCspad = True
    
        self.hasDetInfo = True
        self.setupExperiment()
        self.parent.updateImage()
        if self.parent.args.v >= 1: print "Done updateDetInfo: ", self.parent.detInfo
    
    
    def findEventFromTimestamp(self, secList, nsecList, fidList, sec, nsec, fid):
        eventNumber = (np.where(secList == sec)[0] & np.where(nsecList == nsec)[0] & np.where(fidList == fid)[0])[0]
        return eventNumber

    def convertTimestamp64(self, t):
        _sec = int(t) >> 32
        _nsec = int(t) & 0xFFFFFFFF
        return _sec, _nsec

    def convertSecNanosec(self, sec, nsec):
        _timestamp64 = int(sec >> 32 | nsec)
        return _timestamp64

    def getEvt(self, evtNumber):
        if self.hasRunNumber:
            _evt = self.run.event(self.times[evtNumber])
            return _evt
        else:
            return None

    def getEventID(self, evt):
        if evt is not None:
            _evtid = evt.get(psana.EventId)
            _seconds = _evtid.time()[0]
            _nanoseconds = _evtid.time()[1]
            _fiducials = _evtid.fiducials()
            return _seconds, _nanoseconds, _fiducials

    def updateEventNumber(self, data):
        self.parent.eventNumber = data
        if self.parent.eventNumber >= self.eventTotal:
            self.parent.eventNumber = self.eventTotal - 1
        # update timestamps and fiducial
        self.parent.evt = self.getEvt(self.parent.eventNumber)
        if self.parent.evt is not None:
            sec, nanosec, fid = self.getEventID(self.parent.evt)
            self.eventSeconds = str(sec)
            self.eventNanoseconds = str(nanosec)
            self.eventFiducial = str(fid)
            self.updateEventID(self.eventSeconds, self.eventNanoseconds, self.eventFiducial)
            self.parent.p.param(self.exp_grp, self.exp_evt_str).setValue(self.parent.eventNumber)
            self.parent.updateImage()
        # update labels
        if self.parent.args.mode == "all":
            if self.evtLabels is not None: self.evtLabels.refresh()
        if self.parent.args.v >= 1: print "Done updateEventNumber: ", self.parent.eventNumber

    def hasExpRunInfo(self):
        if self.parent.hasExperimentName and self.parent.hasRunNumber:
            # Check such a run exists
            import glob
            xtcs = glob.glob('/reg/d/psdm/' + self.parent.experimentName[0:3] + '/' + self.parent.experimentName + '/xtc/*-r' + str(
                self.parent.runNumber).zfill(4) + '-*.xtc')
            if len(xtcs) > 0:
                return True
            else:
                # reset run number
                if self.parent.runNumber > 0:
                    print "No such run exists in: ", self.parent.experimentName
                    self.parent.runNumber = 0
                    self.updateRunNumber(self.parent.runNumber)
                    self.parent.p.param(self.exp_grp, self.exp_run_str).setValue(self.parent.runNumber)
                    return False
        return False
     
    def hasExpRunDetInfo(self):
        if self.parent.hasExperimentName and self.parent.hasRunNumber and self.parent.hasDetInfo:
            if self.parent.args.v >= 1: print "hasExpRunDetInfo: True ", self.parent.runNumber
            return True
        else:
            if self.parent.args.v >= 1: print "hasExpRunDetInfo: False ", self.parent.runNumber
            return False 
    
    def getUsername(self):
        process = subprocess.Popen('whoami', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        self.username = out.strip()
    
    def setupPsocake(self):
        self.loggerFile = self.parent.elogDir + '/logger.data'
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
                        myfile.write(self.username)
                    else:
                        myfile.write("NOONE")
            except:
                print "No write access: ", self.parent.elogDir
        else:
            # check if I'm a logger
            with open(self.loggerFile, "r") as myfile:
                content = myfile.readlines()
                if content[0].strip() == self.username:
                    self.logger = True
                    if self.parent.args.v >= 1: print "I'm an elogger"
                else:
                    self.logger = False
                    if self.parent.args.v >= 1: print "I'm not an elogger"
        # Make run folder
        try:
            if os.path.exists(self.parent.psocakeRunDir) is False:
                os.makedirs(self.parent.psocakeRunDir, 0774)
        except:
            print "No write access: ", self.parent.psocakeRunDir
    
    # Launch crawler
    crawlerThread = []
    crawlerThreadCounter = 0
    
    def launchCrawler(self):
        self.crawlerThread.append(LogbookCrawler.LogbookCrawler(self.parent))  # send parent parameters with self
        self.crawlerThread[self.crawlerThreadCounter].updateLogbook(self.parent.experimentName, self.parent.psocakeDir)
        self.crawlerThreadCounter += 1

    def setupExperiment(self):
        if self.parent.args.v >= 1: print "Doing setupExperiment"
        if self.hasExpRunInfo():
            self.getUsername()
            # Set up psocake directory in scratch
            if self.parent.args.outDir is None:
                self.parent.rootDir = '/reg/d/psdm/' + self.parent.experimentName[:3] + '/' + self.parent.experimentName
                self.parent.elogDir = self.parent.rootDir + '/scratch/psocake'
                self.parent.psocakeDir = self.parent.rootDir + '/scratch/' + self.username + '/psocake'
            else:
                self.parent.rootDir = self.parent.args.outDir
                self.parent.elogDir = self.parent.rootDir + '/psocake'
                self.parent.psocakeDir = self.parent.rootDir + '/' + self.username + '/psocake'
            self.parent.psocakeRunDir = self.parent.psocakeDir + '/r' + str(self.parent.runNumber).zfill(4)
    
            # Update peak finder outdir and run number
            self.parent.p3.param(self.parent.pk.hitParam_grp, self.parent.pk.hitParam_outDir_str).setValue(self.parent.psocakeDir)
            self.parent.p3.param(self.parent.pk.hitParam_grp, self.parent.pk.hitParam_runs_str).setValue(self.parent.runNumber)
            # Update powder outdir and run number
            self.parent.p6.param(self.parent.mk.powder_grp, self.parent.mk.powder_outDir_str).setValue(self.parent.psocakeDir)
            self.parent.p6.param(self.parent.mk.powder_grp, self.parent.mk.powder_runs_str).setValue(self.parent.runNumber)
            # Update hit finding outdir, run number
            self.parent.p8.param(self.parent.hf.spiParam_grp, self.parent.hf.spiParam_outDir_str).setValue(self.parent.psocakeDir)
            self.parent.p8.param(self.parent.hf.spiParam_grp, self.parent.hf.spiParam_runs_str).setValue(self.parent.runNumber)
            # Update indexing outdir, run number
            self.parent.p9.param(self.parent.index.launch_grp, self.parent.index.outDir_str).setValue(self.parent.psocakeDir)
            print "@*#$@#*$@#*$*@$*#@*$*#@$*@#*$@*"
            print "#$%#$%#$%#%$# outDir: ", self.parent.psocakeDir
            print "@*#$@#*$@#*$*@$*#@*$*#@$*@#*$@*"
            self.parent.p9.param(self.parent.index.launch_grp, self.parent.index.runs_str).setValue(self.parent.runNumber)
            # Update quantifier filename
            self.parent.pSmall.param(self.parent.small.quantifier_grp, self.parent.small.quantifier_filename_str).setValue(self.parent.psocakeRunDir)
            self.setupPsocake()
    
            # Update hidden CrystFEL files
            self.hiddenCXI = self.parent.psocakeRunDir + '/.temp.cxi'
            self.hiddenCrystfelStream = self.parent.psocakeRunDir + '/.temp.stream'
            self.hiddenCrystfelList = self.parent.psocakeRunDir + '/.temp.lst'
    
            if self.parent.args.localCalib:
                if self.parent.args.v >= 1: print "Using local calib directory"
                psana.setOption('psana.calib-dir', './calib')
    
            try:
                self.ds = psana.DataSource('exp=' + str(self.parent.experimentName) + ':run=' + str(
                    self.parent.runNumber) + ':idx')  # FIXME: psana crashes if runNumber is non-existent
            except:
                print "############# No such datasource exists ###############"
            self.run = self.ds.runs().next()
            self.times = self.run.times()
            self.eventTotal = len(self.times)
            self.parent.spinBox.setMaximum(self.eventTotal - self.parent.stackSize)
            self.parent.p.param(self.exp_grp, self.exp_evt_str).setLimits((0, self.eventTotal - 1))
            self.parent.p.param(self.exp_grp, self.exp_evt_str, self.exp_numEvents_str).setValue(self.eventTotal)
            self.env = self.ds.env()
    
            if self.parent.detInfoList is None:
                self.parent.evt = self.run.event(self.times[0])
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
    
            # Launch e-log crawler
            if self.logger and self.crawlerRunning == False:
                if self.parent.args.v >= 1: print "Launching crawler"
                self.launchCrawler()
                self.crawlerRunning = True
    
        if self.hasExpRunDetInfo():
            self.parent.det = psana.Detector(str(self.parent.detInfo), self.env)
            self.parent.det.do_reshape_2d_to_3d(flag=True)
    
            self.parent.epics = self.ds.env().epicsStore()
            # detector distance
            if 'cspad' in self.parent.detInfo.lower() and 'cxi' in self.parent.experimentName:
                self.parent.clenEpics = str(self.parent.detInfo) + '_z'
                self.parent.clen = self.parent.epics.value(self.parent.clenEpics) / 1000.  # metres
                self.parent.coffset = self.parent.detectorDistance - self.parent.clen
                if self.parent.args.v >= 1:
                    print "clenEpics: ", self.parent.clenEpics
                    print "@detectorDistance (m), self.clen (m), self.coffset (m): ", self.parent.detectorDistance, self.parent.clen, self.parent.coffset
            if 'cspad' in self.parent.detInfo.lower():  # FIXME: increase pixel size list: epix, rayonix
                self.parent.pixelSize = 110e-6  # metres
            elif 'pnccd' in self.parent.detInfo.lower():
                self.parent.pixelSize = 75e-6  # metres
    
            self.parent.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_pixelSize_str).setValue(self.parent.pixelSize)
            # photon energy
            self.parent.ebeam = self.parent.evt.get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
            if self.parent.ebeam:
                self.parent.photonEnergy = self.parent.ebeam.ebeamPhotonEnergy()
            else:
                self.parent.photonEnergy = 0
            self.parent.p1.param(self.parent.geom.geom_grp, self.parent.geom.geom_photonEnergy_str).setValue(self.parent.photonEnergy)
    
            if self.parent.evt is None:
                self.parent.evt = self.run.event(self.times[0])
            self.detGuaranteed = self.parent.det.calib(self.parent.evt)
            if self.detGuaranteed is None:  # image isn't present for this event
                print "No image in this event. Searching for an event..."
                for i in np.arange(len(self.times)):
                    evt = self.run.event(self.times[i])
                    self.detGuaranteed = self.parent.det.calib(evt)
                    if self.detGuaranteed is not None:
                        print "Found an event"
                        break
    
            if self.detGuaranteed is not None:
                self.parent.pixelInd = np.reshape(np.arange(self.detGuaranteed.size) + 1, self.detGuaranteed.shape)
                self.parent.pixelIndAssem = self.parent.getAssembledImage(self.parent.pixelInd)
                self.parent.pixelIndAssem -= 1  # First pixel is 0
    
            # Write a temporary geom file
            if 'cspad' in self.parent.detInfo.lower():
                self.source = Detector.PyDetector.map_alias_to_source(self.parent.detInfo,
                                                                      self.ds.env())  # 'DetInfo(CxiDs2.0:Cspad.0)'
                self.calibSource = self.source.split('(')[-1].split(')')[0]  # 'CxiDs2.0:Cspad.0'
                self.detectorType = gu.det_type_from_source(self.source)  # 1
                self.calibGroup = gu.dic_det_type_to_calib_group[self.detectorType]  # 'CsPad::CalibV1'
                self.detectorName = gu.dic_det_type_to_name[self.detectorType].upper()  # 'CSPAD'
                self.calibPath = "/reg/d/psdm/" + self.parent.experimentName[0:3] + \
                                 "/" + self.parent.experimentName + "/calib/" + \
                                 self.calibGroup + "/" + self.calibSource + "/geometry"
                if self.parent.args.v >= 1: print "### calibPath: ", self.calibPath
    
                # Determine which calib file to use
                geometryFiles = os.listdir(self.calibPath)
                if self.parent.args.v >= 1: print "geom: ", geometryFiles
                calibFile = None
                minDiff = -1e6
                for fname in geometryFiles:
                    if fname.endswith('.data'):
                        endValid = False
                        startNum = int(fname.split('-')[0])
                        endNum = fname.split('-')[-1].split('.data')[0]
                        diff = startNum - self.parent.runNumber
                        # Make sure it's end number is valid too
                        if 'end' in endNum:
                            endValid = True
                        else:
                            try:
                                if self.parent.runNumber <= int(endNum):
                                    endValid = True
                            except:
                                continue
                        if diff <= 0 and diff > minDiff and endValid is True:
                            minDiff = diff
                            calibFile = fname
    
                if calibFile is not None:
                    # Convert psana geometry to crystfel geom
                    self.parent.p9.param(self.parent.index.index_grp, self.parent.index.index_geom_str).setValue(
                        self.parent.psocakeRunDir + '/.temp.geom')
                    cmd = ["python", "/reg/neh/home/yoon82/psgeom/psana2crystfel.py", self.calibPath + '/' + calibFile,
                           self.parent.psocakeRunDir + "/.temp.geom"] # TODO: remove my home
                    if self.parent.args.v >= 1: print "cmd: ", cmd
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                    output = p.communicate()[0]
                    p.stdout.close()
                    if self.parent.args.v >= 1: print "output: ", output
    
        if self.parent.args.v >= 1: print "Done setupExperiment"
    
    def updateLogscale(self, data):
        self.logscaleOn = data
        if self.hasExpRunDetInfo():
            self.parent.firstUpdate = True  # clicking logscale resets plot colorscale
            self.parent.updateImage()
        if self.parent.args.v >= 1: print "Done updateLogscale: ", self.logscaleOn
    
    def updateImageProperty(self, data):
        self.image_property = data
        self.parent.updateImage()
        if self.parent.args.v >= 1: print "Done updateImageProperty: ", self.image_property

    #def updateAduThreshold(self, data):
    #    self.aduThresh = data
    #    if self.hasExpRunDetInfo():
    #        self.parent.updateImage(self.calib)
    #    if self.parent.args.v >= 1: print "Done updateAduThreshold: ", self.aduThresh
    
    def updateCommonModeParam(self, data, ind):
        self.commonModeParams[ind] = data
        self.updateCommonMode(self.applyCommonMode)
        if self.parent.args.v >= 1: print "Done updateCommonModeParam: ", self.commonModeParams

    def updateCommonMode(self, data):
        self.applyCommonMode = data
        if self.applyCommonMode:
            self.parent.commonMode = self.checkCommonMode(self.commonModeParams)
        if self.hasExpRunDetInfo():
            if self.parent.args.v >= 1: print "%%% Redraw image with new common mode: ", self.parent.commonMode
            self.setupExperiment()
            self.parent.updateImage()
        if self.parent.args.v >= 1: print "Done updateCommonMode: ", self.parent.commonMode

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
        self.parent.p.param(self.exp_grp, self.exp_evt_str, self.exp_second_str).setValue(self.eventSeconds)
        self.parent.p.param(self.exp_grp, self.exp_evt_str, self.exp_nanosecond_str).setValue(self.eventNanoseconds)
        self.parent.p.param(self.exp_grp, self.exp_evt_str, self.exp_fiducial_str).setValue(self.eventFiducial)

import numpy as np
from ImgAlgos.PyAlgos import PyAlgos # peak finding
import pyqtgraph as pg
import h5py
from pyqtgraph.dockarea import *
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
import LaunchPeakFinder
import json, os

class PeakFinding(object):
    def __init__(self, parent = None):
        self.parent = parent

        self.d9 = Dock("Peak Finder", size=(1, 1))
        ## Dock 9: Peak finder
        self.w10 = ParameterTree()
        self.d9.addWidget(self.w10)
        #self.w11 = pg.LayoutWidget()
        #self.generatePowderBtn = QtGui.QPushButton('Generate Powder')
        #self.launchBtn = QtGui.QPushButton('Launch peak finder')
        #self.w11.addWidget(self.launchBtn, row=0,col=0)
        #self.w11.addWidget(self.generatePowderBtn, row=0, col=0)
        #self.d9.addWidget(self.w11)

        self.userUpdate = None
        self.doingUpdate = False

        # Peak finding
        self.hitParam_grp = 'Peak finder'
        self.hitParam_showPeaks_str = 'Show peaks found'
        self.hitParam_algorithm_str = 'Algorithm'
        # algorithm 0
        self.hitParam_algorithm0_str = 'None'
        # algorithm 1
        self.hitParam_alg1_npix_min_str = 'npix_min'
        self.hitParam_alg1_npix_max_str = 'npix_max'
        self.hitParam_alg1_amax_thr_str = 'amax_thr'
        self.hitParam_alg1_atot_thr_str = 'atot_thr'
        self.hitParam_alg1_son_min_str = 'son_min'
        self.hitParam_algorithm1_str = 'Droplet'
        self.hitParam_alg1_thr_low_str = 'thr_low'
        self.hitParam_alg1_thr_high_str = 'thr_high'
        self.hitParam_alg1_rank_str = 'rank'
        self.hitParam_alg1_radius_str = 'radius'
        self.hitParam_alg1_dr_str = 'dr'
        # algorithm 2
        self.hitParam_alg2_npix_min_str = 'npix_min'
        self.hitParam_alg2_npix_max_str = 'npix_max'
        self.hitParam_alg2_amax_thr_str = 'amax_thr'
        self.hitParam_alg2_atot_thr_str = 'atot_thr'
        self.hitParam_alg2_son_min_str = 'son_min'
        self.hitParam_algorithm2_str = 'FloodFill'
        self.hitParam_alg2_thr_str = 'thr'
        self.hitParam_alg2_r0_str = 'r0'
        self.hitParam_alg2_dr_str = 'dr'
        # algorithm 3
        self.hitParam_alg3_npix_min_str = 'npix_min'
        self.hitParam_alg3_npix_max_str = 'npix_max'
        self.hitParam_alg3_amax_thr_str = 'amax_thr'
        self.hitParam_alg3_atot_thr_str = 'atot_thr'
        self.hitParam_alg3_son_min_str = 'son_min'
        self.hitParam_algorithm3_str = 'Ranker'
        self.hitParam_alg3_rank_str = 'rank'
        self.hitParam_alg3_r0_str = 'r0'
        self.hitParam_alg3_dr_str = 'dr'
        # algorithm 4
        self.hitParam_alg4_npix_min_str = 'npix_min'
        self.hitParam_alg4_npix_max_str = 'npix_max'
        self.hitParam_alg4_amax_thr_str = 'amax_thr'
        self.hitParam_alg4_atot_thr_str = 'atot_thr'
        self.hitParam_alg4_son_min_str = 'son_min'
        self.hitParam_algorithm4_str = 'iDroplet'
        self.hitParam_alg4_thr_low_str = 'thr_low'
        self.hitParam_alg4_thr_high_str = 'thr_high'
        self.hitParam_alg4_rank_str = 'rank'
        self.hitParam_alg4_r0_str = 'radius'
        self.hitParam_alg4_dr_str = 'dr'

        self.hitParam_outDir_str = 'Output directory'
        self.hitParam_runs_str = 'Run(s)'
        self.hitParam_queue_str = 'queue'
        self.hitParam_cpu_str = 'CPUs'
        self.hitParam_psanaq_str = 'psanaq'
        self.hitParam_psnehq_str = 'psnehq'
        self.hitParam_psfehq_str = 'psfehq'
        self.hitParam_psnehprioq_str = 'psnehprioq'
        self.hitParam_psfehprioq_str = 'psfehprioq'
        self.hitParam_psnehhiprioq_str = 'psnehhiprioq'
        self.hitParam_psfehhiprioq_str = 'psfehhiprioq'
        self.hitParam_psdebugq_str = 'psdebugq'
        self.hitParam_noe_str = 'Number of events to process'
        self.hitParam_threshold_str = 'Indexable number of peaks'
        self.hitParam_launch_str = 'Launch peak finder'
        self.hitParam_extra_str = 'Extra parameters'

        self.save_minPeaks_str = 'Minimum number of peaks'
        self.save_maxPeaks_str = 'Maximum number of peaks'
        self.save_minRes_str = 'Minimum resolution (pixels)'
        self.save_sample_str = 'Sample name'

        self.showPeaks = True
        self.peaks = None
        self.numPeaksFound = 0
        self.algorithm = 0
        self.algInitDone = False
        self.peaksMaxRes = 0
        self.classify = False

        self.hitParam_alg1_npix_min = 2.
        self.hitParam_alg1_npix_max = 20.
        self.hitParam_alg1_amax_thr = 0.
        self.hitParam_alg1_atot_thr = 1000.
        self.hitParam_alg1_son_min = 7.
        self.hitParam_alg1_thr_low = 250.
        self.hitParam_alg1_thr_high = 600.
        self.hitParam_alg1_rank = 2
        self.hitParam_alg1_radius = 2
        self.hitParam_alg1_dr = 1
        # self.hitParam_alg2_npix_min = 1.
        # self.hitParam_alg2_npix_max = 5000.
        # self.hitParam_alg2_amax_thr = 1.
        # self.hitParam_alg2_atot_thr = 1.
        # self.hitParam_alg2_son_min = 1.
        # self.hitParam_alg2_thr = 10.
        # self.hitParam_alg2_r0 = 1.
        # self.hitParam_alg2_dr = 0.05
        # self.hitParam_alg3_npix_min = 5.
        # self.hitParam_alg3_npix_max = 5000.
        # self.hitParam_alg3_amax_thr = 0.
        # self.hitParam_alg3_atot_thr = 0.
        # self.hitParam_alg3_son_min = 4.
        # self.hitParam_alg3_rank = 3
        # self.hitParam_alg3_r0 = 5.
        # self.hitParam_alg3_dr = 0.05
        # self.hitParam_alg4_npix_min = 1.
        # self.hitParam_alg4_npix_max = 45.
        # self.hitParam_alg4_amax_thr = 800.
        # self.hitParam_alg4_atot_thr = 0
        # self.hitParam_alg4_son_min = 7.
        # self.hitParam_alg4_thr_low = 200.
        # self.hitParam_alg4_thr_high = self.hitParam_alg1_thr_high
        # self.hitParam_alg4_rank = 3
        # self.hitParam_alg4_r0 = 2
        # self.hitParam_alg4_dr = 1
        self.hitParam_outDir = self.parent.psocakeDir
        self.hitParam_outDir_overridden = False
        self.hitParam_runs = ''
        self.hitParam_queue = self.hitParam_psanaq_str
        self.hitParam_cpus = 24
        self.hitParam_noe = -1
        self.hitParam_threshold = 15 # usually crystals with less than 15 peaks are not indexable

        self.minPeaks = 15
        self.maxPeaks = 2048
        self.minRes = -1
        self.sample = 'sample'
        self.profile = 0
        self.hitParam_extra = ''

        self.params = [
            {'name': self.hitParam_grp, 'type': 'group', 'children': [
                {'name': self.hitParam_showPeaks_str, 'type': 'bool', 'value': self.showPeaks,
                 'tip': "Show peaks found shot-to-shot"},
                {'name': self.hitParam_algorithm_str, 'type': 'list', 'values': {self.hitParam_algorithm1_str: 1,
                                                                                 self.hitParam_algorithm0_str: 0},
                 'value': self.algorithm},
                {'name': self.hitParam_algorithm1_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "",
                 'readonly': True, 'children': [
                    {'name': self.hitParam_alg1_npix_min_str, 'type': 'float', 'value': self.hitParam_alg1_npix_min,
                     'tip': "Only keep the peak if number of pixels above thr_low is above this value"},
                    {'name': self.hitParam_alg1_npix_max_str, 'type': 'float', 'value': self.hitParam_alg1_npix_max,
                     'tip': "Only keep the peak if number of pixels above thr_low is below this value"},
                    {'name': self.hitParam_alg1_amax_thr_str, 'type': 'float', 'value': self.hitParam_alg1_amax_thr,
                     'tip': "Only keep the peak if max value is above this value"},
                    {'name': self.hitParam_alg1_atot_thr_str, 'type': 'float', 'value': self.hitParam_alg1_atot_thr,
                     'tip': "Only keep the peak if integral inside region of interest is above this value"},
                    {'name': self.hitParam_alg1_son_min_str, 'type': 'float', 'value': self.hitParam_alg1_son_min,
                     'tip': "Only keep the peak if signal-over-noise is above this value"},
                    {'name': self.hitParam_alg1_thr_low_str, 'type': 'float', 'value': self.hitParam_alg1_thr_low,
                     'tip': "Grow a seed peak if above this value"},
                    {'name': self.hitParam_alg1_thr_high_str, 'type': 'float', 'value': self.hitParam_alg1_thr_high,
                     'tip': "Start a seed peak if above this value"},
                    {'name': self.hitParam_alg1_rank_str, 'type': 'int', 'value': self.hitParam_alg1_rank,
                     'tip': "region of integration is a square, (2r+1)x(2r+1)"},
                    {'name': self.hitParam_alg1_radius_str, 'type': 'int', 'value': self.hitParam_alg1_radius,
                     'tip': "region inside the region of interest"},
                    {'name': self.hitParam_alg1_dr_str, 'type': 'float', 'value': self.hitParam_alg1_dr,
                     'tip': "background region outside the region of interest"},
                ]},
                {'name': self.save_minPeaks_str, 'type': 'int', 'value': self.minPeaks,
                 'tip': "Index only if there are more Bragg peaks found"},
                {'name': self.save_maxPeaks_str, 'type': 'int', 'value': self.maxPeaks,
                 'tip': "Index only if there are less Bragg peaks found"},
                {'name': self.save_minRes_str, 'type': 'int', 'value': self.minRes,
                 'tip': "Index only if Bragg peak resolution is at least this"},
                {'name': self.save_sample_str, 'type': 'str', 'value': self.sample,
                 'tip': "Sample name saved inside cxi"},
                {'name': self.hitParam_outDir_str, 'type': 'str', 'value': self.hitParam_outDir},
                {'name': self.hitParam_runs_str, 'type': 'str', 'value': self.hitParam_runs},
                {'name': self.hitParam_queue_str, 'type': 'list', 'values': {self.hitParam_psfehhiprioq_str: 'psfehhiprioq',
                                                                             self.hitParam_psnehhiprioq_str: 'psnehhiprioq',
                                                                             self.hitParam_psfehprioq_str: 'psfehprioq',
                                                                             self.hitParam_psnehprioq_str: 'psnehprioq',
                                                                             self.hitParam_psfehq_str: 'psfehq',
                                                                             self.hitParam_psnehq_str: 'psnehq',
                                                                             self.hitParam_psanaq_str: 'psanaq',
                                                                             self.hitParam_psdebugq_str: 'psdebugq'},
                 'value': self.hitParam_queue, 'tip': "Choose queue"},
                {'name': self.hitParam_cpu_str, 'type': 'int', 'value': self.hitParam_cpus},
                {'name': self.hitParam_noe_str, 'type': 'int', 'value': self.hitParam_noe,
                 'tip': "number of events to process, default=-1 means process all events"},
                {'name': self.hitParam_extra_str, 'type': 'str', 'value': self.hitParam_extra, 'tip': "Extra peak finding flags"},
                {'name': self.hitParam_launch_str, 'type': 'action'},
            ]},
        ]

        self.p3 = Parameter.create(name='paramsPeakFinder', type='group', \
                                   children=self.params, expanded=True)
        self.w10.setParameters(self.p3, showTop=False)
        self.p3.sigTreeStateChanged.connect(self.change)
        #self.parent.connect(self.launchBtn, QtCore.SIGNAL("clicked()"), self.findPeaks)

    def digestRunList(self, runList):
        runsToDo = []
        if not runList:
            print "Run(s) is empty. Please type in the run number(s)."
            return runsToDo
        runLists = str(runList).split(",")
        for list in runLists:
            temp = list.split(":")
            if len(temp) == 2:
                for i in np.arange(int(temp[0]),int(temp[1])+1):
                    runsToDo.append(i)
            elif len(temp) == 1:
                runsToDo.append(int(temp[0]))
        return runsToDo

    def updateParam(self):
        if self.userUpdate is None:
            if self.parent.psocakeRunDir is not None:
                peakParamFname = self.parent.psocakeRunDir + '/peakParam.json'
                if os.path.exists(peakParamFname):
                    with open(peakParamFname) as infile:
                        d = json.load(infile)
                        if d[self.hitParam_algorithm_str] == 1:
                            # Update variables
                            try:
                                self.hitParam_alg1_npix_min = d[self.hitParam_alg1_npix_min_str]
                                self.hitParam_alg1_npix_max = d[self.hitParam_alg1_npix_max_str]
                                self.hitParam_alg1_amax_thr = d[self.hitParam_alg1_amax_thr_str]
                                self.hitParam_alg1_atot_thr = d[self.hitParam_alg1_atot_thr_str]
                                self.hitParam_alg1_son_min = d[self.hitParam_alg1_son_min_str]
                                self.hitParam_alg1_thr_low = d[self.hitParam_alg1_thr_low_str]
                                self.hitParam_alg1_thr_high = d[self.hitParam_alg1_thr_high_str]
                                self.hitParam_alg1_rank = int(d[self.hitParam_alg1_rank_str])
                                self.hitParam_alg1_radius = int(d[self.hitParam_alg1_radius_str])
                                self.hitParam_alg1_dr = d[self.hitParam_alg1_dr_str]
                                # Update GUI
                                self.doingUpdate = True
                                #self.p3.param(self.hitParam_grp, self.hitParam_algorithm_str).setValue(self.algorithm)
                                self.p3.param(self.hitParam_grp, self.hitParam_algorithm1_str, self.hitParam_alg1_npix_min_str).setValue(
                                    self.hitParam_alg1_npix_min)
                                self.p3.param(self.hitParam_grp, self.hitParam_algorithm1_str, self.hitParam_alg1_npix_max_str).setValue(
                                    self.hitParam_alg1_npix_max)
                                self.p3.param(self.hitParam_grp, self.hitParam_algorithm1_str, self.hitParam_alg1_amax_thr_str).setValue(
                                    self.hitParam_alg1_amax_thr)
                                self.p3.param(self.hitParam_grp, self.hitParam_algorithm1_str, self.hitParam_alg1_atot_thr_str).setValue(
                                    self.hitParam_alg1_atot_thr)
                                self.p3.param(self.hitParam_grp, self.hitParam_algorithm1_str, self.hitParam_alg1_son_min_str).setValue(
                                    self.hitParam_alg1_son_min)
                                self.p3.param(self.hitParam_grp, self.hitParam_algorithm1_str, self.hitParam_alg1_thr_low_str).setValue(
                                    self.hitParam_alg1_thr_low)
                                self.p3.param(self.hitParam_grp, self.hitParam_algorithm1_str, self.hitParam_alg1_thr_high_str).setValue(
                                    self.hitParam_alg1_thr_high)
                                self.p3.param(self.hitParam_grp, self.hitParam_algorithm1_str, self.hitParam_alg1_rank_str).setValue(
                                    self.hitParam_alg1_rank)
                                self.p3.param(self.hitParam_grp, self.hitParam_algorithm1_str, self.hitParam_alg1_radius_str).setValue(
                                    self.hitParam_alg1_radius)
                                self.doingUpdate = False
                                self.p3.param(self.hitParam_grp, self.hitParam_algorithm1_str, self.hitParam_alg1_dr_str).setValue(
                                    self.hitParam_alg1_dr)
                            except:
                                pass

    def writeStatus(self, fname, d):
        json.dump(d, open(fname, 'w'))

    # Launch peak finding
    def findPeaks(self):
        self.parent.thread.append(LaunchPeakFinder.LaunchPeakFinder(self.parent)) # send parent parameters with self
        self.parent.thread[self.parent.threadCounter].launch(self.parent.experimentName, self.parent.detInfo)
        self.parent.threadCounter+=1
        # Save peak finding parameters
        runsToDo = self.digestRunList(self.hitParam_runs)
        for run in runsToDo:
            peakParamFname = self.parent.psocakeDir+'/r'+str(run).zfill(4)+'/peakParam.json'
            d = {self.hitParam_algorithm_str: self.algorithm,
                 self.hitParam_alg1_npix_min_str: self.hitParam_alg1_npix_min,
                 self.hitParam_alg1_npix_max_str: self.hitParam_alg1_npix_max,
                 self.hitParam_alg1_amax_thr_str: self.hitParam_alg1_amax_thr,
                 self.hitParam_alg1_atot_thr_str: self.hitParam_alg1_atot_thr,
                 self.hitParam_alg1_son_min_str: self.hitParam_alg1_son_min,
                 self.hitParam_alg1_thr_low_str: self.hitParam_alg1_thr_low,
                 self.hitParam_alg1_thr_high_str: self.hitParam_alg1_thr_high,
                 self.hitParam_alg1_rank_str: self.hitParam_alg1_rank,
                 self.hitParam_alg1_radius_str: self.hitParam_alg1_radius,
                 self.hitParam_alg1_dr_str: self.hitParam_alg1_dr}
            self.writeStatus(peakParamFname, d)

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
        if path[0] == self.hitParam_grp:
            if path[1] == self.hitParam_algorithm_str:
                self.algInitDone = False
                self.updateAlgorithm(data)
            elif path[1] == self.hitParam_showPeaks_str:
                self.showPeaks = data
                self.drawPeaks()
            elif path[1] == self.hitParam_outDir_str:
                self.hitParam_outDir = data
                self.hitParam_outDir_overridden = True
            elif path[1] == self.hitParam_runs_str:
                self.hitParam_runs = data
            elif path[1] == self.hitParam_queue_str:
                self.hitParam_queue = data
            elif path[1] == self.hitParam_cpu_str:
                self.hitParam_cpus = data
            elif path[1] == self.hitParam_noe_str:
                self.hitParam_noe = data
            elif path[1] == self.hitParam_threshold_str:
                self.hitParam_threshold = data
            elif path[1] == self.hitParam_launch_str:
                self.findPeaks()
            elif path[1] == self.save_minPeaks_str:
                self.minPeaks = data
            elif path[1] == self.save_maxPeaks_str:
                self.maxPeaks = data
            elif path[1] == self.save_minRes_str:
                self.minRes = data
            elif path[1] == self.save_sample_str:
                self.sample = data
            elif path[1] == self.hitParam_extra_str:
                self.hitParam_extra = data
            elif path[2] == self.hitParam_alg1_npix_min_str and path[1] == self.hitParam_algorithm1_str:
                self.hitParam_alg1_npix_min = data
                self.algInitDone = False
                self.userUpdate = True
                if self.showPeaks and self.doingUpdate is False:
                    self.updateClassification()
            elif path[2] == self.hitParam_alg1_npix_max_str and path[1] == self.hitParam_algorithm1_str:
                self.hitParam_alg1_npix_max = data
                self.algInitDone = False
                self.userUpdate = True
                if self.showPeaks and self.doingUpdate is False:
                    self.updateClassification()
            elif path[2] == self.hitParam_alg1_amax_thr_str and path[1] == self.hitParam_algorithm1_str:
                self.hitParam_alg1_amax_thr = data
                self.algInitDone = False
                self.userUpdate = True
                if self.showPeaks and self.doingUpdate is False:
                    self.updateClassification()
            elif path[2] == self.hitParam_alg1_atot_thr_str and path[1] == self.hitParam_algorithm1_str:
                self.hitParam_alg1_atot_thr = data
                self.algInitDone = False
                self.userUpdate = True
                if self.showPeaks and self.doingUpdate is False:
                    self.updateClassification()
            elif path[2] == self.hitParam_alg1_son_min_str and path[1] == self.hitParam_algorithm1_str:
                self.hitParam_alg1_son_min = data
                self.algInitDone = False
                self.userUpdate = True
                if self.showPeaks and self.doingUpdate is False:
                    self.updateClassification()
            elif path[2] == self.hitParam_alg1_thr_low_str and path[1] == self.hitParam_algorithm1_str:
                self.hitParam_alg1_thr_low = data
                self.algInitDone = False
                self.userUpdate = True
                if self.showPeaks and self.doingUpdate is False:
                    self.updateClassification()
            elif path[2] == self.hitParam_alg1_thr_high_str and path[1] == self.hitParam_algorithm1_str:
                self.hitParam_alg1_thr_high = data
                self.algInitDone = False
                self.userUpdate = True
                if self.showPeaks and self.doingUpdate is False:
                    self.updateClassification()
            elif path[2] == self.hitParam_alg1_rank_str and path[1] == self.hitParam_algorithm1_str:
                self.hitParam_alg1_rank = data
                self.algInitDone = False
                self.userUpdate = True
                if self.showPeaks and self.doingUpdate is False:
                    self.updateClassification()
            elif path[2] == self.hitParam_alg1_radius_str and path[1] == self.hitParam_algorithm1_str:
                self.hitParam_alg1_radius = data
                self.algInitDone = False
                self.userUpdate = True
                if self.showPeaks and self.doingUpdate is False:
                    self.updateClassification()
            elif path[2] == self.hitParam_alg1_dr_str and path[1] == self.hitParam_algorithm1_str:
                self.hitParam_alg1_dr = data
                self.algInitDone = False
                self.userUpdate = True
                if self.showPeaks and self.doingUpdate is False:
                    self.updateClassification()

    def updateAlgorithm(self, data):
        self.algorithm = data
        self.algInitDone = False
        self.updateClassification()
        if self.parent.args.v >= 1: print "##### Done updateAlgorithm: ", self.algorithm

    def saveCheetahFormat(self, arg):
        if arg == 'lcls':
            if 'cspad' in self.parent.detInfo.lower() and 'cxi' in self.parent.experimentName:
                dim0 = 8 * 185
                dim1 = 4 * 388
            elif 'rayonix' in self.parent.detInfo.lower() and 'mfx' in self.parent.experimentName:
                dim0 = 1920 #FIXME: rayonix can be binned
                dim1 = 1920
            elif 'rayonix' in self.parent.detInfo.lower() and 'xpp' in self.parent.experimentName:
                dim0 = 1920 #FIXME: rayonix can be binned
                dim1 = 1920
            else:
                dim0 = 0
                dim1 = 0

            if dim0 > 0:
                maxNumPeaks = 2048
                if self.parent.index.hiddenCXI is not None:
                    myHdf5 = h5py.File(self.parent.index.hiddenCXI, 'w')
                    grpName = "/entry_1/result_1"
                    dset_nPeaks = "/nPeaks"
                    dset_posX = "/peakXPosRaw"
                    dset_posY = "/peakYPosRaw"
                    dset_atot = "/peakTotalIntensity"
                    if grpName in myHdf5:
                        del myHdf5[grpName]
                    grp = myHdf5.create_group(grpName)
                    myHdf5.create_dataset(grpName + dset_nPeaks, (1,), dtype='int')
                    myHdf5.create_dataset(grpName + dset_posX, (1, maxNumPeaks), dtype='float32', chunks=(1, maxNumPeaks))
                    myHdf5.create_dataset(grpName + dset_posY, (1, maxNumPeaks), dtype='float32', chunks=(1, maxNumPeaks))
                    myHdf5.create_dataset(grpName + dset_atot, (1, maxNumPeaks), dtype='float32', chunks=(1, maxNumPeaks))

                    myHdf5.create_dataset("/LCLS/detector_1/EncoderValue", (1,), dtype=float)
                    myHdf5.create_dataset("/LCLS/photon_energy_eV", (1,), dtype=float)
                    dset = myHdf5.create_dataset("/entry_1/data_1/data", (1, dim0, dim1), dtype=float)

                    # Convert calib image to cheetah image
                    img = np.zeros((dim0, dim1))
                    counter = 0
                    if 'cspad' in self.parent.detInfo.lower() and 'cxi' in self.parent.experimentName:
                        for quad in range(4):
                            for seg in range(8):
                                img[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = self.parent.calib[counter, :, :]
                                counter += 1
                    elif 'rayonix' in self.parent.detInfo.lower() and 'mfx' in self.parent.experimentName:
                        img = self.parent.calib[:, :] # psana format
                    elif 'rayonix' in self.parent.detInfo.lower() and 'xpp' in self.parent.experimentName:
                        img = self.parent.calib[:, :]  # psana format
                    else:
                        print "saveCheetahFormat not implemented"

                    peaks = self.peaks.copy()
                    nPeaks = peaks.shape[0]

                    if nPeaks > maxNumPeaks:
                        peaks = peaks[:maxNumPeaks]
                        nPeaks = maxNumPeaks
                    for i, peak in enumerate(peaks):
                        seg, row, col, npix, amax, atot, rcent, ccent, rsigma, csigma, rmin, rmax, cmin, cmax, bkgd, rms, son = peak[0:17]
                        if 'cspad' in self.parent.detInfo.lower() and 'cxi' in self.parent.experimentName:
                            cheetahRow, cheetahCol = self.convert_peaks_to_cheetah(seg, row, col)
                            myHdf5[grpName + dset_posX][0, i] = cheetahCol
                            myHdf5[grpName + dset_posY][0, i] = cheetahRow
                            myHdf5[grpName + dset_atot][0, i] = atot
                        elif 'rayonix' in self.parent.detInfo.lower() and 'mfx' in self.parent.experimentName:
                            myHdf5[grpName + dset_posX][0, i] = col
                            myHdf5[grpName + dset_posY][0, i] = row
                            myHdf5[grpName + dset_atot][0, i] = atot
                        elif 'rayonix' in self.parent.detInfo.lower() and 'xpp' in self.parent.experimentName:
                            myHdf5[grpName + dset_posX][0, i] = col
                            myHdf5[grpName + dset_posY][0, i] = row
                            myHdf5[grpName + dset_atot][0, i] = atot
                    myHdf5[grpName + dset_nPeaks][0] = nPeaks

                    if self.parent.args.v >= 1: print "hiddenCXI clen (mm): ", self.parent.clen * 1000.
                    myHdf5["/LCLS/detector_1/EncoderValue"][0] = self.parent.clen * 1000.  # mm
                    myHdf5["/LCLS/photon_energy_eV"][0] = self.parent.photonEnergy
                    dset[0, :, :] = img
                    myHdf5.close()

    def updateClassification(self):
        if self.parent.calib is not None:
            if self.parent.mk.streakMaskOn:
                self.parent.mk.initMask()
                self.parent.mk.streakMask = self.parent.mk.StreakMask.getStreakMaskCalib(self.parent.evt)
                if self.parent.mk.streakMask is None:
                    self.parent.mk.streakMaskAssem = None
                else:
                    self.parent.mk.streakMaskAssem = self.parent.det.image(self.parent.evt, self.parent.mk.streakMask)
                self.algInitDone = False

            self.parent.mk.displayMask()

            # update combined mask
            self.parent.mk.combinedMask = np.ones_like(self.parent.calib)
            if self.parent.mk.streakMask is not None and self.parent.mk.streakMaskOn is True:
                self.parent.mk.combinedMask *= self.parent.mk.streakMask
            if self.parent.mk.userMask is not None and self.parent.mk.userMaskOn is True:
                self.parent.mk.combinedMask *= self.parent.mk.userMask
            if self.parent.mk.psanaMask is not None and self.parent.mk.psanaMaskOn is True:
                self.parent.mk.combinedMask *= self.parent.mk.psanaMask

            # Peak output (0-16):
            # 0 seg
            # 1 row
            # 2 col
            # 3 npix: no. of pixels in the ROI intensities above threshold
            # 4 amp_max: max intensity
            # 5 amp_tot: sum of intensities
            # 6,7: row_cgrav: center of mass
            # 8,9: row_sigma
            # 10,11,12,13: minimum bounding box
            # 14: background
            # 15: noise
            # 16: signal over noise
            if self.algorithm == 0: # No peak algorithm
                self.peaks = None
                self.drawPeaks()
            else:
                # Only initialize the hit finder algorithm once
                if self.algInitDone is False:
                    self.windows = None
                    self.alg = []
                    self.alg = PyAlgos(windows=self.windows, mask=self.parent.mk.combinedMask, pbits=0)

                    # set peak-selector parameters:
                    if self.algorithm == 1:
                        self.alg.set_peak_selection_pars(npix_min=self.hitParam_alg1_npix_min, npix_max=self.hitParam_alg1_npix_max, \
                                                amax_thr=self.hitParam_alg1_amax_thr, atot_thr=self.hitParam_alg1_atot_thr, \
                                                son_min=self.hitParam_alg1_son_min)
                    elif self.algorithm == 2:
                        self.alg.set_peak_selection_pars(npix_min=self.hitParam_alg2_npix_min, npix_max=self.hitParam_alg2_npix_max, \
                                                amax_thr=self.hitParam_alg2_amax_thr, atot_thr=self.hitParam_alg2_atot_thr, \
                                                son_min=self.hitParam_alg2_son_min)
                    elif self.algorithm == 3:
                        self.alg.set_peak_selection_pars(npix_min=self.hitParam_alg3_npix_min, npix_max=self.hitParam_alg3_npix_max, \
                                                amax_thr=self.hitParam_alg3_amax_thr, atot_thr=self.hitParam_alg3_atot_thr, \
                                                son_min=self.hitParam_alg3_son_min)
                    elif self.algorithm == 4:
                        self.alg.set_peak_selection_pars(npix_min=self.hitParam_alg4_npix_min, npix_max=self.hitParam_alg4_npix_max, \
                                                amax_thr=self.hitParam_alg4_amax_thr, atot_thr=self.hitParam_alg4_atot_thr, \
                                                son_min=self.hitParam_alg4_son_min)
                    self.algInitDone = True

                self.parent.calib = self.parent.calib * 1.0 # Neccessary when int is returned
                if self.algorithm == 1:
                    # v1 - aka Droplet Finder - two-threshold peak-finding algorithm in restricted region
                    #                           around pixel with maximal intensity.
                    self.peakRadius = int(self.hitParam_alg1_radius)
                    self.peaks = self.alg.peak_finder_v4r2(self.parent.calib,
                                                           thr_low=self.hitParam_alg1_thr_low,
                                                           thr_high=self.hitParam_alg1_thr_high,
                                                           rank=int(self.hitParam_alg1_rank),
                                                           r0=self.peakRadius,
                                                           dr=self.hitParam_alg1_dr)
                elif self.algorithm == 2:
                    # v2 - define peaks for regions of connected pixels above threshold
                    self.peakRadius = int(self.hitParam_alg2_r0)
                    self.peaks = self.alg.peak_finder_v2(self.parent.calib, thr=self.hitParam_alg2_thr, r0=self.peakRadius, dr=self.hitParam_alg2_dr)
                elif self.algorithm == 3:
                    self.peakRadius = int(self.hitParam_alg3_r0)
                    self.peaks = self.alg.peak_finder_v3(self.parent.calib, rank=self.hitParam_alg3_rank, r0=self.peakRadius, dr=self.hitParam_alg3_dr)
                elif self.algorithm == 4:
                    # v4 - aka Droplet Finder - the same as v1, but uses rank and r0 parameters in stead of common radius.
                    self.peakRadius = int(self.hitParam_alg4_r0)
                    self.peaks = self.alg.peak_finder_v4(self.parent.calib, thr_low=self.hitParam_alg4_thr_low, thr_high=self.hitParam_alg4_thr_high,
                                               rank=self.hitParam_alg4_rank, r0=self.peakRadius,  dr=self.hitParam_alg4_dr)
                self.numPeaksFound = self.peaks.shape[0]

                if self.parent.args.v >= 1: print "Num peaks found: ", self.numPeaksFound, self.peaks.shape

                # update clen
                self.parent.geom.updateClen('lcls')

                self.parent.index.clearIndexedPeaks()

                # Save image and peaks in cheetah cxi file
                self.saveCheetahFormat('lcls')

                if self.parent.index.showIndexedPeaks: self.parent.index.updateIndex()

                self.drawPeaks()
            if self.parent.args.v >= 1: print "Done updateClassification"

    def convert_peaks_to_cheetah(self, s, r, c) :
        """Converts seg, row, col assuming (32,185,388)
           to cheetah 2-d table row and col (8*185, 4*388)
        """
        segs, rows, cols = (32,185,388)
        row2d = (int(s)%8) * rows + int(r) # where s%8 is a segment in quad number [0,7]
        col2d = (int(s)/8) * cols + int(c) # where s/8 is a quad number [0,3]
        return row2d, col2d

    def getMaxRes(self, posX, posY, centerX, centerY):
        maxRes = np.max(np.sqrt((posX-centerX)**2 + (posY-centerY)**2))
        if self.parent.args.v >= 1: print "maxRes: ", maxRes
        return maxRes # in pixels

    def drawPeaks(self):
        self.parent.img.clearPeakMessage()
        if self.showPeaks:
            if self.peaks is not None and self.numPeaksFound > 0:
                self.ix = self.parent.det.indexes_x(self.parent.evt)
                self.iy = self.parent.det.indexes_y(self.parent.evt)
                if self.ix is None:
                    (_, dim0, dim1) = self.parent.calib.shape
                    self.iy = np.tile(np.arange(dim0),[dim1, 1])
                    self.ix = np.transpose(self.iy)
                self.iX = np.array(self.ix, dtype=np.int64)
                self.iY = np.array(self.iy, dtype=np.int64)
                if len(self.iX.shape)==2:
                    self.iX = np.expand_dims(self.iX,axis=0)
                    self.iY = np.expand_dims(self.iY,axis=0)
                cenX = self.iX[np.array(self.peaks[:,0],dtype=np.int64),np.array(self.peaks[:,1],dtype=np.int64),np.array(self.peaks[:,2],dtype=np.int64)] + 0.5
                cenY = self.iY[np.array(self.peaks[:,0],dtype=np.int64),np.array(self.peaks[:,1],dtype=np.int64),np.array(self.peaks[:,2],dtype=np.int64)] + 0.5
                self.peaksMaxRes = self.getMaxRes(cenX, cenY, self.parent.cx, self.parent.cy)
                diameter = self.peakRadius*2+1
                self.parent.img.peak_feature.setData(cenX, cenY, symbol='s', \
                                          size=diameter, brush=(255,255,255,0), \
                                          pen=pg.mkPen({'color': "c", 'width': 4}), pxMode=False) #FF0
                # Write number of peaks found
                xMargin = 5 # pixels
                yMargin = 0  # pixels
                maxX = np.max(self.ix) + xMargin
                maxY = np.max(self.iy) - yMargin
                myMessage = '<div style="text-align: center"><span style="color: cyan; font-size: 12pt;">Peaks=' + \
                            str(self.numPeaksFound) + ' <br>Res=' + str(int(self.peaksMaxRes)) + '<br></span></div>'
                self.parent.img.peak_text = pg.TextItem(html=myMessage, anchor=(0, 0))
                self.parent.img.w1.getView().addItem(self.parent.img.peak_text)
                self.parent.img.peak_text.setPos(maxX, maxY)
            else:
                self.parent.img.peak_feature.setData([], [], pxMode=False)
                self.parent.img.peak_text = pg.TextItem(html='', anchor=(0, 0))
                self.parent.img.w1.getView().addItem(self.parent.img.peak_text)
                self.parent.img.peak_text.setPos(0,0)
        else:
            self.parent.img.peak_feature.setData([], [], pxMode=False)
        if self.parent.args.v >= 1: print "Done updatePeaks"
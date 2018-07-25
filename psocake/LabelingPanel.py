import numpy as np
import pyqtgraph as pg
import h5py
from pyqtgraph.dockarea import *
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
import LaunchPeakFinder
import json, os, time
from scipy.spatial.distance import cdist #TODO: clean up unneeded imports
from scipy.spatial import distance
import subprocess

if 'LCLS' in os.environ['PSOCAKE_FACILITY'].upper():
    pass
elif 'PAL' in os.environ['PSOCAKE_FACILITY'].upper():
    pass

class Labeling(object):
    def __init__(self, parent = None):
        self.parent = parent

        ## Dock: Labeler
        self.dock = Dock("Labeling", size=(1, 1))
        self.win = ParameterTree()
        self.dock.addWidget(self.win)

        # TODO: add dropdown for ADD/REMOVE (default: ADD)
        self.labelParam_grp = 'Labeler'
        self.labelParam_shapes_str = 'Shape'
        self.labelParam_pluginDir_str = 'Plugin directory'
        self.labelParam_outDir_str = 'Output directory'
        self.labelParam_runs_str = 'Run(s)'
        self.labelParam_queue_str = 'queue'
        self.labelParam_cpu_str = 'CPUs'
        self.labelParam_psanaq_str = 'psanaq'
        self.labelParam_psnehq_str = 'psnehq'
        self.labelParam_psfehq_str = 'psfehq'
        self.labelParam_psnehprioq_str = 'psnehprioq'
        self.labelParam_psfehprioq_str = 'psfehprioq'
        self.labelParam_psnehhiprioq_str = 'psnehhiprioq'
        self.labelParam_psfehhiprioq_str = 'psfehhiprioq'
        self.labelParam_psdebugq_str = 'psdebugq'
        self.labelParam_noQueue_str = 'N/A'
        self.labelParam_noe_str = 'Number of events to process'
        self.labelParam_launch_str = 'Launch labeler'
        self.labelParam_pluginParam_str = 'Plugin parameters'
        self.tag_str = 'Tag'

        self.labelParam_poly_str = 'Polygon'
        self.labelParam_circ_str = 'Circle'
        self.labelParam_rect_str = 'Rectangle'
        self.labelParam_none_str = 'None'

        self.shape = None
        self.labelParam_pluginParam = ''
        self.tag = ''
        self.labelParam_pluginDir = ''

        if self.parent.facility == self.parent.facilityLCLS:
            self.labelParam_outDir = self.parent.psocakeDir
            self.labelParam_outDir_overridden = False
            self.labelParam_runs = ''
            self.labelParam_queue = self.labelParam_psanaq_str
            self.labelParam_cpus = 24
            self.labelParam_noe = -1
        elif self.parent.facility == self.parent.facilityPAL:
            pass

        # TODO: outDir doesn't display as expected
        # TODO: fix tip
        if self.parent.facility == self.parent.facilityLCLS:
            self.params = [
                {'name': self.labelParam_grp, 'type': 'group', 'children': [
                    {'name': self.labelParam_shapes_str, 'type': 'list', 'values': {self.labelParam_poly_str: 3,
                                                                                    self.labelParam_circ_str: 2,
                                                                                    self.labelParam_rect_str: 1,
                                                                                    self.labelParam_none_str: 0},
                     'value': self.shape},
                    {'name': self.labelParam_pluginDir_str, 'type': 'str', 'value': self.labelParam_pluginDir},
                    {'name': self.labelParam_pluginParam_str, 'type': 'str', 'value': self.labelParam_pluginParam,
                     'tip': "Extra peak finding flags"},
                    {'name': self.tag_str, 'type': 'str', 'value': self.tag,
                     'tip': "attach tag to stream, e.g. cxitut13_0010_tag.stream"},
                    {'name': self.labelParam_outDir_str, 'type': 'str', 'value': self.labelParam_outDir},
                    {'name': self.labelParam_runs_str, 'type': 'str', 'value': self.labelParam_runs},
                    {'name': self.labelParam_queue_str, 'type': 'list', 'values': {self.labelParam_psfehhiprioq_str: 'psfehhiprioq',
                                                                                 self.labelParam_psnehhiprioq_str: 'psnehhiprioq',
                                                                                 self.labelParam_psfehprioq_str: 'psfehprioq',
                                                                                 self.labelParam_psnehprioq_str: 'psnehprioq',
                                                                                 self.labelParam_psfehq_str: 'psfehq',
                                                                                 self.labelParam_psnehq_str: 'psnehq',
                                                                                 self.labelParam_psanaq_str: 'psanaq',
                                                                                 self.labelParam_psdebugq_str: 'psdebugq'},
                     'value': self.labelParam_queue, 'tip': "Choose queue"},
                    {'name': self.labelParam_cpu_str, 'type': 'int', 'decimals': 7, 'value': self.labelParam_cpus},
                    {'name': self.labelParam_noe_str, 'type': 'int', 'decimals': 7, 'value': self.labelParam_noe,
                     'tip': "number of events to process, default=-1 means process all events"},
                    {'name': self.labelParam_launch_str, 'type': 'action'},
                ]},
            ]
        elif self.parent.facility == self.parent.facilityPAL:
            pass

        self.paramWidget = Parameter.create(name='paramsLabel', type='group', \
                                   children=self.params, expanded=True)
        self.win.setParameters(self.paramWidget, showTop=False)
        self.paramWidget.sigTreeStateChanged.connect(self.change)

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
                labelParamFname = self.parent.psocakeRunDir + '/pluginParam'
                if self.tag: labelParamFname += '_' + self.tag
                labelParamFname += '.json'
                if os.path.exists(labelParamFname):
                    with open(labelParamFname) as infile:
                        d = json.load(infile)
                        self.labelParam_pluginParam = d[self.labelParam_pluginParam_str]

    def writeStatus(self, fname, d):
        json.dump(d, open(fname, 'w'))

    # Launch labeling
    def findLabels(self):
        self.parent.thread.append(LaunchPlugin.LaunchPlugin(self.parent)) # send parent parameters with self
        self.parent.thread[self.parent.threadCounter].launch(self.parent.experimentName, self.parent.detInfo)
        self.parent.threadCounter+=1
        # Save peak finding parameters
        runsToDo = self.digestRunList(self.labelParam_runs)
        for run in runsToDo:
            pluginParamFname = self.parent.psocakeDir+'/r'+str(run).zfill(4)+'/pluginParam'
            if self.tag: pluginParamFname += '_'+self.tag
            pluginParamFname += '.json'
            if self.parent.facility == self.parent.facilityLCLS:
                d = {self.labelParam_pluginDir_str: self.labelParam_pluginParam,
                     self.labelParam_pluginParam_str: self.labelParam_pluginParam}
            elif self.parent.facility == self.parent.facilityPAL:
                pass
            if not os.path.exists(self.parent.psocakeDir+'/r'+str(run).zfill(4)):
                os.mkdir(self.parent.psocakeDir+'/r'+str(run).zfill(4))
            self.writeStatus(pluginParamFname, d)

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
        if path[0] == self.labelParam_grp:
            if path[1] == self.labelParam_outDir_str:
                self.labelParam_outDir = data
                self.labelParam_outDir_overridden = True
            elif path[1] == self.labelParam_pluginDir_str:
                self.updateAlgorithm(data)
            elif path[1] == self.labelParam_runs_str:
                self.labelParam_runs = data
            elif path[1] == self.labelParam_queue_str:
                self.labelParam_queue = data
            elif path[1] == self.labelParam_cpu_str:
                self.labelParam_cpus = data
            elif path[1] == self.labelParam_noe_str:
                self.labelParam_noe = data
            elif path[1] == self.labelParam_launch_str:
                self.findLabels()
            elif path[1] == self.tag_str:
                self.updateTag(data)
            elif path[1] == self.labelParam_pluginParam_str:
                self.labelParam_pluginParam = data

    def updateAlgorithm(self, data):
        self.algorithm = data
        self.algInitDone = False
        self.updateLabel()
        if self.parent.args.v >= 1: print "##### Done updateAlgorithm: ", self.algorithm

    def updateTag(self, data):
        self.tag = data

    def updateLabel(self):
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

            # Compute
            if self.algorithm == 0: # No algorithm
                self.labels = None
                self.drawLabels()
            else:
                if self.parent.facility == self.parent.facilityLCLS:
                    # Only initialize the hit finder algorithm once
                    if self.algInitDone is False:
                        # TODO: initialize plugin
                        self.algInitDone = True
                elif self.parent.facility == self.parent.facilityPAL:
                    pass

                # TODO: Run plugin
                if self.parent.args.v >= 1: print "Labels found: ", self.labels

                self.drawLabels()
            if self.parent.args.v >= 1: print "Done updateLabel"

    def drawLabels(self):
        # TODO: add roi objects
        pass

        if self.parent.args.v >= 1: print "Done drawLabels"
        self.parent.geom.drawCentre()

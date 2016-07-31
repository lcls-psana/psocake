from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.dockarea import *
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import LaunchHitFinder

class HitFinder(object):
    def __init__(self, parent = None):
        self.parent = parent

        ## Dock 13: Hit finder
        self.d13 = Dock("Hit Finder", size=(1, 1))
        self.w19 = ParameterTree()
        self.d13.addWidget(self.w19)
        self.w20 = pg.LayoutWidget()
        self.launchSpiBtn = QtGui.QPushButton('Launch hit finder')
        self.w20.addWidget(self.launchSpiBtn, row=1, col=0)
        self.d13.addWidget(self.w20)

        # Hit finding
        self.spiParam_grp = 'Hit finder'
        self.spiParam_algorithm_str = 'Algorithm'
        # algorithm 0
        self.spiParam_algorithm0_str = 'None'
        # algorithm 1
        self.spiParam_algorithm1_str = 'chiSquared'
        self.spiParam_alg1_pruneInterval_str = 'prune interval'
        # algorithm 2
        self.spiParam_algorithm2_str = 'photonFinder'
        self.spiParam_alg2_threshold_str = 'ADU per photon'

        self.spiParam_outDir_str = 'Output directory'
        self.spiParam_tag_str = 'Filename tag'
        self.spiParam_runs_str = 'Run(s)'
        self.spiParam_queue_str = 'queue'
        self.spiParam_cpu_str = 'CPUs'
        self.spiParam_psanaq_str = 'psanaq'
        self.spiParam_psnehq_str = 'psnehq'
        self.spiParam_psfehq_str = 'psfehq'
        self.spiParam_psnehprioq_str = 'psnehprioq'
        self.spiParam_psfehprioq_str = 'psfehprioq'
        self.spiParam_psnehhiprioq_str = 'psnehhiprioq'
        self.spiParam_psfehhiprioq_str = 'psfehhiprioq'
        self.spiParam_noe_str = 'Number of events to process'

        # Init hit finding
        self.spiAlgorithm = 2

        self.spiParam_alg1_pruneInterval = 0
        self.spiParam_alg2_threshold = 100

        self.spiParam_outDir = self.parent.psocakeDir
        self.spiParam_outDir_overridden = False
        self.spiParam_tag = None
        self.spiParam_runs = ''
        self.spiParam_queue = self.spiParam_psanaq_str
        self.spiParam_cpus = 24
        self.spiParam_noe = 0

        self.params = [
            {'name': self.spiParam_grp, 'type': 'group', 'children': [
                {'name': self.spiParam_algorithm_str, 'type': 'list', 'values': {self.spiParam_algorithm2_str: 2,
                                                                            #self.spiParam_algorithm1_str: 1,
                                                                                 self.spiParam_algorithm0_str: 0},
                                                                            'value': self.spiAlgorithm},
                {'name': self.spiParam_algorithm2_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
                    {'name': self.spiParam_alg2_threshold_str, 'type': 'float', 'value': self.spiParam_alg2_threshold, 'tip': "search for pixels above ADU per photon"},
                ]},
                {'name': self.spiParam_outDir_str, 'type': 'str', 'value': self.spiParam_outDir},
                {'name': self.spiParam_tag_str, 'type': 'str', 'value': self.spiParam_tag, 'tip': "(Optional) identifying string to attach to filename"},
                {'name': self.spiParam_runs_str, 'type': 'str', 'value': self.spiParam_runs, 'tip': "comma separated or use colon for a range, e.g. 1,3,5:7 = runs 1,3,5,6,7"},
                {'name': self.spiParam_queue_str, 'type': 'list', 'values': {self.spiParam_psfehhiprioq_str: 'psfehhiprioq',
                                                                             self.spiParam_psnehhiprioq_str: 'psnehhiprioq',
                                                                             self.spiParam_psfehprioq_str: 'psfehprioq',
                                                                             self.spiParam_psnehprioq_str: 'psnehprioq',
                                                                             self.spiParam_psfehq_str: 'psfehq',
                                                                             self.spiParam_psnehq_str: 'psnehq',
                                                                             self.spiParam_psanaq_str: 'psanaq'},
                 'value': self.spiParam_queue, 'tip': "Choose queue"},
                {'name': self.spiParam_cpu_str, 'type': 'int', 'value': self.spiParam_cpus},
                {'name': self.spiParam_noe_str, 'type': 'int', 'value': self.spiParam_noe, 'tip': "number of events to process, default=0 means process all events"},
            ]},
        ]
        self.p8 = Parameter.create(name='paramsHitFinder', type='group', \
                                   children=self.params, expanded=True)
        self.w19.setParameters(self.p8, showTop=False)
        self.p8.sigTreeStateChanged.connect(self.change)

        self.parent.connect(self.launchSpiBtn, QtCore.SIGNAL("clicked()"), self.findHits)

    # Launch hit finding
    def findHits(self):
        self.parent.thread.append(LaunchHitFinder.HitFinder(self))  # send parent parameters with self
        self.parent.thread[self.parent.threadCounter].findHits(self.parent.experimentName, self.parent.runNumber, self.parent.detInfo)
        self.parent.threadCounter += 1

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
        if path[0] == self.spiParam_grp:
            if path[1] == self.spiParam_algorithm_str:
                self.spiAlgorithm = data
            elif path[1] == self.spiParam_outDir_str:
                self.spiParam_outDir = data
                self.spiParam_outDir_overridden = True
            elif path[1] == self.spiParam_tag_str:
                self.spiParam_tag = data
            elif path[1] == self.spiParam_runs_str:
                self.spiParam_runs = data
            elif path[1] == self.spiParam_queue_str:
                self.spiParam_queue = data
            elif path[1] == self.spiParam_cpu_str:
                self.spiParam_cpus = data
            elif path[1] == self.spiParam_noe_str:
                self.spiParam_noe = data
            elif path[2] == self.spiParam_alg1_pruneInterval_str and path[1] == self.spiParam_algorithm1_str:
                self.spiParam_alg1_pruneInterval = data
            elif path[2] == self.spiParam_alg2_threshold_str and path[1] == self.spiParam_algorithm2_str:
                self.spiParam_alg2_threshold = data
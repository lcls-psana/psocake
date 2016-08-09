import numpy as np
import h5py
import os
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.dockarea import *

class Labels(object):
    def __init__(self, parent = None):
        
        #print "init!!!!!"
        self.parent = parent

        ## Dock: Labels
        self.dLabels = Dock("Labels", size=(1, 1))
        self.wLabels = ParameterTree()
        self.wLabels.setWindowTitle('Labels')
        self.dLabels.addWidget(self.wLabels)

        self.labels_grp = 'Labels'
        self.labels_A_str = 'Single'
        self.labels_B_str = 'Multi'
        self.labels_C_str = 'Dunno'

        self.labelA = False
        self.labelB = False
        self.labelC = False
        #######################
        # Mandatory parameter #
        #######################
        self.params = [
            {'name': self.labels_grp, 'type': 'group', 'children': [
                {'name': self.labels_A_str, 'type': 'bool', 'value': self.labelA, 'tip': "Single"},
                {'name': self.labels_B_str, 'type': 'bool', 'value': self.labelB, 'tip': "Multi"},
                {'name': self.labels_C_str, 'type': 'bool', 'value': self.labelC, 'tip': "Dunno"},
            ]},
        ]

        self.pLabels = Parameter.create(name='paramsLabel', type='group', \
                                   children=self.params, expanded=True)
        self.pLabels.sigTreeStateChanged.connect(self.change)
        self.wLabels.setParameters(self.pLabels, showTop=False)

    # If anything changes in the parameter tree, print a message
    def change(self, panel, changes):
        for param, change, data in changes:
            path = panel.childPath(param)
            if self.parent.args.v >= 1:
                print('  path: %s' % path)
                print('  change:    %s' % change)
                print('  data:      %s' % str(data))
                print('  ----------')
            self.paramUpdate(path, data)

    ##############################
    # Mandatory parameter update #
    ##############################
    def paramUpdate(self, path, data):
        global dset
        if path[1] == self.labels_A_str:
            self.labelA = data
            if data:
                dset[self.parent.eventNumber] = 1
            else:
                dset[self.parent.eventNumber] = 0
        elif path[1] == self.labels_B_str:
            self.labelB = data
            if data:
                dset[self.parent.eventNumber] = 2
            else:
                dset[self.parent.eventNumber] = 0
        elif path[1] == self.labels_C_str:
            self.labelC = data
            if data:
                dset[self.parent.eventNumber] = 3
            else:
                dset[self.parent.eventNumber] = 0

    def refresh(self):
        fname = self.parent.psocakeRunDir + '/' + self.parent.experimentName + '_' + str(self.parent.runNumber).zfill(4) + '_labels.h5'
        global dset
        dataSetFound = False
        if self.parent.runNumber > 0:
            if os.path.exists(fname):
               labels = h5py.File(fname, 'r+', dtype = 'i8')
               dataSetFound = True
            else:
               labels = h5py.File(fname, 'x', dtype = 'i8')
            if not dataSetFound:
               dset = labels.create_dataset("labels", (self.parent.exp.eventTotal, 1))
            else:
                try:
                    dset = labels["labels"]
                except: # corrupt dataset, so create a new one
                    dset = labels.create_dataset("labels", (self.parent.exp.eventTotal, 1))
            #print dset.shape
            self.labelA = False
            self.labelB = False
            self.labelC = False
            if dset[self.parent.eventNumber] == 1:
                self.labelA = True
            elif dset[self.parent.eventNumber] == 2:
                self.labelB = True
            elif dset[self.parent.eventNumber] == 3:
                self.labelC = True
            self.pLabels.param(self.labels_grp, self.labels_A_str).setValue(self.labelA)
            self.pLabels.param(self.labels_grp, self.labels_B_str).setValue(self.labelB)
            self.pLabels.param(self.labels_grp, self.labels_C_str).setValue(self.labelC)

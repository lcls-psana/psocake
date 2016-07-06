import numpy as np
import fileinput
import pyqtgraph as pg
import h5py
import os

class Labels(object):
    def __init__(self, parent = None):
        self.parent = parent

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

    ##############################
    # Mandatory parameter update #
    ##############################
    def paramUpdate(self, path, change, data):
        if path[1] == self.labels_A_str:
            self.labelA = data
        elif path[1] == self.labels_B_str:
            self.labelB = data
        elif path[1] == self.labels_C_str:
            self.labelC = data

    def refresh(self):
        self.labelA = False
        self.labelB = False
        self.labelC = False
        self.parent.pLabels.param(self.labels_grp, self.labels_A_str).setValue(self.labelA)
        self.parent.pLabels.param(self.labels_grp, self.labels_B_str).setValue(self.labelB)
        self.parent.pLabels.param(self.labels_grp, self.labels_C_str).setValue(self.labelC)

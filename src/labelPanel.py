import numpy as np
import fileinput
import pyqtgraph as pg
import h5py
import os

class Labels(object):
    def __init__(self, parent = None):
        self.parent = parent

        self.labels_grp = 'Labels'
        self.labels_A_str = 'Label A'
        self.labels_B_str = 'Label B'

        self.labelA = False
        self.labelB = False

        #######################
        # Mandatory parameter #
        #######################
        self.params = [
            {'name': self.labels_grp, 'type': 'group', 'children': [
                {'name': self.labels_A_str, 'type': 'bool', 'value': self.labelA, 'tip': "Label A"},
                {'name': self.labels_B_str, 'type': 'bool', 'value': self.labelB, 'tip': "Label B"},
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

    def refresh(self):
        self.labelA = False
        self.labelB = False
        self.parent.pLabels.param(self.labels_grp, self.labels_A_str).setValue(self.labelA)
        self.parent.pLabels.param(self.labels_grp, self.labels_B_str).setValue(self.labelB)
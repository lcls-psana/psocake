import numpy as np
import fileinput
import pyqtgraph as pg
import h5py
import os

class Labels(object):
    def __init__(self, parent = None):
        
        #print "init!!!!!"
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
    def paramUpdate(self, path, data):
        global dset
        if path[1] == self.labels_A_str:
            self.labelA = data
            if data:
                dset[self.parent.eventNumber] = np.array([1,dset[self.parent.eventNumber][1], dset[self.parent.eventNumber][2]])
            else:
                dset[self.parent.eventNumber] = np.array([0,dset[self.parent.eventNumber][1], dset[self.parent.eventNumber][2]])
        elif path[1] == self.labels_B_str:
            self.labelB = data
            if data:
                dset[self.parent.eventNumber] = np.array([dset[self.parent.eventNumber][0],1, dset[self.parent.eventNumber][2]])
            else:
                dset[self.parent.eventNumber] = np.array([dset[self.parent.eventNumber][0],0, dset[self.parent.eventNumber][2]])
        elif path[1] == self.labels_C_str:
            self.labelC = data
            if data:
                dset[self.parent.eventNumber] = np.array([dset[self.parent.eventNumber][0],dset[self.parent.eventNumber][1], 1])
            else:
                dset[self.parent.eventNumber] = np.array([dset[self.parent.eventNumber][0],dset[self.parent.eventNumber][1], 0])
    def refresh(self):
        global dset
        dataSetFound = False
        if os.path.exists('%s/Exp%sRun%d.hdf5' %(self.parent.psocakeRunDir,self.parent.experimentName,self.parent.runNumber)):
           labels = h5py.File('%s/Exp%sRun%d.hdf5' %(self.parent.psocakeRunDir,self.parent.experimentName,self.parent.runNumber), 'r+', dtype = 'i8')
           dataSetFound = True
           #print "exists in " + self.parent.psocakeRunDir
        else:
           labels = h5py.File('%s/Exp%sRun%d.hdf5' %(self.parent.psocakeRunDir,self.parent.experimentName,self.parent.runNumber), 'x', dtype = 'i8')
        if not dataSetFound:
           dset = labels.create_dataset("labelsDataSet", (self.parent.eventTotal, 3))
        else:
           dset = labels["labelsDataSet"]
        #print dset.shape
        self.labelA = dset[self.parent.eventNumber][0]
        self.labelB = dset[self.parent.eventNumber][1]
        self.labelC = dset[self.parent.eventNumber][2]
        if dset[self.parent.eventNumber][0] == 1:
            self.labelA = True
        elif dset[self.parent.eventNumber][1] == 1:
            self.labelB = True
        elif dset[self.parent.eventNumber][2] == 1:
            self.labelC = True
        self.labelC = False
        self.parent.pLabels.param(self.labels_grp, self.labels_A_str).setValue(self.labelA)
        self.parent.pLabels.param(self.labels_grp, self.labels_B_str).setValue(self.labelB)
        self.parent.pLabels.param(self.labels_grp, self.labels_C_str).setValue(self.labelC)

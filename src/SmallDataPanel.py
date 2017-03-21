import numpy as np
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.dockarea import *
from pyqtgraph.Qt import QtCore, QtGui
import subprocess
#import pandas as pd
import h5py, os
import pyqtgraph as pg

class SmallData(object):
    def __init__(self, parent = None):
        self.parent = parent

        ## Dock 8: Quantifier
        self.dSmall = Dock("Small Data", size=(100, 100))
        self.w8 = ParameterTree()
        self.dSmall.addWidget(self.w8)
        self.w11a = pg.LayoutWidget()
        self.refreshBtn = QtGui.QPushButton('Refresh')
        self.w11a.addWidget(self.refreshBtn, row=0, col=0)
        self.dSmall.addWidget(self.w11a)
        # Add plot
        self.w9 = pg.PlotWidget(title="Metric")
        self.dSmall.addWidget(self.w9)

        # Quantifier parameter tree
        self.quantifier_grp = 'Small data'
        self.quantifier_filename_str = 'filename'
        self.quantifier_dataset_str = 'dataset'
        self.quantifier_sort_str = 'sort'

        # Quantifier
        self.quantifier_filename = ''
        self.quantifier_dataset = '/entry_1/result_1/'
        self.quantifier_sort = False
        self.quantifierFileOpen = False
        self.quantifierHasData = False

        self.params = [
            {'name': self.quantifier_grp, 'type': 'group', 'children': [
                {'name': self.quantifier_filename_str, 'type': 'str', 'value': self.quantifier_filename, 'tip': "Full path Hdf5 filename"},
                {'name': self.quantifier_dataset_str, 'type': 'str', 'value': self.quantifier_dataset, 'tip': "Hdf5 dataset metric, nPeaksAll or nHitsAll"},
                {'name': self.quantifier_sort_str, 'type': 'bool', 'value': self.quantifier_sort, 'tip': "Ascending sort metric"},
            ]},
        ]

        self.pSmall = Parameter.create(name='paramsQuantifier', type='group', \
                                       children=self.params, expanded=True)
        self.w8.setParameters(self.pSmall, showTop=False)
        self.pSmall.sigTreeStateChanged.connect(self.change)
        self.parent.connect(self.refreshBtn, QtCore.SIGNAL("clicked()"), self.reloadQuantifier)

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
        if path[0] == self.quantifier_grp:
            if path[1] == self.quantifier_filename_str:
                self.updateQuantifierFilename(data)
            elif path[1] == self.quantifier_dataset_str:
                self.updateQuantifierDataset(data)
            elif path[1] == self.quantifier_sort_str:
                self.updateQuantifierSort(data)

    ##################################
    ########### Quantifier ###########
    ##################################

    def reloadQuantifier(self):
        self.updateQuantifierFilename(self.quantifier_filename)
        self.updateQuantifierDataset(self.quantifier_dataset)

    def updateQuantifierFilename(self, data):
        self.quantifier_filename = data
        if self.parent.args.v >= 1: print "Done opening metric"

    def updateQuantifierDataset(self, data):
        self.quantifier_dataset = data
        if os.path.isfile(self.quantifier_filename):
            try:
                self.quantifierFile = h5py.File(self.quantifier_filename, 'r')
                self.quantifierMetric = self.quantifierFile[self.quantifier_dataset].value
                try:
                    if self.quantifier_dataset[0] == '/':  # dataset starts with "/"
                        self.quantifier_eventDataset = self.quantifier_dataset.split("/")[1] + "/event"
                    else:  # dataset does not start with "/"
                        self.quantifier_eventDataset = "/" + self.quantifier_dataset.split("/")[0] + "/event"
                    self.quantifierEvent = self.quantifierFile[self.quantifier_eventDataset].value
                except:
                    if self.parent.args.v >= 1: print "Couldn't find /event dataset"
                    self.quantifierEvent = np.arange(len(self.quantifierMetric))
                self.quantifierFile.close()
                self.quantifierInd = np.arange(len(self.quantifierMetric))
                self.updateQuantifierSort(self.quantifier_sort)
            except:
                print "Couldn't read metric"
            if self.parent.args.v >= 1: print "Done reading metric"

    def updateQuantifierSort(self, data):
        self.quantifier_sort = data
        try:
            if self.quantifier_sort is True:
                self.quantifierIndSorted = np.argsort(self.quantifierMetric)
                self.quantifierMetricSorted = self.quantifierMetric[self.quantifierIndSorted]
                #self.quantifierEventSorted = self.quantifierMetric[self.quantifierInd]
                self.updateQuantifierPlot(self.quantifierMetricSorted)
            else:
                self.updateQuantifierPlot(self.quantifierMetric)
        except:
            print "Couldn't sort data"
            pass

    def updateQuantifierPlot(self, metric):
        self.w9.getPlotItem().clear()
        if len(np.where(metric==-1)[0]) > 0:
            self.curve = self.w9.plot(metric, pen=(200, 200, 200), symbolBrush=(255, 0, 0), symbolPen='k') # red
        else: # Every event was processed
            self.curve = self.w9.plot(metric, pen=(200, 200, 200), symbolBrush=(0, 0, 255), symbolPen='k') # blue
        self.w9.setLabel('left', "Small data")
        if self.quantifier_sort:
            self.w9.setLabel('bottom', "Sorted Event Index")
        else:
            self.w9.setLabel('bottom', "Event Index")
        self.curve.curve.setClickable(True)
        self.curve.sigClicked.connect(self.clicked)

    def clicked(self, points):
        with pg.BusyCursor():
            if self.parent.args.v >= 1:
                print("curve clicked", points)
                from pprint import pprint
                pprint(vars(points.scatter))
            for i in range(len(points.scatter.data)):
                if points.scatter.ptsClicked[0] == points.scatter.data[i][7]:
                    ind = i
                    break
            indX = points.scatter.data[i][0]
            indY = points.scatter.data[i][1]
            if self.parent.args.v >= 1: print "x,y: ", indX, indY
            if self.quantifier_sort:
                ind = self.quantifierIndSorted[ind]

            # temp
            self.parent.eventNumber = self.quantifierEvent[ind]

            self.parent.calib, self.parent.data = self.parent.img.getDetImage(self.parent.eventNumber)
            self.parent.img.w1.setImage(self.parent.data, autoRange=False, autoLevels=False, autoHistogramRange=False)
            self.parent.exp.p.param(self.parent.exp.exp_grp, self.parent.exp.exp_evt_str).setValue(self.parent.eventNumber)

from pyqtgraph.Qt import QtCore
import subprocess
import os
import numpy as np

class StackProducer(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        self.exiting = False
        self.parent = parent
        self.startIndex = 0
        self.numImages = 0
        self.evt = None
        self.data = None

    def __del__(self):
        self.exiting = True
        self.wait()

    def load(self, startIndex, numImages):
        self.startIndex = startIndex
        self.numImages = numImages
        self.start()

    def run(self):
        counter = 0
        for i in np.arange(self.startIndex, self.startIndex+self.numImages):
            if counter == 0:
                calib, data = self.parent.img.getDetImage(i, calib=None)
                self.data = np.zeros((self.numImages, data.shape[0], data.shape[1]))
                if data is not None:
                    self.data[counter,:,:] = data
                counter += 1
            else:
                calib, data = self.parent.img.getDetImage(i, calib=None)
                if data is not None:
                    self.data[counter,:,:] = data
                counter += 1

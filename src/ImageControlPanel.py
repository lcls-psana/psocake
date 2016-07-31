from pyqtgraph.dockarea import *
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.pyplot as plt
from PSCalib.GeometryObject import data2x2ToTwo2x1, two2x1ToData2x2

class ImageControl(object):
    def __init__(self, parent = None):
        self.parent = parent

        ## Dock 6: Image Control
        self.nextBtn = QtGui.QPushButton('Next evt')
        self.prevBtn = QtGui.QPushButton('Prev evt')
        self.saveBtn = QtGui.QPushButton('Save evt')
        self.loadBtn = QtGui.QPushButton('Load image')

        #############################
        # Dock 6: Image Control
        #############################
        self.d6 = Dock("Image Control", size=(1, 1))
        self.w6 = pg.LayoutWidget()
        self.w6.addWidget(self.prevBtn, row=0, col=0)
        self.w6.addWidget(self.nextBtn, row=0, col=1)
        self.w6.addWidget(self.saveBtn, row=1, col=0)
        self.w6.addWidget(self.loadBtn, row=1, col=1)
        self.d6.addWidget(self.w6)

        self.nextBtn.clicked.connect(self.nextEvt)
        self.prevBtn.clicked.connect(self.prevEvt)
        self.saveBtn.clicked.connect(self.save)
        self.loadBtn.clicked.connect(self.load)

    def nextEvt(self):
        self.parent.eventNumber += 1
        if self.parent.eventNumber >= self.parent.exp.eventTotal:
            self.parent.eventNumber = self.parent.exp.eventTotal-1
        else:
            self.parent.calib, self.parent.data = self.parent.img.getDetImage(self.parent.eventNumber)
            self.parent.img.w1.setImage(self.parent.data,autoRange=False,autoLevels=False,autoHistogramRange=False)
            self.parent.exp.p.param(self.parent.exp.exp_grp,self.parent.exp.exp_evt_str).setValue(self.parent.eventNumber)

    def prevEvt(self):
        self.parent.eventNumber -= 1
        if self.parent.eventNumber < 0:
            self.parent.eventNumber = 0
        else:
            self.parent.calib, self.parent.data = self.parent.img.getDetImage(self.parent.eventNumber)
            self.parent.img.w1.setImage(self.parent.data,autoRange=False,autoLevels=False,autoHistogramRange=False)
            self.parent.exp.p.param(self.parent.exp.exp_grp,self.parent.exp.exp_evt_str).setValue(self.parent.eventNumber)

    def save(self):
        outputName = self.parent.psocakeRunDir+"/psocake_"+str(self.parent.experimentName)+"_"+str(self.parent.runNumber)+"_"+str(self.parent.detInfo)+"_" \
                     +str(self.parent.eventNumber)+"_"+str(self.parent.exp.eventSeconds)+"_"+str(self.parent.exp.eventNanoseconds)+"_" \
                     +str(self.parent.exp.eventFiducial)+".npy"
        fname = QtGui.QFileDialog.getSaveFileName(self.parent, 'Save file', outputName, 'ndarray image (*.npy)')
        if self.parent.exp.logscaleOn:
            np.save(str(fname),np.log10(abs(self.parent.calib) + self.parent.eps))
        else:
            if self.parent.calib.size==2*185*388: # cspad2x2
                asData2x2 = two2x1ToData2x2(self.parent.calib)
                np.save(str(fname),asData2x2)
                np.savetxt(str(fname).split('.')[0]+".txt", asData2x2.reshape((-1,asData2x2.shape[-1])) ,fmt='%0.18e')
            else:
                np.save(str(fname),self.parent.calib)
                np.savetxt(str(fname).split('.')[0]+".txt", self.parent.calib.reshape((-1,self.parent.calib.shape[-1])) )#,fmt='%0.18e')

    def load(self):
        fname = str(QtGui.QFileDialog.getOpenFileName(self.parent, 'Open file', self.parent.psocakeRunDir, 'ndarray image (*.npy *.npz)'))
        if fname.split('.')[-1] in '.npz':
            temp = np.load(fname)
            self.parent.calib = temp['max']
        else:
            self.parent.calib = np.load(fname)
        self.parent.img.updateImage(self.parent.calib)





from pyqtgraph.dockarea import *
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import os

if 'LCLS' in os.environ['PSOCAKE_FACILITY'].upper():
    from PSCalib.GeometryObject import data2x2ToTwo2x1, two2x1ToData2x2
elif 'PAL' in os.environ['PSOCAKE_FACILITY'].upper():
    pass

class ImageControl(object):
    def __init__(self, parent = None):
        self.parent = parent

        ## Dock: Image Control
        self.nextBtn = QtGui.QPushButton('Next evt')
        self.prevBtn = QtGui.QPushButton('Prev evt')
        self.saveBtn = QtGui.QPushButton('Save evt')
        self.loadBtn = QtGui.QPushButton('Load image')

        #############################
        # Dock: Image Control
        #############################
        self.dock = Dock("Image Control", size=(1, 1))
        self.dock.hideTitleBar()
        self.win = pg.LayoutWidget()
        self.win.addWidget(self.prevBtn, row=0, col=0)
        self.win.addWidget(self.nextBtn, row=0, col=1)
        self.win.addWidget(self.saveBtn, row=1, col=0)
        self.win.addWidget(self.loadBtn, row=1, col=1)
        self.dock.addWidget(self.win)

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
            self.parent.img.win.setImage(self.parent.data,autoRange=False,autoLevels=False,autoHistogramRange=False)
            self.parent.exp.p.param(self.parent.exp.exp_grp,self.parent.exp.exp_evt_str).setValue(self.parent.eventNumber)

    def prevEvt(self):
        self.parent.eventNumber -= 1
        if self.parent.eventNumber < 0:
            self.parent.eventNumber = 0
        else:
            self.parent.calib, self.parent.data = self.parent.img.getDetImage(self.parent.eventNumber)
            self.parent.img.win.setImage(self.parent.data,autoRange=False,autoLevels=False,autoHistogramRange=False)
            self.parent.exp.p.param(self.parent.exp.exp_grp,self.parent.exp.exp_evt_str).setValue(self.parent.eventNumber)

    def save(self):
        output = self.parent.psocakeRunDir+"/psocake_"+str(self.parent.experimentName)+"_"+str(self.parent.runNumber)+"_"+str(self.parent.detInfo)+"_" \
                     +str(self.parent.eventNumber)+"_"+str(self.parent.exp.eventSeconds)+"_"+str(self.parent.exp.eventNanoseconds)+"_" \
                     +str(self.parent.exp.eventFiducial)
        print "##########################################"
        print "Saving unassembled image: ", output + "_unassembled.npy"
        print "Saving unassembled image: ", output + "_unassembled.txt"
        outputUnassem = output + "_unassembled.npy"
        if self.parent.calib.size==2*185*388: # cspad2x2
            asData2x2 = two2x1ToData2x2(self.parent.calib)
            np.save(str(outputUnassem),asData2x2)
            np.savetxt(str(outputUnassem).split('.npy')[0]+".txt", asData2x2.reshape((-1,asData2x2.shape[-1])) ,fmt='%0.18e')
        else:
            np.save(str(outputUnassem),self.parent.calib)
            np.savetxt(str(outputUnassem).split('.npy')[0]+".txt", self.parent.calib.reshape((-1,self.parent.calib.shape[-1])) )#,fmt='%0.18e')
        # Save assembled
        outputAssem = output + "_assembled.npy"
        print "##########################################"
        print "Saving assembled image: ", outputAssem
        print "##########################################"
        np.save(str(outputAssem), self.parent.det.image(self.parent.evt, self.parent.calib))

    def load(self):
        fname = str(QtGui.QFileDialog.getOpenFileName(self.parent, 'Open file', self.parent.psocakeRunDir, 'ndarray image (*.npy *.npz)'))
        if fname.split('.')[-1] in '.npz':
            temp = np.load(fname)
            self.parent.calib = temp['max']
        else:
            self.parent.calib = np.load(fname)
        self.parent.firstUpdate = True
        #self.parent.pk.userUpdate = None
        self.parent.img.updateImage(self.parent.calib)
        self.parent.pk.updateClassification()





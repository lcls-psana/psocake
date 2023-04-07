from pyqtgraph.dockarea import *
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets
import os

from PSCalib.GeometryObject import data2x2ToTwo2x1, two2x1ToData2x2

class ImageControl(object):
    def __init__(self, parent = None):
        self.parent = parent

        ## Dock: Image Control
        self.nextBtn = QtWidgets.QPushButton('Next evt')
        self.prevBtn = QtWidgets.QPushButton('Prev evt')
        self.saveBtn = QtWidgets.QPushButton('Save evt')
        self.loadBtn = QtWidgets.QPushButton('Load image')

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
            self.parent.exp.p.param(self.parent.exp.exp_grp,self.parent.exp.exp_evt_str).setValue(self.parent.eventNumber)
        if 'label' in self.parent.args.mode:
            self.parent.labeling.updateText()

    def prevEvt(self):
        self.parent.eventNumber -= 1
        if self.parent.eventNumber < 0:
            self.parent.eventNumber = 0
        else:
            self.parent.calib, self.parent.data = self.parent.img.getDetImage(self.parent.eventNumber)
            self.parent.exp.p.param(self.parent.exp.exp_grp,self.parent.exp.exp_evt_str).setValue(self.parent.eventNumber)
        if 'label' in self.parent.args.mode:
            self.parent.labeling.updateText()

    def save(self):
        output = self.parent.psocakeRunDir+"/psocake_"+str(self.parent.experimentName)+"_"+str(self.parent.runNumber)+"_"+str(self.parent.detInfo)+"_" \
                     +str(self.parent.eventNumber)+"_"+str(self.parent.exp.eventSeconds)+"_"+str(self.parent.exp.eventNanoseconds)+"_" \
                     +str(self.parent.exp.eventFiducial)
        print("##########################################")
        print("Saving unassembled image: ", output + "_unassembled.npy")
        print("Saving unassembled image: ", output + "_unassembled.txt")
        outputUnassem = output + "_unassembled.npy"
        _calib = self.parent.calib.copy()

        if _calib.size==2*185*388: # cspad2x2
            asData2x2 = two2x1ToData2x2(_calib)
            np.save(str(outputUnassem),asData2x2)
            np.savetxt(str(outputUnassem).split('.npy')[0]+".txt", asData2x2.reshape((-1,asData2x2.shape[-1])) ,fmt='%0.18e')
        else:
            np.save(str(outputUnassem), _calib)
            np.savetxt(str(outputUnassem).split('.npy')[0]+".txt", _calib.reshape((-1,_calib.shape[-1])) )#,fmt='%0.18e')
        # Save assembled
        outputAssem = output + "_assembled.npy"
        print("##########################################")
        print("Saving assembled image: ", outputAssem)
        print("##########################################")
        np.save(str(outputAssem), self.parent.det.image(self.parent.evt, _calib))

        # Save publication quality images
        vmin, vmax = self.parent.img.win.getLevels()
        dimY,dimX = self.parent.data.shape
        distEdge = min(dimX - self.parent.cx, dimY - self.parent.cy)
        thetaMax = np.arctan(distEdge * self.parent.pixelSize / self.parent.detectorDistance)
        qMax_crystal = 2 / self.parent.wavelength * np.sin(thetaMax / 2)
        dMin_crystal = 1 / qMax_crystal
        res = '%.3g A' % (dMin_crystal * 1e10)
        resColor = '#0497cb'
        textSize = 24
        move = 3 * textSize  # move text to the left
        img = self.parent.det.image(self.parent.evt, self.parent.calib * self.parent.mk.combinedMask)
        cx = self.parent.cx
        cy = self.parent.cy
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        outputName = output + "_img.png"
        print("##########################################")
        print("Saving image as png: ", outputName)
        print("##########################################")
        fix, ax = plt.subplots(1, figsize=(10, 10))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        ax.imshow(img, cmap='binary', vmax=vmax, vmin=vmin)
        plt.savefig(outputName, dpi=300)

        outputName = output + "_pks.png"
        if self.parent.pk.peaks is not None and self.parent.pk.numPeaksFound > 0:
            print("##########################################")
            print("Saving peak image as png: ", outputName)
            print("##########################################")
            cenX, cenY = self.parent.pk.assemblePeakPos(self.parent.pk.peaks)
            fix, ax = plt.subplots(1, figsize=(10, 10))
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            ax.imshow(img, cmap='binary', vmax=vmax, vmin=vmin)
            boxDim = self.parent.pk.hitParam_alg1_rank * 2
            for i in range(self.parent.pk.peaks.shape[0]):
                rect = patches.Rectangle((cenY[i] - self.parent.pk.hitParam_alg1_rank, cenX[i] - self.parent.pk.hitParam_alg1_rank), boxDim, boxDim, linewidth=0.5,
                                         edgecolor='#0497cb', facecolor='none')
                ax.add_patch(rect)
            # Resolution ring
            circ = patches.Circle((cx, cy), radius=distEdge, linewidth=1, edgecolor=resColor, facecolor='none')
            ax.add_patch(circ)
            plt.text(cx - textSize / 2 - move, cy + distEdge, res, size=textSize, color=resColor)
            plt.savefig(outputName, dpi=300)

        outputName = output + "_idx.png"
        if self.parent.index.indexedPeaks is not None:
            print("##########################################")
            print("Saving indexed image as png: ", outputName)
            print("##########################################")
            numIndexedPeaksFound = self.parent.index.indexedPeaks.shape[0]
            intRadius = self.parent.index.intRadius
            cenX1 = self.parent.index.indexedPeaks[:, 0] + 0.5
            cenY1 = self.parent.index.indexedPeaks[:, 1] + 0.5
            diameter0 = np.ones_like(cenX1)
            diameter1 = np.ones_like(cenX1)
            diameter2 = np.ones_like(cenX1)
            diameter0[0:numIndexedPeaksFound] = float(intRadius.split(',')[0]) * 2
            diameter1[0:numIndexedPeaksFound] = float(intRadius.split(',')[1]) * 2
            diameter2[0:numIndexedPeaksFound] = float(intRadius.split(',')[2]) * 2
            fix, ax = plt.subplots(1, figsize=(10, 10))
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            ax.imshow(img, cmap='binary', vmax=vmax, vmin=vmin)
            for i in range(numIndexedPeaksFound):
                circ = patches.Circle((cenY1[i], cenX1[i]), radius=diameter1[i] / 2., linewidth=0.5, edgecolor='#c84c6b',
                                      facecolor='none')
                ax.add_patch(circ)
            # Resolution ring
            circ = patches.Circle((cx, cy), radius=distEdge, linewidth=1, edgecolor=resColor, facecolor='none')
            ax.add_patch(circ)
            plt.text(cx - textSize / 2 - move, cy + distEdge, res, size=textSize, color=resColor)
            plt.savefig(outputName, dpi=300)

    def load(self):
        fname = str(QtWidgets.QFileDialog.getOpenFileName(self.parent, 'Open file', self.parent.psocakeRunDir, 'ndarray image (*.npy *.npz)')[0])
        if fname:
            if '.npz' in str(fname.split('.')[-1]):
                temp = np.load(fname)
                self.parent.calib = temp['max']
            else:
                self.parent.calib = np.load(fname)
            self.parent.firstUpdate = True
            self.parent.img.updateImage(self.parent.calib)
            self.parent.pk.updateClassification()
            if 'label' in self.parent.args.mode:
                self.parent.label.updateText()
        else:
            print("No such file or directory")





from pyqtgraph.dockarea import *
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
try:
    from PyQt5.QtWidgets import *
    using_pyqt4 = False
except ImportError:
    using_pyqt4 = True
    pass

from psocake import LaunchStackProducer

class ImageStack(object):
    def __init__(self, parent = None):
        self.parent = parent

        self.stackStart = 0
        self.stackSizeMax = 120
        self.stackSize = 20

        ## Dock 3: Image Stack
        self.dock = Dock("Image Scroll", size=(1, 1))
        self.winL = pg.LayoutWidget()
        self.win = pg.ImageView(view=pg.PlotItem())
        self.win.getView().invertY(False)
        self.scroll = np.random.random((5, 10, 10))
        self.win.setImage(self.scroll, xvals=np.linspace(0., self.scroll.shape[0] - 1, self.scroll.shape[0]))
        self.spinBox = QtGui.QSpinBox()
        self.spinBox.setValue(0)
        self.label = QtGui.QLabel("Event Number:")
        self.stackSizeBox = QtGui.QSpinBox()
        self.stackSizeBox.setMaximum(self.stackSizeMax)
        self.stackSizeBox.setValue(self.stackSize)
        self.startBtn = QtGui.QPushButton("&Load image stack")
        # Connect listeners to functions
        self.winL.addWidget(self.win, row=0, colspan=4)
        self.winL.addWidget(self.label, 1, 0)
        self.winL.addWidget(self.spinBox, 1, 1)
        self.winL.addWidget(self.stackSizeBox, 1, 2)
        self.winL.addWidget(self.startBtn, 1, 3)
        self.dock.addWidget(self.winL)

        self.threadpool = LaunchStackProducer.StackProducer(self.parent) # send parent parameters
        if using_pyqt4:
            self.parent.connect(self.threadpool, QtCore.SIGNAL("finished()"), self.displayImageStack)
            self.parent.connect(self.startBtn, QtCore.SIGNAL("clicked()"), self.loadStack)
        else:
            self.threadpool.finished.connect(self.displayImageStack)
            self.startBtn.clicked.connect(self.loadStack)

    # Loading image stack
    def displayImageStack(self):
        if self.parent.exp.logscaleOn:
            self.win.setImage(np.log10(abs(self.threadpool.data) + self.parent.eps), xvals=np.linspace(self.stackStart,
                                                                                               self.stackStart +
                                                                                               self.threadpool.data.shape[0] - 1,
                                                                                               self.threadpool.data.shape[0]))
        else:
            self.win.setImage(self.threadpool.data, xvals=np.linspace(self.stackStart,
                                                                     self.stackStart + self.threadpool.data.shape[0] - 1,
                                                                     self.threadpool.data.shape[0]))
        self.startBtn.setEnabled(True)
        if self.parent.args.v >= 1: print("Done display image stack!!!!!")

    def loadStack(self):
        self.stackStart = self.spinBox.value()
        self.stackSize = self.stackSizeBox.value()
        self.threadpool.load(self.stackStart, self.stackSize)
        self.startBtn.setEnabled(False)
        self.win.getView().setTitle("exp=" + self.parent.experimentName + ":run=" + str(self.parent.runNumber) +
                                   ":evt" + str(self.stackStart) + "-" + str(self.stackStart + self.stackSize))


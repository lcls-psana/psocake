# GUI for browsing LCLS area detectors. Tune hit finding parameters and common mode correction.

# TODO: Multiple subplots or grid of images
# TODO: dropdown menu for available detectors, dropdown menu for small data
# TODO: When front and back detectors given, display both
# TODO: Display acqiris
# TODO: Downsampler
# TODO: Radial background, polarization correction
# TODO: Run from ffb
# TODO: script for histogram of lit pixels, percentile display
# TODO: hit threshold on hit finding panel

# TODO: Launch jobs based on image_property
# TODO: Manifold embed for ice events
# TODO: update ROI on every event, but skip if ROI is outside
# TODO: update clen

import sys, signal
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.console
from pyqtgraph.dockarea import *
from pyqtgraph.dockarea.Dock import DockLabel
from pyqtgraph.parametertree import Parameter, ParameterTree#, ParameterItem, registerParameterType
import psana
import h5py
import argparse
import time
import subprocess
import os.path

# Panel modules
import diffractionGeometryPanel, crystalIndexingPanel, SmallDataPanel, ExperimentPanel
import PeakFindingPanel, HitFinderPanel, MaskPanel, LabelPanel, ImagePanel, RoiPanel
import ImageControlPanel, MousePanel, ImageStackPanel
import LaunchPeakFinder, LaunchIndexer, LaunchStackProducer
import matplotlib.pyplot as plt
import _colorScheme as color
from _version import __version__

parser = argparse.ArgumentParser()
parser.add_argument('expRun', nargs='?', default=None, help="Psana-style experiment/run string in the format (e.g. exp=cxi06216:run=22). This option trumps -e and -r options.")
parser.add_argument("-e","--exp", help="Experiment name (e.g. cxis0813 ). This option is ignored if expRun option is used.", default="", type=str)
parser.add_argument("-r","--run", help="Run number. This option is ignored if expRun option is used.",default=0, type=int)
parser.add_argument("-d","--det", help="Detector alias or DAQ name (e.g. DscCsPad or CxiDs1.0:Cspad.0), default=''",default="", type=str)
parser.add_argument("-n","--evt", help="Event number (e.g. 1), default=0",default=0, type=int)
parser.add_argument("--localCalib", help="Use local calib directory. A calib directory must exist in your current working directory.", action='store_true')
parser.add_argument("-o","--outDir", help="Use this directory for output instead.", default=None, type=str)
parser.add_argument("-v", help="verbosity level, default=0",default=0, type=int)
parser.add_argument('--version', action='version',
                    version='%(prog)s {version}'.format(version=__version__))
parser.add_argument("-m","--mode", help="Mode sets the combination of panels available on the GUI, options: {lite,sfx,spi,all}",default="lite", type=str)
args = parser.parse_args()

class Window(QtGui.QMainWindow):
    global ex

    def previewEvent(self, eventNumber):
        ex.eventNumber = eventNumber
        ex.calib, ex.data = ex.img.getDetImage(ex.eventNumber)
        ex.img.w1.setImage(ex.data, autoRange=False, autoLevels=False, autoHistogramRange=False)
        ex.exp.p.param(ex.exp.exp_grp, ex.exp.exp_evt_str).setValue(ex.eventNumber)

    def keyPressEvent(self, event):
        super(Window, self).keyPressEvent(event)
        if args.mode == "all":
            if type(event) == QtGui.QKeyEvent:
                path = ["", ""]
                if event.key() == QtCore.Qt.Key_1:
                    path[1] = "Single"
                    data = True
                    if ex.evtLabels.labelA == True : data = False
                    ex.evtLabels.paramUpdate(path, data)
                elif event.key() == QtCore.Qt.Key_2:
                    path[1] = "Multi"
                    data = True
                    if ex.evtLabels.labelB == True : data = False
                    ex.evtLabels.paramUpdate(path, data)
                elif event.key() == QtCore.Qt.Key_3:
                    path[1] = "Dunno"
                    data = True
                    if ex.evtLabels.labelC == True : data = False
                    ex.evtLabels.paramUpdate(path, data)
                elif event.key() == QtCore.Qt.Key_Period:
                    if ex.small.w9.getPlotItem().listDataItems() != []:
                        idx = -1
                        array = np.where(ex.small.quantifierEvent >= ex.eventNumber)
                        if array[0].size != 0:
                            idx = array[0][0]
                            if ex.small.quantifierEvent[idx] == ex.eventNumber: idx += 1
                            if idx < (ex.small.quantifierEvent.size): self.previewEvent(ex.small.quantifierEvent[idx])
                elif event.key() == QtCore.Qt.Key_N:
                    if ex.eventNumber < (ex.exp.eventTotal - 1): self.previewEvent(ex.eventNumber+1)
                elif event.key() == QtCore.Qt.Key_Comma:
                    if ex.small.w9.getPlotItem().listDataItems() != []:
                        idx = -1
                        array = np.where(ex.small.quantifierEvent <= ex.eventNumber)
                        if array[0].size != 0:
                            idx = array[0][array[0].size - 1]
                            if ex.small.quantifierEvent[idx] == ex.eventNumber: idx -= 1
                            if ex.small.quantifierEvent[idx] != 0: self.previewEvent(ex.small.quantifierEvent[idx])
                elif event.key() == QtCore.Qt.Key_P:
                    if ex.eventNumber != 0: self.previewEvent(ex.eventNumber-1)
                ex.evtLabels.refresh()

class MainFrame(QtGui.QWidget):
    """
    The main frame of the application
    """        
    def __init__(self, arg_list):
        super(MainFrame, self).__init__()
        self.args = args

        # Set up tolerance
        self.eps = np.finfo("float64").eps
        self.firstUpdate = True
        self.operationModeChoices = ['none','masking']
        self.operationMode =  self.operationModeChoices[0] # Masking mode, Peak finding mode
        # Init experiment parameters from args
        if args.expRun is not None and ':run=' in args.expRun:
            self.experimentName = args.expRun.split('exp=')[-1].split(':')[0]
            self.runNumber = int(args.expRun.split('run=')[-1])
        else:
            self.experimentName = args.exp
            self.runNumber = int(args.run)
        self.detInfo = args.det
        self.eventNumber = int(args.evt)

        # Directories
        self.psocakeDir = None
        self.psocakeRunDir = None
        self.elogDir = None
        self.rootDir = None

        # Init variables
        self.det = None
        self.detnames = None
        self.detInfoList = None
        self.isCspad = False
        self.evt = None
        self.eventID = ""
        self.hasExperimentName = False
        self.hasRunNumber = False
        self.hasDetInfo = False
        self.pixelIndAssem = None

        # Init diffraction geometry parameters
        self.coffset = 0.0
        self.clen = 0.0
        self.clenEpics = 0.0
        self.epics = None
        self.detectorDistance = 0.0
        self.photonEnergy = None
        self.wavelength = None
        self.pixelSize = None
        self.resolutionRingsOn = False
        self.resolution = None
        self.resolutionUnits = 0

        # Init variables
        self.calib = None # ndarray detector image
        self.data = None # assembled detector image
        self.cx = 0 # detector centre x
        self.cy = 0 # detector centre y

        ########################################
        # Instantiate panels
        ########################################
        self.exp = ExperimentPanel.ExperimentInfo(self)
        self.geom = diffractionGeometryPanel.DiffractionGeometry(self)
        self.index = crystalIndexingPanel.CrystalIndexing(self)
        self.small = SmallDataPanel.SmallData(self)
        self.evtLabels = LabelPanel.Labels(self)
        self.pk = PeakFindingPanel.PeakFinding(self)
        self.hf = HitFinderPanel.HitFinder(self)
        self.mk = MaskPanel.MaskMaker(self)
        self.img = ImagePanel.ImageViewer(self)
        self.roi = RoiPanel.RoiHistogram(self)
        self.control = ImageControlPanel.ImageControl(self)
        self.mouse = MousePanel.Mouse(self)
        self.stack = ImageStackPanel.ImageStack(self)

        self.initUI()

    def initUI(self):
        ## Define a top-level widget to hold everything
        self.win = Window()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1300,650)
        self.win.setWindowTitle('psocake')

        # Set the color scheme
        def updateStylePatched(self):
            r = '3px'
            fg = color.cardinalRed_hex
            bg = color.sandstone100_rgb
            border = "white"

            if self.orientation == 'vertical':
                self.vStyle = """DockLabel {
                    background-color : %s;
                    color : %s;
                    border-top-right-radius: 0px;
                    border-top-left-radius: %s;
                    border-bottom-right-radius: 0px;
                    border-bottom-left-radius: %s;
                    border-width: 0px;
                    border-right: 2px solid %s;
                    padding-top: 3px;
                    padding-bottom: 3px;
                    font-size: 18px;
                }""" % (bg, fg, r, r, border)
                self.setStyleSheet(self.vStyle)
            else:
                self.hStyle = """DockLabel {
                    background-color : %s;
                    color : %s;
                    border-top-right-radius: %s;
                    border-top-left-radius: %s;
                    border-bottom-right-radius: 0px;
                    border-bottom-left-radius: 0px;
                    border-width: 0px;
                    border-bottom: 2px solid %s;
                    padding-left: 13px;
                    padding-right: 13px;
                    font-size: 18px
                }""" % (bg, fg, r, r, border)
                self.setStyleSheet(self.hStyle)
        DockLabel.updateStyle = updateStylePatched

        if args.mode == 'sfx':
            # Dock positions on the main frame
            self.area.addDock(self.mouse.d5, 'left')  ## place d5 at left edge of d1
            self.area.addDock(self.control.d6, 'bottom', self.mouse.d5)    ## place d1 at left edge of dock area
            self.area.addDock(self.stack.d7, 'bottom', self.mouse.d5)
            self.area.addDock(self.img.d1, 'bottom', self.mouse.d5)    ## place d1 at left edge of dock area
            self.area.moveDock(self.img.d1, 'above', self.stack.d7)

            self.area.addDock(self.exp.d2, 'right')     ## place d2 at right edge of dock area
            self.area.addDock(self.pk.d9, 'bottom', self.exp.d2)
            self.area.addDock(self.mk.d12, 'bottom',self.exp.d2)
            self.area.addDock(self.index.d14, 'bottom', self.exp.d2)
            self.area.moveDock(self.pk.d9, 'above', self.mk.d12)
            self.area.moveDock(self.index.d14, 'above', self.mk.d12)

            self.area.addDock(self.geom.d3, 'bottom', self.exp.d2)    ## place d3 at bottom edge of d1
            self.area.addDock(self.roi.d4, 'bottom', self.exp.d2)    ## place d4 at right edge of dock area
            self.area.moveDock(self.geom.d3, 'above', self.exp.d2)
            self.area.moveDock(self.roi.d4, 'above', self.exp.d2)

            self.area.addDock(self.small.dSmall, 'right')#, self.exp.d2)
            self.area.moveDock(self.exp.d2, 'above', self.geom.d3)
        elif args.mode == 'spi':
            # Dock positions on the main frame
            self.area.addDock(self.mouse.d5, 'left')  ## place d5 at left edge of d1
            self.area.addDock(self.control.d6, 'bottom', self.mouse.d5)    ## place d1 at left edge of dock area
            self.area.addDock(self.stack.d7, 'bottom', self.mouse.d5)
            self.area.addDock(self.img.d1, 'bottom', self.mouse.d5)    ## place d1 at left edge of dock area
            self.area.moveDock(self.img.d1, 'above', self.stack.d7)

            self.area.addDock(self.exp.d2, 'right')     ## place d2 at right edge of dock area
            self.area.addDock(self.mk.d12, 'bottom',self.exp.d2)
            self.area.addDock(self.hf.d13, 'bottom', self.exp.d2)
            self.area.moveDock(self.hf.d13, 'above', self.mk.d12)

            self.area.addDock(self.geom.d3, 'bottom', self.exp.d2)    ## place d3 at bottom edge of d1
            self.area.addDock(self.roi.d4, 'bottom', self.exp.d2)    ## place d4 at right edge of dock area
            self.area.moveDock(self.geom.d3, 'above', self.exp.d2)
            self.area.moveDock(self.roi.d4, 'above', self.exp.d2)

            self.area.addDock(self.small.dSmall, 'right')#, self.exp.d2)
            self.area.moveDock(self.exp.d2, 'above', self.geom.d3)
        elif args.mode == 'all':
            # Dock positions on the main frame
            self.area.addDock(self.mouse.d5, 'left')  ## place d5 at left edge of d1
            self.area.addDock(self.control.d6, 'bottom', self.mouse.d5)  ## place d1 at left edge of dock area
            self.area.addDock(self.stack.d7, 'bottom', self.mouse.d5)
            self.area.addDock(self.img.d1, 'bottom', self.mouse.d5)  ## place d1 at left edge of dock area
            self.area.moveDock(self.img.d1, 'above', self.stack.d7)

            self.area.addDock(self.exp.d2, 'right')  ## place d2 at right edge of dock area
            self.area.addDock(self.pk.d9, 'bottom', self.exp.d2)
            self.area.addDock(self.mk.d12, 'bottom', self.exp.d2)
            self.area.addDock(self.hf.d13, 'bottom', self.exp.d2)
            self.area.addDock(self.index.d14, 'bottom', self.exp.d2)
            self.area.moveDock(self.pk.d9, 'above', self.mk.d12)
            self.area.moveDock(self.hf.d13, 'above', self.mk.d12)
            self.area.moveDock(self.index.d14, 'above', self.mk.d12)

            self.area.addDock(self.geom.d3, 'bottom', self.exp.d2)  ## place d3 at bottom edge of d1
            self.area.addDock(self.roi.d4, 'bottom', self.exp.d2)  ## place d4 at right edge of dock area
            self.area.moveDock(self.geom.d3, 'above', self.exp.d2)
            self.area.moveDock(self.roi.d4, 'above', self.exp.d2)

            self.area.addDock(self.small.dSmall, 'right')  # , self.exp.d2)
            self.area.moveDock(self.exp.d2, 'above', self.geom.d3)

            self.area.addDock(self.evtLabels.dLabels, 'bottom', self.small.dSmall)
        else: # lite
            # Dock positions on the main frame
            self.area.addDock(self.mouse.d5, 'left')  ## place d5 at left edge of d1
            self.area.addDock(self.control.d6, 'bottom', self.mouse.d5)  ## place d1 at left edge of dock area
            self.area.addDock(self.stack.d7, 'bottom', self.mouse.d5)
            self.area.addDock(self.img.d1, 'bottom', self.mouse.d5)  ## place d1 at left edge of dock area
            self.area.moveDock(self.img.d1, 'above', self.stack.d7)

            self.area.addDock(self.exp.d2, 'right')  ## place d2 at right edge of dock area
            self.area.addDock(self.mk.d12, 'bottom', self.exp.d2)
            self.area.addDock(self.roi.d4, 'bottom', self.exp.d2)  ## place d4 at right edge of dock area





        ###############
        ### Threads ###
        ###############
        # Making powder patterns
        self.thread = []
        self.threadCounter = 0


        # Deploy psana geometry
        self.connect(self.geom.deployGeomBtn, QtCore.SIGNAL("clicked()"), self.geom.deploy)



        # Setup input parameters
        if self.experimentName is not "":
            self.hasExperimentName = True
            self.exp.p.param(self.exp.exp_grp, self.exp.exp_name_str).setValue(self.experimentName)
            self.exp.updateExpName(self.experimentName)
        if self.runNumber is not 0:
            self.hasRunNumber = True
            self.exp.p.param(self.exp.exp_grp, self.exp.exp_run_str).setValue(self.runNumber)
            self.exp.updateRunNumber(self.runNumber)
        if self.detInfo is not "":
            self.hasDetInfo = True
            self.exp.p.param(self.exp.exp_grp, self.exp.exp_detInfo_str).setValue(self.detInfo)
            self.exp.updateDetInfo(self.detInfo)
        self.exp.p.param(self.exp.exp_grp, self.exp.exp_evt_str).setValue(self.eventNumber)
        self.exp.updateEventNumber(self.eventNumber)

        # Indicate centre of detector
        self.geom.drawCentre()

        # Try mouse over crosshair
        self.xhair = self.img.w1.getView()
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.xhair.addItem(self.vLine, ignoreBounds=True)
        self.xhair.addItem(self.hLine, ignoreBounds=True)
        self.vb = self.xhair.vb
        self.label = pg.LabelItem()
        self.mouse.w5.addItem(self.label)

        def mouseMoved(evt):
            pos = evt[0]  ## using signal proxy turns original arguments into a tuple
            if self.xhair.sceneBoundingRect().contains(pos):
                mousePoint = self.vb.mapSceneToView(pos)
                indexX = int(mousePoint.x())
                indexY = int(mousePoint.y())

                # update crosshair position
                self.vLine.setPos(mousePoint.x())
                self.hLine.setPos(mousePoint.y())
                # get pixel value, if data exists
                if self.data is not None:
                    if indexX >= 0 and indexX < self.data.shape[0] \
                       and indexY >= 0 and indexY < self.data.shape[1]:
                        if self.mk.maskingMode > 0:
                            modeInfo = self.mk.masking_mode_message
                        else:
                            modeInfo = ""
                        pixelInfo = "<span style='color: " + color.cardinalRed_hex + "; font-size: 24pt;'>x=%0.1f y=%0.1f I=%0.1f </span>"
                        self.label.setText(modeInfo + pixelInfo % (mousePoint.x(), mousePoint.y(), self.data[indexX,indexY]))

        def mouseClicked(evt):
            mousePoint = self.vb.mapSceneToView(evt[0].scenePos())
            indexX = int(mousePoint.x())
            indexY = int(mousePoint.y())

            if self.data is not None:
                # Mouse click
                if indexX >= 0 and indexX < self.data.shape[0] \
                    and indexY >= 0 and indexY < self.data.shape[1]:
                    print "mouse clicked: ", mousePoint.x(), mousePoint.y(), self.data[indexX,indexY]
                    if self.mk.maskingMode > 0:
                        self.initMask()
                        if self.mk.maskingMode == 1:
                            # masking mode
                            self.mk.userMaskAssem[indexX,indexY] = 0
                        elif self.mk.maskingMode == 2:
                            # unmasking mode
                            self.mk.userMaskAssem[indexX,indexY] = 1
                        elif self.mk.maskingMode == 3:
                            # toggle mode
                            self.mk.userMaskAssem[indexX,indexY] = (1-self.mk.userMaskAssem[indexX,indexY])
                        self.displayMask()

                        self.mk.userMask = self.det.ndarray_from_image(self.evt,self.mk.userMaskAssem, pix_scale_size_um=None, xy0_off_pix=None)
                        self.algInitDone = False
                        self.parent.pk.updateClassification()

        # Signal proxy
        self.proxy_move = pg.SignalProxy(self.xhair.scene().sigMouseMoved, rateLimit=30, slot=mouseMoved)
        self.proxy_click = pg.SignalProxy(self.xhair.scene().sigMouseClicked, slot=mouseClicked)
        self.win.show()

    # If anything changes in the parameter tree, print a message
    def change(self, panel, changes):
        for param, change, data in changes:
            path = panel.childPath(param)
            if args.v >= 1:
                print('  path: %s'% path)
                print('  change:    %s'% change)
                print('  data:      %s'% str(data))
                print('  ----------')
            self.update(path,change,data)

    def update(self, path, change, data):
        if args.v >= 1: print "path: ", path
        ################################################
        # experiment parameters
        ################################################
        if path[0] == self.exp.exp_grp or path[0] == self.exp.disp_grp:
            self.exp.paramUpdate(path, change, data)
        ################################################
        # peak finder parameters
        ################################################
        if path[0] == self.pk.hitParam_grp:
            self.pk.paramUpdate(path, change, data)
        ################################################
        # hit finder parameters
        ################################################
        if path[0] == self.hf.spiParam_grp:
            self.hf.paramUpdate(path, change, data)
        ################################################
        # diffraction geometry parameters
        ################################################
        if path[0] == self.geom.geom_grp:
            self.geom.paramUpdate(path, change, data)
        ################################################
        # quantifier parameters
        ################################################
        if path[0] == self.small.quantifier_grp:
            self.small.paramUpdate(path, change, data)
        ################################################
        # manifold parameters
        ################################################
        #if path[0] == manifold_grp:
        #    if path[1] == manifold_filename_str:
        #        self.updateManifoldFilename(data)
        #    elif path[1] == manifold_dataset_str:
        #        self.updateManifoldDataset(data)
        #    elif path[1] == manifold_sigma_str:
        #        self.updateManifoldSigma(data)
        ################################################
        # masking parameters
        ################################################
        if path[0] == self.mk.mask_grp or path[0] == self.mk.powder_grp:
            self.mk.paramUpdate(path, change, data)
        ################################################
        # crystal indexing parameters
        ################################################
        if path[0] == self.index.index_grp or path[0] == self.index.launch_grp:
            self.index.paramUpdate(path, change, data)
        ################################################
        # label parameters
        ################################################
        if path[0] == self.evtLabels.labels_grp:
            self.evtLabels.paramUpdate(path, change, data)

def main():
    global ex
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtGui.QApplication(sys.argv)
    ex = MainFrame(sys.argv)
    sys.exit(app.exec_())

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    main()

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
from PSCalib.GeometryObject import data2x2ToTwo2x1, two2x1ToData2x2
# Panel modules
import diffractionGeometryPanel, crystalIndexingPanel, SmallDataPanel, ExperimentPanel
import PeakFindingPanel, HitFinderPanel, MaskPanel, LabelPanel, ImagePanel
import LaunchPeakFinder, LaunchPowderProducer, LaunchIndexer, LaunchHitFinder, LaunchStackProducer
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

# PerPixelHistogram parameter tree
#perPixelHistogram_grp = 'Per Pixel Histogram'
#perPixelHistogram_filename_str = 'filename'
#perPixelHistogram_adu_str = 'ADU'

# Manifold parameter tree
#manifold_grp = 'Manifold'
#manifold_filename_str = 'filename'
#manifold_dataset_str = 'eigenvector_dataset'
#manifold_sigma_str = 'sigma'

# Detector correction parameter tree
#correction_grp = 'Detector correction'
#correction_radialBackground_str = "Use radial background correction"
#correction_polarization_str = "Use polarization correction"

class Window(QtGui.QMainWindow):
    global ex

    def previewEvent(self, eventNumber):
        ex.eventNumber = eventNumber
        ex.calib, ex.data = ex.img.getDetImage(ex.eventNumber)
        ex.img.w1.setImage(ex.data, autoRange=False, autoLevels=False, autoHistogramRange=False)
        ex.p.param(self.exp.exp_grp, self.exp.exp_evt_str).setValue(ex.eventNumber)

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
                    if ex.w9.getPlotItem().listDataItems() != []:
                        idx = -1
                        array = np.where(ex.quantifierEvent >= ex.eventNumber)
                        if array[0].size != 0:
                            idx = array[0][0]
                            if ex.quantifierEvent[idx] == ex.eventNumber: idx += 1
                            if idx < (ex.quantifierEvent.size): self.previewEvent(ex.quantifierEvent[idx])
                elif event.key() == QtCore.Qt.Key_N:
                    if ex.eventNumber < (ex.exp.eventTotal - 1): self.previewEvent(ex.eventNumber+1)
                elif event.key() == QtCore.Qt.Key_Comma:
                    if ex.w9.getPlotItem().listDataItems() != []:
                        idx = -1
                        array = np.where(ex.quantifierEvent <= ex.eventNumber)
                        if array[0].size != 0:
                            idx = array[0][array[0].size - 1]
                            if ex.quantifierEvent[idx] == ex.eventNumber: idx -= 1
                            if ex.quantifierEvent[idx] != 0: self.previewEvent(ex.quantifierEvent[idx])
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

        # Instantiate panels
        self.exp = ExperimentPanel.ExperimentInfo(self)
        self.geom = diffractionGeometryPanel.DiffractionGeometry(self)
        self.index = crystalIndexingPanel.CrystalIndexing(self)
        self.small = SmallDataPanel.SmallData(self)
        self.evtLabels = LabelPanel.Labels(self)
        self.pk = PeakFindingPanel.PeakFinding(self)
        self.hf = HitFinderPanel.HitFinder(self)
        self.mk = MaskPanel.MaskMaker(self)
        self.img = ImagePanel.ImageViewer(self)

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

        # Display params
        #self.maxPercentile = 0
        #self.minPercentile = 0
        #self.displayMaxPercentile = 99.0
        #self.displayMinPercentile = 1.0


        #self.hasCommonMode = False


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
        self.updateRoiStatus = True

        # Threads
        self.stackStart = 0
        self.stackSizeMax = 120
        self.stackSize = 20

        #self.perPixelHistogram_filename = ''
        #self.perPixelHistogram_adu = 20
        #self.perPixelHistogramFileOpen = False

        #self.manifold_filename = ''
        #self.manifold_dataset = ''
        #self.manifold_sigma = 0
        #self.manifoldFileOpen = False
        #self.manifoldHasData = False

        #self.correction_radialBackground = False
        #self.correction_polarization = False

        #self.paramsManifold = [
        #    {'name': manifold_grp, 'type': 'group', 'children': [
        #        {'name': manifold_filename_str, 'type': 'str', 'value': self.manifold_filename, 'tip': "Full path Hdf5 filename"},
        #        {'name': manifold_dataset_str, 'type': 'str', 'value': self.manifold_dataset, 'tip': "Hdf5 dataset metric"},
        #        {'name': manifold_sigma_str, 'type': 'float', 'value': self.manifold_sigma, 'tip': "kernel sigma"},
        #    ]},
        #]

        #self.paramsCorrection = [
        #    {'name': correction_grp, 'type': 'group', 'children': [
        #        {'name': correction_radialBackground_str, 'type': 'str', 'value': self.correction_radialBackground, 'tip': "Use radial background correction"},
        #        {'name': correction_polarization_str, 'type': 'str', 'value': self.correction_polarization, 'tip': "Use polarization correction"},
        #    ]},
        #]

        self.initUI()

    def initUI(self):
        ## Define a top-level widget to hold everything
        self.win = Window()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1300,650)
        self.win.setWindowTitle('psocake')

        ## Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', \
                                  children=self.exp.params, expanded=True)
        self.p1 = Parameter.create(name='paramsDiffractionGeometry', type='group', \
                                  children=self.geom.params, expanded=True)
        self.pSmall = Parameter.create(name='paramsQuantifier', type='group', \
                                  children=self.small.params, expanded=True)
        self.p3 = Parameter.create(name='paramsPeakFinder', type='group', \
                                  children=self.pk.params, expanded=True)
        #self.p4 = Parameter.create(name='paramsManifold', type='group', \
        #                          children=self.paramsManifold, expanded=True)
        #self.p5 = Parameter.create(name='paramsPerPixelHistogram', type='group', \
        #                          children=self.paramsPerPixelHistogram, expanded=True)
        self.p6 = Parameter.create(name='paramsMask', type='group', \
                                  children=self.mk.params, expanded=True)
        #self.p7 = Parameter.create(name='paramsCorrection', type='group', \
        #                           children=self.paramsCorrection, expanded=True)
        self.p8 = Parameter.create(name='paramsHitFinder', type='group', \
                                   children=self.hf.params, expanded=True)
        self.p9 = Parameter.create(name='paramsCrystalIndexing', type='group', \
                                   children=self.index.params, expanded=True)
        self.pLabels = Parameter.create(name='paramsLabel', type='group', \
                                   children=self.evtLabels.params, expanded=True)
        self.p.sigTreeStateChanged.connect(self.change)
        self.p1.sigTreeStateChanged.connect(self.change)
        self.pSmall.sigTreeStateChanged.connect(self.change)
        self.p3.sigTreeStateChanged.connect(self.change)
        #self.p4.sigTreeStateChanged.connect(self.change)
        #self.p5.sigTreeStateChanged.connect(self.change)
        self.p6.sigTreeStateChanged.connect(self.change)
        #self.p7.sigTreeStateChanged.connect(self.change)
        self.p8.sigTreeStateChanged.connect(self.change)
        self.p9.sigTreeStateChanged.connect(self.change)
        self.pLabels.sigTreeStateChanged.connect(self.change)

        ## Create docks, place them into the window one at a time.
        ## Note that size arguments are only a suggestion; docks will still have to
        ## fill the entire dock area and obey the limits of their internal widgets.
             ## give this dock the minimum possible size
        self.d2 = Dock("Experiment Parameters", size=(1, 1))
        self.d3 = Dock("Diffraction Geometry", size=(1, 1))
        self.d4 = Dock("ROI Histogram", size=(1, 1))
        self.d5 = Dock("Mouse", size=(500, 75), closable=False)
        self.d6 = Dock("Image Control", size=(1, 1))
        self.d7 = Dock("Image Scroll", size=(1, 1))
        self.dSmall = Dock("Small Data", size=(100, 100))
        self.d9 = Dock("Peak Finder", size=(1, 1))
        #self.d10 = Dock("Manifold", size=(1, 1))
        self.d12 = Dock("Mask Panel", size=(1, 1))
        self.d13 = Dock("Hit Finder", size=(1, 1))
        self.d14 = Dock("Indexing", size=(1, 1))
        self.dLabels = Dock("Labels", size=(1, 1))

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
            self.area.addDock(self.d5, 'left')  ## place d5 at left edge of d1
            self.area.addDock(self.d6, 'bottom', self.d5)    ## place d1 at left edge of dock area
            self.area.addDock(self.d7, 'bottom', self.d5)
            self.area.addDock(self.img.d1, 'bottom', self.d5)    ## place d1 at left edge of dock area
            self.area.moveDock(self.img.d1, 'above', self.d7)

            self.area.addDock(self.d2, 'right')     ## place d2 at right edge of dock area
            self.area.addDock(self.d9, 'bottom', self.d2)
            self.area.addDock(self.d12, 'bottom',self.d2)
            self.area.addDock(self.d14, 'bottom', self.d2)
            self.area.moveDock(self.d9, 'above', self.d12)
            self.area.moveDock(self.d14, 'above', self.d12)

            self.area.addDock(self.d3, 'bottom', self.d2)    ## place d3 at bottom edge of d1
            self.area.addDock(self.d4, 'bottom', self.d2)    ## place d4 at right edge of dock area
            self.area.moveDock(self.d3, 'above', self.d2)
            self.area.moveDock(self.d4, 'above', self.d2)

            self.area.addDock(self.dSmall, 'right')#, self.d2)
            self.area.moveDock(self.d2, 'above', self.d3)
        elif args.mode == 'spi':
            # Dock positions on the main frame
            self.area.addDock(self.d5, 'left')  ## place d5 at left edge of d1
            self.area.addDock(self.d6, 'bottom', self.d5)    ## place d1 at left edge of dock area
            self.area.addDock(self.d7, 'bottom', self.d5)
            self.area.addDock(self.img.d1, 'bottom', self.d5)    ## place d1 at left edge of dock area
            self.area.moveDock(self.img.d1, 'above', self.d7)

            self.area.addDock(self.d2, 'right')     ## place d2 at right edge of dock area
            self.area.addDock(self.d12, 'bottom',self.d2)
            self.area.addDock(self.d13, 'bottom', self.d2)
            self.area.moveDock(self.d13, 'above', self.d12)

            self.area.addDock(self.d3, 'bottom', self.d2)    ## place d3 at bottom edge of d1
            self.area.addDock(self.d4, 'bottom', self.d2)    ## place d4 at right edge of dock area
            self.area.moveDock(self.d3, 'above', self.d2)
            self.area.moveDock(self.d4, 'above', self.d2)

            self.area.addDock(self.dSmall, 'right')#, self.d2)
            self.area.moveDock(self.d2, 'above', self.d3)
        elif args.mode == 'all':
            # Dock positions on the main frame
            self.area.addDock(self.d5, 'left')  ## place d5 at left edge of d1
            self.area.addDock(self.d6, 'bottom', self.d5)  ## place d1 at left edge of dock area
            self.area.addDock(self.d7, 'bottom', self.d5)
            self.area.addDock(self.img.d1, 'bottom', self.d5)  ## place d1 at left edge of dock area
            self.area.moveDock(self.img.d1, 'above', self.d7)

            self.area.addDock(self.d2, 'right')  ## place d2 at right edge of dock area
            self.area.addDock(self.d9, 'bottom', self.d2)
            self.area.addDock(self.d12, 'bottom', self.d2)
            self.area.addDock(self.d13, 'bottom', self.d2)
            self.area.addDock(self.d14, 'bottom', self.d2)
            self.area.moveDock(self.d9, 'above', self.d12)
            self.area.moveDock(self.d13, 'above', self.d12)
            self.area.moveDock(self.d14, 'above', self.d12)

            self.area.addDock(self.d3, 'bottom', self.d2)  ## place d3 at bottom edge of d1
            self.area.addDock(self.d4, 'bottom', self.d2)  ## place d4 at right edge of dock area
            self.area.moveDock(self.d3, 'above', self.d2)
            self.area.moveDock(self.d4, 'above', self.d2)

            self.area.addDock(self.dSmall, 'right')  # , self.d2)
            self.area.moveDock(self.d2, 'above', self.d3)

            self.area.addDock(self.dLabels, 'bottom', self.dSmall)
        else: # lite
            # Dock positions on the main frame
            self.area.addDock(self.d5, 'left')  ## place d5 at left edge of d1
            self.area.addDock(self.d6, 'bottom', self.d5)  ## place d1 at left edge of dock area
            self.area.addDock(self.d7, 'bottom', self.d5)
            self.area.addDock(self.img.d1, 'bottom', self.d5)  ## place d1 at left edge of dock area
            self.area.moveDock(self.img.d1, 'above', self.d7)

            self.area.addDock(self.d2, 'right')  ## place d2 at right edge of dock area
            self.area.addDock(self.d12, 'bottom', self.d2)
            self.area.addDock(self.d4, 'bottom', self.d2)  ## place d4 at right edge of dock area

        # Custom ROI for selecting an image region
        self.roi = pg.ROI(pos=[0, -250], size=[200, 200], snapSize=1.0, scaleSnap=True, translateSnap=True,
                          pen={'color': 'g', 'width': 4, 'style': QtCore.Qt.DashLine})
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0], [1, 1]) # bottom,left handles scaling both vertically and horizontally
        self.roi.addScaleHandle([1, 1], [0, 0])  # top,right handles scaling both vertically and horizontally
        self.roi.addScaleHandle([1, 0], [0, 1])  # bottom,right handles scaling both vertically and horizontally
        self.roi.name = 'rect'
        self.img.w1.getView().addItem(self.roi)
        self.roiPoly = pg.PolyLineROI([[300, -250], [300,-50], [500,-50], [500,-150], [375,-150], [375,-250]],
                                      closed=True, snapSize=1.0, scaleSnap=True, translateSnap=True,
                                      pen={'color': 'g', 'width': 4, 'style': QtCore.Qt.DashLine})
        self.roiPoly.name = 'poly'
        self.img.w1.getView().addItem(self.roiPoly)
        self.roiCircle = pg.CircleROI([600, -250], size=[200, 200], snapSize=0.1, scaleSnap=False, translateSnap=False,
                                        pen={'color': 'g', 'width': 4, 'style': QtCore.Qt.DashLine})
        self.roiCircle.name = 'circ'
        self.img.w1.getView().addItem(self.roiCircle)

        # Callbacks for handling user interaction
        def updateRoiHistogram():
            if self.data is not None:
                selected, coord = self.roi.getArrayRegion(self.data, self.img.w1.getImageItem(), returnMappedCoords=True)
                hist,bin = np.histogram(selected.flatten(), bins=1000)
                self.w4.plot(bin, hist, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True)

        def updateRoi(roi):
            if self.data is not None:
                if self.updateRoiStatus == True:
                    calib = np.ones_like(self.calib)
                    img = self.det.image(self.evt, calib)
                    pixelsExist = roi.getArrayRegion(img, self.img.w1.getImageItem())
                    if roi.name == 'poly':
                        self.ret = roi.getArrayRegion(self.data, self.img.w1.getImageItem(), returnMappedCoords=True)
                        self.ret = self.ret[np.where(pixelsExist == 1)]
                    elif roi.name == 'circ':
                        self.ret = roi.getArrayRegion(self.data, self.img.w1.getImageItem())
                        self.ret = self.ret[np.where(pixelsExist == 1)]
                    else:
                        self.ret = roi.getArrayRegion(self.data, self.img.w1.getImageItem(), returnMappedCoords=True)

                    if roi.name == 'rect':  # isinstance(self.ret,tuple): # rectangle
                        selected, coord = self.ret
                        self.x0 = int(coord[0][0][0])
                        self.x1 = int(coord[0][-1][0]) + 1
                        self.y0 = int(coord[1][0][0])
                        self.y1 = int(coord[1][0][-1]) + 1

                        # Limit coordinates to inside the assembled image
                        if self.x0 < 0: self.x0 = 0
                        if self.y0 < 0: self.y0 = 0
                        if self.x1 > self.data.shape[0]: self.x1 = self.data.shape[0]
                        if self.y1 > self.data.shape[1]: self.y1 = self.data.shape[1]
                        print "######################################################"
                        print "Assembled ROI: [" + str(self.x0) + ":" + str(self.x1) + "," + str(self.y0) + ":" + str(
                            self.y1) + "]"  # Note: self.data[x0:x1,y0:y1]
                        selected = selected[np.where(pixelsExist == 1)]

                        mask_roi = np.zeros_like(self.data)
                        mask_roi[self.x0:self.x1, self.y0:self.y1] = 1
                        self.nda = self.det.ndarray_from_image(self.evt, mask_roi, pix_scale_size_um=None,
                                                               xy0_off_pix=None)
                        for itile, tile in enumerate(self.nda):
                            if tile.sum() > 0:
                                ax0 = np.arange(0, tile.sum(axis=0).shape[0])[tile.sum(axis=0) > 0]
                                ax1 = np.arange(0, tile.sum(axis=1).shape[0])[tile.sum(axis=1) > 0]
                                print 'Unassembled ROI: [[%i,%i], [%i,%i], [%i,%i]]' % (
                                    itile, itile + 1, ax1.min(), ax1.max(), ax0.min(), ax0.max())
                                if args.v >= 1:
                                    fig = plt.figure(figsize=(6, 6))
                                    plt.imshow(self.calib[itile, ax1.min():ax1.max(), ax0.min():ax0.max()],
                                               interpolation='none')
                                    plt.show()
                        print "######################################################"
                    elif roi.name == 'circ':
                        selected = self.ret
                        self.centreX = roi.x() + roi.size().x() / 2
                        self.centreY = roi.y() + roi.size().y() / 2
                        print "###########################################"
                        print "Centre: [" + str(self.centreX) + "," + str(self.centreY) + "]"
                        print "###########################################"

                    else:
                        selected = self.ret
                    hist, bin = np.histogram(selected.flatten(), bins=1000)
                    self.w4.plot(bin, hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150), clear=True)
                else: # update ROI off
                    if roi.name == 'rect':  # isinstance(self.ret,tuple): # rectangle
                        self.ret = roi.getArrayRegion(self.data, self.img.w1.getImageItem(), returnMappedCoords=True)
                        selected, coord = self.ret
                        self.x0 = int(coord[0][0][0])
                        self.x1 = int(coord[0][-1][0]) + 1
                        self.y0 = int(coord[1][0][0])
                        self.y1 = int(coord[1][0][-1]) + 1

                        # Limit coordinates to inside the assembled image
                        if self.x0 < 0: self.x0 = 0
                        if self.y0 < 0: self.y0 = 0
                        if self.x1 > self.data.shape[0]: self.x1 = self.data.shape[0]
                        if self.y1 > self.data.shape[1]: self.y1 = self.data.shape[1]
                        print "######################################################"
                        print "Assembled ROI: [" + str(self.x0) + ":" + str(self.x1) + "," + str(self.y0) + ":" + str(
                            self.y1) + "]"  # Note: self.data[x0:x1,y0:y1]
                        mask_roi = np.zeros_like(self.data)
                        mask_roi[self.x0:self.x1, self.y0:self.y1] = 1
                        self.nda = self.det.ndarray_from_image(self.evt, mask_roi, pix_scale_size_um=None,
                                                               xy0_off_pix=None)
                        for itile, tile in enumerate(self.nda):
                            if tile.sum() > 0:
                                ax0 = np.arange(0, tile.sum(axis=0).shape[0])[tile.sum(axis=0) > 0]
                                ax1 = np.arange(0, tile.sum(axis=1).shape[0])[tile.sum(axis=1) > 0]
                                print 'Unassembled ROI: [[%i,%i], [%i,%i], [%i,%i]]' % (
                                    itile, itile + 1, ax1.min(), ax1.max(), ax0.min(), ax0.max())
                        print "######################################################"
                    elif roi.name == 'circ':
                        self.centreX = roi.x() + roi.size().x() / 2
                        self.centreY = roi.y() + roi.size().y() / 2
                        print "###########################################"
                        print "Centre: [" + str(self.centreX) + "," + str(self.centreY) + "]"
                        print "###########################################"

        def updateRoiStatus():
            if self.roiCheckbox.checkState() == 0:
                self.updateRoiStatus = False
            else:
                self.updateRoiStatus = True

        self.rois = []
        self.rois.append(self.roi)
        self.rois.append(self.roiPoly)
        self.rois.append(self.roiCircle)
        for roi in self.rois:
            roi.sigRegionChangeFinished.connect(updateRoi)

        # Connect listeners to functions
        self.img.d1.addWidget(self.img.w1)

        ## Dock 2: parameter
        self.w2 = ParameterTree()
        self.w2.setParameters(self.p, showTop=False)
        self.w2.setWindowTitle('Parameters')
        self.d2.addWidget(self.w2)

        ## Dock 3: Diffraction geometry
        self.w3 = ParameterTree()
        self.w3.setParameters(self.p1, showTop=False)
        self.w3.setWindowTitle('Diffraction geometry')
        self.d3.addWidget(self.w3)
        self.w3a = pg.LayoutWidget()
        self.deployGeomBtn = QtGui.QPushButton('Deploy centred psana geometry')
        self.w3a.addWidget(self.deployGeomBtn, row=0, col=0)
        self.d3.addWidget(self.w3a)

        ## Dock 4: ROI histogram
        self.w4 = pg.PlotWidget(title="ROI histogram")
        hist,bin = np.histogram(np.random.random(1000), bins=1000)
        self.w4.plot(bin, hist, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True)
        self.d4.addWidget(self.w4)
        self.roiCheckbox = QtGui.QCheckBox('Update ROI')
        self.roiCheckbox.setCheckState(True)
        self.roiCheckbox.setTristate(False)
        self.roiCheckbox.stateChanged.connect(updateRoiStatus)
        # Layout
        self.w4a = pg.LayoutWidget()
        self.w4a.addWidget(self.roiCheckbox, row=0, col=0)
        self.d4.addWidget(self.w4a)

        ## Dock 5 - mouse intensity display
        #self.d5.hideTitleBar()
        self.w5 = pg.GraphicsView(background=pg.mkColor(color.sandstone100_rgb))
        self.d5.addWidget(self.w5)

        ## Dock 6: Image Control
        self.nextBtn = QtGui.QPushButton('Next evt')
        self.prevBtn = QtGui.QPushButton('Prev evt')
        self.saveBtn = QtGui.QPushButton('Save evt')
        self.loadBtn = QtGui.QPushButton('Load image')

        def next():
            self.eventNumber += 1
            if self.eventNumber >= self.exp.eventTotal:
                self.eventNumber = self.exp.eventTotal-1
            else:
                self.calib, self.data = self.img.getDetImage(self.eventNumber)
                self.img.w1.setImage(self.data,autoRange=False,autoLevels=False,autoHistogramRange=False)
                self.p.param(self.exp.exp_grp,self.exp.exp_evt_str).setValue(self.eventNumber)
        def prev():
            self.eventNumber -= 1
            if self.eventNumber < 0:
                self.eventNumber = 0
            else:
                self.calib, self.data = self.img.getDetImage(self.eventNumber)
                self.img.w1.setImage(self.data,autoRange=False,autoLevels=False,autoHistogramRange=False)
                self.p.param(self.exp.exp_grp,self.exp.exp_evt_str).setValue(self.eventNumber)
        def save():
            outputName = self.psocakeRunDir+"/psocake_"+str(self.experimentName)+"_"+str(self.runNumber)+"_"+str(self.detInfo)+"_" \
                         +str(self.eventNumber)+"_"+str(self.exp.eventSeconds)+"_"+str(self.exp.eventNanoseconds)+"_" \
                         +str(self.exp.eventFiducial)+".npy"
            fname = QtGui.QFileDialog.getSaveFileName(self, 'Save file', outputName, 'ndarray image (*.npy)')
            if self.exp.logscaleOn:
                np.save(str(fname),np.log10(abs(self.calib) + self.eps))
            else:
                if self.calib.size==2*185*388: # cspad2x2
                    asData2x2 = two2x1ToData2x2(self.calib)
                    np.save(str(fname),asData2x2)
                    np.savetxt(str(fname).split('.')[0]+".txt", asData2x2.reshape((-1,asData2x2.shape[-1])) ,fmt='%0.18e')
                else:
                    np.save(str(fname),self.calib)
                    np.savetxt(str(fname).split('.')[0]+".txt", self.calib.reshape((-1,self.calib.shape[-1])) )#,fmt='%0.18e')
        def load():
            fname = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.psocakeRunDir, 'ndarray image (*.npy *.npz)'))
            if fname.split('.')[-1] in '.npz':
                temp = np.load(fname)
                self.calib = temp['max']
            else:
                self.calib = np.load(fname)
            self.img.updateImage(self.calib)

        self.nextBtn.clicked.connect(next)
        self.prevBtn.clicked.connect(prev)
        self.saveBtn.clicked.connect(save)
        self.loadBtn.clicked.connect(load)

        # Layout
        self.w6 = pg.LayoutWidget()
        self.w6.addWidget(self.prevBtn, row=0, col=0)
        self.w6.addWidget(self.nextBtn, row=0, col=1)
        self.w6.addWidget(self.saveBtn, row=1, col=0)#colspan=2)
        self.w6.addWidget(self.loadBtn, row=1, col=1)
        self.d6.addWidget(self.w6)

        ## Dock 7: Image Stack
        self.w7L = pg.LayoutWidget()
        self.w7 = pg.ImageView(view=pg.PlotItem())
        self.w7.getView().invertY(False)
        self.scroll = np.random.random((5,10,10))
        self.w7.setImage(self.scroll, xvals=np.linspace(0., self.scroll.shape[0]-1, self.scroll.shape[0]))
        self.spinBox = QtGui.QSpinBox()
        self.spinBox.setValue(0)
        self.label = QtGui.QLabel("Event Number:")
        self.stackSizeBox = QtGui.QSpinBox()
        self.stackSizeBox.setMaximum(self.stackSizeMax)
        self.stackSizeBox.setValue(self.stackSize)
        self.startBtn = QtGui.QPushButton("&Load image stack")
        # Connect listeners to functions
        self.w7L.addWidget(self.w7, row=0, colspan=4)
        self.w7L.addWidget(self.label, 1, 0)
        self.w7L.addWidget(self.spinBox, 1, 1)
        self.w7L.addWidget(self.stackSizeBox, 1, 2)
        self.w7L.addWidget(self.startBtn, 1, 3)
        self.d7.addWidget(self.w7L)

        ## Dock 8: Quantifier
        self.w8 = ParameterTree()
        self.w8.setParameters(self.pSmall, showTop=False)
        self.dSmall.addWidget(self.w8)
        self.w11a = pg.LayoutWidget()
        self.refreshBtn = QtGui.QPushButton('Refresh')
        self.w11a.addWidget(self.refreshBtn, row=0, col=0)
        self.dSmall.addWidget(self.w11a)
        # Add plot
        self.w9 = pg.PlotWidget(title="Metric")
        self.dSmall.addWidget(self.w9)

        ## Dock 9: Peak finder
        self.w10 = ParameterTree()
        self.w10.setParameters(self.p3, showTop=False)
        self.d9.addWidget(self.w10)
        self.w11 = pg.LayoutWidget()
        #self.generatePowderBtn = QtGui.QPushButton('Generate Powder')
        self.launchBtn = QtGui.QPushButton('Launch peak finder')
        self.w11.addWidget(self.launchBtn, row=0,col=0)
        #self.w11.addWidget(self.generatePowderBtn, row=0, col=0)
        self.d9.addWidget(self.w11)

        ## Dock 10: Manifold
        #self.w12 = ParameterTree()
        #self.w12.setParameters(self.p4, showTop=False)
        #self.d10.addWidget(self.w12)
        # Add plot
        #self.w13 = pg.PlotWidget(title="Manifold!!!!")
        #self.d10.addWidget(self.w13)

        ## Dock 12: Mask Panel
        self.w17 = ParameterTree()
        self.w17.setParameters(self.p6, showTop=False)
        self.d12.addWidget(self.w17)
        self.w18 = pg.LayoutWidget()
        self.maskRectBtn = QtGui.QPushButton('Stamp rectangular mask')
        self.w18.addWidget(self.maskRectBtn, row=0, col=0, colspan=2)
        self.maskCircleBtn = QtGui.QPushButton('Stamp circular mask')
        self.w18.addWidget(self.maskCircleBtn, row=1, col=0, colspan=2)
        self.maskThreshBtn = QtGui.QPushButton('Mask outside histogram')
        self.w18.addWidget(self.maskThreshBtn, row=2, col=0, colspan=2)
        #self.maskPolyBtn = QtGui.QPushButton('Stamp polygon mask')
        #self.w18.addWidget(self.maskPolyBtn, row=2, col=0, colspan=2)
        self.deployMaskBtn = QtGui.QPushButton()
        self.deployMaskBtn.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')
        self.deployMaskBtn.setText('Save user-defined mask')
        self.w18.addWidget(self.deployMaskBtn, row=3, col=0)
        self.loadMaskBtn = QtGui.QPushButton()
        self.loadMaskBtn.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')
        self.loadMaskBtn.setText('Load user-defined mask')
        self.w18.addWidget(self.loadMaskBtn, row=3, col=1)
        self.generatePowderBtn = QtGui.QPushButton('Generate Average Image')
        self.w18.addWidget(self.generatePowderBtn, row=4, col=0, colspan=2)
        # Connect listeners to functions
        self.d12.addWidget(self.w18)

        ## Dock 13: Hit finder
        self.w19 = ParameterTree()
        self.w19.setParameters(self.p8, showTop=False)
        self.d13.addWidget(self.w19)
        self.w20 = pg.LayoutWidget()
        self.launchSpiBtn = QtGui.QPushButton('Launch hit finder')
        self.w20.addWidget(self.launchSpiBtn, row=1, col=0)
        self.d13.addWidget(self.w20)

        ## Dock 14: Indexing
        self.w21 = ParameterTree()
        self.w21.setParameters(self.p9, showTop=False)
        self.w21.setWindowTitle('Indexing')
        self.d14.addWidget(self.w21)
        self.w22 = pg.LayoutWidget()
        self.launchIndexBtn = QtGui.QPushButton('Launch indexing')
        self.w22.addWidget(self.launchIndexBtn, row=0, col=0)
        self.d14.addWidget(self.w22)

        ## Dock: Labels
        self.wLabels = ParameterTree()
        self.wLabels.setParameters(self.pLabels, showTop=False)
        self.wLabels.setWindowTitle('Labels')
        self.dLabels.addWidget(self.wLabels)

        self.connect(self.maskRectBtn, QtCore.SIGNAL("clicked()"), self.mk.makeMaskRect)
        self.connect(self.maskCircleBtn, QtCore.SIGNAL("clicked()"), self.mk.makeMaskCircle)
        self.connect(self.maskThreshBtn, QtCore.SIGNAL("clicked()"), self.mk.makeMaskThresh)
        #self.connect(self.maskPolyBtn, QtCore.SIGNAL("clicked()"), self.mk.makeMaskPoly)
        self.connect(self.deployMaskBtn, QtCore.SIGNAL("clicked()"), self.mk.deployMask)
        self.connect(self.loadMaskBtn, QtCore.SIGNAL("clicked()"), self.mk.loadMask)

        ###############
        ### Threads ###
        ###############
        # Making powder patterns
        self.thread = []
        self.threadCounter = 0
        def makePowder():
            self.thread.append(LaunchPowderProducer.PowderProducer(self)) # send parent parameters with self
            self.thread[self.threadCounter].computePowder(self.experimentName, self.runNumber, self.detInfo)
            self.threadCounter+=1
        self.connect(self.generatePowderBtn, QtCore.SIGNAL("clicked()"), makePowder)
        # Launch peak finding
        def findPeaks():
            self.thread.append(LaunchPeakFinder.LaunchPeakFinder(self)) # send parent parameters with self
            self.thread[self.threadCounter].launch(self.experimentName, self.detInfo)
            self.threadCounter+=1
        self.connect(self.launchBtn, QtCore.SIGNAL("clicked()"), findPeaks)
        # Launch hit finding
        def findHits():
            self.thread.append(LaunchHitFinder.HitFinder(self)) # send parent parameters with self
            self.thread[self.threadCounter].findHits(self.experimentName,self.runNumber,self.detInfo)
            self.threadCounter+=1
        self.connect(self.launchSpiBtn, QtCore.SIGNAL("clicked()"), findHits)
        # Launch indexing
        def indexPeaks():
            self.thread.append(LaunchIndexer.LaunchIndexer(self)) # send parent parameters with self
            self.thread[self.threadCounter].launch(self.experimentName, self.detInfo)
            self.threadCounter+=1
        self.connect(self.launchIndexBtn, QtCore.SIGNAL("clicked()"), indexPeaks)
        # Deploy psana geometry
        self.connect(self.deployGeomBtn, QtCore.SIGNAL("clicked()"), self.geom.deploy)

        # Loading image stack
        def displayImageStack():
            if self.exp.logscaleOn:
                self.w7.setImage(np.log10(abs(self.threadpool.data) + self.eps), xvals=np.linspace(self.stackStart,
                                                                     self.stackStart+self.threadpool.data.shape[0]-1,
                                                                     self.threadpool.data.shape[0]))
            else:
                self.w7.setImage(self.threadpool.data, xvals=np.linspace(self.stackStart,
                                                                     self.stackStart+self.threadpool.data.shape[0]-1,
                                                                     self.threadpool.data.shape[0]))
            self.startBtn.setEnabled(True)
            if args.v >= 1:
                print "Done display image stack!!!!!"
        def loadStack():
            self.stackStart = self.spinBox.value()
            self.stackSize = self.stackSizeBox.value()
            self.threadpool.load(self.stackStart,self.stackSize)
            self.startBtn.setEnabled(False)
            self.w7.getView().setTitle("exp="+self.experimentName+":run="+str(self.runNumber)+":evt"+str(self.stackStart)+"-"
                                       +str(self.stackStart+self.stackSize))
        self.threadpool = LaunchStackProducer.StackProducer(self) # send parent parameters
        self.connect(self.threadpool, QtCore.SIGNAL("finished()"), displayImageStack)
        self.connect(self.startBtn, QtCore.SIGNAL("clicked()"), loadStack)

        self.connect(self.refreshBtn, QtCore.SIGNAL("clicked()"), self.small.reloadQuantifier)

        # Setup input parameters
        if self.experimentName is not "":
            self.hasExperimentName = True
            self.p.param(self.exp.exp_grp, self.exp.exp_name_str).setValue(self.experimentName)
            self.exp.updateExpName(self.experimentName)
        if self.runNumber is not 0:
            self.hasRunNumber = True
            self.p.param(self.exp.exp_grp, self.exp.exp_run_str).setValue(self.runNumber)
            self.exp.updateRunNumber(self.runNumber)
        if self.detInfo is not "":
            self.hasDetInfo = True
            self.p.param(self.exp.exp_grp, self.exp.exp_detInfo_str).setValue(self.detInfo)
            self.exp.updateDetInfo(self.detInfo)
        self.p.param(self.exp.exp_grp, self.exp.exp_evt_str).setValue(self.eventNumber)
        self.exp.updateEventNumber(self.eventNumber)

        self.img.drawLabCoordinates() # FIXME: This does not match the lab coordinates yet!

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
        self.w5.addItem(self.label)

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
                            self.userMaskAssem[indexX,indexY] = 0
                        elif self.mk.maskingMode == 2:
                            # unmasking mode
                            self.userMaskAssem[indexX,indexY] = 1
                        elif self.mk.maskingMode == 3:
                            # toggle mode
                            self.userMaskAssem[indexX,indexY] = (1-self.userMaskAssem[indexX,indexY])
                        self.displayMask()

                        self.userMask = self.det.ndarray_from_image(self.evt,self.userMaskAssem, pix_scale_size_um=None, xy0_off_pix=None)
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

    ##################################
    ###### Per Pixel Histogram #######
    ##################################

    # def updatePerPixelHistogramFilename(self, data):
    #     # close previously open file
    #     if self.perPixelHistogram_filename is not data and self.perPixelHistogramFileOpen:
    #         self.perPixelHistogramFile.close()
    #     self.perPixelHistogram_filename = data
    #     self.perPixelHistogramFile = h5py.File(self.perPixelHistogram_filename,'r')
    #
    #     self.valid_min = self.perPixelHistogramFile['/dataHist/histogram'].attrs['valid_min'] # ADU
    #     self.valid_max = self.perPixelHistogramFile['/dataHist/histogram'].attrs['valid_max'] # ADU
    #     self.bin_size = self.perPixelHistogramFile['/dataHist/histogram'].attrs['bin_size'] # ADU
    #     units = np.linspace(self.valid_min,self.valid_max,self.valid_max-self.valid_min+1) # ADU
    #     start = np.mean(units[0:self.bin_size]) # ADU
    #     finish = np.mean(units[len(units)-self.bin_size:len(units)]) # ADU
    #     numBins = (self.valid_max-self.valid_min+1)/self.bin_size # ADU
    #     self.histogram_adu = np.linspace(start,finish,numBins) # ADU
    #
    #     self.perPixelHistograms = self.perPixelHistogramFile['/dataHist/histogram'].value
    #     self.histogram1D = self.perPixelHistograms.reshape((-1,self.perPixelHistograms.shape[-1]))
    #
    #     self.perPixelHistogramFileOpen = True
    #     if args.v >= 1: print "Done opening perPixelHistogram file"
    #
    # # FIXME: I don't think pixelIndex is correct
    # def updatePerPixelHistogramAdu(self, data):
    #     self.perPixelHistogram_adu = data
    #     if self.perPixelHistogramFileOpen:
    #         self.updatePerPixelHistogramSlice(self.perPixelHistogram_adu)
    #     if args.v >= 1: print "Done perPixelHistogram adu", self.perPixelHistogram_adu
    #
    # def updatePerPixelHistogramSlice(self,adu):
    #     self.histogramAduIndex = self.getHistogramIndex(adu)
    #     self.calib = np.squeeze(self.perPixelHistogramFile['/dataHist/histogram'][:,:,:,self.histogramAduIndex])
    #     self.updateImage(calib=self.calib)
    #
    # def getHistogramIndex(self,adu):
    #     histogramIndex = np.argmin(abs(self.histogram_adu - adu))
    #     return histogramIndex
    #
    # def updatePerPixelHistogram(self, pixelIndex):
    #     self.w16.getPlotItem().clear()
    #     if pixelIndex >= 0:
    #         self.perPixelHistogram = self.histogram1D[pixelIndex,1:-1]
    #         self.w16.plot(self.histogram_adu, self.perPixelHistogram, pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')
    #     self.w16.setLabel('left', "Counts")
    #     self.w16.setLabel('bottom', "ADU")

    ##################################
    ########### Manifold #############
    ##################################
    # # FIXME: manifold is incomplete
    # def updateManifoldFilename(self, data):
    #     # close previously open file
    #     if self.manifold_filename is not data and self.manifoldFileOpen:
    #         self.manifoldFile.close()
    #     self.manifold_filename = data
    #     self.manifoldFile = h5py.File(self.manifold_filename,'r')
    #     self.manifoldFileOpen = True
    #     if args.v >= 1: print "Done opening manifold"
    #
    # def updateManifoldDataset(self, data):
    #     self.manifold_dataset = data
    #     if self.manifoldFileOpen:
    #         self.manifoldEigs = self.manifoldFile[self.manifold_dataset].value
    #         (self.manifoldNumHits,self.manifoldNumEigs) = self.manifoldEigs.shape
    #         self.manifoldHasData = True
    #         self.updateManifoldPlot(self.manifoldEigs)
    #         self.manifoldInd = np.arange(self.manifoldNumHits)
    #         try:
    #             eventDataset = "/" + self.manifold_dataset.split("/")[1] + "/event"
    #             self.manifoldEvent = self.manifoldFile[eventDataset].value
    #         except:
    #             self.manifoldEvent = np.arange(self.manifoldNumHits)
    #         if args.v >= 1: print "Done reading manifold"
    #
    # def updateManifoldSigma(self, data):
    #     self.manifold_sigma = data
    #     if self.manifoldHasData:
    #         self.updateManifoldPlot(self.manifoldInd,self.manifoldEigs)
    #
    # def updateManifoldPlot(self,ind,eigenvectors):
    #     self.lastClicked = []
    #     self.w13.getPlotItem().clear()
    #     pos = np.random.normal(size=(2,10000), scale=1e-9)
    #     self.curve = self.w13.plot(pos[0],pos[1], pen=None, symbol='o',symbolPen=None,symbolSize=10,symbolBrush=(100,100,255,50))
    #     self.curve.curve.setClickable(True)
    #     self.curve.sigClicked.connect(self.clicked1)
    #
    # def clicked1(self,points): # manifold click
    #     print("curve clicked",points)
    #     from pprint import pprint
    #     pprint(vars(points.scatter))
    #     for i in range(len(points.scatter.data)):
    #         if points.scatter.ptsClicked[0] == points.scatter.data[i][7]:
    #             ind = i
    #             break
    #     indX = points.scatter.data[i][0]
    #     indY = points.scatter.data[i][1]
    #     if args.v >= 1: print "x,y: ", indX, indY
    #
    #     ind = self.manifoldInd[ind]
    #
    #     # temp
    #     self.eventNumber = self.manifoldEvent[ind]
    #
    #     self.calib, self.data = self.getDetImage(self.eventNumber)
    #     self.w1.setImage(self.data,autoRange=False,autoLevels=False,autoHistogramRange=False)
    #     self.p.param(self.exp.exp_grp, self.exp.exp_evt_str).setValue(self.eventNumber)

def main():
    global ex
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtGui.QApplication(sys.argv)
    ex = MainFrame(sys.argv)
    sys.exit(app.exec_())

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    main()

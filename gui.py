# GUI for browsing LCLS area detectors. Tune hit finding parameters and common mode correction.

# TODO: Zoom in area / numbers
# TODO: Multiple subplots
# TODO: powder pattern generator
# TODO: 20 ADU display
# TODO: dropdown menu for available detectors

import sys, signal
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.console
import numpy as np
from pyqtgraph.dockarea import *
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import psana
import h5py
from ImgAlgos.PyAlgos import PyAlgos # peak finding
import matplotlib.pyplot as plt
from optics import *
from pyqtgraph import Point
import argparse
import Detector.PyDetector

import sys
import logging
import multiprocessing as mp
from psmon import app, config, log_level_parse
from IPython import embed


parser = argparse.ArgumentParser()
parser.add_argument("-e","--exp", help="experiment name (e.g. cxis0813), default=''",default="", type=str)
parser.add_argument("-r","--run", help="run number (e.g. 5), default=0",default=0, type=int)
parser.add_argument("-d","--det", help="detector name (e.g. CxiDs1.0:Cspad.0), default=''",default="", type=str)
parser.add_argument("-n","--evt", help="event number (e.g. 1), default=0",default=0, type=int)
parser.add_argument("--localCalib", help="use local calib directory, default=False", action='store_true')
args = parser.parse_args()

# Set up tolerance
eps = np.finfo("float64").eps
resolutionRingList = np.array([100.,300.,500.,700.,900.,1100.])

# Set up list of parameters
exp_grp = 'Experiment information'
exp_name_str = 'Experiment Name'
exp_run_str = 'Run Number'
exp_det_str = 'DetInfo'
exp_evt_str = 'Event Number'
exp_second_str = 'Seconds'
exp_nanosecond_str = 'Nanoseconds'
exp_fiducial_str = 'Fiducial'
exp_numEvents_str = 'Total Events'
exp_detInfo_str = 'Detector ID'

disp_grp = 'Display'
disp_log_str = 'Logscale'
disp_resolutionRings_str = 'Resolution rings'
disp_resolution_str = 'Resolution (m)'
disp_commonMode_str = 'Common mode (override)'
disp_overrideCommonMode_str = 'Apply common mode (override)'
disp_commonModeParam0_str = 'parameters 0'
disp_commonModeParam1_str = 'parameters 1'
disp_commonModeParam2_str = 'parameters 2'
disp_commonModeParam3_str = 'parameters 3'

hitParam_grp = 'Hit finder'
hitParam_classify_str = 'Classify'
hitParam_algorithm_str = 'Algorithm'
hitParam_alg_npix_min_str = 'npix_min'
hitParam_alg_npix_max_str = 'npix_max'
hitParam_alg_amax_thr_str = 'amax_thr'
hitParam_alg_atot_thr_str = 'atot_thr'
hitParam_alg_son_min_str = 'son_min'
# algorithm 1
hitParam_algorithm1_str = 'Droplet'
hitParam_alg1_thr_low_str = 'thr_low'
hitParam_alg1_thr_high_str = 'thr_high'
hitParam_alg1_radius_str = 'radius'
hitParam_alg1_dr_str = 'dr'
# algorithm 2
hitParam_algorithm2_str = 'Flood-fill'
hitParam_alg2_thr_str = 'thr'
hitParam_alg2_r0_str = 'r0'
hitParam_alg2_dr_str = 'dr'

geom_grp = 'Diffraction geometry'
geom_detectorDistance_str = 'Detector distance'
geom_photonEnergy_str = 'Photon energy'
geom_wavelength_str = "Wavelength"
geom_pixelSize_str = 'Pixel size'

paramsDiffractionGeometry = [
    {'name': geom_grp, 'type': 'group', 'children': [
        {'name': geom_detectorDistance_str, 'type': 'float', 'value': 0.0, 'precision': 12, 'minVal': 1e-6, 'siFormat': (6,6), 'siPrefix': True, 'suffix': 'm'},
        {'name': geom_photonEnergy_str, 'type': 'float', 'value': 0.0, 'step': 1e-6, 'siPrefix': True, 'suffix': 'eV'},
        {'name': geom_wavelength_str, 'type': 'float', 'value': 0.0, 'step': 1e-6, 'siPrefix': True, 'suffix': 'm', 'readonly': True},
        {'name': geom_pixelSize_str, 'type': 'float', 'value': 0.0, 'precision': 12, 'minVal': 1e-6, 'siPrefix': True, 'suffix': 'm'},
    ]},
]

# Color scheme
sandstone100_rgb = (221,207,153) # Sandstone
cardinalRed_hex = str("#8C1515") # Cardinal red

class MainFrame(QtGui.QWidget):
    """
    The main frame of the application
    """        
    def __init__(self, arg_list):
        super(MainFrame, self).__init__()
        # Init experiment parameters
        self.experimentName = args.exp
        self.runNumber = int(args.run)
        #self.detInfoList = [1,2,3,4]
        self.detInfo = args.det
        self.eventNumber = int(args.evt)
        print "$$$ num: ", self.eventNumber
        self.eventSeconds = ""
        self.eventNanoseconds = ""
        self.eventFiducial = ""
        self.eventTotal = 0
        self.hasExperimentName = False
        self.hasRunNumber = False
        self.hasDetInfo = False
        # Init display parameters
        self.logscaleOn = True
        self.resolutionRingsOn = False
        self.resolution = None
        self.hasUserDefinedResolution = False
        self.hasCommonMode = False
        self.applyCommonMode = False
        self.commonModeParams = np.array([0,0,0,0])
        # Init diffraction geometry parameters
        self.detectorDistance = None
        self.photonEnergy = None
        self.wavelength = None
        self.pixelSize = None
        self.resolutionText = []
        # Init variables
        self.data = None # assembled detector image
        self.cx = None
        self.cy = None
        self.calib = None # ndarray detector image
        # Init hit finding parameters
        self.algInitDone = False
        self.algorithm = 1
        self.classify = False
        self.hitParam_alg_npix_min = 5.
        self.hitParam_alg_npix_max = 5000.
        self.hitParam_alg_amax_thr = 0.
        self.hitParam_alg_atot_thr = 0.
        self.hitParam_alg_son_min = 10.
        self.hitParam_alg1_thr_low = 10.
        self.hitParam_alg1_thr_high = 150.
        self.hitParam_alg1_radius = 5
        self.hitParam_alg1_dr = 0.05
        self.hitParam_alg2_thr = 10.
        self.hitParam_alg2_r0 = 5.
        self.hitParam_alg2_dr = 0.05
        self.params = [
            {'name': exp_grp, 'type': 'group', 'children': [
                {'name': exp_name_str, 'type': 'str', 'value': self.experimentName},
                {'name': exp_run_str, 'type': 'int', 'value': self.runNumber},
                #{'name': exp_det_str, 'type': 'list', 'values': self.detInfoList, 'value': 0},
                {'name': exp_detInfo_str, 'type': 'str', 'value': self.detInfo},
                {'name': exp_evt_str, 'type': 'int', 'value': self.eventNumber, 'limits': (0, 0), 'children': [
                    {'name': exp_second_str, 'type': 'str', 'value': self.eventSeconds, 'readonly': True},
                    {'name': exp_nanosecond_str, 'type': 'str', 'value': self.eventNanoseconds, 'readonly': True},
                    {'name': exp_fiducial_str, 'type': 'str', 'value': self.eventFiducial, 'readonly': True},
                    {'name': exp_numEvents_str, 'type': 'str', 'value': self.eventTotal, 'readonly': True},
                ]},
            ]},
            {'name': disp_grp, 'type': 'group', 'children': [
                {'name': disp_log_str, 'type': 'bool', 'value': self.logscaleOn, 'tip': "Display in log10"},
                {'name': disp_resolutionRings_str, 'type': 'bool', 'value': self.resolutionRingsOn, 'tip': "Display resolution rings", 'children': [
                    {'name': disp_resolution_str, 'type': 'str', 'value': self.resolution},
                ]},
                {'name': disp_commonMode_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
                    {'name': disp_overrideCommonMode_str, 'type': 'bool', 'value': self.applyCommonMode, 'tip': "Apply common mode (override)"},
                    {'name': disp_commonModeParam0_str, 'type': 'int', 'value': self.commonModeParams[0]},
                    {'name': disp_commonModeParam1_str, 'type': 'int', 'value': self.commonModeParams[1]},
                    {'name': disp_commonModeParam2_str, 'type': 'int', 'value': self.commonModeParams[2]},
                    {'name': disp_commonModeParam3_str, 'type': 'int', 'value': self.commonModeParams[3]},
                ]},
            ]},
            {'name': hitParam_grp, 'type': 'group', 'children': [
                {'name': hitParam_classify_str, 'type': 'bool', 'value': self.classify, 'tip': "Classify current image as hit or miss"},
                {'name': hitParam_algorithm_str, 'type': 'list', 'values': {hitParam_algorithm2_str: 2, hitParam_algorithm1_str: 1}, 'value': self.algorithm},
                {'name': hitParam_algorithm1_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
                    {'name': hitParam_alg_npix_min_str, 'type': 'float', 'value': self.hitParam_alg_npix_min},
                    {'name': hitParam_alg_npix_max_str, 'type': 'float', 'value': self.hitParam_alg_npix_max},
                    {'name': hitParam_alg_amax_thr_str, 'type': 'float', 'value': self.hitParam_alg_amax_thr},
                    {'name': hitParam_alg_atot_thr_str, 'type': 'float', 'value': self.hitParam_alg_atot_thr},
                    {'name': hitParam_alg_son_min_str, 'type': 'float', 'value': self.hitParam_alg_son_min},
                    {'name': hitParam_alg1_thr_low_str, 'type': 'float', 'value': self.hitParam_alg1_thr_low},
                    {'name': hitParam_alg1_thr_high_str, 'type': 'float', 'value': self.hitParam_alg1_thr_high},
                    {'name': hitParam_alg1_radius_str, 'type': 'int', 'value': self.hitParam_alg1_radius},
                    {'name': hitParam_alg1_dr_str, 'type': 'float', 'value': self.hitParam_alg1_dr},
                ]},
                {'name': hitParam_algorithm2_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
                    {'name': hitParam_alg2_thr_str, 'type': 'float', 'value': self.hitParam_alg2_thr},
                    {'name': hitParam_alg2_r0_str, 'type': 'float', 'value': self.hitParam_alg2_r0},
                    {'name': hitParam_alg2_dr_str, 'type': 'float', 'value': self.hitParam_alg2_dr},
                ]},
            ]},
        ]

        self.initUI()

    def initUI(self):
        ## Define a top-level widget to hold everything
        self.win = QtGui.QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1300,1400)
        self.win.setWindowTitle('psocake')

        ## Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', \
                                  children=self.params, expanded=True)
        self.p1 = Parameter.create(name='paramsDiffractionGeometry', type='group', \
                                  children=paramsDiffractionGeometry, expanded=True)
        self.p.sigTreeStateChanged.connect(self.change)
        self.p1.sigTreeStateChanged.connect(self.changeGeomParam)

        #self.p.param('Basic parameter data types', 'Event Number').sigTreeStateChanged.connect(self.change)
        #self.p.param('Basic parameter data types', 'Float').sigTreeStateChanged.connect(save)
        
        ## Create docks, place them into the window one at a time.
        ## Note that size arguments are only a suggestion; docks will still have to
        ## fill the entire dock area and obey the limits of their internal widgets.
        self.d1 = Dock("Image Panel", size=(900, 900))     ## give this dock the minimum possible size
        self.d2 = Dock("Dock2 - Parameters", size=(500,300))
        self.d3 = Dock("Dock3", size=(200,200))
        self.d4 = Dock("Dock4 (tabbed) - Plot", size=(200,200))
        self.d5 = Dock("Dock5 ", size=(50,50))
        self.d6 = Dock("Dock6 (tabbed) - Plot", size=(900,200))
        self.d7 = Dock("Dock7 - Console", size=(200,200), closable=True)

        self.area.addDock(self.d1, 'left')      ## place d1 at left edge of dock area
        self.area.addDock(self.d2, 'right')     ## place d2 at right edge of dock area
        self.area.addDock(self.d3, 'bottom', self.d2)## place d3 at bottom edge of d1
        self.area.addDock(self.d4, 'right')     ## place d4 at right edge of dock area
        self.area.addDock(self.d5, 'top', self.d1)  ## place d5 at left edge of d1
        self.area.addDock(self.d6, 'bottom')   ## place d6 at top edge of d4
        self.area.addDock(self.d7, 'bottom', self.d4)   ## place d7 at left edge of d5

        ## Dock 1: Image Panel
        self.nextBtn = QtGui.QPushButton('Next evt')
        self.prevBtn = QtGui.QPushButton('Prev evt')
        self.saveBtn = QtGui.QPushButton('Save evt')
        self.wQ = pg.LayoutWidget()
        self.w1 = pg.ImageView(view=pg.PlotItem())
        self.w1.getView().invertY(False)
        self.ring_feature = pg.ScatterPlotItem()
        self.peak_feature = pg.ScatterPlotItem()
        self.z_direction = pg.ScatterPlotItem()
        self.z_direction1 = pg.ScatterPlotItem()
        self.w1.getView().addItem(self.ring_feature)
        self.w1.getView().addItem(self.peak_feature)
        self.w1.getView().addItem(self.z_direction)
        self.w1.getView().addItem(self.z_direction1)
        self.wQ.addWidget(self.w1, row=1, colspan=2)
        self.wQ.addWidget(self.prevBtn, row=2, col=0)
        self.wQ.addWidget(self.nextBtn, row=2, col=1)
        self.wQ.addWidget(self.saveBtn, row=3, col=0)

        # Custom ROI for selecting an image region
        self.roi = pg.ROI(pos=[900, 900], size=[50, 50], snapSize=1.0, scaleSnap=True, translateSnap=True)
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.w1.getView().addItem(self.roi)
        #self.roi.setZValue(10)  # make sure ROI is drawn above image

        # Callbacks for handling user interaction
        def updateRoiHistogram():
            if self.data is not None:
                selected, coord = self.roi.getArrayRegion(self.data, self.w1.getImageItem(), returnMappedCoords=True)
                hist,bin = np.histogram(selected.flatten(), bins=1000)
                self.w4.plot(bin, hist, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True)
        self.roi.sigRegionChanged.connect(updateRoiHistogram)

        def next():
            self.eventNumber += 1
            if self.eventNumber >= self.eventTotal:
                self.eventNumber = self.eventTotal-1
            else:
                self.calib, self.data = self.getDetImage(self.eventNumber)
                self.w1.setImage(self.data-20)
                self.p.param(exp_grp,exp_evt_str).setValue(self.eventNumber)

        def prev():
            self.eventNumber -= 1
            if self.eventNumber < 0:
                self.eventNumber = 0
            else:
                self.calib, self.data = self.getDetImage(self.eventNumber)
                self.w1.setImage(self.data)
                self.p.param(exp_grp,exp_evt_str).setValue(self.eventNumber)

        def save():
            outputName = "psocake_"+str(self.experimentName)+"_"+str(self.runNumber)+"_"+str(self.detInfo)+"_" \
                         +str(self.eventNumber)+"_"+str(self.eventSeconds)+"_"+str(self.eventNanoseconds)+"_" \
                         +str(self.eventFiducial)+".npy"
            print "Saving detector image as numpy ndarray: ", outputName
            np.save(outputName,self.calib)

          ##### Polar #####
            #plot = pg.plot()
            #plot.setAspectLocked()
            ## Add polar grid lines
            #plot.addLine(x=10, pen='r') # y-line
            #plot.addLine(y=0, pen='r') # x-line
            #for r in range(2, 20, 2):
            #    circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r*2, r*2)
            #    circle.setPen(pg.mkPen('y', width=0.5, style=QtCore.Qt.DashLine))
            #    plot.addItem(circle)
            # make polar data
            #theta = np.linspace(0, 2*np.pi, 100)
            #radius = np.random.normal(loc=10, size=100)
            # Transform to cartesian and plot
            #x = radius * np.cos(theta)
            #y = radius * np.sin(theta)
            #plot.plot(x, y)
            #self.wQ.addWidget(plot, row=0, colspan=2)

        # Reading from HDF5
        #f = h5py.File('/reg/d/psdm/amo/amo86615/scratch/yoon82/amo86615_89_139_class_v1.h5','r')
        #data = np.array(f['/hitClass/adu'])
        #f.close()

        # Reading from events
        #data = np.random.normal(size=(100, 200, 200))

        data = np.zeros((10,500,500))
        self.w1.setImage(data, xvals=np.linspace(1., data.shape[0], data.shape[0]))

        self.nextBtn.clicked.connect(next)
        self.prevBtn.clicked.connect(prev)
        self.saveBtn.clicked.connect(save)
        self.d1.addWidget(self.wQ)

        ## Dock 2: parameter
        self.w2 = ParameterTree()
        self.w2.setParameters(self.p, showTop=False)
        self.w2.setWindowTitle('Parameters')
        self.d2.addWidget(self.w2)

        ## Dock 3
        self.w3 = ParameterTree()
        self.w3.setParameters(self.p1, showTop=False)
        self.w3.setWindowTitle('Diffraction geometry')
        self.d3.addWidget(self.w3)

        ## Dock 4
        self.w4 = pg.PlotWidget(title="Plot inside dock with no title bar")
        self.w4.plot(np.random.normal(size=100))
        self.d4.addWidget(self.w4)

        ## Dock 5 - intensity display
        self.d5.hideTitleBar()
        self.w5 = pg.GraphicsView(background=pg.mkColor(sandstone100_rgb))
        self.d5.addWidget(self.w5)

        ## Dock 6
        self.w6 = pg.ImageView(view=pg.PlotItem())

        ## Scan mirrors
        self.scanx = 250
        self.scany = 20
        self.m1 = Mirror(dia=4.2, d=0.001, pos=(self.scanx, 0), angle=315)
        self.m2 = Mirror(dia=8.4, d=0.001, pos=(self.scanx, self.scany), angle=135)

        ## Scan lenses
        self.l3 = Lens(r1=23.0, r2=0, d=5.8, pos=(self.scanx+50, self.scany), glass='Corning7980')  ## 50mm  UVFS  (LA4148)
        self.l4 = Lens(r1=0, r2=69.0, d=3.2, pos=(self.scanx+250, self.scany), glass='Corning7980')  ## 150mm UVFS  (LA4874)

        ## Objective
        self.obj = Lens(r1=15, r2=15, d=10, dia=8, pos=(self.scanx+400, self.scany), glass='Corning7980')

        self.IROptics = [self.m1, self.m2, self.l3, self.l4, self.obj]

        for o in set(self.IROptics):
            self.w6.getView().addItem(o)

        self.IRRays = []
        for dy in [-0.4, -0.15, 0, 0.15, 0.4]:
            self.IRRays.append(Ray(start=Point(-50, dy), dir=(1, 0), wl=780))
        for r in set(self.IRRays):
            self.w6.getView().addItem(r)

        self.IRTracer = Tracer(self.IRRays, self.IROptics)

        self.d6.addWidget(self.w6, row=0, colspan=2)

        ## Dock 7: console
        self.w7 = pg.console.ConsoleWidget()
        self.d7.addWidget(self.w7)

        # Setup input parameters
        if self.experimentName is not "":
            self.hasExperimentName = True
            self.p.param(exp_grp,exp_name_str).setValue(self.experimentName)
            self.updateExpName(self.experimentName)
        if self.runNumber is not 0:
            self.hasRunNumber = True
            self.p.param(exp_grp,exp_run_str).setValue(self.runNumber)
            self.updateRunNumber(self.runNumber)
        if self.detInfo is not "":
            self.hasDetInfo = True
            self.p.param(exp_grp,exp_detInfo_str).setValue(self.detInfo)
            self.updateDetInfo(self.detInfo)
        self.p.param(exp_grp,exp_evt_str).setValue(self.eventNumber)
        self.updateEventNumber(self.eventNumber)

        self.drawLabCoordinates() # FIXME: This does not match the lab coordinates yet!

        # Try mouse over crosshair
        self.xhair = self.w1.getView()
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
                    textInfo = "<span style='color: " + cardinalRed_hex + "; font-size: 24pt;'>x=%0.1f y=%0.1f I=%0.1f </span>"
                    self.label.setText(textInfo % (mousePoint.x(), mousePoint.y(), self.data[indexX,indexY]))

        self.proxy = pg.SignalProxy(self.xhair.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)

        self.win.show()
        #embed()

    def drawLabCoordinates(self):
        # Draw xy arrows
        symbolSize = 40
        cutoff=symbolSize/2
        headLen=30
        tailLen=30-cutoff
        xArrow = pg.ArrowItem(angle=180, tipAngle=30, baseAngle=20, headLen=headLen, tailLen=tailLen, tailWidth=8, pen=None, brush='b', pxMode=False)
        xArrow.setPos(2*headLen,0)
        self.w1.getView().addItem(xArrow)
        yArrow = pg.ArrowItem(angle=-90, tipAngle=30, baseAngle=20, headLen=headLen, tailLen=tailLen, tailWidth=8, pen=None, brush='r', pxMode=False)
        yArrow.setPos(0,2*headLen)
        self.w1.getView().addItem(yArrow)

        # z-direction
        self.z_direction.setData([0], [0], symbol='o', \
                                 size=symbolSize, brush='w', \
                                 pen={'color': 'k', 'width': 4}, pxMode=False)
        self.z_direction1.setData([0], [0], symbol='o', \
                                 size=symbolSize/6, brush='k', \
                                 pen={'color': 'k', 'width': 4}, pxMode=False)
        # Add xyz text
        self.x_text = pg.TextItem(html='<div style="text-align: center"><span style="color: #0000FF; font-size: 16pt;">x</span></div>', anchor=(0,0))
        self.w1.getView().addItem(self.x_text)
        self.x_text.setPos(2*headLen, 0)
        self.y_text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FF0000; font-size: 16pt;">y</span></div>', anchor=(1,1))
        self.w1.getView().addItem(self.y_text)
        self.y_text.setPos(0, 2*headLen)
        self.z_text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFFFFF; font-size: 16pt;">z</span></div>', anchor=(1,0))
        self.w1.getView().addItem(self.z_text)
        self.z_text.setPos(-headLen, 0)

        # Label xy axes
        self.x_axis = self.w1.getView().getAxis('bottom')
        self.x_axis.setLabel('X-axis (pixels)')
        self.y_axis = self.w1.getView().getAxis('left')
        self.y_axis.setLabel('Y-axis (pixels)')

    def updateClassification(self):
        print("Running hit finder")
        # Peak output (0-16):
        # 0 seg
        # 1 row
        # 2 col
        # 3 npix: no. of pixels in the ROI intensities above threshold
        # 4 amp_max: max intensity
        # 5 amp_tot: sum of intensities
        # 6,7: row_cgrav: center of mass
        # 8,9: row_sigma
        # 10,11,12,13: minimum bounding box
        # 14: background
        # 15: noise
        # 16: signal over noise

        # Only initialize the hit finder algorithm once
        if self.algInitDone is False:
            winds = None
            mask = None
            self.alg = PyAlgos(windows=winds, mask=mask, pbits=0)

            # set peak-selector parameters:
            self.alg.set_peak_selection_pars(npix_min=self.hitParam_alg_npix_min, npix_max=self.hitParam_alg_npix_max, \
                                        amax_thr=self.hitParam_alg_amax_thr, atot_thr=self.hitParam_alg_atot_thr, \
                                        son_min=self.hitParam_alg_son_min)
            self.algInitDone = True

        if self.algorithm == 1:
            # v1 - aka Droplet Finder - two-threshold peak-finding algorithm in restricted region
            #                           around pixel with maximal intensity.
            self.peaks = self.alg.peak_finder_v1(self.calib, thr_low=self.hitParam_alg1_thr_low, thr_high=self.hitParam_alg1_thr_high, \
                                       radius=int(self.hitParam_alg1_radius), dr=self.hitParam_alg1_dr)
        elif self.algorithm == 2:
            # v2 - define peaks for regions of connected pixels above threshold
            self.peaks = self.alg.peak_finder_v2(self.calib, thr=self.hitParam_alg2_thr, r0=self.hitParam_alg2_r0, dr=self.hitParam_alg2_dr)

        self.numPeaksFound = self.peaks.shape[0]
        print "peaks: ", self.peaks
        print "num peaks found: ", self.numPeaksFound, self.peaks.shape
        #sys.stdout.flush()
        self.drawPeaks()

    def drawPeaks(self):
        if self.peaks is not None and self.numPeaksFound > 0:
            iX  = np.array(self.det.indexes_x(self.evt), dtype=np.int64)
            iY  = np.array(self.det.indexes_y(self.evt), dtype=np.int64)
            cenX = iX[np.array(self.peaks[:,0],dtype=np.int64),np.array(self.peaks[:,1],dtype=np.int64),np.array(self.peaks[:,2],dtype=np.int64)]
            cenY = iY[np.array(self.peaks[:,0],dtype=np.int64),np.array(self.peaks[:,1],dtype=np.int64),np.array(self.peaks[:,2],dtype=np.int64)]
            diameter = 5
            self.peak_feature.setData(cenX, cenY, symbol='o', \
                                      size=diameter, brush=(255,255,255,0), \
                                      pen='r', pxMode=False)
        print "Done updatePeaks"

    def updateImage(self):
        if self.hasExperimentName and self.hasRunNumber and self.hasDetInfo:
            self.calib, self.data = self.getDetImage(self.eventNumber)
            if self.logscaleOn:
                self.w1.setImage(np.log10(abs(self.data)+eps))
            else:
                self.w1.setImage(self.data)
        print "Done updateImage"

    def updateRings(self):
        if self.resolutionRingsOn:
            self.clearRings()
            cenx = np.ones_like(self.myResolutionRingList)*self.cx
            ceny = np.ones_like(self.myResolutionRingList)*self.cy
            diameter = 2*self.myResolutionRingList
            print "self.myResolutionRingList, diameter: ", self.myResolutionRingList, diameter
            self.ring_feature.setData(cenx, ceny, symbol='o', \
                                      size=diameter, brush=(255,255,255,0), \
                                      pen='r', pxMode=False)
            for i,val in enumerate(self.dMin):
                self.resolutionText.append(pg.TextItem(text='%s A' % float('%.3g' % (val*1e10)), border='w', fill=(0, 0, 255, 100)))
                self.w1.getView().addItem(self.resolutionText[i])
                self.resolutionText[i].setPos(self.myResolutionRingList[i]+self.cx, self.cy)

        else:
            self.clearRings()
        print "Done updateRings"

    def clearRings(self):
        if self.resolutionText:
            print "going to clear rings: ", self.resolutionText, len(self.resolutionText)
            cen = [0,]
            self.ring_feature.setData(cen, cen, size=0)
            for i,val in enumerate(self.resolutionText):
                self.w1.getView().removeItem(self.resolutionText[i])
            self.resolutionText = []

    def getEvt(self,evtNumber):
        print "getEvt: ", evtNumber
        if self.hasRunNumber: #self.run is not None:
            evt = self.run.event(self.times[evtNumber])
            return evt
        else:
            return None

    def getDetImage(self,evtNumber):
        if self.run is not None:
            self.evt = self.getEvt(evtNumber)
            if self.applyCommonMode:
                if self.commonMode[0] == 5: # Algorithm 5
                    calib = self.det.calib(self.evt, cmpars=(self.commonMode[0],self.commonMode[1]))
                else: # Algorithms 1 to 4
                    print "### Overriding common mode: ", self.commonMode
                    calib = self.det.calib(self.evt, cmpars=(self.commonMode[0],self.commonMode[1],self.commonMode[2],self.commonMode[3]))
            else:
                calib = self.det.calib(self.evt)

        if calib is not None:
            calib *= self.det.gain(self.evt)
            data = self.det.image(self.evt, calib)

            self.cx, self.cy = self.getCentre(data.shape)
            print "cx,cy: ", self.cx, self.cy

            return calib, data
        else:
            return None

    def getCentre(self, dim):
        cx = dim[0]/2
        cy = dim[1]/2
        return cx,cy

    def getEventID(self,evt):
        if evt is not None:
            evtid = evt.get(psana.EventId)
            seconds = evtid.time()[0]
            nanoseconds = evtid.time()[1]
            fiducials = evtid.fiducials()
            return seconds, nanoseconds, fiducials

    # If anything changes in the parameter tree, print a message
    def change(self, param, changes):
        for param, change, data in changes:
            path = self.p.childPath(param)
            print('  path: %s'% path)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')
            self.update(path,change,data)

    def changeGeomParam(self, param, changes):
        for param, change, data in changes:
            path = self.p1.childPath(param)
            print('  path: %s'% path)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')
            self.update(path,data)

    def update(self, path, change, data):
        print "path: ", path
        if path[0] == exp_grp:
            if path[1] == exp_name_str:
                self.updateExpName(data)
                if self.classify:
                    self.updateClassification()
            elif path[1] == exp_run_str:
                self.updateRunNumber(data)
                if self.classify:
                    self.updateClassification()
            elif path[1] == exp_detInfo_str:
                self.updateDetInfo(data)
                if self.classify:
                    self.updateClassification()
            elif path[1] == exp_evt_str and len(path) == 2 and change is 'value':
                self.updateEventNumber(data)
                if self.classify:
                    self.updateClassification()
        if path[0] == disp_grp:
            if path[1] == disp_log_str:
                self.updateLogscale(data)
            elif path[1] == disp_resolutionRings_str and len(path) == 2:
                self.updateResolutionRings(data)
            elif path[2] == disp_resolution_str:
                self.updateResolution(data)
            elif path[2] == disp_commonModeParam0_str:
                self.updateCommonModeParam(data, 0)
            elif path[2] == disp_commonModeParam1_str:
                self.updateCommonModeParam(data, 1)
            elif path[2] == disp_commonModeParam2_str:
                self.updateCommonModeParam(data, 2)
            elif path[2] == disp_commonModeParam3_str:
                self.updateCommonModeParam(data, 3)
            elif path[2] == disp_overrideCommonMode_str:
                self.updateCommonMode(data)
        if path[0] == hitParam_grp:
            if path[1] == hitParam_algorithm_str:
                self.updateAlgorithm(data)
            elif path[1] == hitParam_classify_str:
                self.updateClassify(data)
            elif path[2] == hitParam_alg_npix_min_str:
                self.hitParam_alg_npix_min = data
                self.algInitDone = False
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg_npix_max_str:
                self.hitParam_alg_npix_max = data
                self.algInitDone = False
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg_amax_thr_str:
                self.hitParam_alg_amax_thr = data
                self.algInitDone = False
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg_atot_thr_str:
                self.hitParam_alg_atot_thr = data
                self.algInitDone = False
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg_son_min_str:
                self.hitParam_alg_son_min = data
                self.algInitDone = False
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg1_thr_low_str:
                self.hitParam_alg1_thr_low = data
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg1_thr_high_str:
                self.hitParam_alg1_thr_high = data
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg1_radius_str:
                self.hitParam_alg1_radius = data
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg1_dr_str:
                self.hitParam_alg1_dr = data
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg2_thr_str:
                self.hitParam_alg2_thr = data
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg2_r0_str:
                self.hitParam_alg2_r0 = data
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg2_dr_str:
                self.hitParam_alg2_dr = data
                if self.classify:
                    self.updateClassification()
        if path[0] == geom_grp:
            if path[1] == geom_detectorDistance_str:
                self.updateDetectorDistance(data)
            elif path[1] == geom_photonEnergy_str:
                self.updatePhotonEnergy(data)
            elif path[1] == geom_pixelSize_str:
                self.updatePixelSize(data)

    ###################################
    ###### Experiment Parameters ######
    ###################################

    def updateExpName(self, data):
        self.experimentName = data
        self.hasExperimentName = True

        self.setupExperiment()

        self.updateImage()
        print "Done updateExperimentName:", self.experimentName

    def updateRunNumber(self, data):
        self.runNumber = data
        self.hasRunNumber = True

        self.setupExperiment()

        self.updateImage()
        print "Done updateRunNumber: ", self.runNumber

    def updateDetInfo(self, data):
        self.detInfo = data
        self.hasDetInfo = True
        self.setupExperiment()
        self.updateImage()
        print "Done updateDetInfo: ", self.detInfo

    def updateEventNumber(self, data):
        self.eventNumber = data
        if self.eventNumber >= self.eventTotal:
            self.eventNumber = self.eventTotal-1
        # update timestamps and fiducial
        self.evt = self.getEvt(self.eventNumber)
        if self.evt is not None:
            sec, nanosec, fid = self.getEventID(self.evt)
            self.eventSeconds = str(sec)
            self.eventNanoseconds = str(nanosec)
            self.eventFiducial = str(fid)
            self.updateEventID(self.eventSeconds, self.eventNanoseconds, self.eventFiducial)
            self.p.param(exp_grp,exp_evt_str).setValue(self.eventNumber)
            self.updateImage()
        print "Done updateEventNumber: ", self.eventNumber

    def hasExpRunInfo(self):
        if self.hasExperimentName and self.hasRunNumber:
            print "hasExpRunInfo: True"
            return True
        else:
            print "hasExpRunInfo: False"
            return False

    def hasExpRunDetInfo(self):
        if self.hasExperimentName and self.hasRunNumber and self.hasDetInfo:
            print "hasExpRunDetInfo: True"
            return True
        else:
            print "hasExpRunDetInfo: False"
            return False

    def setupExperiment(self):
        if self.hasExpRunInfo():
            if args.localCalib:
                print "Using local calib directory"
                psana.setOption('psana.calib-dir','./calib')
            self.ds = psana.DataSource('exp='+str(self.experimentName)+':run='+str(self.runNumber)+':idx')
            self.run = self.ds.runs().next()
            self.times = self.run.times()
            self.eventTotal = len(self.times)
            self.p.param(exp_grp,exp_evt_str).setLimits((0,self.eventTotal-1))
            self.p.param(exp_grp,exp_evt_str,exp_numEvents_str).setValue(self.eventTotal)
            self.env = self.ds.env()

            evt = self.run.event(self.times[0])
            myAreaDetectors = []
            for k in evt.keys():
                if Detector.PyDetector.isAreaDetector(k.src()):
                    myAreaDetectors.append(k.alias())
            self.detInfoList = list(set(myAreaDetectors))
            print "# Available detectors: ", self.detInfoList

        if self.hasExpRunDetInfo():
            self.det = psana.Detector(str(self.detInfo), self.env)

            # Get epics variable, clen
            if "cxi" in self.experimentName:
                self.epics = self.ds.env().epicsStore()
                self.clen = self.epics.value('CXI:DS1:MMS:06.RBV')
                print "clen: ", self.clen
            print "Done setupExperiment"

    def updateLogscale(self, data):
        self.logscaleOn = data
        if self.hasExpRunDetInfo():
            self.updateImage()
        print "Done updateLogscale: ", self.logscaleOn

    def updateResolutionRings(self, data):
        self.resolutionRingsOn = data
        if self.hasExpRunDetInfo():
            self.updateRings()
        print "Done updateResolutionRings: ", self.resolutionRingsOn

    def updateResolution(self, data):
        # convert to array of floats
        _resolution = data.split(',')
        self.resolution = np.zeros((len(_resolution,)))
        for i in range(len(_resolution)):
            self.resolution[i] = float(_resolution[i])

        if len(_resolution) > 0:
            self.hasUserDefinedResolution = True
        else:
            self.hasUserDefinedResolution = False

        if self.hasGeometryInfo():
            self.updateGeometry()
        if self.hasExpRunDetInfo():
            self.updateRings()
        print "Done updateResolution: ", self.resolution, self.hasUserDefinedResolution

    def updateCommonModeParam(self, data, ind):
        self.commonModeParams[ind] = data
        self.updateCommonMode(self.applyCommonMode)
        print "Done updateCommonModeParam: ", self.commonModeParams

    def updateCommonMode(self, data):
        self.applyCommonMode = data
        if self.applyCommonMode:
            self.commonMode = self.checkCommonMode(self.commonModeParams)
        if self.hasExpRunDetInfo():
            self.setupExperiment()
            self.updateImage()
        print "Done updateCommonMode: ", self.commonMode

    def checkCommonMode(self, _commonMode):
        _alg = int(_commonMode[0])
        if _alg >= 1 and _alg <= 4:
            _param1 = int(_commonMode[1])
            _param2 = int(_commonMode[2])
            _param3 = int(_commonMode[3])
            return (_alg,_param1,_param2,_param3)
        elif _alg == 5 and _numParams == 2:
            _param1 = int(_commonMode[1])
            return (_alg,_param1)
        else:
            print "Undefined common mode algorithm"
            return None


    def updateEventID(self, sec, nanosec, fid):
        print "eventID: ", sec, nanosec, fid
        self.p.param(exp_grp,exp_evt_str,exp_second_str).setValue(self.eventSeconds)
        self.p.param(exp_grp,exp_evt_str,exp_nanosecond_str).setValue(self.eventNanoseconds)
        self.p.param(exp_grp,exp_evt_str,exp_fiducial_str).setValue(self.eventFiducial)

    ########################
    ###### Hit finder ######
    ########################

    def updateAlgorithm(self, data):
        self.algorithm = data
        if self.classify:
            self.updateClassification()
        print "##### Done updateAlgorithm: ", self.algorithm

    def updateClassify(self, data):
        self.classify = data
        if self.classify:
            self.updateClassification()
        print "Done updateClassify: ", self.classify

    ##################################
    ###### Diffraction Geometry ######
    ##################################

    def updateDetectorDistance(self, data):
        self.detectorDistance = data
        if self.hasGeometryInfo():
            self.updateGeometry()

    def updatePhotonEnergy(self, data):
        self.photonEnergy = data
        # E = hc/lambda
        h = 6.626070e-34 # J.m
        c = 2.99792458e8 # m/s
        joulesPerEv = 1.602176621e-19 #J/eV
        self.wavelength = (h/joulesPerEv*c)/self.photonEnergy
        print "wavelength: ", self.wavelength
        self.p1.param(geom_grp,geom_wavelength_str).setValue(self.wavelength)
        if self.hasGeometryInfo():
            self.updateGeometry()

    def updatePixelSize(self, data):
        self.pixelSize = data
        if self.hasGeometryInfo():
            self.updateGeometry()

    def hasGeometryInfo(self):
        if self.detectorDistance is not None \
           and self.photonEnergy is not None \
           and self.pixelSize is not None:
            return True
        else:
            return False

    def updateGeometry(self):
        if self.hasUserDefinedResolution:
            self.myResolutionRingList = self.resolution
        else:
            self.myResolutionRingList = resolutionRingList
        self.dMin = np.zeros_like(self.myResolutionRingList)
        for i, pix in enumerate(self.myResolutionRingList):
            thetaMax = np.arctan(pix*self.pixelSize/self.detectorDistance)
            qMax = 2/self.wavelength*np.sin(thetaMax/2)
            self.dMin[i] = 1/(2*qMax)
            print "updateGeometry: ", i, thetaMax, qMax, self.dMin[i]
            if self.resolutionRingsOn:
                self.updateRings()

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtGui.QApplication(sys.argv)
    ex = MainFrame(sys.argv)
    sys.exit(app.exec_())

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    main()
    #import sys
    #if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #    QtGui.QApplication.instance().exec_()

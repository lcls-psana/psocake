# TODO: Zoom in area
# Multiple subplots

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
from Detector.PyDetector import PyDetector
from ImgAlgos.PyAlgos import PyAlgos # peak finding
import matplotlib.pyplot as plt

from optics import *
from pyqtgraph import Point

# Set up tolerance
eps = np.finfo("float64").eps
resolutionRingList = np.array([100.,300.,500.,700.,900.,1100.])
dMin = np.zeros_like(resolutionRingList)

# Set up list of parameters
exp_grp = 'Experiment information'
exp_name_str = 'Experiment Name'
exp_run_str = 'Run Number'
exp_evt_str = 'Event Number'
exp_detInfo_str = 'Detector ID'

disp_grp = 'Display'
disp_log_str = 'Logscale'
disp_resolutionRings_str = 'Resolution rings'

hitParam_grp = 'Hit finder'
hitParam_classify_str = 'Classify'
hitParam_algorithm_str = 'Algorithm'
# algorithm 1
hitParam_algorithm1_str = 'Droplet'
hitParam_alg1_npix_min_str = 'npix_min'
hitParam_alg1_npix_max_str = 'npix_max'
hitParam_alg1_amax_thr_str = 'amax_thr'
hitParam_alg1_atot_thr_str = 'atot_thr'
hitParam_alg1_son_min_str = 'son_min'
hitParam_alg1_thr_low_str = 'thr_low'
hitParam_alg1_thr_high_str = 'thr_high'
hitParam_alg1_radius_str = 'radius'
hitParam_alg1_dr_str = 'dr'
# algorithm 2
hitParam_algorithm2_str = 'Rank'
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

class MainFrame(QtGui.QWidget):
    """
    The main frame of the application
    """        
    def __init__(self, arg_list):
        super(MainFrame, self).__init__()
        # Init experiment parameters
        self.experimentName = "None"
        self.runNumber = 0
        self.eventNumber = 0
        self.detInfo = "None"
        self.hasExperimentName = False
        self.hasRunNumber = False
        self.hasEventNumber = False
        self.hasDetInfo = False
        # Init display parameters
        self.logscaleOn = False
        self.resolutionRingsOn = False
        # Init diffraction geometry parameters
        self.detectorDistance = None
        self.photonEnergy = None
        self.wavelength = None
        self.pixelSize = None
        # Init variables
        self.eventNumber = 0
        self.data = None # assembled detector image
        self.calib = None # ndarray detector image
        # Init hit finding parameters
        self.algorithm = 1
        self.hitParam_alg1_npix_min = 5.
        self.hitParam_alg1_npix_max = 5000.
        self.hitParam_alg1_amax_thr = 0.
        self.hitParam_alg1_atot_thr = 0.
        self.hitParam_alg1_son_min = 10.
        self.hitParam_alg1_thr_low = 10.
        self.hitParam_alg1_thr_high = 150.
        self.hitParam_alg1_radius = 5
        self.hitParam_alg1_dr = 0.05
        self.hitParam_alg2_thr = 10.
        self.hitParam_alg2_r0 = 5.
        self.hitParam_alg2_dr = 0.05
        self.params = [
            {'name': exp_grp, 'type': 'group', 'children': [
                {'name': exp_name_str, 'type': 'str', 'value': ""},
                {'name': exp_run_str, 'type': 'int', 'value': -1},
                {'name': exp_evt_str, 'type': 'int', 'value': -1},
                {'name': exp_detInfo_str, 'type': 'str', 'value': ""},
            ]},
            {'name': disp_grp, 'type': 'group', 'children': [
                {'name': disp_log_str, 'type': 'bool', 'value': False, 'tip': "Display in log10"},
                {'name': disp_resolutionRings_str, 'type': 'bool', 'value': False, 'tip': "Display resolution rings"},
            ]},
            {'name': hitParam_grp, 'type': 'group', 'children': [
                {'name': hitParam_classify_str, 'type': 'bool', 'value': False, 'tip': "Classify current image as hit or miss"},
                {'name': hitParam_algorithm_str, 'type': 'list', 'values': {"Rank": 2, "Droplet": 1}, 'value': self.algorithm},
                {'name': hitParam_algorithm1_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
                    {'name': hitParam_alg1_npix_min_str, 'type': 'float', 'value': self.hitParam_alg1_npix_min},
                    {'name': hitParam_alg1_npix_max_str, 'type': 'float', 'value': self.hitParam_alg1_npix_max},
                    {'name': hitParam_alg1_amax_thr_str, 'type': 'float', 'value': self.hitParam_alg1_amax_thr},
                    {'name': hitParam_alg1_atot_thr_str, 'type': 'float', 'value': self.hitParam_alg1_atot_thr},
                    {'name': hitParam_alg1_son_min_str, 'type': 'float', 'value': self.hitParam_alg1_son_min},
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
        self.d5 = Dock("Dock5 ", size=(200,200))
        self.d6 = Dock("Dock6 (tabbed) - Plot", size=(900,200))
        self.d7 = Dock("Dock7 - Console", size=(200,200), closable=True)

        self.area.addDock(self.d1, 'left')      ## place d1 at left edge of dock area
        self.area.addDock(self.d2, 'right')     ## place d2 at right edge of dock area
        self.area.addDock(self.d3, 'bottom', self.d2)## place d3 at bottom edge of d1
        self.area.addDock(self.d4, 'right')     ## place d4 at right edge of dock area
        self.area.addDock(self.d5, 'bottom', self.d4)  ## place d5 at left edge of d1
        self.area.addDock(self.d6, 'bottom')   ## place d6 at top edge of d4
        self.area.addDock(self.d7, 'bottom', self.d4)   ## place d7 at left edge of d5

        ## Dock 1: Image Panel
        self.nextBtn = QtGui.QPushButton('Next evt')
        self.prevBtn = QtGui.QPushButton('Prev evt')
        self.wQ = pg.LayoutWidget()
        self.w1 = pg.ImageView(view=pg.PlotItem())
        self.w1.getView().invertY(True)#False)
        self.ring_feature = pg.ScatterPlotItem()
        self.z_direction = pg.ScatterPlotItem()
        self.z_direction1 = pg.ScatterPlotItem()
        #self.xy_feature = pg.PlotItem()
        self.w1.getView().addItem(self.ring_feature)
        self.w1.getView().addItem(self.z_direction)
        self.w1.getView().addItem(self.z_direction1)
        #self.w5.getView().addItem(self.xy_feature)
        self.wQ.addWidget(self.w1, row=0, colspan=2)
        self.wQ.addWidget(self.prevBtn, row=2, col=0)
        self.wQ.addWidget(self.nextBtn, row=2, col=1)

        def next():
            self.eventNumber += 1
            self.calib, self.data = self.getEvt(self.eventNumber)
            self.w1.setImage(self.data,autoLevels=None)
            self.p.param(exp_grp,exp_evt_str).setValue(self.eventNumber)

        def prev():
            self.eventNumber -= 1
            self.calib, self.data = self.getEvt(self.eventNumber)
            self.w1.setImage(self.data)
            self.p.param(exp_grp,exp_evt_str).setValue(self.eventNumber)

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
        data = np.random.normal(size=(100, 200, 200))

        self.w1.setImage(data, xvals=np.linspace(1., data.shape[0], data.shape[0]))
        self.nextBtn.clicked.connect(next)
        self.prevBtn.clicked.connect(prev)
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

        ## Hide title bar on dock 4
        self.d4.hideTitleBar()
        self.w4 = pg.PlotWidget(title="Plot inside dock with no title bar")
        self.w4.plot(np.random.normal(size=100))
        self.d4.addWidget(self.w4)

        ## Dock 5
        self.w5 = pg.PlotWidget(title="Dock 5 plot")
        self.w5.plot(np.random.normal(size=100))
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
        if len(sys.argv) == 5:
            self.experimentName = str(sys.argv[1])
            self.hasExperientName = True
            self.p.param(exp_grp,exp_name_str).setValue(self.experimentName)

            self.runNumber = int(sys.argv[2])
            self.hasRunNumber = True
            self.p.param(exp_grp,exp_run_str).setValue(self.runNumber)

            self.eventNumber = int(sys.argv[3])
            self.hasEventNumber = True
            self.p.param(exp_grp,exp_evt_str).setValue(self.eventNumber)

            self.detInfo = str(sys.argv[4])
            self.hasDetInfo = True
            self.p.param(exp_grp,exp_detInfo_str).setValue(self.detInfo)

        self.drawLabCoordinates()

        # Try mouse over crosshair
        self.xhair = self.w1.getView()
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.xhair.addItem(self.vLine, ignoreBounds=True)
        self.xhair.addItem(self.hLine, ignoreBounds=True)
        self.vb = self.xhair.vb
        self.label = pg.LabelItem(justify='right')
        self.xhair.addItem(self.label)

        def mouseMoved(evt):
            pos = evt[0]  ## using signal proxy turns original arguments into a tuple
            if self.xhair.sceneBoundingRect().contains(pos):
                mousePoint = self.vb.mapSceneToView(pos)
                indexX = int(mousePoint.x())
                indexY = int(mousePoint.y())
                #if index > 0:# and index < len(data1):
                
            # update crosshair position
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            # get pixel value, if data exists
            if self.data is not None:
                # FIXME: Assume data is 2D for now
                if indexX >= 0 and indexX < self.data.shape[0] \
                   and indexY >= 0 and indexY < self.data.shape[1]:
                    self.label.setText("<span style='font-size: 36pt'>x=%0.1f y=%0.1f I=%0.1f </span>" % (mousePoint.x(), mousePoint.y(), self.data[indexX,indexY]))

        self.proxy = pg.SignalProxy(self.xhair.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)

        self.win.show()

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

        # Draw xy axis at (0,0)
        #self.w1.getView().addLine(x=0, pen='w')
        #self.w1.getView().addLine(y=0, pen='w')

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

        winds = None
        mask = None
        alg = PyAlgos(windows=winds, mask=mask, pbits=0)

        print "params: ", self.hitParam_alg1_thr_low, self.hitParam_alg2_thr
        # set peak-selector parameters: #FIXME: is this unique to algorithm 1?
        alg.set_peak_selection_pars(npix_min=self.hitParam_alg1_npix_min, npix_max=self.hitParam_alg1_npix_max, \
                                    amax_thr=self.hitParam_alg1_amax_thr, atot_thr=self.hitParam_alg1_atot_thr, \
                                    son_min=self.hitParam_alg1_son_min)

        if self.algorithm == 1:
            # v1 - aka Droplet Finder - two-threshold peak-finding algorithm in restricted region
            #                           around pixel with maximal intensity.
            self.peaks = alg.peak_finder_v1(self.calib, thr_low=self.hitParam_alg1_thr_low, thr_high=self.hitParam_alg1_thr_high, \
                                       radius=int(self.hitParam_alg1_radius), dr=self.hitParam_alg1_dr)
        elif self.algorithm == 2:
            # v2 - define peaks for regions of connected pixels above threshold
            self.peaks = alg.peak_finder_v2(self.calib, thr=self.hitParam_alg2_thr, r0=self.hitParam_alg2_r0, dr=self.hitParam_alg2_dr)

        print "num peaks found: ", self.peaks.shape[0]
        sys.stdout.flush()
        self.drawPeaks()

    def drawPeaks(self):
        if self.peaks is not None:
            # Pixel image indexes
            iX  = np.array(self.det.indexes_x(self.evt), dtype=np.int64) #- xoffset
            iY  = np.array(self.det.indexes_y(self.evt), dtype=np.int64) #- yoffset
            cenX = iX[np.array(self.peaks[:,0],dtype=np.int64),np.array(self.peaks[:,1],dtype=np.int64),np.array(self.peaks[:,2],dtype=np.int64)]
            cenY = iY[np.array(self.peaks[:,0],dtype=np.int64),np.array(self.peaks[:,1],dtype=np.int64),np.array(self.peaks[:,2],dtype=np.int64)]
            diameter = 5
            self.ring_feature.setData(cenX, cenY, symbol='o', \
                                      size=diameter, brush=(255,255,255,0), \
                                      pen='r', pxMode=False)
            self.resolutionText = []
            for i,val in enumerate(dMin):
                self.resolutionText.append(pg.TextItem(text='%s A' % float('%.3g' % (val*1e10)), border='w', fill=(0, 0, 255, 100)))
                self.w1.getView().addItem(self.resolutionText[i])
                self.resolutionText[i].setPos(resolutionRingList[i]+detCenX, detCenY)
        else:
            cen = [0,]
            self.ring_feature.setData(cen, cen, size=0)
            for i,val in enumerate(dMin):
                self.w1.getView().removeItem(self.resolutionText[i])
        print "Done updateRings"

    def updateImage(self):
        self.calib, self.data = self.getEvt(self.eventNumber)
        if self.logscaleOn:
            self.w1.setImage(np.log10(abs(self.data)+eps))
        else:
            self.w1.setImage(self.data)
        print "Done updateImage"

    def updateRings(self):
        if self.resolutionRingsOn:
            detCenX = detCenY = 512 # TODO: find centre of detector
            cen = np.ones_like(resolutionRingList)*detCenX
            diameter = 2*resolutionRingList
            self.ring_feature.setData(cen, cen, symbol='o', \
                                      size=diameter, brush=(255,255,255,0), \
                                      pen='r', pxMode=False)
            self.resolutionText = []
            for i,val in enumerate(dMin):
                self.resolutionText.append(pg.TextItem(text='%s A' % float('%.3g' % (val*1e10)), border='w', fill=(0, 0, 255, 100)))
                self.w1.getView().addItem(self.resolutionText[i])
                self.resolutionText[i].setPos(resolutionRingList[i]+detCenX, detCenY)
        else:
            print "let's remove resolutionText"
            cen = [0,]
            self.ring_feature.setData(cen, cen, size=0)
            for i,val in enumerate(dMin):
                self.w1.getView().removeItem(self.resolutionText[i])
        print "Done updateRings"

    def getEvt(self,evtNumber):
        if self.run is not None:
            self.evt = self.run.event(self.times[evtNumber])
            calib = self.det.calib(self.evt)
            #self.det.common_mode_apply(self.evt, self.calib)
        if calib is not None: 
            data = self.det.image(self.evt,calib)
            return calib, data
        else:
            return None
    
    ## If anything changes in the parameter tree, print a message
    def change(self, param, changes):
        for param, change, data in changes:
            path = self.p.childPath(param)
            print('  path: %s'% path)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')
            self.update(path,data)

    def changeGeomParam(self, param, changes):
        for param, change, data in changes:
            path = self.p1.childPath(param)
            print('  path: %s'% path)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')
            self.update(path,data)

    def update(self, path, data):
        print "path: ", path[0], path[1]
        if path[0] == exp_grp:
            if path[1] == exp_name_str:
                self.updateEventName(data)
            elif path[1] == exp_run_str:
                self.updateRunNumber(data)
            elif path[1] == exp_evt_str:
                self.updateEventNumber(data)
            elif path[1] == exp_detInfo_str:
                self.updateDetInfo(data)
        if path[0] == disp_grp:
            if path[1] == disp_log_str:
                self.updateLogscale(data)
            elif path[1] == disp_resolutionRings_str:
                self.updateResolutionRings(data)
        if path[0] == hitParam_grp:
            if path[1] == hitParam_algorithm_str:
                self.updateAlgorithm(data)
            elif path[1] == hitParam_classify_str:
                self.updateClassify(data)
            elif path[2] == hitParam_alg1_npix_min_str:
                self.hitParam_alg1_npix_min = data
            elif path[2] == hitParam_alg1_npix_max_str:
                self.hitParam_alg1_npix_max = data
            elif path[2] == hitParam_alg1_amax_thr_str:
                self.hitParam_alg1_amax_thr = data
            elif path[2] == hitParam_alg1_atot_thr_str:
                self.hitParam_alg1_atot_thr = data
            elif path[2] == hitParam_alg1_son_min_str:
                self.hitParam_alg1_son_min = data
            elif path[2] == hitParam_alg1_thr_low_str:
                self.hitParam_alg1_thr_low = data
            elif path[2] == hitParam_alg1_thr_high_str:
                self.hitParam_alg1_thr_high = data
            elif path[2] == hitParam_alg1_radius_str:
                self.hitParam_alg1_radius = data
            elif path[2] == hitParam_alg1_dr_str:
                self.hitParam_alg1_dr = data
            elif path[2] == hitParam_alg2_thr_str:
                self.hitParam_alg2_thr = data
            elif path[2] == hitParam_alg2_r0_str:
                self.hitParam_alg2_r0 = data
            elif path[2] == hitParam_alg2_dr_str:
                self.hitParam_alg2_dr = data
        if path[0] == geom_grp:
            if path[1] == geom_detectorDistance_str:
                self.updateDetectorDistance(data)
            elif path[1] == geom_photonEnergy_str:
                self.updatePhotonEnergy(data)
            elif path[1] == geom_pixelSize_str:
                self.updatePixelSize(data)

    ###### Experiment Parameters ######

    def updateEventName(self, data):
        self.experimentName = data
        self.hasExperimentName = True
        self.setupExperiment()
        if self.hasExperimentInfo():
            self.updateImage()
        print "Done updateEventName:", self.experimentName

    def updateRunNumber(self, data):
        self.runNumber = data
        self.hasRunNumber = True
        self.setupExperiment()
        if self.hasExperimentInfo():
            self.updateImage()
        print "Done updateRunNumber: ", self.runNumber

    def updateEventNumber(self, data):
        self.eventNumber = data
        self.hasEventNumber = True
        if self.hasExperimentInfo():
            self.updateImage()
        print "Done updateEventNumber: ", self.eventNumber

    def updateDetInfo(self, data):
        self.detInfo = data
        self.hasDetInfo = True
        self.setupExperiment()
        if self.hasExperimentInfo():
            self.updateImage()
        print "Done updateDetInfo: ", self.detInfo

    def hasExperimentInfo(self):
        if self.hasExperimentName and self.hasRunNumber \
           and self.hasEventNumber and self.hasDetInfo:
            return True
        else:
            return False

    def setupExperiment(self):
        if self.hasExperimentInfo():
            self.ds = psana.DataSource('exp='+str(self.experimentName)+':run='+str(self.runNumber)+':idx') #('exp=amo86615:run=190:idx')
            self.src = psana.Source('DetInfo('+str(self.detInfo)+')') #('DetInfo(Camp.0:pnCCD.1)')
            self.run = self.ds.runs().next()
            self.times = self.run.times()
            self.totalEvents = len(self.times)
            self.env = self.ds.env()
            self.det = PyDetector(self.src, self.env, pbits=0)
            print "Done setupExperiment"     

    def updateLogscale(self, data):
        self.logscaleOn = data
        if self.hasExperimentInfo():
            self.updateImage()
        print "Done updateLogscale: ", self.logscaleOn

    def updateResolutionRings(self, data):
        self.resolutionRingsOn = data
        if self.hasExperimentInfo():
            self.updateRings()
        print "Done updateResolutionRings: ", self.resolutionRingsOn

    ###### Hit finder ######

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

    ###### Diffraction Geometry ######

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
        for i, pix in enumerate(resolutionRingList):
            thetaMax = np.arctan(pix*self.pixelSize/self.detectorDistance)
            qMax = 2/self.wavelength*np.sin(thetaMax/2)
            dMin[i] = 1/(2*qMax)
            print i, thetaMax, qMax, dMin[i]

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

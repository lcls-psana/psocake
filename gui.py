#self.ds = psana.DataSource('exp=cxif5315:run=169:idx') #('exp=amo86615:run=190:idx')
#self.src = psana.Source('DetInfo(CxiDs2.0:Cspad.0)')#('DetInfo(Camp.0:pnCCD.1)')

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

geom_grp = 'Diffraction geometry'
geom_detectorDistance_str = 'Detector distance'
geom_photonEnergy_str = 'Photon energy'
geom_wavelength_str = "Wavelength"
geom_pixelSize_str = 'Pixel size'

params = [
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
]
params1 = [
    {'name': geom_grp, 'type': 'group', 'children': [
        {'name': geom_detectorDistance_str, 'type': 'float', 'value': 0.0, 'precision': 12, 'minVal': 1e-6, 'khsiFormat': (6,6), 'siPrefix': True, 'suffix': 'm'},
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

        self.experimentName = "None"
        self.runNumber = 0
        self.eventNumber = 0
        self.detInfo = "None"

        self.logscaleOn = False
        self.resolutionRingsOn = False

        self.detectorDistance = None
        self.photonEnergy = None
        self.wavelength = None
        self.pixelSize = None

        self.hasExperientName = False
        self.hasRunNumber = False
        self.hasEventNumber = False
        self.hasDetInfo = False
        self.counter = 0
        self.data = None # assembled detector image
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
                                  children=params, expanded=True)
        self.p1 = Parameter.create(name='params1', type='group', \
                                  children=params1, expanded=True)
        self.p.sigTreeStateChanged.connect(self.change)
        self.p1.sigTreeStateChanged.connect(self.change1)

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
        self.d6 = Dock("Dock6 (tabbed) - Plot", size=(200,200))
        self.d7 = Dock("Dock7 - Console", size=(200,200), closable=True)

        self.area.addDock(self.d1, 'left')      ## place d1 at left edge of dock area
        self.area.addDock(self.d2, 'right')     ## place d2 at right edge of dock area
        self.area.addDock(self.d3, 'bottom', self.d2)## place d3 at bottom edge of d1
        self.area.addDock(self.d4, 'right')     ## place d4 at right edge of dock area
        self.area.addDock(self.d5, 'bottom', self.d4)  ## place d5 at left edge of d1
        self.area.addDock(self.d6, 'top', self.d4)   ## place d6 at top edge of d4
        self.area.addDock(self.d7, 'bottom', self.d4)   ## place d7 at left edge of d5

        ## Dock 1
        self.nextBtn = QtGui.QPushButton('Next evt')
        self.prevBtn = QtGui.QPushButton('Prev evt')
        self.wQ = pg.LayoutWidget()
        self.w1 = pg.ImageView(view=pg.PlotItem())
        self.w1.getView().invertY(False)
        self.ring_feature = pg.ScatterPlotItem()
        self.z_direction = pg.ScatterPlotItem()
        self.z_direction1 = pg.ScatterPlotItem()
        #self.xy_feature = pg.PlotItem()
        self.w1.getView().addItem(self.ring_feature)
        self.w1.getView().addItem(self.z_direction)
        self.w1.getView().addItem(self.z_direction1)
        #self.w5.getView().addItem(self.xy_feature)
        self.wQ.addWidget(self.w1, row=0, colspan=2)
        self.wQ.addWidget(self.prevBtn, row=1, col=0)
        self.wQ.addWidget(self.nextBtn, row=1, col=1)

        def next():
            self.eventNumber += 1
            self.data = self.getEvt(self.eventNumber)
            self.w1.setImage(self.data)
            self.p.param(exp_grp,exp_evt_str).setValue(self.eventNumber)

          ##### Hit finder #####
            #print("Running hit finder")
            #winds = None
            #mask = None
            #alg = PyAlgos(windows=winds, mask=mask, pbits=0)
            # set peak-selector parameters:
            #alg.set_peak_selection_pars(npix_min=5, npix_max=5000, amax_thr=0, atot_thr=0, son_min=10)
            # min number 
            # v1 - aka Droplet Finder - two-threshold peak-finding algorithm in restricted region
            #                           around pixel with maximal intensity.
            #peaks = alg.peak_finder_v1(self.data, thr_low=10, thr_high=150, radius=5, dr=0.05)
            # v2 - define peaks for regoins of connected pixels above threshold
            #peaks = alg.peak_finder_v2(self.data, thr=10, r0=5, dr=0.05)
            #print "peaks: ", peaks, peaks.shape
            #sys.stdout.flush()
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

        def prev():
            self.eventNumber -= 1
            self.data = self.getEvt(self.eventNumber)
            self.w1.setImage(self.data)
            self.p.param(exp_grp,exp_evt_str).setValue(self.eventNumber)

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
        self.w6 = pg.PlotWidget(title="Dock 6 plot")
        self.w6.plot(np.random.normal(size=100))
        self.d6.addWidget(self.w6)

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

    def updateImage(self):
        self.data = self.getEvt(self.eventNumber)
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
            self.text = []
            for i,val in enumerate(dMin):
                self.text.append(pg.TextItem(text='%s A' % float('%.3g' % (val*1e10)), border='w', fill=(0, 0, 255, 100)))
                self.w1.getView().addItem(self.text[i])
                self.text[i].setPos(resolutionRingList[i]+detCenX, detCenY)
        else:
            cen = [0,]
            self.ring_feature.setData(cen, cen, size=0)
            self.w1.getView().removeItem(self.text)
        print "Done updateResolutionRings"

    def getEvt(self,evtNumber):
        if self.run is not None:
            evt = self.run.event(self.times[evtNumber])
            calib = self.det.calib(evt)
            self.det.common_mode_apply(evt, calib)
        if calib is not None: 
            self.data = self.det.image(evt,calib)
            return self.data
        else:
            return None
    
    ## If anything changes in the parameter tree, print a message
    def change(self, param, changes):
        for param, change, data in changes:
            path = self.p.childPath(param)
            #if path is not None:
            #    childName = '.'.join(path)
            #else:
            #    childName = param.name()
            print('  path: %s'% path)
            #print('  parameter: %s'% childName)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')
            self.update(path,data)

    def change1(self, param, changes):
        for param, change, data in changes:
            path = self.p1.childPath(param)
            print('  path: %s'% path)
            #print('  parameter: %s'% childName)
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
        if path[0] == geom_grp:
            if path[1] == geom_detectorDistance_str:
                self.updateDetectorDistance(data)
            elif path[1] == geom_photonEnergy_str:
                self.updatePhotonEnergy(data)
                print "photonEnergy: ", self.photonEnergy                   
            elif path[1] == geom_pixelSize_str:
                self.updatePixelSize(data)

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
#    import sys
#    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#        QtGui.QApplication.instance().exec_()

# GUI for browsing LCLS area detectors. Tune hit finding parameters and common mode correction.

# TODO: Zoom in area / numbers
# TODO: Multiple subplots
# TODO: grid of images
# TODO: powder pattern generator
# TODO: dropdown menu for available detectors
# TODO: When front and back detectors given, display both

import sys, signal
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.console
from pyqtgraph.dockarea import *
from pyqtgraph.dockarea.Dock import DockLabel
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import psana
import h5py
from ImgAlgos.PyAlgos import PyAlgos # peak finding
import matplotlib.pyplot as plt
from pyqtgraph import Point
import argparse
import Detector.PyDetector
import logging
import multiprocessing as mp
import time
import subprocess
import os.path

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
disp_image_str = 'Image properties'
disp_adu_str = 'adu'
disp_gain_str = 'gain'
disp_coordx_str = 'coord_x'
disp_coordy_str = 'coord_y'
disp_quad_str = 'quad number'
disp_seg_str = 'seg number'
disp_row_str = 'row number'
disp_col_str = 'col number'
disp_aduThresh_str = 'ADU threshold'

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
# algorithm 3
hitParam_alg3_npix_min_str = 'npix_min'
hitParam_alg3_npix_max_str = 'npix_max'
hitParam_alg3_amax_thr_str = 'amax_thr'
hitParam_alg3_atot_thr_str = 'atot_thr'
hitParam_alg3_son_min_str = 'son_min'
hitParam_algorithm3_str = 'Ranker'
hitParam_alg3_rank_str = 'rank'
hitParam_alg3_r0_str = 'r0'
hitParam_alg3_dr_str = 'dr'

hitParam_outDir_str = 'Output directory'
hitParam_runs_str = 'Run(s)'
hitParam_queue_str = 'queue'
hitParam_cpu_str = 'CPUs'
hitParam_psanaq_str = 'psanaq'
hitParam_psnehq_str = 'psnehq'
hitParam_psfehq_str = 'psfehq'
hitParam_psnehprioq_str = 'psnehprioq'
hitParam_psfehprioq_str = 'psfehprioq'
hitParam_psnehhiprioq_str = 'psnehhiprioq'
hitParam_psfehhiprioq_str = 'psfehhiprioq'
hitParam_noe_str = 'Number of events to process'

# Diffraction geometry parameter tree
geom_grp = 'Diffraction geometry'
geom_detectorDistance_str = 'Detector distance'
geom_photonEnergy_str = 'Photon energy'
geom_wavelength_str = "Wavelength"
geom_pixelSize_str = 'Pixel size'
geom_resolutionRings_str = 'Resolution rings'
geom_resolution_str = 'Resolution (pixels)'

paramsDiffractionGeometry = [
    {'name': geom_grp, 'type': 'group', 'children': [
        {'name': geom_detectorDistance_str, 'type': 'float', 'value': 0.0, 'precision': 12, 'minVal': 1e-6, 'siFormat': (6,6), 'siPrefix': True, 'suffix': 'm'},
        {'name': geom_photonEnergy_str, 'type': 'float', 'value': 0.0, 'step': 1e-6, 'siPrefix': True, 'suffix': 'eV'},
        {'name': geom_wavelength_str, 'type': 'float', 'value': 0.0, 'step': 1e-6, 'siPrefix': True, 'suffix': 'm', 'readonly': True},
        {'name': geom_pixelSize_str, 'type': 'float', 'value': 0.0, 'precision': 12, 'minVal': 1e-6, 'siPrefix': True, 'suffix': 'm'},
        {'name': geom_resolutionRings_str, 'type': 'bool', 'value': False, 'tip': "Display resolution rings", 'children': [
            {'name': geom_resolution_str, 'type': 'str', 'value': None},
        ]},
    ]},
]

# Quantifier parameter tree
quantifier_grp = 'Quantifier'
quantifier_filename_str = 'filename'
quantifier_dataset_str = 'metric_dataset'
quantifier_sort_str = 'sort'

# Color scheme
sandstone100_rgb = (221,207,153) # Sandstone
cardinalRed_hex = str("#8C1515") # Cardinal red

class MainFrame(QtGui.QWidget):
    """
    The main frame of the application
    """        
    def __init__(self, arg_list):
        super(MainFrame, self).__init__()
        self.firstUpdate = True
        # Init experiment parameters
        self.experimentName = args.exp
        self.runNumber = int(args.run)
        #self.detInfoList = [1,2,3,4]
        self.detInfo = args.det
        self.isCspad = False
        self.eventNumber = int(args.evt)
        self.eventSeconds = ""
        self.eventNanoseconds = ""
        self.eventFiducial = ""
        self.eventTotal = 0
        self.hasExperimentName = False
        self.hasRunNumber = False
        self.hasDetInfo = False
        # Init display parameters
        self.logscaleOn = False
        self.image_property = 1
        self.aduThresh = 0.

        self.hasUserDefinedResolution = False
        self.hasCommonMode = False
        self.applyCommonMode = False
        self.commonMode = np.array([0,0,0,0])
        self.commonModeParams = np.array([0,0,0,0])
        # Init diffraction geometry parameters
        self.detectorDistance = None
        self.photonEnergy = None
        self.wavelength = None
        self.pixelSize = None
        self.resolutionRingsOn = False
        self.resolution = None
        self.resolutionText = []
        # Init variables
        self.data = None # assembled detector image
        self.cx = None
        self.cy = None
        self.calib = None # ndarray detector image
        self.mask = None
        # Init hit finding parameters
        self.algInitDone = False
        self.algorithm = 1
        self.classify = False
        self.hitParam_alg_npix_min = 1.
        self.hitParam_alg_npix_max = 45.
        self.hitParam_alg_amax_thr = 250.
        self.hitParam_alg_atot_thr = 330.
        self.hitParam_alg_son_min = 10.
        self.hitParam_alg1_thr_low = 80.
        self.hitParam_alg1_thr_high = 270.
        self.hitParam_alg1_radius = 3
        self.hitParam_alg1_dr = 1
        self.hitParam_alg3_npix_min = 5.
        self.hitParam_alg3_npix_max = 5000.
        self.hitParam_alg3_amax_thr = 0.
        self.hitParam_alg3_atot_thr = 0.
        self.hitParam_alg3_son_min = 10.
        self.hitParam_alg3_rank = 3
        self.hitParam_alg3_r0 = 5.
        self.hitParam_alg3_dr = 0.05

        self.hitParam_outDir = os.getcwd()
        self.hitParam_runs = ''
        self.hitParam_queue = hitParam_psanaq_str
        self.hitParam_cpus = 32
        self.hitParam_noe = 0

        self.quantifier_filename = ''
        self.quantifier_dataset = ''
        self.quantifier_sort = False
        self.quantifierFileOpen = False
        self.quantifierHasData = False
        # Threads
        self.stackStart = 0
        self.stackSize = 60
        self.params = [
            {'name': exp_grp, 'type': 'group', 'children': [
                {'name': exp_name_str, 'type': 'str', 'value': self.experimentName, 'tip': "Experiment name, .e.g. cxic0415"},
                {'name': exp_run_str, 'type': 'int', 'value': self.runNumber, 'tip': "Run number, e.g. 15"},
                {'name': exp_detInfo_str, 'type': 'str', 'value': self.detInfo, 'tip': "Detector ID. Look at the terminal for available area detectors, e.g. DscCsPad"},
                {'name': exp_evt_str, 'type': 'int', 'value': self.eventNumber, 'tip': "Event number, first event is 0", 'children': [
                    {'name': exp_second_str, 'type': 'str', 'value': self.eventSeconds, 'readonly': True},
                    {'name': exp_nanosecond_str, 'type': 'str', 'value': self.eventNanoseconds, 'readonly': True},
                    {'name': exp_fiducial_str, 'type': 'str', 'value': self.eventFiducial, 'readonly': True},
                    {'name': exp_numEvents_str, 'type': 'str', 'value': self.eventTotal, 'readonly': True},
                ]},
            ]},
            {'name': disp_grp, 'type': 'group', 'children': [
                {'name': disp_log_str, 'type': 'bool', 'value': self.logscaleOn, 'tip': "Display in log10"},
                {'name': disp_image_str, 'type': 'list', 'values': {disp_col_str: 8,
                                                                    disp_row_str: 7,
                                                                    disp_seg_str: 6,
                                                                    disp_quad_str: 5,
                                                                    disp_gain_str: 4,
                                                                    disp_coordy_str: 3,
                                                                    disp_coordx_str: 2,
                                                                    disp_adu_str: 1},
                 'value': self.image_property, 'tip': "Choose image property to display"},
                {'name': disp_aduThresh_str, 'type': 'float', 'value': self.aduThresh, 'tip': "Only display ADUs above this threshold"},
                {'name': disp_commonMode_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
                    {'name': disp_overrideCommonMode_str, 'type': 'bool', 'value': self.applyCommonMode,
                     'tip': "Click to play around with common mode settings.\n This does not change your deployed calib file."},
                    {'name': disp_commonModeParam0_str, 'type': 'int', 'value': self.commonModeParams[0]},
                    {'name': disp_commonModeParam1_str, 'type': 'int', 'value': self.commonModeParams[1]},
                    {'name': disp_commonModeParam2_str, 'type': 'int', 'value': self.commonModeParams[2]},
                    {'name': disp_commonModeParam3_str, 'type': 'int', 'value': self.commonModeParams[3]},
                ]},
            ]},
        ]
        self.paramsPeakFinder = [
            {'name': hitParam_grp, 'type': 'group', 'children': [
                {'name': hitParam_classify_str, 'type': 'bool', 'value': self.classify, 'tip': "Classify current image as hit or miss"},
                {'name': hitParam_algorithm_str, 'type': 'list', 'values': {hitParam_algorithm3_str: 3,
                                                                            hitParam_algorithm1_str: 1},
                 'value': self.algorithm},
                {'name': hitParam_algorithm1_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
                    {'name': hitParam_alg_npix_min_str, 'type': 'float', 'value': self.hitParam_alg_npix_min, 'tip': "Only keep the peak if number of pixels above thr_low is above this value"},
                    {'name': hitParam_alg_npix_max_str, 'type': 'float', 'value': self.hitParam_alg_npix_max, 'tip': "Only keep the peak if number of pixels above thr_low is below this value"},
                    {'name': hitParam_alg_amax_thr_str, 'type': 'float', 'value': self.hitParam_alg_amax_thr, 'tip': "Only keep the peak if max value is above this value"},
                    {'name': hitParam_alg_atot_thr_str, 'type': 'float', 'value': self.hitParam_alg_atot_thr, 'tip': "Only keep the peak if integral inside region of interest is above this value"},
                    {'name': hitParam_alg_son_min_str, 'type': 'float', 'value': self.hitParam_alg_son_min, 'tip': "Only keep the peak if signal-over-noise is above this value"},
                    {'name': hitParam_alg1_thr_low_str, 'type': 'float', 'value': self.hitParam_alg1_thr_low, 'tip': "Only consider values above this value"},
                    {'name': hitParam_alg1_thr_high_str, 'type': 'float', 'value': self.hitParam_alg1_thr_high, 'tip': "Only keep the peak if max value is above this value"},
                    {'name': hitParam_alg1_radius_str, 'type': 'int', 'value': self.hitParam_alg1_radius, 'tip': "region of integration is a square, (2r+1)x(2r+1)"},
                    {'name': hitParam_alg1_dr_str, 'type': 'float', 'value': self.hitParam_alg1_dr, 'tip': "background region outside the region of interest"},
                ]},
                {'name': hitParam_algorithm3_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
                    {'name': hitParam_alg3_npix_min_str, 'type': 'float', 'value': self.hitParam_alg3_npix_min},
                    {'name': hitParam_alg3_npix_max_str, 'type': 'float', 'value': self.hitParam_alg3_npix_max},
                    {'name': hitParam_alg3_amax_thr_str, 'type': 'float', 'value': self.hitParam_alg3_amax_thr},
                    {'name': hitParam_alg3_atot_thr_str, 'type': 'float', 'value': self.hitParam_alg3_atot_thr},
                    {'name': hitParam_alg3_son_min_str, 'type': 'float', 'value': self.hitParam_alg3_son_min},
                    {'name': hitParam_alg3_rank_str, 'type': 'int', 'value': self.hitParam_alg3_rank},
                    {'name': hitParam_alg3_r0_str, 'type': 'float', 'value': self.hitParam_alg3_r0},
                    {'name': hitParam_alg3_dr_str, 'type': 'float', 'value': self.hitParam_alg3_dr},
                ]},
                {'name': hitParam_outDir_str, 'type': 'str', 'value': self.hitParam_outDir},
                {'name': hitParam_runs_str, 'type': 'str', 'value': self.hitParam_runs},
                {'name': hitParam_queue_str, 'type': 'list', 'values': {hitParam_psfehhiprioq_str: 'psfehhiprioq',
                                                                        hitParam_psnehhiprioq_str: 'psnehhiprioq',
                                                                        hitParam_psfehprioq_str: 'psfehprioq',
                                                                        hitParam_psnehprioq_str: 'psnehprioq',
                                                                        hitParam_psfehq_str: 'psfehq',
                                                                        hitParam_psnehq_str: 'psnehq',
                                                                        hitParam_psanaq_str: 'psanaq'},
                 'value': self.hitParam_queue, 'tip': "Choose queue"},
                {'name': hitParam_cpu_str, 'type': 'int', 'value': self.hitParam_cpus},
                {'name': hitParam_noe_str, 'type': 'int', 'value': self.hitParam_noe, 'tip': "number of events to process, default=0 means process all events"},
            ]},
        ]
        self.paramsQuantifier = [
            {'name': quantifier_grp, 'type': 'group', 'children': [
                {'name': quantifier_filename_str, 'type': 'str', 'value': self.quantifier_filename, 'tip': "Full path Hdf5 filename"},
                {'name': quantifier_dataset_str, 'type': 'str', 'value': self.quantifier_dataset, 'tip': "Hdf5 dataset metric"},
                {'name': quantifier_sort_str, 'type': 'bool', 'value': self.quantifier_sort, 'tip': "Ascending sort metric"},
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
        self.p2 = Parameter.create(name='paramsQuantifier', type='group', \
                                  children=self.paramsQuantifier, expanded=True)
        self.p3 = Parameter.create(name='paramsPeakFinder', type='group', \
                                  children=self.paramsPeakFinder, expanded=True)
        self.p.sigTreeStateChanged.connect(self.change)
        self.p1.sigTreeStateChanged.connect(self.changeGeomParam)
        self.p2.sigTreeStateChanged.connect(self.changeMetric)
        self.p3.sigTreeStateChanged.connect(self.changePeakFinder)

        ## Create docks, place them into the window one at a time.
        ## Note that size arguments are only a suggestion; docks will still have to
        ## fill the entire dock area and obey the limits of their internal widgets.
        self.d1 = Dock("Image Panel", size=(1100, 1100))     ## give this dock the minimum possible size
        self.d2 = Dock("Experiment Parameters", size=(300,150))
        self.d3 = Dock("Diffraction Geometry", size=(150,150))
        self.d4 = Dock("ROI Histogram", size=(200,150))
        self.d5 = Dock("Mouse", size=(100,50), closable=False)
        self.d6 = Dock("Image Control", size=(100, 150))
        self.d7 = Dock("Image Scroll", size=(500,150))
        self.d8 = Dock("Quantifier", size=(300,150))
        self.d9 = Dock("Peak Finder", size=(300,150))

        # Set the color scheme
        def updateStylePatched(self):
            r = '3px'
            if self.dim:
                fg = cardinalRed_hex
                bg = sandstone100_rgb
                border = "white"
                pass
            else:
                fg = cardinalRed_hex
                bg = sandstone100_rgb
                border = "white" #sandstone100_rgb

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

        # Dock positions on the main frame
        self.area.addDock(self.d1, 'left')      ## place d1 at left edge of dock area
        self.area.addDock(self.d6, 'bottom', self.d1)      ## place d1 at left edge of dock area
        self.area.addDock(self.d2, 'right')     ## place d2 at right edge of dock area
        self.area.addDock(self.d3, 'bottom', self.d2)## place d3 at bottom edge of d1
        self.area.addDock(self.d4, 'right')     ## place d4 at right edge of dock area
        self.area.addDock(self.d5, 'top', self.d1)  ## place d5 at left edge of d1
        self.area.addDock(self.d7, 'bottom', self.d4) ## place d7 below d4
        self.area.addDock(self.d8, 'bottom', self.d3)
        self.area.addDock(self.d9, 'bottom', self.d4)

        ## Dock 1: Image Panel
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
        # Custom ROI for selecting an image region
        self.roi = pg.ROI(pos=[900, 900], size=[50, 50], snapSize=1.0, scaleSnap=True, translateSnap=True, pen={'color': 'g', 'width': 6})
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.w1.getView().addItem(self.roi)
        # Beam mask
        #self.beamROI = pg.ROI([800,800], size=[150, 50], pen={'color': 'r', 'width': 6})
        #self.beamROI.addRotateHandle([1, 1], [0.5, 0.5])
        #self.beamROI.addScaleHandle([0.5, 1], [0.5, 0.5])
        #self.beamROI.addScaleHandle([0, 0.5], [0.5, 0.5])
        #self.w1.getView().addItem(self.beamROI)

        # Callbacks for handling user interaction
        def updateRoiHistogram():
            if self.data is not None:
                selected, coord = self.roi.getArrayRegion(self.data, self.w1.getImageItem(), returnMappedCoords=True)
                hist,bin = np.histogram(selected.flatten(), bins=1000)
                self.w4.plot(bin, hist, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True)
        self.roi.sigRegionChanged.connect(updateRoiHistogram)

        #def updateBeamMask():
        #    if self.data is not None:
        #        beamSelected, beamCoord = self.beamROI.getArrayRegion(self.data, self.w1.getImageItem(), returnMappedCoords=True)
        #        print "beamSelected: ", beamSelected, beamSelected.shape
        #        print "beamCoord: ", beamCoord, beamCoord.shape
        #        row=beamCoord[0,:,:].flatten()
        #        col=beamCoord[1,:,:].flatten()
        #        print "rows: ", row
        #        print "cols: ", col
        #        for i,val in enumerate(row):
        #            self.data[row[i],col[i]] = 0
        #        self.updateImage(calib=self.calib)
        #self.beamROI.sigRegionChanged.connect(updateBeamMask)

        # Connect listeners to functions
        self.d1.addWidget(self.w1)

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

        ## Dock 4: ROI histogram
        self.w4 = pg.PlotWidget(title="ROI histogram")
        hist,bin = np.histogram(np.random.random(1000), bins=1000)
        self.w4.plot(bin, hist, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True)
        self.d4.addWidget(self.w4)

        ## Dock 5 - mouse intensity display
        #self.d5.hideTitleBar()
        self.w5 = pg.GraphicsView(background=pg.mkColor(sandstone100_rgb))
        self.d5.addWidget(self.w5)

        ## Dock 6: Image Control
        self.nextBtn = QtGui.QPushButton('Next evt')
        self.prevBtn = QtGui.QPushButton('Prev evt')
        self.saveBtn = QtGui.QPushButton('Save evt')
        self.loadBtn = QtGui.QPushButton('Load image')

        def next():
            self.eventNumber += 1
            if self.eventNumber >= self.eventTotal:
                self.eventNumber = self.eventTotal-1
            else:
                self.calib, self.data = self.getDetImage(self.eventNumber)
                self.w1.setImage(self.data,autoRange=False,autoLevels=False,autoHistogramRange=False)
                self.p.param(exp_grp,exp_evt_str).setValue(self.eventNumber)
        def prev():
            self.eventNumber -= 1
            if self.eventNumber < 0:
                self.eventNumber = 0
            else:
                self.calib, self.data = self.getDetImage(self.eventNumber)
                self.w1.setImage(self.data,autoRange=False,autoLevels=False,autoHistogramRange=False)
                self.p.param(exp_grp,exp_evt_str).setValue(self.eventNumber)
        def save():
            outputName = "psocake_"+str(self.experimentName)+"_"+str(self.runNumber)+"_"+str(self.detInfo)+"_" \
                         +str(self.eventNumber)+"_"+str(self.eventSeconds)+"_"+str(self.eventNanoseconds)+"_" \
                         +str(self.eventFiducial)+".npy"
            fname = QtGui.QFileDialog.getSaveFileName(self, 'Save file', outputName, 'ndarray image (*.npy)')
            if self.logscaleOn:
                np.save(str(fname),np.log10(abs(self.calib)+eps))
            else:
                np.save(str(fname),self.calib)
        def load():
            fname = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file', './', 'ndarray image (*.npy *.npz)'))
            print "fname: ", fname, fname.split('.')[-1]
            if fname.split('.')[-1] in '.npz':
                print "got npz"
                temp = np.load(fname)
                self.calib = temp['max']
            else:
                print "got npy"
                self.calib = np.load(fname)
            self.updateImage(self.calib)

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

        #self.label = QtGui.QLabel("Event Number:")
        self.spinBox = QtGui.QSpinBox()
        self.spinBox.setValue(0)
        self.label = QtGui.QLabel("Event Number:")
        self.stackSizeBox = QtGui.QSpinBox()
        self.stackSizeBox.setMaximum(self.stackSize)
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
        self.w8.setParameters(self.p2, showTop=False)
        self.d8.addWidget(self.w8)
        # Add plot
        self.w9 = pg.PlotWidget(title="Metric")
        self.d8.addWidget(self.w9)

        ## Dock 9: Peak finder
        self.w10 = ParameterTree()
        self.w10.setParameters(self.p3, showTop=False)
        self.d9.addWidget(self.w10)
        self.w11 = pg.LayoutWidget()
        self.generatePowderBtn = QtGui.QPushButton('Generate Powder')
        self.launchBtn = QtGui.QPushButton('Launch peak finder')
        self.w11.addWidget(self.launchBtn, row=1, col=0)
        self.w11.addWidget(self.generatePowderBtn, row=0, col=0)
        self.d9.addWidget(self.w11)

        ###############
        ### Threads ###
        ###############
        # Making powder patterns
        self.thread = []
        self.threadCounter = 0
        def makePowder():
            print "makePowder!!!!!!"
            self.thread.append(PowderProducer(self)) # send parent parameters with self
            self.thread[self.threadCounter].computePowder(self.experimentName,self.runNumber,self.detInfo)
            self.threadCounter+=1
            #self.generatePowderBtn.setEnabled(False)
            print "done makePowder!!!!!!"
        self.connect(self.generatePowderBtn, QtCore.SIGNAL("clicked()"), makePowder)
        # Launch hit finding
        def findPeaks():
            print "find hits!!!!!!"
            self.thread.append(PeakFinder(self)) # send parent parameters with self
            self.thread[self.threadCounter].findPeaks(self.experimentName,self.runNumber,self.detInfo)
            self.threadCounter+=1
            print "done finding hits!!!!!!"
        self.connect(self.launchBtn, QtCore.SIGNAL("clicked()"), findPeaks)
        # Loading image stack
        def displayImageStack():
            print "display image stack!!!!!!"
            if self.logscaleOn:
                print "applying logscale..."
                self.w7.setImage(np.log10(abs(self.threadpool.data)+eps), xvals=np.linspace(self.stackStart,
                                                                     self.stackStart+self.threadpool.data.shape[0]-1,
                                                                     self.threadpool.data.shape[0]))
                print "done applying logscale"
            else:
                self.w7.setImage(self.threadpool.data, xvals=np.linspace(self.stackStart,
                                                                     self.stackStart+self.threadpool.data.shape[0]-1,
                                                                     self.threadpool.data.shape[0]))
            self.startBtn.setEnabled(True)
            print "Done display image stack!!!!!"
        def loadStack():
            print "loading stack!!!!!!"
            self.stackStart = self.spinBox.value()
            self.stackSize = self.stackSizeBox.value()
            self.threadpool.load(self.stackStart,self.stackSize)
            self.startBtn.setEnabled(False)
            self.w7.getView().setTitle("exp="+self.experimentName+":run="+str(self.runNumber)+":evt"+str(self.stackStart)+"-"
                                       +str(self.stackStart+self.stackSize))
            print "done loading stack!!!!!!"
        self.threadpool = stackProducer(self) # send parent parameters
        self.connect(self.threadpool, QtCore.SIGNAL("finished()"), displayImageStack)
        self.connect(self.startBtn, QtCore.SIGNAL("clicked()"), loadStack)

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
        (cenX,cenY) = (0,0) # no offset
        # Draw xy arrows
        symbolSize = 40
        cutoff=symbolSize/2
        headLen=30
        tailLen=30-cutoff
        xArrow = pg.ArrowItem(angle=180, tipAngle=30, baseAngle=20, headLen=headLen, tailLen=tailLen, tailWidth=8, pen=None, brush='b', pxMode=False)
        xArrow.setPos(2*headLen+cenX,0+cenY)
        self.w1.getView().addItem(xArrow)
        yArrow = pg.ArrowItem(angle=-90, tipAngle=30, baseAngle=20, headLen=headLen, tailLen=tailLen, tailWidth=8, pen=None, brush='r', pxMode=False)
        yArrow.setPos(0+cenX,2*headLen+cenY)
        self.w1.getView().addItem(yArrow)

        # z-direction
        self.z_direction.setData([0+cenX], [0+cenY], symbol='o', \
                                 size=symbolSize, brush='w', \
                                 pen={'color': 'k', 'width': 4}, pxMode=False)
        self.z_direction1.setData([0+cenX], [0+cenY], symbol='o', \
                                 size=symbolSize/6, brush='k', \
                                 pen={'color': 'k', 'width': 4}, pxMode=False)
        # Add xyz text
        self.x_text = pg.TextItem(html='<div style="text-align: center"><span style="color: #0000FF; font-size: 16pt;">x</span></div>', anchor=(0,0))
        self.w1.getView().addItem(self.x_text)
        self.x_text.setPos(2*headLen+cenX, 0+cenY)
        self.y_text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FF0000; font-size: 16pt;">y</span></div>', anchor=(1,1))
        self.w1.getView().addItem(self.y_text)
        self.y_text.setPos(0+cenX, 2*headLen+cenY)
        self.z_text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFFFFF; font-size: 16pt;">z</span></div>', anchor=(1,0))
        self.w1.getView().addItem(self.z_text)
        self.z_text.setPos(-headLen+cenX, 0+cenY)

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
            self.windows = None
            self.alg = []
            self.alg = PyAlgos(windows=self.windows, mask=self.mask, pbits=0)

            # set peak-selector parameters:
            #alg.set_peak_selection_pars(npix_min=2, npix_max=50, amax_thr=10, atot_thr=20, son_min=5)
            if self.algorithm == 1:
                self.alg.set_peak_selection_pars(npix_min=self.hitParam_alg_npix_min, npix_max=self.hitParam_alg_npix_max, \
                                        amax_thr=self.hitParam_alg_amax_thr, atot_thr=self.hitParam_alg_atot_thr, \
                                        son_min=self.hitParam_alg_son_min)
            elif self.algorithm == 3:
                self.alg.set_peak_selection_pars(npix_min=self.hitParam_alg3_npix_min, npix_max=self.hitParam_alg3_npix_max, \
                                        amax_thr=self.hitParam_alg3_amax_thr, atot_thr=self.hitParam_alg3_atot_thr, \
                                        son_min=self.hitParam_alg3_son_min)

            self.algInitDone = True

        if self.algorithm == 1:
            # v1 - aka Droplet Finder - two-threshold peak-finding algorithm in restricted region
            #                           around pixel with maximal intensity.
            #peaks = alg.peak_finder_v1(nda, thr_low=5, thr_high=30, radius=5, dr=0.05)
            self.peakRadius = int(self.hitParam_alg1_radius)
            self.peaks = self.alg.peak_finder_v1(self.calib, thr_low=self.hitParam_alg1_thr_low, thr_high=self.hitParam_alg1_thr_high, \
                                       radius=self.peakRadius, dr=self.hitParam_alg1_dr)
        #elif self.algorithm == 2:
        #    # v2 - define peaks for regions of connected pixels above threshold
        #    self.peaks = self.alg.peak_finder_v2(self.calib, thr=self.hitParam_alg2_thr, r0=self.hitParam_alg2_r0, dr=self.hitParam_alg2_dr)
        elif self.algorithm == 3:
            print "#$@#$ got here"
            self.peakRadius = int(self.hitParam_alg3_r0)
            print "peakRadius: ", self.peakRadius, self.hitParam_alg3_r0
            self.peaks = self.alg.peak_finder_v3(self.calib, rank=self.hitParam_alg3_rank, r0=self.peakRadius, dr=self.hitParam_alg3_dr)

        self.numPeaksFound = self.peaks.shape[0]

        fmt = '%3d %4d %4d  %4d %8.1f %6.1f %6.1f %6.2f %6.2f  %6.2f %4d %4d %4d %4d  %6.2f  %6.2f  %6.2f'
        for peak in self.peaks :
                seg,row,col,npix,amax,atot,rcent,ccent,rsigma,csigma,rmin,rmax,cmin,cmax,bkgd,rms,son = peak[0:17]

                print fmt % (seg, row, col, npix, amax, atot, rcent, ccent, rsigma, csigma,\
                             rmin, rmax, cmin, cmax, bkgd, rms, son)
                if self.isCspad:
                    cheetahRow,cheetahCol = self.convert_peaks_to_cheetah(seg,row,col)
                    print "cheetahRow,Col", cheetahRow, cheetahCol, atot
                    print "^^^^"

        print "num peaks found: ", self.numPeaksFound, self.peaks.shape
        self.drawPeaks()

    def convert_peaks_to_cheetah(self, s, r, c) :
        """Converts seg, row, col assuming (32,185,388)
           to cheetah 2-d table row and col (8*185, 4*388)
        """
        segs, rows, cols = (32,185,388)
        row2d = (int(s)%8) * rows + int(r) # where s%8 is a segment in quad number [0,7]
        col2d = (int(s)/8) * cols + int(c) # where s/8 is a quad number [0,3]
        return row2d, col2d

    def drawPeaks(self):
        if self.peaks is not None and self.numPeaksFound > 0:
            iX  = np.array(self.det.indexes_x(self.evt), dtype=np.int64)
            iY  = np.array(self.det.indexes_y(self.evt), dtype=np.int64)
            cenX = iX[np.array(self.peaks[:,0],dtype=np.int64),np.array(self.peaks[:,1],dtype=np.int64),np.array(self.peaks[:,2],dtype=np.int64)] + 0.5
            cenY = iY[np.array(self.peaks[:,0],dtype=np.int64),np.array(self.peaks[:,1],dtype=np.int64),np.array(self.peaks[:,2],dtype=np.int64)] + 0.5
            diameter = self.peakRadius*2+1
            print "cenX: ", cenX
            print "cenY: ", cenY
            print "diameter: ", diameter, self.peakRadius
            self.peak_feature.setData(cenX, cenY, symbol='s', \
                                      size=diameter, brush=(255,255,255,0), \
                                      pen=pg.mkPen({'color': "FF0", 'width': 4}), pxMode=False)
        else:
            self.peak_feature.setData([], [], pxMode=False)
        print "Done updatePeaks"

    def updateImage(self,calib=None):
        if self.hasExperimentName and self.hasRunNumber and self.hasDetInfo:
            if calib is None:
                self.calib, self.data = self.getDetImage(self.eventNumber)
            else:
                self.calib, self.data = self.getDetImage(self.eventNumber,calib=calib)

            if self.firstUpdate:
                if self.logscaleOn:
                    print "################################# 11"
                    self.w1.setImage(np.log10(abs(self.data)+eps))
                    self.firstUpdate = False
                else:
                    print "################################# 22"
                    self.w1.setImage(self.data)
                    self.firstUpdate = False
            else:
                if self.logscaleOn:
                    print "################################# 1"
                    self.w1.setImage(np.log10(abs(self.data)+eps),autoRange=False,autoLevels=False,autoHistogramRange=False)
                else:
                    print "################################# 2"
                    self.w1.setImage(self.data,autoRange=False,autoLevels=False,autoHistogramRange=False)
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

    def getCalib(self,evtNumber):
        if self.run is not None:
            self.evt = self.getEvt(evtNumber)
            if self.applyCommonMode: # play with different common mode
                if self.commonMode[0] == 5: # Algorithm 5
                    calib = self.det.calib(self.evt, cmpars=(self.commonMode[0],self.commonMode[1]))
                else: # Algorithms 1 to 4
                    print "### Overriding common mode: ", self.commonMode
                    calib = self.det.calib(self.evt, cmpars=(self.commonMode[0],self.commonMode[1],self.commonMode[2],self.commonMode[3]))
            else:
                calib = self.det.calib(self.evt)
            return calib
        else:
            return None

    def getAssembledImage(self,calib):
        _calib = calib.copy() # this is important
        # Apply gain if available
        if self.det.gain(self.evt) is not None:
            _calib *= self.det.gain(self.evt)
        # Do not display ADUs below threshold
        _calib[np.where(_calib<self.aduThresh)]=0
        tic = time.time()
        data = self.det.image(self.evt, _calib)
        toc = time.time()
        print "time assemble: ", toc-tic
        return data

    def getDetImage(self,evtNumber,calib=None):
        if self.image_property == 1 and calib is None:
            calib = self.getCalib(evtNumber)
        elif self.image_property == 2: # coords_x
            calib = self.det.coords_x(self.evt)
        elif self.image_property == 3: # coords_y
            calib = self.det.coords_y(self.evt)
        elif self.image_property == 4: # gain
            calib = self.det.gain(self.evt)
        elif self.image_property == 5 and self.isCspad: # quad ind
            calib = np.zeros((32,185,388))
            for i in range(32):
                calib[i,:,:] = int(i)%8
        elif self.image_property == 6 and self.isCspad: # seg ind
            calib = np.zeros((32,185,388))
            for i in range(32):
                calib[i,:,:] = int(i)/8
        elif self.image_property == 7 and self.isCspad: # row ind
            calib = np.zeros((32,185,388))
            for i in range(185):
                    calib[:,i,:] = i
        elif self.image_property == 8 and self.isCspad: # col ind
            calib = np.zeros((32,185,388))
            for i in range(388):
                    calib[:,:,i] = i

        if calib is not None:
            # apply mask
            if self.mask is not None:
                calib *= self.mask
            # assemble image
            data = self.getAssembledImage(calib)
            self.cx, self.cy = self.getCentre(data.shape)
            return calib, data
        else: # TODO: this is a hack that assumes opal is the only detector without calib
            # we have an opal
            data = self.det.raw(self.evt)
            # Do not display ADUs below threshold
            data[np.where(data<self.aduThresh)]=0
            self.cx, self.cy = self.getCentre(data.shape)
            return data, data

    def getCentre(self,shape):
        cx = shape[1]/2
        cy = shape[0]/2
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
            self.update(path,change,data)

    def changeMetric(self, param, changes):
        for param, change, data in changes:
            path = self.p2.childPath(param)
            print('  path: %s'% path)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')
            self.update(path,change,data)

    def changePeakFinder(self, param, changes):
        for param, change, data in changes:
            path = self.p3.childPath(param)
            print('  path: %s'% path)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')
            self.update(path,change,data)

    def update(self, path, change, data):
        print "path: ", path
        ################################################
        # experiment parameters
        ################################################
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
        ################################################
        # display parameters
        ################################################
        if path[0] == disp_grp:
            if path[1] == disp_log_str:
                self.updateLogscale(data)
            elif path[1] == disp_image_str:
                self.updateImageProperty(data)
            elif path[1] == disp_aduThresh_str:
                self.updateAduThreshold(data)
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
        ################################################
        # peak finder parameters
        ################################################
        if path[0] == hitParam_grp:
            if path[1] == hitParam_algorithm_str:
                self.updateAlgorithm(data)
            elif path[1] == hitParam_classify_str:
                self.updateClassify(data)

            elif path[1] == hitParam_outDir_str:
                self.hitParam_outDir = data
            elif path[1] == hitParam_runs_str:
                self.hitParam_runs = data
            elif path[1] == hitParam_queue_str:
                self.hitParam_queue = data
            elif path[1] == hitParam_cpu_str:
                self.hitParam_cpus = data
            elif path[1] == hitParam_noe_str:
                self.hitParam_noe = data

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
            elif path[2] == hitParam_alg3_npix_min_str:
                self.hitParam_alg3_npix_min = data
                self.algInitDone = False
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_npix_max_str:
                self.hitParam_alg3_npix_max = data
                self.algInitDone = False
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_amax_thr_str:
                self.hitParam_alg3_amax_thr = data
                self.algInitDone = False
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_atot_thr_str:
                self.hitParam_alg3_atot_thr = data
                self.algInitDone = False
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_son_min_str:
                self.hitParam_alg3_son_min = data
                self.algInitDone = False
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_rank_str:
                self.hitParam_alg3_rank = data
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_r0_str:
                self.hitParam_alg3_r0 = data
                if self.classify:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_dr_str:
                self.hitParam_alg3_dr = data
                if self.classify:
                    self.updateClassification()
        ################################################
        # diffraction geometry parameters
        ################################################
        if path[0] == geom_grp:
            if path[1] == geom_detectorDistance_str:
                self.updateDetectorDistance(data)
            elif path[1] == geom_photonEnergy_str:
                self.updatePhotonEnergy(data)
            elif path[1] == geom_pixelSize_str:
                self.updatePixelSize(data)
            elif path[1] == geom_wavelength_str:
                pass
            elif path[1] == geom_resolutionRings_str and len(path) == 2:
                self.updateResolutionRings(data)
            elif path[2] == geom_resolution_str:
                self.updateResolution(data)
        ################################################
        # quantifier parameters
        ################################################
        if path[0] == quantifier_grp:
            if path[1] == quantifier_filename_str:
                self.updateQuantifierFilename(data)
            elif path[1] == quantifier_dataset_str:
                self.updateQuantifierDataset(data)
            elif path[1] == quantifier_sort_str:
                self.updateQuantifierSort(data)


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
        if data == 'DscCsPad' or data == 'DsdCsPad':
            self.isCspad = True
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
            self.ds = psana.DataSource('exp='+str(self.experimentName)+':run='+str(self.runNumber)+':idx') # FIXME: psana crashes if runNumber is non-existent
            self.run = self.ds.runs().next()
            self.times = self.run.times()
            self.eventTotal = len(self.times)
            self.spinBox.setMaximum(self.eventTotal-self.stackSize)
            self.p.param(exp_grp,exp_evt_str).setLimits((0,self.eventTotal-1))
            self.p.param(exp_grp,exp_evt_str,exp_numEvents_str).setValue(self.eventTotal)
            self.env = self.ds.env()

            evt = self.run.event(self.times[0])
            myAreaDetectors = []
            for k in evt.keys():
                try:
                    if Detector.PyDetector.dettype(k.alias(), self.env) == Detector.AreaDetector.AreaDetector:
                        myAreaDetectors.append(k.alias())
                except ValueError:
                    continue
            self.detInfoList = list(set(myAreaDetectors))
            print "# Available detectors: ", self.detInfoList

        if self.hasExpRunDetInfo():
            self.det = psana.Detector(str(self.detInfo), self.env)

            # Get epics variable, clen
            if "cxi" in self.experimentName:
                self.epics = self.ds.env().epicsStore()
                self.clen = self.epics.value('CXI:DS1:MMS:06.RBV')
                print "clen: ", self.clen
                self.mask = self.det.mask(evt, calib=True, status=True, edges=True, central=True, unbond=True, unbondnbrs=True)
            print "Done setupExperiment"

    def updateLogscale(self, data):
        self.logscaleOn = data
        if self.hasExpRunDetInfo():
            self.firstUpdate = True # clicking logscale resets plot colorscale
            self.updateImage()
        print "Done updateLogscale: ", self.logscaleOn

    def updateImageProperty(self, data):
        self.image_property = data
        self.updateImage()
        print "##### Done updateImageProperty: ", self.image_property

    def updateAduThreshold(self, data):
        self.aduThresh = data
        if self.hasExpRunDetInfo():
            self.updateImage(self.calib)
        print "Done updateAduThreshold: ", self.aduThresh

    def updateResolutionRings(self, data):
        self.resolutionRingsOn = data
        if self.hasExpRunDetInfo():
            self.updateRings()
        print "Done updateResolutionRings: ", self.resolutionRingsOn

    def updateResolution(self, data):
        # convert to array of floats
        _resolution = data.split(',')
        self.resolution = np.zeros((len(_resolution,)))

        a = ['a','b','c','d','e','k','m','n','r','s']
        myStr = a[5]+a[8]+a[0]+a[5]+a[4]+a[7]
        if myStr in data:
            self.d42 = Dock("Console", size=(100,100))
            # build an initial namespace for console commands to be executed in (this is optional;
            # the user can always import these modules manually)
            namespace = {'pg': pg, 'np': np, 'self': self}
            # initial text to display in the console
            text = "You have awoken the "+myStr+"\nWelcome to psocake IPython: dir(self)"
            self.w42 = pg.console.ConsoleWidget(parent=None,namespace=namespace, text=text)
            self.d42.addWidget(self.w42)
            self.area.addDock(self.d42, 'bottom')
            data = ''

        if data != '':
            for i in range(len(_resolution)):
                self.resolution[i] = float(_resolution[i])

        if data != '':
            self.hasUserDefinedResolution = True
        else:
            self.hasUserDefinedResolution = False

        self.myResolutionRingList = self.resolution
        self.dMin = np.zeros_like(self.myResolutionRingList)
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
        # TODO: cspad2x2 can only use algorithms 1 and 5
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
            self.dMin[i] = 1/qMax
            print "updateGeometry: ", i, thetaMax, qMax, self.dMin[i]
            if self.resolutionRingsOn:
                self.updateRings()

    def updateQuantifierFilename(self, data):
        # close previously open file
        if self.quantifier_filename is not data and self.quantifierFileOpen:
            self.quantifierFile.close()
        self.quantifier_filename = data
        self.quantifierFile = h5py.File(self.quantifier_filename,'r')
        self.quantifierFileOpen = True
        print "Done opening metric"

    def updateQuantifierDataset(self, data):
        self.quantifier_dataset = data
        if self.quantifierFileOpen:
            self.quantifierMetric = self.quantifierFile[self.quantifier_dataset].value
            self.quantifierInd = np.arange(len(self.quantifierMetric))
            self.quantifierHasData = True
            self.updateQuantifierPlot(self.quantifierInd,self.quantifierMetric)
            print "Done reading metric"

    def updateQuantifierSort(self, data):
        self.quantifier_sort = data
        if self.quantifierHasData:
            if data is True:
                self.quantifierInd = np.argsort(self.quantifierFile[self.quantifier_dataset].value)
                self.quantifierMetric = self.quantifierFile[self.quantifier_dataset].value[self.quantifierInd]
                self.updateQuantifierPlot(self.quantifierInd,self.quantifierMetric)
            else:
                self.quantifierMetric = self.quantifierFile[self.quantifier_dataset].value
                self.quantifierInd = np.arange(len(self.quantifierMetric))
                self.updateQuantifierPlot(self.quantifierInd,self.quantifierMetric)
        print "metric: ", self.quantifierMetric
        print "ind: ", self.quantifierInd

    def updateQuantifierPlot(self,ind,metric):
        #self.curve = None
        #self.w9.plotItem.clear()
        self.w9.getPlotItem().clear()
        self.curve = self.w9.plot(metric, pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')
        #self.curve = self.w9.plot(ind,metric, pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')
        self.curve.curve.setClickable(True)
        self.curve.sigClicked.connect(self.clicked)

    def clicked(self,points):
        print("curve clicked",points)
        from pprint import pprint
        pprint(vars(points.scatter))
        for i in range(len(points.scatter.data)):
            if points.scatter.ptsClicked[0] == points.scatter.data[i][7]:
                ind = i
                break
        indX = points.scatter.data[i][0]
        indY = points.scatter.data[i][1]
        print "indX: ", ind, indX, indY
        if self.quantifier_sort:
            ind = self.quantifierInd[ind]
        self.eventNumber = ind
        self.calib, self.data = self.getDetImage(self.eventNumber)
        self.w1.setImage(self.data,autoRange=False,autoLevels=False,autoHistogramRange=False)
        self.p.param(exp_grp,exp_evt_str).setValue(self.eventNumber)

class PowderProducer(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        print "WORKER!!!!!!!!!!"
        self.parent = parent
        self.experimentName = None
        self.runNumber = None
        self.detInfo = None

    def __del__(self):
        print "del PowderProducer #$!@#$!#"
        self.exiting = True
        self.wait()

    def computePowder(self,experimentName,runNumber,detInfo):
        self.experimentName = experimentName
        self.runNumber = runNumber
        self.detInfo = detInfo
        self.start()

    def digestRunList(self,runList):
        runsToDo = []
        if not runList:
            print "Run(s) is empty. Please type in the run number(s)."
            return runsToDo
        runLists = runList.split(",")
        for list in runLists:
            temp = list.split(":")
            if len(temp) == 2:
                for i in np.arange(int(temp[0]),int(temp[1])+1):
                    runsToDo.append(i)
            elif len(temp) == 1:
                runsToDo.append(int(temp[0]))
        return runsToDo

    def run(self):
        print "Generating powder!!!!!!!!!!!!"
        runsToDo = self.digestRunList(self.parent.hitParam_runs)
        print runsToDo
        for run in runsToDo:
            # Command for submitting to batch
            cmd = "bsub -q "+self.parent.hitParam_queue+" -a mympi -n "+str(self.parent.hitParam_cpus)+\
                  " -o %J.log python generatePowder.py exp="+self.experimentName+\
                  ":run="+str(run)+" -d "+self.detInfo+\
                  " -o "+str(self.parent.hitParam_outDir)
            if self.parent.hitParam_noe > 0:
                cmd += " -n "+str(self.parent.hitParam_noe)
            print "Submitting batch job: ", cmd
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            jobid = out.split("<")[1].split(">")[0]
            myLog = jobid+".log"
            print "bsub log filename: ", myLog
            # myKeyString = "The output (if any) is above this job summary."
            # mySuccessString = "Successfully completed."
            # notDone = 1
            # havePowder = 0
            # while notDone:
            #     if os.path.isfile(myLog):
            #         p = subprocess.Popen(["grep", myKeyString, myLog],stdout=subprocess.PIPE)
            #         output = p.communicate()[0]
            #         p.stdout.close()
            #         if myKeyString in output: # job has finished
            #             # check job was a success or a failure
            #             p = subprocess.Popen(["grep", mySuccessString, myLog], stdout=subprocess.PIPE)
            #             output = p.communicate()[0]
            #             p.stdout.close()
            #             if mySuccessString in output: # success
            #                 print "successfully done"
            #                 havePowder = 1
            #             else:
            #                 print "failed attempt"
            #             notDone = 0
            #         else:
            #             print "job hasn't finished yet"
            #             time.sleep(10)
            #     else:
            #         print "no such file yet"
            #         time.sleep(10)


class stackProducer(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        print "stack producer !!!!!!!!!!"
        self.exiting = False
        self.parent = parent
        self.startIndex = 0
        self.numImages = 0
        self.evt = None
        self.data = None

    def __del__(self):
        print "del stackProducer #$!@#$!#"
        self.exiting = True
        self.wait()

    def load(self, startIndex, numImages):
        self.startIndex = startIndex
        self.numImages = numImages
        self.start()

    def run(self):
        print "Doing WORK!!!!!!!!!!!!: ", self.startIndex,self.startIndex+self.numImages
        counter = 0
        for i in np.arange(self.startIndex,self.startIndex+self.numImages):
            if counter == 0:
                calib,data = self.parent.getDetImage(i,calib=None)
                self.data = np.zeros((self.numImages,data.shape[0],data.shape[1]))
                if data is not None:
                    self.data[counter,:,:] = data
                counter += 1
            else:
                calib,data = self.parent.getDetImage(i,calib=None)
                if data is not None:
                    self.data[counter,:,:] = data
                counter += 1
        #self.emit(QtCore.SIGNAL("done"))
        #time.sleep(1)

class PeakFinder(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        print "PeakFinder!!!!!!!!!!"
        self.parent = parent
        self.experimentName = None
        self.runNumber = None
        self.detInfo = None

    def __del__(self):
        print "del PeakFinder #$!@#$!#"
        self.exiting = True
        self.wait()

    def findPeaks(self,experimentName,runNumber,detInfo): # Pass in peak parameters
        self.experimentName = experimentName
        self.runNumber = runNumber
        self.detInfo = detInfo
        self.start()

    def digestRunList(self,runList):
        runsToDo = []
        if not runList:
            print "Run(s) is empty. Please type in the run number(s)."
            return runsToDo
        runLists = runList.split(",")
        for list in runLists:
            temp = list.split(":")
            if len(temp) == 2:
                for i in np.arange(int(temp[0]),int(temp[1])+1):
                    runsToDo.append(i)
            elif len(temp) == 1:
                runsToDo.append(int(temp[0]))
        return runsToDo

    def run(self):
        print "Finding peaks for all events!!!!!!!!!!!!"
        # Digest the run list
        runsToDo = self.digestRunList(self.parent.hitParam_runs)
        print runsToDo
        for run in runsToDo:
            # Command for submitting to batch
            if self.parent.algorithm == 1:
                cmd = "bsub -q "+self.parent.hitParam_queue+" -a mympi -n "+str(self.parent.hitParam_cpus)+\
                      " -o %J.log python findPeaks.py -e "+self.experimentName+\
                      " -r "+str(run)+" -d "+self.detInfo+\
                      " --outDir "+str(self.parent.hitParam_outDir)+\
                      " --algorithm "+str(self.parent.algorithm)+\
                      " --alg_npix_min "+str(self.parent.hitParam_alg_npix_min)+\
                      " --alg_npix_max "+str(self.parent.hitParam_alg_npix_max)+\
                      " --alg_amax_thr "+str(self.parent.hitParam_alg_amax_thr)+\
                      " --alg_atot_thr "+str(self.parent.hitParam_alg_atot_thr)+\
                      " --alg_son_min "+str(self.parent.hitParam_alg_son_min)+\
                      " --alg1_thr_low "+str(self.parent.hitParam_alg1_thr_low)+\
                      " --alg1_thr_high "+str(self.parent.hitParam_alg1_thr_high)+\
                      " --alg1_radius "+str(self.parent.hitParam_alg1_radius)+\
                      " --alg1_dr "+str(self.parent.hitParam_alg1_dr)
            elif self.parent.algorithm == 3:
                cmd = "bsub -q "+self.parent.hitParam_queue+" -a mympi -n "+str(self.parent.hitParam_cpus)+\
                      " -o %J.log python findPeaks.py -e "+self.experimentName+\
                      " -r "+str(run)+" -d "+self.detInfo+\
                      " --outDir "+str(self.parent.hitParam_outDir)+\
                      " --algorithm "+str(self.parent.algorithm)+\
                      " --alg_npix_min "+str(self.parent.hitParam_alg_npix_min)+\
                      " --alg_npix_max "+str(self.parent.hitParam_alg_npix_max)+\
                      " --alg_amax_thr "+str(self.parent.hitParam_alg_amax_thr)+\
                      " --alg_atot_thr "+str(self.parent.hitParam_alg_atot_thr)+\
                      " --alg_son_min "+str(self.parent.hitParam_alg_son_min)+\
                      " --alg3_rank "+str(self.parent.hitParam_alg3_rank)+\
                      " --alg3_r0 "+str(self.parent.hitParam_alg3_r0)+\
                      " --alg3_dr "+str(self.parent.hitParam_alg3_dr)
            if self.parent.hitParam_noe > 0:
                cmd += " --noe "+str(self.parent.hitParam_noe)
            print "Submitting batch job: ", cmd
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            jobid = out.split("<")[1].split(">")[0]
            myLog = jobid+".log"
            print "*******************"
            print "bsub log filename: ", myLog
        # myKeyString = "The output (if any) is above this job summary."
        # mySuccessString = "Successfully completed."
        # notDone = 1
        # haveFinished = 0
        # while notDone:
        #     if os.path.isfile(myLog):
        #         p = subprocess.Popen(["grep", myKeyString, myLog],stdout=subprocess.PIPE)
        #         output = p.communicate()[0]
        #         p.stdout.close()
        #         if myKeyString in output: # job has finished
        #             # check job was a success or a failure
        #             p = subprocess.Popen(["grep", mySuccessString, myLog], stdout=subprocess.PIPE)
        #             output = p.communicate()[0]
        #             p.stdout.close()
        #             if mySuccessString in output: # success
        #                 print "successfully done"
        #                 haveFinished = 1
        #             else:
        #                 print "failed attempt"
        #             notDone = 0
        #         else:
        #             print "job hasn't finished yet"
        #             time.sleep(10)
        #     else:
        #         print "no such file yet"
        #         time.sleep(10)

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtGui.QApplication(sys.argv)
    ex = MainFrame(sys.argv)
    sys.exit(app.exec_())

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    main()

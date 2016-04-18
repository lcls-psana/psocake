# GUI for browsing LCLS area detectors. Tune hit finding parameters and common mode correction.

# TODO: Zoom in area / view matrix of numbers
# TODO: Multiple subplots
# TODO: grid of images
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
import myskbeam
from PSCalib.GeometryObject import data2x2ToTwo2x1, two2x1ToData2x2

parser = argparse.ArgumentParser()
parser.add_argument("-e","--exp", help="experiment name (e.g. cxis0813), default=''",default="", type=str)
parser.add_argument("-r","--run", help="run number (e.g. 5), default=0",default=0, type=int)
parser.add_argument("-d","--det", help="detector name (e.g. CxiDs1.0:Cspad.0), default=''",default="", type=str)
parser.add_argument("-n","--evt", help="event number (e.g. 1), default=0",default=0, type=int)
parser.add_argument("--localCalib", help="use local calib directory, default=False", action='store_true')
parser.add_argument("--more", help="display more panels", action='store_true')
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
hitParam_showPeaks_str = 'Show peaks found'
hitParam_algorithm_str = 'Algorithm'
hitParam_alg_npix_min_str = 'npix_min'
hitParam_alg_npix_max_str = 'npix_max'
hitParam_alg_amax_thr_str = 'amax_thr'
hitParam_alg_atot_thr_str = 'atot_thr'
hitParam_alg_son_min_str = 'son_min'
# algorithm 0
hitParam_algorithm0_str = 'None'
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
# algorithm 4
hitParam_alg4_npix_min_str = 'npix_min'
hitParam_alg4_npix_max_str = 'npix_max'
hitParam_alg4_amax_thr_str = 'amax_thr'
hitParam_alg4_atot_thr_str = 'atot_thr'
hitParam_alg4_son_min_str = 'son_min'
hitParam_algorithm4_str = 'iDroplet'
hitParam_alg4_thr_low_str = 'thr_low'
hitParam_alg4_thr_high_str = 'thr_high'
hitParam_alg4_rank_str = 'rank'
hitParam_alg4_radius_str = 'radius'
hitParam_alg4_dr_str = 'dr'

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

# PerPixelHistogram parameter tree
perPixelHistogram_grp = 'Per Pixel Histogram'
perPixelHistogram_filename_str = 'filename'
perPixelHistogram_adu_str = 'ADU'

# Manifold parameter tree
manifold_grp = 'Manifold'
manifold_filename_str = 'filename'
manifold_dataset_str = 'eigenvector_dataset'
manifold_sigma_str = 'sigma'

mask_grp = 'Mask'
mask_mode_str = 'Masking mode'
do_nothing_str = 'Off'
do_toggle_str = 'Toggle'
do_mask_str = 'Mask'
do_unmask_str = 'Unmask'
streak_mask_str = 'Use jet streak mask'
streak_width_str = 'maximum streak length'
streak_sigma_str = 'sigma'
psana_mask_str = 'Use psana mask'
user_mask_str = 'Use user-defined mask'
mask_calib_str = 'calib pixels'
mask_status_str = 'status pixels'
mask_edges_str = 'edge pixels'
mask_central_str = 'central pixels'
mask_unbond_str = 'unbonded pixels'
mask_unbondnrs_str = 'unbonded pixel neighbors'

# Color scheme
cardinalRed_hex = str("#8C1515") # Cardinal red
darkRed_hex = str("#820000") # dark red
black_hex = str("#2e2d29") # black
black80_hex = str("#585754") # black 80%
gray_hex = str("#3f3c30") # gray
gray90_hex = str("#565347") # gray 90%
gray60_hex = str("#8a887d") # gray 60%
sandstone100_rgb = (221,207,153) # Sandstone
beige_hex = ("#9d9573") # beige
masking_mode_message = "<span style='color: " + black_hex + "; font-size: 24pt;'>Masking mode <br> </span>"

class MainFrame(QtGui.QWidget):
    """
    The main frame of the application
    """        
    def __init__(self, arg_list):
        super(MainFrame, self).__init__()
        self.firstUpdate = True
        self.operationModeChoices = ['none','masking']
        self.operationMode =  self.operationModeChoices[0] # Masking mode, Peak finding mode
        # Init experiment parameters
        self.experimentName = args.exp
        self.runNumber = int(args.run)
        self.detInfo = args.det
        self.isCspad = False
        self.isCamera = False
        self.evt = None
        self.eventNumber = int(args.evt)
        self.eventSeconds = ""
        self.eventNanoseconds = ""
        self.eventFiducial = ""
        self.eventTotal = 0
        self.hasExperimentName = False
        self.hasRunNumber = False
        self.hasDetInfo = False
        self.pixelIndAssem = None
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
        self.cx = 0
        self.cy = 0
        self.calib = None # ndarray detector image
        self.psanaMask = None # psana mask
        self.psanaMaskAssem = None
        self.userMask = None # user-defined mask
        self.userMaskAssem = None
        self.streakMask = None # jet streak mask
        self.streakMaskAssem = None
        self.combinedMask = None # combined mask
        self.gapAssemInd = None
        # Init hit finding parameters
        self.algInitDone = False
        self.algorithm = 0
        self.classify = False

        self.showPeaks = True
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
        self.hitParam_alg4_npix_min = 1.
        self.hitParam_alg4_npix_max = 45.
        self.hitParam_alg4_amax_thr = 250.
        self.hitParam_alg4_atot_thr = 330.
        self.hitParam_alg4_son_min = 10.
        self.hitParam_alg4_thr_low = 80.
        self.hitParam_alg4_thr_high = 270.
        self.hitParam_alg4_rank = 3
        self.hitParam_alg4_radius = 3
        self.hitParam_alg4_dr = 1

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

        self.perPixelHistogram_filename = ''
        self.perPixelHistogram_adu = 20
        self.perPixelHistogramFileOpen = False

        self.manifold_filename = ''
        self.manifold_dataset = ''
        self.manifold_sigma = 0
        self.manifoldFileOpen = False
        self.manifoldHasData = False

        self.maskingMode = 0
        self.userMaskOn = False
        self.streakMaskOn = False
        self.streak_sigma = 1
        self.streak_width = 250
        self.psanaMaskOn = False
        self.mask_calibOn = False
        self.mask_statusOn = False
        self.mask_edgesOn = False
        self.mask_centralOn = False
        self.mask_unbondOn = False
        self.mask_unbondnrsOn = False
        self.display_data = None
        self.roi_rect = None
        self.roi_circle = None

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
                {'name': hitParam_showPeaks_str, 'type': 'bool', 'value': self.showPeaks, 'tip': "Show peaks found shot-to-shot"},
                {'name': hitParam_algorithm_str, 'type': 'list', 'values': {hitParam_algorithm4_str: 4,
                                                                            hitParam_algorithm3_str: 3,
                                                                            hitParam_algorithm1_str: 1,
                                                                            hitParam_algorithm0_str: 0},
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
                {'name': hitParam_algorithm4_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
                    {'name': hitParam_alg4_npix_min_str, 'type': 'float', 'value': self.hitParam_alg4_npix_min, 'tip': "Only keep the peak if number of pixels above thr_low is above this value"},
                    {'name': hitParam_alg4_npix_max_str, 'type': 'float', 'value': self.hitParam_alg4_npix_max, 'tip': "Only keep the peak if number of pixels above thr_low is below this value"},
                    {'name': hitParam_alg4_amax_thr_str, 'type': 'float', 'value': self.hitParam_alg4_amax_thr, 'tip': "Only keep the peak if max value is above this value"},
                    {'name': hitParam_alg4_atot_thr_str, 'type': 'float', 'value': self.hitParam_alg4_atot_thr, 'tip': "Only keep the peak if integral inside region of interest is above this value"},
                    {'name': hitParam_alg4_son_min_str, 'type': 'float', 'value': self.hitParam_alg4_son_min, 'tip': "Only keep the peak if signal-over-noise is above this value"},
                    {'name': hitParam_alg4_thr_low_str, 'type': 'float', 'value': self.hitParam_alg4_thr_low, 'tip': "Only consider values above this value"},
                    {'name': hitParam_alg4_thr_high_str, 'type': 'float', 'value': self.hitParam_alg4_thr_high, 'tip': "Only keep the peak if max value is above this value"},
                    {'name': hitParam_alg4_rank_str, 'type': 'int', 'value': self.hitParam_alg4_rank, 'tip': "region of integration is a square, (2r+1)x(2r+1)"},
                    {'name': hitParam_alg4_radius_str, 'type': 'int', 'value': self.hitParam_alg4_radius, 'tip': "region of integration is a square, (2r+1)x(2r+1)"},
                    {'name': hitParam_alg4_dr_str, 'type': 'float', 'value': self.hitParam_alg4_dr, 'tip': "background region outside the region of interest"},
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
        self.paramsPerPixelHistogram = [
            {'name': perPixelHistogram_grp, 'type': 'group', 'children': [
                {'name': perPixelHistogram_filename_str, 'type': 'str', 'value': self.perPixelHistogram_filename, 'tip': "Full path Hdf5 filename"},
                {'name': perPixelHistogram_adu_str, 'type': 'float', 'value': self.perPixelHistogram_adu, 'tip': "histogram value at this adu"},
            ]},
        ]
        self.paramsManifold = [
            {'name': manifold_grp, 'type': 'group', 'children': [
                {'name': manifold_filename_str, 'type': 'str', 'value': self.manifold_filename, 'tip': "Full path Hdf5 filename"},
                {'name': manifold_dataset_str, 'type': 'str', 'value': self.manifold_dataset, 'tip': "Hdf5 dataset metric"},
                {'name': manifold_sigma_str, 'type': 'float', 'value': self.manifold_sigma, 'tip': "kernel sigma"},
            ]},
        ]
        self.paramsMask = [
            {'name': mask_grp, 'type': 'group', 'children': [
                {'name': user_mask_str, 'type': 'bool', 'value': self.userMaskOn, 'tip': "Mask areas defined by user", 'children':[
                    {'name': mask_mode_str, 'type': 'list', 'values': {do_toggle_str: 3,
                                                                       do_unmask_str: 2,
                                                                       do_mask_str: 1,
                                                                       do_nothing_str: 0},
                                                                       'value': self.maskingMode,
                                                                       'tip': "Choose masking mode"},
                ]},
                {'name': streak_mask_str, 'type': 'bool', 'value': self.streakMaskOn, 'tip': "Mask jet streaks shot-to-shot", 'children':[
                    {'name': streak_width_str, 'type': 'float', 'value': self.streak_width, 'tip': "set maximum length of streak"},
                    {'name': streak_sigma_str, 'type': 'float', 'value': self.streak_sigma, 'tip': "set number of sigma to threshold"},
                ]},
                {'name': psana_mask_str, 'type': 'bool', 'value': self.psanaMaskOn, 'tip': "Mask edges and unbonded pixels etc", 'children': [
                    {'name': mask_calib_str, 'type': 'bool', 'value': self.mask_calibOn, 'tip': "use custom mask deployed in calibdir"},
                    {'name': mask_status_str, 'type': 'bool', 'value': self.mask_statusOn, 'tip': "mask bad pixel status"},
                    {'name': mask_edges_str, 'type': 'bool', 'value': self.mask_edgesOn, 'tip': "mask edge pixels"},
                    {'name': mask_central_str, 'type': 'bool', 'value': self.mask_centralOn, 'tip': "mask central edge pixels inside asic2x1"},
                    {'name': mask_unbond_str, 'type': 'bool', 'value': self.mask_unbondOn, 'tip': "mask unbonded pixels (cspad only)"},
                    {'name': mask_unbondnrs_str, 'type': 'bool', 'value': self.mask_unbondnrsOn, 'tip': "mask unbonded pixel neighbors (cspad only)"},
                ]},
            ]}
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
        self.p4 = Parameter.create(name='paramsManifold', type='group', \
                                  children=self.paramsManifold, expanded=True)
        self.p5 = Parameter.create(name='paramsPerPixelHistogram', type='group', \
                                  children=self.paramsPerPixelHistogram, expanded=True)
        self.p6 = Parameter.create(name='paramsMask', type='group', \
                                  children=self.paramsMask, expanded=True)
        self.p.sigTreeStateChanged.connect(self.change)
        self.p1.sigTreeStateChanged.connect(self.changeGeomParam)
        self.p2.sigTreeStateChanged.connect(self.changeMetric)
        self.p3.sigTreeStateChanged.connect(self.changePeakFinder)
        self.p4.sigTreeStateChanged.connect(self.changeManifold)
        self.p5.sigTreeStateChanged.connect(self.changePerPixelHistogram)
        self.p6.sigTreeStateChanged.connect(self.changeMask)

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
        self.d10 = Dock("Manifold", size=(300,150))
        self.d11 = Dock("Per Pixel Histogram", size=(300,150))
        self.d12 = Dock("Mask Panel", size=(300, 150))

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
        self.area.addDock(self.d11, 'bottom', self.d4)
        self.area.addDock(self.d12, 'bottom',self.d4)
        if args.more:
            self.area.addDock(self.d10, 'bottom', self.d6)

        ## Dock 1: Image Panel
        self.w1 = pg.ImageView(view=pg.PlotItem())
        self.w1.getView().invertY(False)

        self.img_feature = pg.ImageItem()
        self.w1.getView().addItem(self.img_feature)

        self.ring_feature = pg.ScatterPlotItem()
        self.peak_feature = pg.ScatterPlotItem()
        self.z_direction = pg.ScatterPlotItem()
        self.z_direction1 = pg.ScatterPlotItem()
        self.w1.getView().addItem(self.ring_feature)
        self.w1.getView().addItem(self.peak_feature)
        self.w1.getView().addItem(self.z_direction)
        self.w1.getView().addItem(self.z_direction1)

        # Custom ROI for selecting an image region
        self.roi = pg.ROI(pos=[900, 900], size=[50, 50], snapSize=1.0, scaleSnap=True, translateSnap=True, pen={'color': 'g', 'width': 4})
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.roi.addRotateHandle([0.5, 0.5], [1, 1])
        self.w1.getView().addItem(self.roi)
        # Callbacks for handling user interaction
        def updateRoiHistogram():
            if self.data is not None:
                selected, coord = self.roi.getArrayRegion(self.data, self.w1.getImageItem(), returnMappedCoords=True)
                hist,bin = np.histogram(selected.flatten(), bins=1000)
                self.w4.plot(bin, hist, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True)
        self.roi.sigRegionChangeFinished.connect(updateRoiHistogram)

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
                if self.calib.size==2*185*388: # cspad2x2
                    asData2x2 = two2x1ToData2x2(self.calib)
                    np.save(str(fname),asData2x2)
                    np.savetxt(str(fname).split('.')[0]+".txt", asData2x2.reshape((-1,asData2x2.shape[-1])) ,fmt='%0.18e')
                else:
                    np.save(str(fname),self.calib)
                    np.savetxt(str(fname).split('.')[0]+".txt", self.calib.reshape((-1,self.calib[-1])) ,fmt='%0.18e')
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

        ## Dock 10: Manifold
        self.w12 = ParameterTree()
        self.w12.setParameters(self.p4, showTop=False)
        self.d10.addWidget(self.w12)
        # Add plot
        self.w13 = pg.PlotWidget(title="Manifold!!!!")
        self.d10.addWidget(self.w13)

        ## Dock 11: Per Pixel Histogram
        self.w14 = ParameterTree()
        self.w14.setParameters(self.p5, showTop=False)
        self.d11.addWidget(self.w14)
        # Add buttons
        self.w15 = pg.LayoutWidget()
        self.fitBtn = QtGui.QPushButton('Fit histogram')
        self.w15.addWidget(self.fitBtn, row=0, col=0)
        self.d11.addWidget(self.w15)
        # Add plot
        self.w16 = pg.PlotWidget(title="Per Pixel Histogram")
        self.d11.addWidget(self.w16)

        ## Dock 12: Mask Panel
        self.w17 = ParameterTree()
        self.w17.setParameters(self.p6, showTop=False)
        self.d12.addWidget(self.w17)
        self.w18 = pg.LayoutWidget()
        self.maskRectBtn = QtGui.QPushButton('mask rectangular ROI')
        self.w18.addWidget(self.maskRectBtn, row=0, col=0)
        self.maskCircleBtn = QtGui.QPushButton('mask circular ROI')
        self.w18.addWidget(self.maskCircleBtn, row=1, col=0)
        self.deployMaskBtn = QtGui.QPushButton()
        self.deployMaskBtn.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')
        self.deployMaskBtn.setText('Save user-defined mask')
        self.w18.addWidget(self.deployMaskBtn, row=2, col=0)
        # Connect listeners to functions
        self.d12.addWidget(self.w18)

        # mask
        def makeMaskRect():
            print "makeMaskRect!!!!!!"
            self.initMask()
            if self.data is not None and self.maskingMode > 0:
                print "makeMaskRect_data: ", self.data.shape
                selected, coord = self.roi_rect.getArrayRegion(self.data, self.w1.getImageItem(), returnMappedCoords=True)
                _mask = np.ones_like(self.data)
                _mask[coord[0].ravel().astype('int'),coord[1].ravel().astype('int')] = 0
                if self.maskingMode == 1: # masking mode
                    self.userMaskAssem *= _mask
                elif self.maskingMode == 2: # unmasking mode
                    self.userMaskAssem[coord[0].ravel().astype('int'),coord[1].ravel().astype('int')] = 1
                elif self.maskingMode == 3: # toggle mode
                    self.userMaskAssem[coord[0].ravel().astype('int'),coord[1].ravel().astype('int')] = (1-self.userMaskAssem[coord[0].ravel().astype('int'),coord[1].ravel().astype('int')])

                # update userMask
                self.userMask = self.det.ndarray_from_image(self.evt,self.userMaskAssem, pix_scale_size_um=None, xy0_off_pix=None)

                self.displayMask()
                self.algInitDone = False
            print "done makeMaskRect!!!!!!"
        self.connect(self.maskRectBtn, QtCore.SIGNAL("clicked()"), makeMaskRect)

        def makeMaskCircle():
            print "makeMaskCircle!!!!!!"
            self.initMask()
            if self.data is not None and self.maskingMode > 0:
                (radiusX,radiusY) = self.roi_circle.size()
                (cornerX,cornerY) = self.roi_circle.pos()
                i0, j0 = np.meshgrid(range(int(radiusY)),
                                     range(int(radiusX)), indexing = 'ij')
                r = np.sqrt(np.square((i0 - radiusY/2).astype(np.float)) +
                            np.square((j0 - radiusX/2).astype(np.float)))
                i0 = np.rint(i0[np.where(r < radiusY/2.)] + cornerY).astype(np.int)
                j0 = np.rint(j0[np.where(r < radiusX/2.)] + cornerX).astype(np.int)
                i01 = i0[(i0>=0) & (i0<self.data.shape[1]) & (j0>=0) & (j0<self.data.shape[0])]
                j01 = j0[(i0>=0) & (i0<self.data.shape[1]) & (j0>=0) & (j0<self.data.shape[0])]

                _mask = np.ones_like(self.data)
                _mask[j01,i01] = 0
                if self.maskingMode == 1: # masking mode
                    self.userMaskAssem *= _mask
                elif self.maskingMode == 2: # unmasking mode
                    self.userMaskAssem[j01,i01] = 1
                elif self.maskingMode == 3: # toggle mode
                    self.userMaskAssem[j01,i01] = (1-self.userMaskAssem[j01,i01])

                # update userMask
                self.userMask = self.det.ndarray_from_image(self.evt,self.userMaskAssem, pix_scale_size_um=None, xy0_off_pix=None)

                self.displayMask()
                self.algInitDone = False
            print "done makeMaskCircle!!!!!!"
        self.connect(self.maskCircleBtn, QtCore.SIGNAL("clicked()"), makeMaskCircle)

        def deployMask():
            print "*** deploy user-defined mask as mask.txt and mask.npy ***"
            print "userMask: ", self.userMask.shape
            if self.userMask is not None:
                if self.userMask.size==2*185*388: # cspad2x2
                    asData2x2 = two2x1ToData2x2(self.userMask)
                    np.save("mask.npy",asData2x2)
                    np.savetxt("mask.txt", asData2x2.reshape((-1,asData2x2.shape[-1])) ,fmt='%0.18e')
                else:
                    np.save("mask.npy",self.userMask)
                    np.savetxt("mask.txt", self.userMask.reshape((-1,self.userMask.shape[-1])) ,fmt='%0.18e')
            else:
                print "user mask is not defined"
        self.connect(self.deployMaskBtn, QtCore.SIGNAL("clicked()"), deployMask)

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
                        if self.maskingMode > 0:
                            modeInfo = masking_mode_message
                        else:
                            modeInfo = ""
                        pixelInfo = "<span style='color: " + cardinalRed_hex + "; font-size: 24pt;'>x=%0.1f y=%0.1f I=%0.1f </span>"
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
                    if self.maskingMode > 0:
                        self.initMask()
                        if self.maskingMode == 1:
                            # masking mode
                            self.userMaskAssem[indexX,indexY] = 0
                        elif self.maskingMode == 2:
                            # unmasking mode
                            self.userMaskAssem[indexX,indexY] = 1
                        elif self.maskingMode == 3:
                            # toggle mode
                            self.userMaskAssem[indexX,indexY] = (1-self.userMaskAssem[indexX,indexY])
                        self.displayMask()

                # Per pixel histogram
                if self.pixelIndAssem is not None and self.perPixelHistogramFileOpen:
                    self.pixelInd = self.pixelIndAssem[indexX,indexY]
                    print "pixel index: ", self.pixelInd
                    self.updatePerPixelHistogram(self.pixelInd)

        # Signal proxy
        self.proxy_move = pg.SignalProxy(self.xhair.scene().sigMouseMoved, rateLimit=30, slot=mouseMoved)
        self.proxy_click = pg.SignalProxy(self.xhair.scene().sigMouseClicked, slot=mouseClicked)

        self.win.show()

    def initMask(self):
        print "initMask"
        if self.gapAssemInd is None:
            self.gapAssem = self.det.image(self.evt,np.ones_like(self.calib,dtype='int'))
            self.gapAssemInd = np.where(self.gapAssem==0)
        if self.userMask is None and self.data is not None:
            print "make user mask: ", self.data.shape, self.userMask
            # initialize
            self.userMaskAssem = np.ones_like(self.data,dtype='int')
            self.userMask = self.det.ndarray_from_image(self.evt,self.userMaskAssem, pix_scale_size_um=None, xy0_off_pix=None)
        print "Done initMask"

    def displayMask(self):
        print "displayMask"
        # convert to RGB
        print "mask on:", self.userMaskOn, self.streakMaskOn, self.psanaMaskOn
        if self.userMaskOn is False and self.streakMaskOn is False and self.psanaMaskOn is False:
            print "No mask is being used"
            self.display_data = self.data
        elif self.userMaskAssem is None and self.streakMaskAssem is None and self.psanaMaskAssem is None:
            print "No mask exists"
            self.display_data = self.data
        elif self.data is not None:
            print "Mask exists"
            self.display_data = np.zeros((self.data.shape[0], self.data.shape[1], 3), dtype = self.data.dtype)
            self.display_data[:,:,0] = self.data
            self.display_data[:,:,1] = self.data
            self.display_data[:,:,2] = self.data
            # update streak mask as red
            if self.streakMaskOn is True and self.streakMaskAssem is not None:
                self.streakMaskAssem[self.gapAssemInd] = 1
                _streakMaskInd = np.where(self.streakMaskAssem==0)
                self.display_data[_streakMaskInd[0], _streakMaskInd[1], 0] = self.data[_streakMaskInd] + (np.max(self.data) - self.data[_streakMaskInd]) * (1-self.streakMaskAssem[_streakMaskInd])
                self.display_data[_streakMaskInd[0], _streakMaskInd[1], 1] = self.data[_streakMaskInd] * self.streakMaskAssem[_streakMaskInd]
                self.display_data[_streakMaskInd[0], _streakMaskInd[1], 2] = self.data[_streakMaskInd] * self.streakMaskAssem[_streakMaskInd]
            # update psana mask as green
            if self.psanaMaskOn is True and self.psanaMaskAssem is not None:
                self.psanaMaskAssem[self.gapAssemInd] = 1
                _psanaMaskInd = np.where(self.psanaMaskAssem==0)
                self.display_data[_psanaMaskInd[0], _psanaMaskInd[1], 0] = self.data[_psanaMaskInd] * self.psanaMaskAssem[_psanaMaskInd]
                self.display_data[_psanaMaskInd[0], _psanaMaskInd[1], 1] = self.data[_psanaMaskInd] + (np.max(self.data) - self.data[_psanaMaskInd]) * (1-self.psanaMaskAssem[_psanaMaskInd])
                self.display_data[_psanaMaskInd[0], _psanaMaskInd[1], 2] = self.data[_psanaMaskInd] * self.psanaMaskAssem[_psanaMaskInd]
            # update user mask as blue
            if self.userMaskOn is True and self.userMaskAssem is not None:
                self.userMaskAssem[self.gapAssemInd] = 1
                _userMaskInd = np.where(self.userMaskAssem==0)
                self.display_data[_userMaskInd[0], _userMaskInd[1], 0] = self.data[_userMaskInd] * self.userMaskAssem[_userMaskInd]
                self.display_data[_userMaskInd[0], _userMaskInd[1], 1] = self.data[_userMaskInd] * self.userMaskAssem[_userMaskInd]
                self.display_data[_userMaskInd[0], _userMaskInd[1], 2] = self.data[_userMaskInd] + (np.max(self.data) - self.data[_userMaskInd]) * (1-self.userMaskAssem[_userMaskInd])
            print "display_data: ", self.display_data.shape
        self.w1.setImage(self.display_data,autoRange=False,autoLevels=False,autoHistogramRange=False)
        print "Done drawing"

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
        print("**** Running hit finder ****")

        if self.streakMaskOn:
            print "Getting streak mask!!!"
            self.initMask()
            self.streakMask = myskbeam.getStreakMaskCalib(self.det,self.evt,width=self.streak_width,sigma=self.streak_sigma)
            self.streakMaskAssem = self.det.image(self.evt,self.streakMask)
            self.algInitDone = False

        self.displayMask()

        # update combined mask
        if self.combinedMask is None:
            self.combinedMask = np.ones_like(self.calib)
        if self.streakMask is not None:
            self.combinedMask *= self.streakMask
        if self.userMask is not None:
            self.combinedMask *= self.userMask
        if self.psanaMask is not None:
            self.combinedMask *= self.psanaMask

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
        if self.algorithm == 0: # No peak algorithm
            self.peaks = None
            self.drawPeaks()
        else:
            # Only initialize the hit finder algorithm once
            if self.algInitDone is False:
                print("** Initializing peak finding algorithm **")
                self.windows = None
                self.alg = []
                self.alg = PyAlgos(windows=self.windows, mask=self.combinedMask, pbits=0)

                # set peak-selector parameters:
                if self.algorithm == 1:
                    self.alg.set_peak_selection_pars(npix_min=self.hitParam_alg_npix_min, npix_max=self.hitParam_alg_npix_max, \
                                            amax_thr=self.hitParam_alg_amax_thr, atot_thr=self.hitParam_alg_atot_thr, \
                                            son_min=self.hitParam_alg_son_min)
                elif self.algorithm == 3:
                    self.alg.set_peak_selection_pars(npix_min=self.hitParam_alg3_npix_min, npix_max=self.hitParam_alg3_npix_max, \
                                            amax_thr=self.hitParam_alg3_amax_thr, atot_thr=self.hitParam_alg3_atot_thr, \
                                            son_min=self.hitParam_alg3_son_min)
                elif self.algorithm == 4:
                    self.alg.set_peak_selection_pars(npix_min=self.hitParam_alg4_npix_min, npix_max=self.hitParam_alg4_npix_max, \
                                            amax_thr=self.hitParam_alg4_amax_thr, atot_thr=self.hitParam_alg4_atot_thr, \
                                            son_min=self.hitParam_alg4_son_min)
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
                self.peakRadius = int(self.hitParam_alg3_r0)
                self.peaks = self.alg.peak_finder_v3(self.calib, rank=self.hitParam_alg3_rank, r0=self.peakRadius, dr=self.hitParam_alg3_dr)
            elif self.algorithm == 4:
                print("* Running peak finder 4 *")
                # v4 - aka Droplet Finder - the same as v1, but uses rank and r0 parameters in stead of common radius.
                self.peakRadius = int(self.hitParam_alg4_radius)
                print(self.hitParam_alg4_thr_low,self.hitParam_alg4_thr_high,self.hitParam_alg4_rank,self.peakRadius,self.hitParam_alg4_dr)
                self.peaks = self.alg.peak_finder_v4(self.calib, thr_low=self.hitParam_alg4_thr_low, thr_high=self.hitParam_alg4_thr_high,
                                           rank=self.hitParam_alg4_rank, r0=self.peakRadius,  dr=self.hitParam_alg4_dr)
            self.numPeaksFound = self.peaks.shape[0]

            fmt = '%3d %4d %4d  %4d %8.1f %6.1f %6.1f %6.2f %6.2f %6.2f %4d %4d %4d %4d %6.2f %6.2f %6.2f'
            for peak in self.peaks :
                    seg,row,col,npix,amax,atot,rcent,ccent,rsigma,csigma,rmin,rmax,cmin,cmax,bkgd,rms,son = peak[0:17]
                    print fmt % (seg, row, col, npix, amax, atot, rcent, ccent, rsigma, csigma,\
                                 rmin, rmax, cmin, cmax, bkgd, rms, son)
                    if self.isCspad:
                        cheetahRow,cheetahCol = self.convert_peaks_to_cheetah(seg,row,col)
                        print "cheetahRow,Col", cheetahRow, cheetahCol, atot

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
        if self.showPeaks:
            if self.peaks is not None and self.numPeaksFound > 0:
                iX  = np.array(self.det.indexes_x(self.evt), dtype=np.int64)
                iY  = np.array(self.det.indexes_y(self.evt), dtype=np.int64)
                if len(iX.shape)==2:
                    iX = np.expand_dims(iX,axis=0)
                    iY = np.expand_dims(iY,axis=0)
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
        else:
            self.peak_feature.setData([], [], pxMode=False)
        print "Done updatePeaks"

    def updateImage(self,calib=None):
        if self.hasExperimentName and self.hasRunNumber and self.hasDetInfo:
            if calib is None:
                self.calib, self.data = self.getDetImage(self.eventNumber)
            else:
                print "Got here getDetImage"
                _, self.data = self.getDetImage(self.eventNumber,calib=calib)

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
                calib = self.det.calib(self.evt) #self.det.raw(self.evt) - self.det.pedestals(self.evt)
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
        if data is None:
            data = _calib
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
            # assemble image
            data = self.getAssembledImage(calib)
            self.cx, self.cy = self.det.point_indexes(self.evt,pxy_um=(0,0))
            return calib, data
        else: # TODO: this is a hack that assumes opal is the only detector without calib
            # we have an opal
            data = self.det.raw(self.evt)
            if data is not None:
                data = data.copy()
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

    def changeManifold(self, param, changes):
        for param, change, data in changes:
            path = self.p4.childPath(param)
            print('  path: %s'% path)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')
            self.update(path,change,data)

    def changePerPixelHistogram(self, param, changes):
        for param, change, data in changes:
            path = self.p5.childPath(param)
            print('  path: %s'% path)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')
            self.update(path,change,data)

    def changeMask(self, param, changes):
        for param, change, data in changes:
            path = self.p6.childPath(param)
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
                if self.showPeaks:
                    self.updateClassification()
            elif path[1] == exp_run_str:
                self.updateRunNumber(data)
                if self.showPeaks:
                    self.updateClassification()
            elif path[1] == exp_detInfo_str:
                self.updateDetInfo(data)
                if self.showPeaks:
                    self.updateClassification()
            elif path[1] == exp_evt_str and len(path) == 2 and change is 'value':
                self.updateEventNumber(data)
                if self.showPeaks:
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
                self.algInitDone = False
                self.updateAlgorithm(data)
            elif path[1] == hitParam_showPeaks_str:
                self.showPeaks = data
                self.drawPeaks()
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

            elif path[2] == hitParam_alg_npix_min_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg_npix_min = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg_npix_max_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg_npix_max = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg_amax_thr_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg_amax_thr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg_atot_thr_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg_atot_thr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg_son_min_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg_son_min = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg1_thr_low_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg1_thr_low = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg1_thr_high_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg1_thr_high = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg1_radius_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg1_radius = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg1_dr_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg1_dr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_npix_min_str and path[1] == hitParam_algorithm3_str:
                self.hitParam_alg3_npix_min = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_npix_max_str and path[1] == hitParam_algorithm3_str:
                self.hitParam_alg3_npix_max = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_amax_thr_str and path[1] == hitParam_algorithm3_str:
                self.hitParam_alg3_amax_thr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_atot_thr_str and path[1] == hitParam_algorithm3_str:
                self.hitParam_alg3_atot_thr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_son_min_str and path[1] == hitParam_algorithm3_str:
                self.hitParam_alg3_son_min = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_rank_str and path[1] == hitParam_algorithm3_str:
                self.hitParam_alg3_rank = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_r0_str and path[1] == hitParam_algorithm3_str:
                self.hitParam_alg3_r0 = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg3_dr_str and path[1] == hitParam_algorithm3_str:
                self.hitParam_alg3_dr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg4_npix_min_str and path[1] == hitParam_algorithm4_str:
                self.hitParam_alg4_npix_min = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg4_npix_max_str and path[1] == hitParam_algorithm4_str:
                self.hitParam_alg4_npix_max = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg4_amax_thr_str and path[1] == hitParam_algorithm4_str:
                self.hitParam_alg_amax_thr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg4_atot_thr_str and path[1] == hitParam_algorithm4_str:
                self.hitParam_alg4_atot_thr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg4_son_min_str and path[1] == hitParam_algorithm4_str:
                self.hitParam_alg4_son_min = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg4_thr_low_str and path[1] == hitParam_algorithm4_str:
                self.hitParam_alg4_thr_low = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg4_thr_high_str and path[1] == hitParam_algorithm4_str:
                self.hitParam_alg4_thr_high = data
                print("### alg4_thr_high ###: ",data)
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg4_rank_str and path[1] == hitParam_algorithm4_str:
                self.hitParam_alg4_rank = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg4_radius_str and path[1] == hitParam_algorithm4_str:
                self.hitParam_alg4_radius = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg4_dr_str and path[1] == hitParam_algorithm4_str:
                self.hitParam_alg4_dr = data
                self.algInitDone = False
                if self.showPeaks:
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
        ################################################
        # manifold parameters
        ################################################
        if path[0] == manifold_grp:
            if path[1] == manifold_filename_str:
                self.updateManifoldFilename(data)
            elif path[1] == manifold_dataset_str:
                self.updateManifoldDataset(data)
            elif path[1] == manifold_sigma_str:
                self.updateManifoldSigma(data)
        ################################################
        # per pixel histogram parameters
        ################################################
        if path[0] == perPixelHistogram_grp:
            if path[1] == perPixelHistogram_filename_str:
                self.updatePerPixelHistogramFilename(data)
            elif path[1] == perPixelHistogram_adu_str:
                self.updatePerPixelHistogramAdu(data)
        ################################################
        # masking parameters
        ################################################
        if path[0] == mask_grp:
            if path[1] == user_mask_str and len(path) == 2:
                self.updateUserMask(data)
                self.algInitDone = False
            elif path[1] == streak_mask_str and len(path) == 2:
                self.updateStreakMask(data)
                self.algInitDone = False
            elif path[1] == psana_mask_str and len(path) == 2:
                self.updatePsanaMask(data)
                self.algInitDone = False
            if len(path) == 3:
                if path[2] == mask_mode_str:
                    self.algInitDone = False
                    self.updateMaskingMode(data)
                if path[2] == streak_width_str:
                    self.algInitDone = False
                    self.updateStreakWidth(data)
                if path[2] == streak_sigma_str:
                    self.algInitDone = False
                    self.updateStreakSigma(data)
                if path[2] == mask_calib_str:
                    self.algInitDone = False
                    self.updatePsanaMaskFlag(path[2],data)
                elif path[2] == mask_status_str:
                    self.algInitDone = False
                    self.updatePsanaMaskFlag(path[2],data)
                elif path[2] == mask_edges_str:
                    self.algInitDone = False
                    self.updatePsanaMaskFlag(path[2],data)
                elif path[2] == mask_central_str:
                    self.algInitDone = False
                    self.updatePsanaMaskFlag(path[2],data)
                elif path[2] == mask_unbond_str:
                    self.algInitDone = False
                    self.updatePsanaMaskFlag(path[2],data)
                elif path[2] == mask_unbondnrs_str:
                    self.algInitDone = False
                    self.updatePsanaMaskFlag(path[2],data)
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
        if self.hasDetInfo is False or self.detInfo is not data:
            self.resetMasks()
            self.calib = None
            self.data = None
            self.firstUpdate = True

        self.detInfo = data
        if data == 'DscCsPad' or data == 'DsdCsPad' or data == 'DsaCsPad':
            self.isCspad = True
        if data == 'Sc2Questar':
            self.isCamera = True
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

    def resetMasks(self):
        self.userMask = None
        self.psanaMask = None
        self.streakMask = None
        self.userMaskAssem = None
        self.psanaMaskAssem = None
        self.streakMaskAssem = None
        self.combinedMask = None
        self.gapAssemInd = None
        self.gapAssem = None
        self.userMaskOn = False
        self.psanaMaskOn = False
        self.streakMaskOn = False
        self.maskingMode = 0
        self.p6.param(mask_grp,user_mask_str,mask_mode_str).setValue(0)
        self.p6.param(mask_grp,user_mask_str).setValue(0)
        self.p6.param(mask_grp,psana_mask_str).setValue(0)
        self.p6.param(mask_grp,streak_mask_str).setValue(0)

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
                    source_string = Detector.PyDetector.map_alias_to_source(k.alias(), self.env)
                    if source_string is not '':
                        if Detector.PyDetector.dettype(source_string, self.env) == Detector.AreaDetector.AreaDetector:
                            myAreaDetectors.append(k.alias())
                except ValueError:
                    continue
            self.detInfoList = list(set(myAreaDetectors))
            print "#######################"
            print "# Available detectors: ", self.detInfoList
            print "#######################"

        if self.hasExpRunDetInfo():
            self.det = psana.Detector(str(self.detInfo), self.env)

            if self.evt is None:
                self.evt = self.run.event(self.times[0])
            print "Setting up pixelInd"
            temp = self.det.calib(self.evt)
            if temp is None:
                _temp = self.det.raw(self.evt) # why is this read-only?
                temp = _temp.copy()
            if temp is not None:
                self.pixelInd = np.reshape(np.arange(temp.size)+1,temp.shape)
                self.pixelIndAssem = self.getAssembledImage(self.pixelInd)
                self.pixelIndAssem -= 1 # First pixel is 0
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
        self.updateClassification()
        print "##### Done updateAlgorithm: ", self.algorithm

    def updateUserMask(self, data):
        self.userMaskOn = data
        self.updateClassification()
        print "Done updateUserMask: ", self.userMaskOn

    def updateStreakMask(self, data):
        self.streakMaskOn = data
        self.updateClassification()
        print "Done updateStreakMask: ", self.streakMaskOn

    def updateStreakWidth(self, data):
        self.streak_width = data
        self.updateClassification()
        print "Done updateStreakWidth: ", self.streak_width

    def updateStreakSigma(self, data):
        self.streak_sigma = data
        self.updateClassification()
        print "Done updateStreakSigma: ", self.streak_sigma

    def updatePsanaMask(self, data):
        self.psanaMaskOn = data
        self.updatePsanaMaskOn()
        print "Done updatePsanaMask: ", self.psanaMaskOn

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

    ##################################
    ########### Quantifier ###########
    ##################################

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

            # temp
            try:
                eventDataset = "/" + self.quantifier_dataset.split("/")[1] + "/event"
                print eventDataset
                self.quantifierEvent = self.quantifierFile[eventDataset].value
            except:
                self.quantifierEvent = np.arange(len(self.quantifierMetric))

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
        print "x,y: ", indX, indY
        if self.quantifier_sort:
            ind = self.quantifierInd[ind]

        # temp
        self.eventNumber = self.quantifierEvent[ind]
        #self.eventNumber = ind

        self.calib, self.data = self.getDetImage(self.eventNumber)
        self.w1.setImage(self.data,autoRange=False,autoLevels=False,autoHistogramRange=False)
        self.p.param(exp_grp,exp_evt_str).setValue(self.eventNumber)

    ##################################
    ###### Per Pixel Histogram #######
    ##################################

    def updatePerPixelHistogramFilename(self, data):
        # close previously open file
        if self.perPixelHistogram_filename is not data and self.perPixelHistogramFileOpen:
            self.perPixelHistogramFile.close()
        self.perPixelHistogram_filename = data
        self.perPixelHistogramFile = h5py.File(self.perPixelHistogram_filename,'r')

        self.valid_min = self.perPixelHistogramFile['/dataHist/histogram'].attrs['valid_min'] # ADU
        self.valid_max = self.perPixelHistogramFile['/dataHist/histogram'].attrs['valid_max'] # ADU
        self.bin_size = self.perPixelHistogramFile['/dataHist/histogram'].attrs['bin_size'] # ADU
        units = np.linspace(self.valid_min,self.valid_max,self.valid_max-self.valid_min+1) # ADU
        start = np.mean(units[0:self.bin_size]) # ADU
        finish = np.mean(units[len(units)-self.bin_size:len(units)]) # ADU
        numBins = (self.valid_max-self.valid_min+1)/self.bin_size # ADU
        self.histogram_adu = np.linspace(start,finish,numBins) # ADU

        self.perPixelHistograms = self.perPixelHistogramFile['/dataHist/histogram'].value
        self.histogram1D = self.perPixelHistograms.reshape((-1,self.perPixelHistograms.shape[-1]))

        self.perPixelHistogramFileOpen = True
        print "Done opening perPixelHistogram file"

    # FIXME: I don't think pixelIndex is correct
    def updatePerPixelHistogramAdu(self, data):
        print "$$$$$$ money: ", data
        self.perPixelHistogram_adu = data
        if self.perPixelHistogramFileOpen:
            print "update slice"
            self.updatePerPixelHistogramSlice(self.perPixelHistogram_adu)
        print "Done perPixelHistogram adu", self.perPixelHistogram_adu

    def updatePerPixelHistogramSlice(self,adu):
        print "%%% got ", adu
        self.histogramAduIndex = self.getHistogramIndex(adu)
        print "^^^ ", self.histogramAduIndex
        self.calib = np.squeeze(self.perPixelHistogramFile['/dataHist/histogram'][:,:,:,self.histogramAduIndex])
        print "updatePerPixelHistogramSlice: ", self.calib.shape
        self.updateImage(calib=self.calib)

    def getHistogramIndex(self,adu):
        histogramIndex = np.argmin(abs(self.histogram_adu - adu))
        print "histogramIndex: ", self.histogram_adu, histogramIndex
        return histogramIndex

    def updatePerPixelHistogram(self, pixelIndex):
        self.w16.getPlotItem().clear()
        if pixelIndex >= 0:
            self.perPixelHistogram = self.histogram1D[pixelIndex,1:-1]
            self.w16.plot(self.histogram_adu, self.perPixelHistogram, pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')
        self.w16.setLabel('left', "Counts")
        self.w16.setLabel('bottom', "ADU")

    ##################################
    ############ Masking #############
    ##################################

    def updateMaskingMode(self, data):
        self.maskingMode = data
        if self.maskingMode == 0:
            # display text
            self.label.setText("")
            # do not display user mask
            self.displayMask()
            # remove ROIs
            self.w1.getView().removeItem(self.roi_rect)
            self.w1.getView().removeItem(self.roi_circle)
        else:
            # display text
            self.label.setText(masking_mode_message)
            # display user mask
            self.displayMask()
            # init masks
            if self.roi_rect is None:
                # Rect mask
                self.roi_rect = pg.ROI(pos=[self.cx+100,self.cy], size=[50, 50], snapSize=1.0, scaleSnap=True, translateSnap=True, pen={'color': 'b', 'width': 4})
                self.roi_rect.addScaleHandle([0.5, 1], [0.5, 0.5])
                self.roi_rect.addScaleHandle([0, 0.5], [0.5, 0.5])
                self.roi_rect.addRotateHandle([0.5, 0.5], [1, 1])
                # Circular mask
                self.roi_circle = pg.CircleROI([self.cx,self.cy], size=[50, 50], snapSize=1.0, scaleSnap=True, translateSnap=True, pen={'color': 'b', 'width': 4})

            # add ROIs
            self.w1.getView().addItem(self.roi_rect)
            self.w1.getView().addItem(self.roi_circle)
        print "Done updateMaskingMode: ", self.maskingMode

    def updatePsanaMaskFlag(self, flag, data):
        print "Update psana mask flag"
        if flag == mask_calib_str:
            self.mask_calibOn = data
        elif flag == mask_status_str:
            self.mask_statusOn = data
        elif flag == mask_central_str:
            self.mask_centralOn = data
        elif flag == mask_edges_str:
            self.mask_edgesOn = data
        elif flag == mask_unbond_str:
            self.mask_unbondOn = data
        elif flag == mask_unbondnrs_str:
            self.mask_unbondnrsOn = data
        self.updatePsanaMaskOn()

    def updatePsanaMaskOn(self):
        print "Making psana mask"
        self.initMask()
        self.psanaMask = self.det.mask(self.evt, calib=self.mask_calibOn, status=self.mask_statusOn,
                                      edges=self.mask_edgesOn, central=self.mask_centralOn,
                                      unbond=self.mask_unbondOn, unbondnbrs=self.mask_unbondnrsOn)
        if self.psanaMask is not None:
            self.psanaMaskAssem = self.det.image(self.evt,self.psanaMask)
        else:
            self.psanaMaskAssem = None
        self.updateClassification()

    ##################################
    ########### Manifold #############
    ##################################
    # FIXME: manifold is incomplete
    def updateManifoldFilename(self, data):
        # close previously open file
        if self.manifold_filename is not data and self.manifoldFileOpen:
            self.manifoldFile.close()
        self.manifold_filename = data
        self.manifoldFile = h5py.File(self.manifold_filename,'r')
        self.manifoldFileOpen = True
        print "Done opening manifold"

    def updateManifoldDataset(self, data):
        self.manifold_dataset = data
        if self.manifoldFileOpen:
            self.manifoldEigs = self.manifoldFile[self.manifold_dataset].value
            (self.manifoldNumHits,self.manifoldNumEigs) = self.manifoldEigs.shape
            self.manifoldHasData = True
            self.updateManifoldPlot(self.manifoldEigs)
            self.manifoldInd = np.arange(self.manifoldNumHits)

            try:
                eventDataset = "/" + self.manifold_dataset.split("/")[1] + "/event"
                self.manifoldEvent = self.manifoldFile[eventDataset].value
            except:
                self.manifoldEvent = np.arange(self.manifoldNumHits)

            print "Done reading manifold"

    def updateManifoldSigma(self, data):
        self.manifold_sigma = data
        if self.manifoldHasData:
            self.updateManifoldPlot(self.manifoldInd,self.manifoldEigs)

    def updateManifoldPlot(self,ind,eigenvectors):
        print "updateManifoldPlot"
        self.lastClicked = []
        self.w13.getPlotItem().clear()
        pos = np.random.normal(size=(2,10000), scale=1e-9)
        self.curve = self.w13.plot(pos[0],pos[1], pen=None, symbol='o',symbolPen=None,symbolSize=10,symbolBrush=(100,100,255,50))
        #self.curve = self.w9.plot(ind,metric, pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')
        self.curve.curve.setClickable(True)
        self.curve.sigClicked.connect(self.clicked1)

    def clicked1(self,points): # manifold click
        print("curve clicked",points)
        from pprint import pprint
        pprint(vars(points.scatter))
        for i in range(len(points.scatter.data)):
            if points.scatter.ptsClicked[0] == points.scatter.data[i][7]:
                ind = i
                break
        indX = points.scatter.data[i][0]
        indY = points.scatter.data[i][1]
        print "x,y: ", indX, indY

        ind = self.manifoldInd[ind]

        # temp
        self.eventNumber = self.manifoldEvent[ind]

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
                  " -o %J.log generatePowder exp="+self.experimentName+\
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
        print "Finding peaks!!!!!!!!!!!!"
        # Digest the run list
        runsToDo = self.digestRunList(self.parent.hitParam_runs)
        print runsToDo

        for run in runsToDo:
            cmd = "bsub -q "+self.parent.hitParam_queue+\
              " -a mympi -n "+str(self.parent.hitParam_cpus)+\
              " -o %J.log findPeaks -e "+self.experimentName+\
              " -r "+str(run)+" -d "+self.detInfo+\
              " --outDir "+str(self.parent.hitParam_outDir)+\
              " --algorithm "+str(self.parent.algorithm)+\
              " --alg_npix_min "+str(self.parent.hitParam_alg_npix_min)+\
              " --alg_npix_max "+str(self.parent.hitParam_alg_npix_max)+\
              " --alg_amax_thr "+str(self.parent.hitParam_alg_amax_thr)+\
              " --alg_atot_thr "+str(self.parent.hitParam_alg_atot_thr)+\
              " --alg_son_min "+str(self.parent.hitParam_alg_son_min)

            if self.parent.algorithm == 1:
                cmd += " --alg1_thr_low "+str(self.parent.hitParam_alg1_thr_low)+\
                         " --alg1_thr_high "+str(self.parent.hitParam_alg1_thr_high)+\
                         " --alg1_radius "+str(self.parent.hitParam_alg1_radius)+\
                         " --alg1_dr "+str(self.parent.hitParam_alg1_dr)
            elif self.parent.algorithm == 3:
                cmd += " --alg3_rank "+str(self.parent.hitParam_alg3_rank)+\
                         " --alg3_r0 "+str(self.parent.hitParam_alg3_r0)+\
                         " --alg3_dr "+str(self.parent.hitParam_alg3_dr)
            # Save user mask to a deterministic path
            if self.parent.userMaskOn:
                tempFilename = "tempUserMask.npy"
                np.save(tempFilename,self.parent.userMask) # TODO: save
                cmd += " --userMask_path "+str(tempFilename)
            if self.parent.streakMaskOn:
                cmd += " --streakMask_sigma "+str(self.parent.streak_sigma)+\
                   " --streakMask_width "+str(self.parent.streak_width)
            if self.parent.psanaMaskOn:
                cmd += " --psanaMask_calib "+str(self.parent.mask_calibOn)+" "+\
                   " --psanaMask_status "+str(self.parent.mask_statusOn)+" "+\
                   " --psanaMask_edges "+str(self.parent.mask_edgesOn)+" "+\
                   " --psanaMask_central "+str(self.parent.mask_centralOn)+" "+\
                   " --psanaMask_unbond "+str(self.parent.mask_unbondOn)+" "+\
                   " --psanaMask_unbondnrs "+str(self.parent.mask_unbondnrsOn)

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

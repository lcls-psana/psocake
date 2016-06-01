# GUI for browsing LCLS area detectors. Tune hit finding parameters and common mode correction.

# TODO: Zoom in area / view matrix of numbers
# TODO: Multiple subplots or grid of images
# TODO: dropdown menu for available detectors
# TODO: When front and back detectors given, display both
# TODO: Radial average panel
# TODO: Show raw, pedestal-corrected, commonMode-corrected, gain-corrected
# TODO: Display xtcav, acqiris
# TODO: Downsampler
# TODO: Radial background, polarization correction

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
from ImgAlgos.PyAlgos import PyAlgos # peak finding
import argparse
import Detector.PyDetector
import time
import subprocess
import os.path
import myskbeam
from PSCalib.GeometryObject import data2x2ToTwo2x1, two2x1ToData2x2
# Panel modules
import diffractionGeometryPanel
import crystalIndexingPanel

parser = argparse.ArgumentParser()
parser.add_argument('expRun', nargs='?', default=None, help="psana experiment/run string in the format of exp=<experiment name>:run=<run number> (e.g. exp=cxi06216:run=22)")
parser.add_argument("-e","--exp", help="experiment name only or psana-style experiment and run (e.g. cxis0813 )", default="", type=str)
parser.add_argument("-r","--run", help="run number (e.g. 5), default=0",default=0, type=int)
parser.add_argument("-d","--det", help="detector name (e.g. CxiDs1.0:Cspad.0), default=''",default="", type=str)
parser.add_argument("-n","--evt", help="event number (e.g. 1), default=0",default=0, type=int)
parser.add_argument("--localCalib", help="use local calib directory, default=False", action='store_true')
#parser.add_argument("--more", help="display more panels", action='store_true')
args = parser.parse_args()

# Set up tolerance
eps = np.finfo("float64").eps

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
disp_adu_str = 'gain corrected ADU'
disp_gain_str = 'gain'
disp_coordx_str = 'coord_x'
disp_coordy_str = 'coord_y'
disp_quad_str = 'quad number'
disp_seg_str = 'seg number'
disp_row_str = 'row number'
disp_col_str = 'col number'
disp_raw_str = 'raw ADU'
disp_pedestalCorrected_str = 'pedestal corrected ADU'
disp_commonModeCorrected_str = 'common mode corrected ADU'
disp_photons_str = 'photon counts'
disp_rms_str = 'pixel rms'
disp_status_str = 'pixel status'
disp_pedestal_str = 'pedestal'
disp_commonMode_str = 'common mode'
disp_aduThresh_str = 'ADU threshold'

disp_commonModeOverride_str = 'Common mode (override)'
disp_overrideCommonMode_str = 'Apply common mode (override)'
disp_commonModeParam0_str = 'parameters 0'
disp_commonModeParam1_str = 'parameters 1'
disp_commonModeParam2_str = 'parameters 2'
disp_commonModeParam3_str = 'parameters 3'

# Peak finding
hitParam_grp = 'Peak finder'
hitParam_showPeaks_str = 'Show peaks found'
hitParam_algorithm_str = 'Algorithm'
# algorithm 0
hitParam_algorithm0_str = 'None'
# algorithm 1
hitParam_alg1_npix_min_str = 'npix_min'
hitParam_alg1_npix_max_str = 'npix_max'
hitParam_alg1_amax_thr_str = 'amax_thr'
hitParam_alg1_atot_thr_str = 'atot_thr'
hitParam_alg1_son_min_str = 'son_min'
hitParam_algorithm1_str = 'Droplet'
hitParam_alg1_thr_low_str = 'thr_low'
hitParam_alg1_thr_high_str = 'thr_high'
hitParam_alg1_radius_str = 'radius'
hitParam_alg1_dr_str = 'dr'
# algorithm 2
hitParam_alg2_npix_min_str = 'npix_min'
hitParam_alg2_npix_max_str = 'npix_max'
hitParam_alg2_amax_thr_str = 'amax_thr'
hitParam_alg2_atot_thr_str = 'atot_thr'
hitParam_alg2_son_min_str = 'son_min'
hitParam_algorithm2_str = 'FloodFill'
hitParam_alg2_thr_str = 'thr'
hitParam_alg2_r0_str = 'r0'
hitParam_alg2_dr_str = 'dr'
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
hitParam_alg4_r0_str = 'radius'
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

# Hit finding
spiParam_grp = 'Hit finder'
spiParam_algorithm_str = 'Algorithm'
# algorithm 0
spiParam_algorithm0_str = 'None'
# algorithm 1
spiParam_algorithm1_str = 'chiSquared'
spiParam_alg1_pruneInterval_str = 'prune interval'
# algorithm 2
spiParam_algorithm2_str = 'photonFinder'
spiParam_alg2_threshold_str = 'ADU per photon'

spiParam_outDir_str = 'Output directory'
spiParam_tag_str = 'Filename tag'
spiParam_runs_str = 'Run(s)'
spiParam_queue_str = 'queue'
spiParam_cpu_str = 'CPUs'
spiParam_psanaq_str = 'psanaq'
spiParam_psnehq_str = 'psnehq'
spiParam_psfehq_str = 'psfehq'
spiParam_psnehprioq_str = 'psnehprioq'
spiParam_psfehprioq_str = 'psfehprioq'
spiParam_psnehhiprioq_str = 'psnehhiprioq'
spiParam_psfehhiprioq_str = 'psfehhiprioq'
spiParam_noe_str = 'Number of events to process'

# Quantifier parameter tree
quantifier_grp = 'Small data'
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

# Detector correction parameter tree
correction_grp = 'Detector correction'
correction_radialBackground_str = "Use radial background correction"
correction_polarization_str = "Use polarization correction"

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
        self.psocakeDir = os.getcwd()
        # Init experiment parameters
        if args.expRun is not None and ':run=' in args.expRun[0]:
            self.experimentName = args.expRun[0].split('exp=')[-1].split(':')[0]
            self.runNumber = args.expRun[0].split('run=')[-1]
            self.psocakeDir = '/reg/d/psdm/'+self.experimentName[:3]+'/'+self.experimentName+'/scratch/psocake'
        else:
            self.experimentName = args.exp
            self.runNumber = int(args.run)
            self.psocakeDir = '/reg/d/psdm/'+self.experimentName[:3]+'/'+self.experimentName+'/scratch/psocake'
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
        self.aduThresh = -100.
        self.displayMaxPercentile = 99.0

        self.hasUserDefinedResolution = False
        self.hasCommonMode = False
        self.applyCommonMode = False
        self.commonMode = np.array([0,0,0,0])
        self.commonModeParams = np.array([0,0,0,0])
        # Init diffraction geometry parameters
        self.detectorDistance = 0.0
        self.photonEnergy = None
        self.wavelength = None
        self.pixelSize = None
        self.resolutionRingsOn = False
        self.resolution = None
        #self.resolutionText = []
        self.resolutionUnits = 0
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
        self.StreakMask = None # streak mask class
        self.streakMaskAssem = None
        self.combinedMask = None # combined mask
        self.gapAssemInd = None
        # Init peak finding parameters
        self.algInitDone = False
        self.algorithm = 0
        self.classify = False

        self.showPeaks = True
        self.peaks = None
        self.hitParam_alg1_npix_min = 1.
        self.hitParam_alg1_npix_max = 45.
        self.hitParam_alg1_amax_thr = 250.
        self.hitParam_alg1_atot_thr = 330.
        self.hitParam_alg1_son_min = 10.
        self.hitParam_alg1_thr_low = 80.
        self.hitParam_alg1_thr_high = 270.
        self.hitParam_alg1_radius = 3
        self.hitParam_alg1_dr = 1
        self.hitParam_alg2_npix_min = 1.
        self.hitParam_alg2_npix_max = 5000.
        self.hitParam_alg2_amax_thr = 1.
        self.hitParam_alg2_atot_thr = 1.
        self.hitParam_alg2_son_min = 1.
        self.hitParam_alg2_thr = 10.
        self.hitParam_alg2_r0 = 1.
        self.hitParam_alg2_dr = 0.05
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
        self.hitParam_alg4_r0 = 3
        self.hitParam_alg4_dr = 1

        self.hitParam_outDir = self.psocakeDir
        self.hitParam_runs = ''
        self.hitParam_queue = hitParam_psanaq_str
        self.hitParam_cpus = 32
        self.hitParam_noe = 0

        # Indexing
        self.showIndexedPeaks = True
        self.indexedPeaks = None
        self.hiddenCXI = '.temp.cxi'
        self.hiddenCrystfelStream = '.temp.stream'
        self.hiddenCrystfelList = '.temp.lst'
        if os.path.isfile(self.hiddenCXI): os.remove(self.hiddenCXI)
        if os.path.isfile(self.hiddenCrystfelStream): os.remove(self.hiddenCrystfelStream)
        if os.path.isfile(self.hiddenCrystfelList): os.remove(self.hiddenCrystfelList)

        # Init hit finding
        self.spiAlgorithm = 1

        self.spiParam_alg1_pruneInterval = 0
        self.spiParam_alg2_threshold = 100

        self.spiParam_outDir = self.psocakeDir
        self.spiParam_tag = ''
        self.spiParam_runs = ''
        self.spiParam_queue = spiParam_psanaq_str
        self.spiParam_cpus = 32
        self.spiParam_noe = 0

        # Quantifier
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

        self.correction_radialBackground = False
        self.correction_polarization = False

        self.maskingMode = 0
        self.userMaskOn = False
        self.streakMaskOn = False
        self.streak_sigma = 1
        self.streak_width = 250
        self.psanaMaskOn = False
        self.mask_calibOn = True
        self.mask_statusOn = True
        self.mask_edgesOn = True
        self.mask_centralOn = True
        self.mask_unbondOn = True
        self.mask_unbondnrsOn = True
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
                {'name': disp_image_str, 'type': 'list', 'values': {disp_coordy_str: 16,
                                                                    disp_coordx_str: 15,
                                                                    disp_col_str: 14,
                                                                    disp_row_str: 13,
                                                                    disp_seg_str: 12,
                                                                    disp_quad_str: 11,
                                                                    disp_gain_str: 10,
                                                                    disp_commonMode_str: 9,
                                                                    disp_rms_str: 8,
                                                                    disp_status_str: 7,
                                                                    disp_pedestal_str: 6,
                                                                    disp_photons_str: 5,
                                                                    disp_raw_str: 4,
                                                                    disp_pedestalCorrected_str: 3,
                                                                    disp_commonModeCorrected_str: 2,
                                                                    disp_adu_str: 1},
                 'value': self.image_property, 'tip': "Choose image property to display"},
                {'name': disp_aduThresh_str, 'type': 'float', 'value': self.aduThresh, 'tip': "Only display ADUs above this threshold"},
                {'name': disp_commonModeOverride_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
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
                    {'name': hitParam_alg1_npix_min_str, 'type': 'float', 'value': self.hitParam_alg1_npix_min, 'tip': "Only keep the peak if number of pixels above thr_low is above this value"},
                    {'name': hitParam_alg1_npix_max_str, 'type': 'float', 'value': self.hitParam_alg1_npix_max, 'tip': "Only keep the peak if number of pixels above thr_low is below this value"},
                    {'name': hitParam_alg1_amax_thr_str, 'type': 'float', 'value': self.hitParam_alg1_amax_thr, 'tip': "Only keep the peak if max value is above this value"},
                    {'name': hitParam_alg1_atot_thr_str, 'type': 'float', 'value': self.hitParam_alg1_atot_thr, 'tip': "Only keep the peak if integral inside region of interest is above this value"},
                    {'name': hitParam_alg1_son_min_str, 'type': 'float', 'value': self.hitParam_alg1_son_min, 'tip': "Only keep the peak if signal-over-noise is above this value"},
                    {'name': hitParam_alg1_thr_low_str, 'type': 'float', 'value': self.hitParam_alg1_thr_low, 'tip': "Only consider values above this value"},
                    {'name': hitParam_alg1_thr_high_str, 'type': 'float', 'value': self.hitParam_alg1_thr_high, 'tip': "Only keep the peak if max value is above this value"},
                    {'name': hitParam_alg1_radius_str, 'type': 'int', 'value': self.hitParam_alg1_radius, 'tip': "region of integration is a square, (2r+1)x(2r+1)"},
                    {'name': hitParam_alg1_dr_str, 'type': 'float', 'value': self.hitParam_alg1_dr, 'tip': "background region outside the region of interest"},
                ]},
#                {'name': hitParam_algorithm2_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
#                    #{'name': hitParam_alg2_npix_min_str, 'type': 'float', 'value': self.hitParam_alg2_npix_min, 'tip': "Only keep the peak if number of pixels above thr_low is above this value"},
#                    #{'name': hitParam_alg2_npix_max_str, 'type': 'float', 'value': self.hitParam_alg2_npix_max, 'tip': "Only keep the peak if number of pixels above thr_low is below this value"},
#                    #{'name': hitParam_alg2_amax_thr_str, 'type': 'float', 'value': self.hitParam_alg2_amax_thr, 'tip': "Only keep the peak if max value is above this value"},
#                    #{'name': hitParam_alg2_atot_thr_str, 'type': 'float', 'value': self.hitParam_alg2_atot_thr, 'tip': "Only keep the peak if integral inside region of interest is above this value"},
#                    #{'name': hitParam_alg2_son_min_str, 'type': 'float', 'value': self.hitParam_alg2_son_min, 'tip': "Only keep the peak if signal-over-noise is above this value"},
#                    {'name': hitParam_alg2_thr_str, 'type': 'float', 'value': self.hitParam_alg2_thr, 'tip': "Only keep the peak if max value is above this value"},
#                    {'name': hitParam_alg2_r0_str, 'type': 'float', 'value': self.hitParam_alg2_r0, 'tip': "region of integration is a square, (2r+1)x(2r+1)"},
#                    #{'name': hitParam_alg2_dr_str, 'type': 'float', 'value': self.hitParam_alg2_dr, 'tip': "background region outside the region of interest"},
#                ]},
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
                    {'name': hitParam_alg4_r0_str, 'type': 'int', 'value': self.hitParam_alg4_r0, 'tip': "region of integration is a square, (2r+1)x(2r+1)"},
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
        self.paramsCorrection = [
            {'name': correction_grp, 'type': 'group', 'children': [
                {'name': correction_radialBackground_str, 'type': 'str', 'value': self.correction_radialBackground, 'tip': "Use radial background correction"},
                {'name': correction_polarization_str, 'type': 'str', 'value': self.correction_polarization, 'tip': "Use polarization correction"},
            ]},
        ]
        self.paramsHitFinder = [
            {'name': spiParam_grp, 'type': 'group', 'children': [
                {'name': spiParam_algorithm_str, 'type': 'list', 'values': {spiParam_algorithm2_str: 2,
                                                                            #spiParam_algorithm1_str: 1,
                                                                            spiParam_algorithm0_str: 0},
                                                                            'value': self.spiAlgorithm},
                #{'name': spiParam_algorithm1_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
                #    {'name': spiParam_alg1_pruneInterval_str, 'type': 'float', 'value': self.spiParam_alg1_pruneInterval, 'tip': "Only keep the peak if number of pixels above thr_low is above this value"},
                #]},
                {'name': spiParam_algorithm2_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
                    {'name': spiParam_alg2_threshold_str, 'type': 'float', 'value': self.spiParam_alg2_threshold, 'tip': "search for pixels above ADU per photon"},
                ]},
                {'name': spiParam_outDir_str, 'type': 'str', 'value': self.spiParam_outDir},
                {'name': spiParam_tag_str, 'type': 'str', 'value': self.spiParam_tag, 'tip': "(Optional) identifying string to attach to filename"},
                {'name': spiParam_runs_str, 'type': 'str', 'value': self.spiParam_runs, 'tip': "comma separated or use colon for a range, e.g. 1,3,5:7 = runs 1,3,5,6,7"},
                {'name': spiParam_queue_str, 'type': 'list', 'values': {spiParam_psfehhiprioq_str: 'psfehhiprioq',
                                                                        spiParam_psnehhiprioq_str: 'psnehhiprioq',
                                                                        spiParam_psfehprioq_str: 'psfehprioq',
                                                                        spiParam_psnehprioq_str: 'psnehprioq',
                                                                        spiParam_psfehq_str: 'psfehq',
                                                                        spiParam_psnehq_str: 'psnehq',
                                                                        spiParam_psanaq_str: 'psanaq'},
                 'value': self.spiParam_queue, 'tip': "Choose queue"},
                {'name': spiParam_cpu_str, 'type': 'int', 'value': self.spiParam_cpus},
                {'name': spiParam_noe_str, 'type': 'int', 'value': self.spiParam_noe, 'tip': "number of events to process, default=0 means process all events"},
            ]},
        ]

        # Instantiate panels
        self.geom = diffractionGeometryPanel.DiffractionGeometry(self)
        self.index = crystalIndexingPanel.CrystalIndexing(self)

        self.initUI()

    def initUI(self):
        ## Define a top-level widget to hold everything
        self.win = QtGui.QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1300,650)
        self.win.setWindowTitle('psocake')

        ## Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', \
                                  children=self.params, expanded=True)
        self.p1 = Parameter.create(name='paramsDiffractionGeometry', type='group', \
                                  children=self.geom.params, expanded=True)
        self.p2 = Parameter.create(name='paramsQuantifier', type='group', \
                                  children=self.paramsQuantifier, expanded=True)
        self.p3 = Parameter.create(name='paramsPeakFinder', type='group', \
                                  children=self.paramsPeakFinder, expanded=True)
        self.p4 = Parameter.create(name='paramsManifold', type='group', \
                                  children=self.paramsManifold, expanded=True)
        #self.p5 = Parameter.create(name='paramsPerPixelHistogram', type='group', \
        #                          children=self.paramsPerPixelHistogram, expanded=True)
        self.p6 = Parameter.create(name='paramsMask', type='group', \
                                  children=self.paramsMask, expanded=True)
        self.p7 = Parameter.create(name='paramsCorrection', type='group', \
                                   children=self.paramsCorrection, expanded=True)
        self.p8 = Parameter.create(name='paramsHitFinder', type='group', \
                                   children=self.paramsHitFinder, expanded=True)
        self.p9 = Parameter.create(name='paramsCrystalIndexing', type='group', \
                                   children=self.index.params, expanded=True)
        self.p.sigTreeStateChanged.connect(self.change)
        self.p1.sigTreeStateChanged.connect(self.change)
        self.p2.sigTreeStateChanged.connect(self.change)
        self.p3.sigTreeStateChanged.connect(self.change)
        self.p4.sigTreeStateChanged.connect(self.change)
        #self.p5.sigTreeStateChanged.connect(self.change)
        self.p6.sigTreeStateChanged.connect(self.change)
        self.p7.sigTreeStateChanged.connect(self.change)
        self.p8.sigTreeStateChanged.connect(self.change)
        self.p9.sigTreeStateChanged.connect(self.change)

        ## Create docks, place them into the window one at a time.
        ## Note that size arguments are only a suggestion; docks will still have to
        ## fill the entire dock area and obey the limits of their internal widgets.
        self.d1 = Dock("Image Panel", size=(400, 400))     ## give this dock the minimum possible size
        self.d2 = Dock("Experiment Parameters", size=(1, 1))
        self.d3 = Dock("Diffraction Geometry", size=(1, 1))
        self.d4 = Dock("ROI Histogram", size=(1, 1))
        self.d5 = Dock("Mouse", size=(400, 75), closable=False)
        self.d6 = Dock("Image Control", size=(1, 1))
        self.d7 = Dock("Image Scroll", size=(1, 1))
        self.d8 = Dock("Small Data", size=(1, 1))
        self.d9 = Dock("Peak Finder", size=(1, 1))
        self.d10 = Dock("Manifold", size=(1, 1))
        self.d12 = Dock("Mask Panel", size=(1, 1))
        self.d13 = Dock("Detector Correction", size=(1, 1))
        self.d14 = Dock("Hit Finder", size=(1, 1))
        self.d15 = Dock("Indexing", size=(1, 1))

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
        self.area.addDock(self.d5, 'left')  ## place d5 at left edge of d1
        self.area.addDock(self.d6, 'bottom', self.d5)    ## place d1 at left edge of dock area
        self.area.addDock(self.d7, 'bottom', self.d5)
        self.area.addDock(self.d1, 'bottom', self.d5)    ## place d1 at left edge of dock area
        self.area.moveDock(self.d1, 'above', self.d7)

        self.area.addDock(self.d2, 'right')     ## place d2 at right edge of dock area
        self.area.addDock(self.d9, 'bottom', self.d2)
        self.area.addDock(self.d12, 'bottom',self.d2)
        self.area.addDock(self.d14, 'bottom', self.d2)
        self.area.addDock(self.d15, 'bottom', self.d2)
        self.area.moveDock(self.d9, 'above', self.d12)
        self.area.moveDock(self.d14, 'above', self.d12)
        self.area.moveDock(self.d15, 'above', self.d12)

        self.area.addDock(self.d3, 'bottom', self.d2)    ## place d3 at bottom edge of d1
        self.area.addDock(self.d4, 'bottom', self.d2)    ## place d4 at right edge of dock area
        self.area.addDock(self.d8, 'bottom', self.d2)
        self.area.moveDock(self.d3, 'above', self.d8)
        self.area.moveDock(self.d4, 'above', self.d8)

        #if args.more:
        #    self.area.addDock(self.d10, 'bottom', self.d6)

        ## Dock 1: Image Panel
        self.w1 = pg.ImageView(view=pg.PlotItem())
        self.w1.getView().invertY(False)

        self.img_feature = pg.ImageItem()
        self.w1.getView().addItem(self.img_feature)

        self.ring_feature = pg.ScatterPlotItem()
        self.peak_feature = pg.ScatterPlotItem()
        self.indexedPeak_feature = pg.ScatterPlotItem()
        self.z_direction = pg.ScatterPlotItem()
        self.z_direction1 = pg.ScatterPlotItem()
        self.w1.getView().addItem(self.ring_feature)
        self.w1.getView().addItem(self.peak_feature)
        self.w1.getView().addItem(self.indexedPeak_feature)
        self.w1.getView().addItem(self.z_direction)
        self.w1.getView().addItem(self.z_direction1)
        self.abc_text = pg.TextItem(html='', anchor=(0,0))
        self.w1.getView().addItem(self.abc_text)

        # Custom ROI for selecting an image region
        self.roi = pg.ROI(pos=[0, -200], size=[100, 100], snapSize=1.0, scaleSnap=True, translateSnap=True, pen={'color': 'g', 'width': 4})
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
                    np.savetxt(str(fname).split('.')[0]+".txt", self.calib.reshape((-1,self.calib.shape[-1])) )#,fmt='%0.18e')
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
        self.w11a = pg.LayoutWidget()
        self.refreshBtn = QtGui.QPushButton('Refresh')
        self.w11a.addWidget(self.refreshBtn, row=0, col=0)
        self.d8.addWidget(self.w11a)
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

        ## Dock 12: Mask Panel
        self.w17 = ParameterTree()
        self.w17.setParameters(self.p6, showTop=False)
        self.d12.addWidget(self.w17)
        self.w18 = pg.LayoutWidget()
        self.maskRectBtn = QtGui.QPushButton('mask rectangular ROI')
        self.w18.addWidget(self.maskRectBtn, row=0, col=0, colspan=2)
        self.maskCircleBtn = QtGui.QPushButton('mask circular ROI')
        self.w18.addWidget(self.maskCircleBtn, row=1, col=0, colspan=2)
        self.deployMaskBtn = QtGui.QPushButton()
        self.deployMaskBtn.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')
        self.deployMaskBtn.setText('Save user-defined mask')
        self.w18.addWidget(self.deployMaskBtn, row=2, col=0)
        self.loadMaskBtn = QtGui.QPushButton()
        self.loadMaskBtn.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')
        self.loadMaskBtn.setText('Load user-defined mask')
        self.w18.addWidget(self.loadMaskBtn, row=2, col=1)
        # Connect listeners to functions
        self.d12.addWidget(self.w18)

        ## Dock 13: Correction Panel

        ## Dock 14: Hit finder
        self.w19 = ParameterTree()
        self.w19.setParameters(self.p8, showTop=False)
        self.d14.addWidget(self.w19)
        self.w20 = pg.LayoutWidget()
        self.launchSpiBtn = QtGui.QPushButton('Launch hit finder')
        self.w20.addWidget(self.launchSpiBtn, row=1, col=0)
        self.d14.addWidget(self.w20)

        ## Dock 15: Indexing
        self.w21 = ParameterTree()
        self.w21.setParameters(self.p9, showTop=False)
        self.w21.setWindowTitle('Indexing')
        self.d15.addWidget(self.w21)
        self.w22 = pg.LayoutWidget()
        self.launchIndexBtn = QtGui.QPushButton('Launch indexing')
        self.w22.addWidget(self.launchIndexBtn, row=0, col=0)
        self.d15.addWidget(self.w22)

        # mask
        def makeMaskRect():
            print "makeMaskRect!!!!!!"
            self.initMask()
            if self.data is not None and self.maskingMode > 0:
                print "makeMaskRect_data: ", self.data.shape
                selected, coord = self.roi_rect.getArrayRegion(self.data, self.w1.getImageItem(), returnMappedCoords=True)
                print "selected, coord: ", np.max(coord[0]), np.max(coord[1]), coord.shape
                # Remove mask elements outside data
                coord_row = coord[0,(coord[0]>=0) & (coord[0]<self.data.shape[0]) & (coord[1]>=0) & (coord[1]<self.data.shape[1])].ravel()
                coord_col = coord[1,(coord[0]>=0) & (coord[0]<self.data.shape[0]) & (coord[1]>=0) & (coord[1]<self.data.shape[1])].ravel()
                _mask = np.ones_like(self.data)
                _mask[coord_row.astype('int'),coord_col.astype('int')] = 0
                if self.maskingMode == 1: # masking mode
                    self.userMaskAssem *= _mask
                elif self.maskingMode == 2: # unmasking mode
                    self.userMaskAssem[coord_row.astype('int'),coord_col.astype('int')] = 1
                elif self.maskingMode == 3: # toggle mode
                    self.userMaskAssem[coord_row.astype('int'),coord_col.astype('int')] = (1-self.userMaskAssem[coord_row.astype('int'),coord_col.astype('int')])

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
            print "*** deploy user-defined mask as mask.txt and mask.npy as DAQ shape ***"
            print "*** deploy user-defined mask as mask_natural_shape.npy as natural shape ***"
            print "userMask: ", self.userMask.shape
            if self.userMask is not None:
                if self.userMask.size==2*185*388: # cspad2x2
                    # DAQ shape
                    asData2x2 = two2x1ToData2x2(self.userMask)
                    np.save("mask.npy",asData2x2)
                    np.savetxt("mask.txt", asData2x2.reshape((-1,asData2x2.shape[-1])) ,fmt='%0.18e')
                    # Natural shape
                    np.save("mask_natural_shape.npy",self.userMask)
                else:
                    np.save("mask.npy",self.userMask)
                    np.savetxt("mask.txt", self.userMask.reshape((-1,self.userMask.shape[-1])) ,fmt='%0.18e')
            else:
                print "user mask is not defined"
        self.connect(self.deployMaskBtn, QtCore.SIGNAL("clicked()"), deployMask)

        def loadMask():
            fname = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file', './', 'ndarray image (*.npy *.npz)'))
            print "fname: ", fname, fname.split('.')[-1]
            self.initMask()
            self.userMask = np.load(fname)
            if self.userMask.shape != self.calib.shape:
                self.userMask = None
            if self.userMask is not None:
                self.userMaskAssem = self.det.image(self.evt,self.userMask)
            else:
                self.userMaskAssem = None
            self.updateClassification()
            self.userMaskOn = True
            self.p6.param(mask_grp,user_mask_str).setValue(self.userMaskOn)
        self.connect(self.loadMaskBtn, QtCore.SIGNAL("clicked()"), loadMask)

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
        # Launch peak finding
        def findPeaks():
            print "find peaks!!!!!!"
            self.thread.append(PeakFinder(self)) # send parent parameters with self
            self.thread[self.threadCounter].findPeaks(self.experimentName,self.runNumber,self.detInfo)
            self.threadCounter+=1
            print "done finding peaks!!!!!!"
        self.connect(self.launchBtn, QtCore.SIGNAL("clicked()"), findPeaks)
        # Launch hit finding
        def findHits():
            print "find hits!!!!!!"
            self.thread.append(HitFinder(self)) # send parent parameters with self
            self.thread[self.threadCounter].findHits(self.experimentName,self.runNumber,self.detInfo)
            self.threadCounter+=1
            print "done finding hits!!!!!!"
        self.connect(self.launchSpiBtn, QtCore.SIGNAL("clicked()"), findHits)

        # Launch indexing
        def digestRunList(runList):
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

        def indexPeaks():
            print "index.runs: ", self.index.runs
            runsToDo = digestRunList(self.index.runs)
            print "runsToDo: ", runsToDo
            for run in runsToDo:
                print "index peaks!!!!!!"
                self.thread.append(self.index) # send parent parameters with self
                print "self.index: ", self.index.outDir, self.index.intRadius
                self.thread[self.threadCounter].launchIndexing(run)
                self.threadCounter+=1
                print "done indexing peaks!!!!!!"
        self.connect(self.launchIndexBtn, QtCore.SIGNAL("clicked()"), indexPeaks)
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
        if self.streakMask is None:
            self.StreakMask = myskbeam.StreakMask(self.det, self.evt, width=self.streak_width, sigma=self.streak_sigma)
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

        # Lab coordinates: Add z-direction
        self.z_direction.setData([0+cenX], [0+cenY], symbol='o', \
                                 size=symbolSize, brush='w', \
                                 pen={'color': 'k', 'width': 4}, pxMode=False)
        self.z_direction1.setData([0+cenX], [0+cenY], symbol='o', \
                                 size=symbolSize/6, brush='k', \
                                 pen={'color': 'k', 'width': 4}, pxMode=False)
        # Lab coordinates: Add xyz text
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
            self.streakMask = self.StreakMask.getStreakMaskCalib(self.evt)
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
                    self.alg.set_peak_selection_pars(npix_min=self.hitParam_alg1_npix_min, npix_max=self.hitParam_alg1_npix_max, \
                                            amax_thr=self.hitParam_alg1_amax_thr, atot_thr=self.hitParam_alg1_atot_thr, \
                                            son_min=self.hitParam_alg1_son_min)
                elif self.algorithm == 2:
                    self.alg.set_peak_selection_pars(npix_min=self.hitParam_alg2_npix_min, npix_max=self.hitParam_alg2_npix_max, \
                                            amax_thr=self.hitParam_alg2_amax_thr, atot_thr=self.hitParam_alg2_atot_thr, \
                                            son_min=self.hitParam_alg2_son_min)
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
            elif self.algorithm == 2:
                # v2 - define peaks for regions of connected pixels above threshold
                self.peakRadius = int(self.hitParam_alg2_r0)
                self.peaks = self.alg.peak_finder_v2(self.calib, thr=self.hitParam_alg2_thr, r0=self.peakRadius, dr=self.hitParam_alg2_dr)
            elif self.algorithm == 3:
                self.peakRadius = int(self.hitParam_alg3_r0)
                self.peaks = self.alg.peak_finder_v3(self.calib, rank=self.hitParam_alg3_rank, r0=self.peakRadius, dr=self.hitParam_alg3_dr)
            elif self.algorithm == 4:
                print("* Running peak finder 4 *")
                # v4 - aka Droplet Finder - the same as v1, but uses rank and r0 parameters in stead of common radius.
                self.peakRadius = int(self.hitParam_alg4_r0)
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
            if self.showIndexedPeaks:
                self.clearIndexedPeaks()


                print "########################## Save peak finding"
                #self.index.psana
                #saveCxiLite(filename,expName,runNumber,detInfo,eventList)
                maxNumPeaks = 2048
                myHdf5 = h5py.File(self.hiddenCXI, 'w')
                grpName = "/entry_1/result_1"
                dset_nPeaks = "/nPeaks"
                dset_posX = "/peakXPosRaw"
                dset_posY = "/peakYPosRaw"
                dset_atot = "/peakTotalIntensity"
                if grpName in myHdf5:
                    del myHdf5[grpName]
                grp = myHdf5.create_group(grpName)
                myHdf5.create_dataset(grpName+dset_nPeaks, (1,), dtype='int')
                myHdf5.create_dataset(grpName+dset_posX, (1,maxNumPeaks), dtype='float32', chunks=(1,maxNumPeaks))
                myHdf5.create_dataset(grpName+dset_posY, (1,maxNumPeaks), dtype='float32', chunks=(1,maxNumPeaks))
                myHdf5.create_dataset(grpName+dset_atot, (1,maxNumPeaks), dtype='float32', chunks=(1,maxNumPeaks))

                myHdf5.create_dataset("/LCLS/detector_1/EncoderValue", (1,), dtype=float)
                myHdf5.create_dataset("/LCLS/photon_energy_eV", (1,), dtype=float)
                dim0 = 8*185
                dim1 = 4*388
                dset = myHdf5.create_dataset("/entry_1/data_1/data",(1,dim0,dim1),dtype=float)

                # Convert calib image to cheetah image
                img = np.zeros((dim0, dim1))
                counter = 0
                for quad in range(4):
                    for seg in range(8):
                        img[seg*185:(seg+1)*185,quad*388:(quad+1)*388] = self.calib[counter,:,:]
                        counter += 1

                peaks = self.peaks.copy()
                nPeaks = peaks.shape[0]
                print "Number of peaks found: ", nPeaks

                if nPeaks > maxNumPeaks:
                    peaks = peaks[:maxNumPeaks]
                    nPeaks = maxNumPeaks
                for i,peak in enumerate(peaks):
                    seg,row,col,npix,amax,atot,rcent,ccent,rsigma,csigma,rmin,rmax,cmin,cmax,bkgd,rms,son = peak[0:17]
                    cheetahRow,cheetahCol = self.convert_peaks_to_cheetah(seg,row,col)
                    myHdf5[grpName+dset_posX][0,i] = cheetahCol
                    myHdf5[grpName+dset_posY][0,i] = cheetahRow
                    myHdf5[grpName+dset_atot][0,i] = atot
                myHdf5[grpName+dset_nPeaks][0] = nPeaks

                if 'cspad' in self.detInfo.lower():
                    myHdf5["/LCLS/detector_1/EncoderValue"][0] = self.clen #-419.9938 # FIXME
                myHdf5["/LCLS/photon_energy_eV"][0] = self.photonEnergy# 8203.9019 #FIXME
                dset[0,:,:] = img
                myHdf5.close()
                print "########################## Done save peak finding"

                self.index.updateIndex()

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
                                          pen=pg.mkPen({'color': "c", 'width': 4}), pxMode=False) #FF0
                print "number of peaks drawn: ", len(cenX)
            else:
                self.peak_feature.setData([], [], pxMode=False)
        else:
            self.peak_feature.setData([], [], pxMode=False)
        print "Done updatePeaks"

    def clearIndexedPeaks(self):
        self.w1.getView().removeItem(self.abc_text)
        self.indexedPeak_feature.setData([], [], pxMode=False)
        print "Done clearIndexedPeaks"

    def drawIndexedPeaks(self,unitCell=None):
        self.clearIndexedPeaks()
        if self.showIndexedPeaks:
            if self.indexedPeaks is not None and self.numIndexedPeaksFound > 0:
                cenX = self.indexedPeaks[:,0]+0.5
                cenY = self.indexedPeaks[:,1]+0.5
                cenX = np.concatenate((cenX,cenX,cenX))
                cenY = np.concatenate((cenY,cenY,cenY))
                diameter = np.ones_like(cenX)
                diameter[0:self.numIndexedPeaksFound] = float(self.index.intRadius.split(',')[0])*2
                diameter[self.numIndexedPeaksFound:2*self.numIndexedPeaksFound] = float(self.index.intRadius.split(',')[1])*2
                diameter[2*self.numIndexedPeaksFound:3*self.numIndexedPeaksFound] = float(self.index.intRadius.split(',')[2])*2
                print "cenX: ", cenX
                print "cenY: ", cenY
                print "diameter: ", diameter#, self.peakRadius
                self.indexedPeak_feature.setData(cenX, cenY, symbol='o', \
                                          size=diameter, brush=(255,255,255,0), \
                                          pen=pg.mkPen({'color': "#FF00FF", 'width': 3}), pxMode=False)
                print "number of peaks drawn: ", len(cenX)

                # Write unit cell parameters
                xMargin = 30 # pixels
                maxX   = np.max(self.det.indexes_x(self.evt))+xMargin
                maxY   = np.max(self.det.indexes_y(self.evt))
                print "maxX, maxY: ", maxX, maxY, unitCell
                myMessage = '<div style="text-align: center"><span style="color: #FF00FF; font-size: 16pt;">a='+\
                            str(round(float(unitCell[0]),2))+'nm <br>b='+str(round(float(unitCell[1]),2))+'nm <br>c='+\
                            str(round(float(unitCell[2]),2))+'nm <br>&alpha;='+str(round(float(unitCell[3]),2))+\
                            '&deg; <br>&beta;='+str(round(float(unitCell[4]),2))+'&deg; <br>&gamma;='+\
                            str(round(float(unitCell[5]),2))+'&deg; <br></span></div>'

                self.abc_text = pg.TextItem(html=myMessage, anchor=(0,0))
                self.w1.getView().addItem(self.abc_text)
                self.abc_text.setPos(maxX, maxY)
            else:
                xMargin = 30 # pixels
                maxX   = np.max(self.det.indexes_x(self.evt))+xMargin
                maxY   = np.max(self.det.indexes_y(self.evt))
                # Draw a big X
                cenX = np.array((self.cx,))+0.5
                cenY = np.array((self.cy,))+0.5
                diameter = 256 #self.peakRadius*2+1
                self.indexedPeak_feature.setData(cenX, cenY, symbol='x', \
                                          size=diameter, brush=(255,255,255,0), \
                                          pen=pg.mkPen({'color': "#FF00FF", 'width': 3}), pxMode=False)
                #self.w1.getView().removeItem(self.abc_text)

                self.abc_text = pg.TextItem(html='', anchor=(0,0))
                self.w1.getView().addItem(self.abc_text)
                self.abc_text.setPos(maxX,maxY)
        else:
            self.indexedPeak_feature.setData([], [], pxMode=False)
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
                    self.w1.setImage(self.data,levels=(0,np.percentile(self.data,self.displayMaxPercentile)))
                    self.firstUpdate = False
            else:
                if self.logscaleOn:
                    print "################################# 1"
                    self.w1.setImage(np.log10(abs(self.data)+eps),autoRange=False,autoLevels=False,autoHistogramRange=False)
                else:
                    print "################################# 2"
                    self.w1.setImage(self.data,autoRange=False,autoLevels=False,autoHistogramRange=False)
        print "Done updateImage"

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

    def getCommonMode(self,evtNumber):
        if self.run is not None:
            self.evt = self.getEvt(evtNumber)
            pedestalCorrected = self.det.raw(self.evt)-self.det.pedestals(self.evt)
            if self.applyCommonMode: # play with different common mode
                print "### Overriding common mode: ", self.commonMode
                if self.commonMode[0] == 5: # Algorithm 5
                    cm = self.det.common_mode_correction(self.evt, pedestalCorrected, cmpars=(self.commonMode[0],self.commonMode[1]))
                else: # Algorithms 1 to 4
                    cm = self.det.common_mode_correction(self.evt, pedestalCorrected, cmpars=(self.commonMode[0],self.commonMode[1],self.commonMode[2],self.commonMode[3]))
            else:
                cm = self.det.common_mode_correction(self.evt, pedestalCorrected)
            return cm
        else:
            return None

    def getAssembledImage(self,calib):
        _calib = calib.copy() # this is important
        # Do not display ADUs below threshold
        if self.image_property == 1:
            _calib[np.where(_calib<self.aduThresh)]=0
        tic = time.time()
        data = self.det.image(self.evt, _calib)
        if data is None:
            data = _calib
        toc = time.time()
        print "time assemble: ", toc-tic
        return data

    def getDetImage(self,evtNumber,calib=None):
        if calib is None:
            if self.image_property == 1: # gain corrected
                calib = self.getCalib(evtNumber) * self.det.gain(self.evt)
            elif self.image_property == 2: # common mode corrected
                print "$$$ image property: 2"
                calib = self.getCalib(evtNumber)
            elif self.image_property == 3: # pedestal corrected
                calib = self.det.raw(self.evt) - self.det.pedestals(self.evt)
            elif self.image_property == 4: # raw
                calib = self.det.raw(self.evt)
            elif self.image_property == 5: # photon counts
                print "Sorry, this feature is not available"
            elif self.image_property == 6: # pedestal
                calib = self.det.pedestals(self.evt)
            elif self.image_property == 7: # status
                calib = self.det.status(self.evt)
            elif self.image_property == 8: # rms
                calib = self.det.rms(self.evt)
            elif self.image_property == 9: # common mode
                calib = self.getCommonMode(evtNumber)
            elif self.image_property == 10: # gain
                calib = self.det.gain(self.evt)
            elif self.image_property == 15: # coords_x
                calib = self.det.coords_x(self.evt)
            elif self.image_property == 16: # coords_y
                calib = self.det.coords_y(self.evt)

            shape = self.det.shape(self.evt)
            if len(shape) == 3:
                if self.image_property == 11: # quad ind
                    calib = np.zeros(shape)
                    for i in range(shape[0]):
                        # TODO: handle detectors properly
                        if shape[0] == 32: # cspad
                            calib[i,:,:] = int(i)%8
                        elif shape[0] == 2: # cspad2x2
                            calib[i,:,:] = int(i)%2
                        elif shape[0] == 4: # pnccd
                            calib[i,:,:] = int(i)%4
                elif self.image_property == 12: # seg ind
                    calib = np.zeros(shape)
                    if shape[0] == 32: # cspad
                        for i in range(32):
                            calib[i,:,:] = int(i)/8
                    elif shape[0] == 2: # cspad2x2
                        for i in range(2):
                            calib[i,:,:] = int(i)
                    elif shape[0] == 4: # pnccd
                        for i in range(4):
                            calib[i,:,:] = int(i)
                elif self.image_property == 13: # row ind
                    calib = np.zeros(shape)
                    if shape[0] == 32: # cspad
                        for i in range(185):
                            calib[:,i,:] = i
                    elif shape[0] == 2: # cspad2x2
                        for i in range(185):
                            calib[:,i,:] = i
                    elif shape[0] == 4: # pnccd
                        for i in range(512):
                            calib[:,i,:] = i
                elif self.image_property == 14: # col ind
                    calib = np.zeros(shape)
                    if shape[0] == 32: # cspad
                        for i in range(388):
                            calib[:,:,i] = i
                    elif shape[0] == 2: # cspad2x2
                        for i in range(388):
                            calib[:,:,i] = i
                    elif shape[0] == 4: # pnccd
                        for i in range(512):
                            calib[:,:,i] = i
            else:
                print "psocake can't handle this detector"

        # Update photon energy
        self.ebeam = self.evt.get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
        self.photonEnergy = self.ebeam.ebeamPhotonEnergy()
        self.p1.param(self.geom.geom_grp,self.geom.geom_photonEnergy_str).setValue(self.photonEnergy)

        if calib is not None:
            # assemble image
            data = self.getAssembledImage(calib)
            self.cx, self.cy = self.det.point_indexes(self.evt,pxy_um=(0,0))
            return calib, data
        else: # TODO: this is a hack that assumes opal is the only detector without calib
            # we have an opal / epix
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
    def change(self, panel, changes):
        for param, change, data in changes:
            path = panel.childPath(param)
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

            elif path[2] == hitParam_alg1_npix_min_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg1_npix_min = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg1_npix_max_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg1_npix_max = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg1_amax_thr_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg1_amax_thr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg1_atot_thr_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg1_atot_thr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg1_son_min_str and path[1] == hitParam_algorithm1_str:
                self.hitParam_alg1_son_min = data
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
            elif path[2] == hitParam_alg2_npix_min_str and path[1] == hitParam_algorithm2_str:
                self.hitParam_alg2_npix_min = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg2_npix_max_str and path[1] == hitParam_algorithm2_str:
                self.hitParam_alg2_npix_max = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg2_amax_thr_str and path[1] == hitParam_algorithm2_str:
                self.hitParam_alg2_amax_thr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg2_atot_thr_str and path[1] == hitParam_algorithm2_str:
                self.hitParam_alg2_atot_thr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg2_son_min_str and path[1] == hitParam_algorithm2_str:
                self.hitParam_alg2_son_min = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg2_thr_str and path[1] == hitParam_algorithm2_str:
                self.hitParam_alg2_thr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg2_r0_str and path[1] == hitParam_algorithm2_str:
                self.hitParam_alg2_r0 = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg2_dr_str and path[1] == hitParam_algorithm2_str:
                self.hitParam_alg2_dr = data
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
                self.hitParam_alg4_amax_thr = data
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
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg4_rank_str and path[1] == hitParam_algorithm4_str:
                self.hitParam_alg4_rank = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg4_r0_str and path[1] == hitParam_algorithm4_str:
                self.hitParam_alg4_r0 = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
            elif path[2] == hitParam_alg4_dr_str and path[1] == hitParam_algorithm4_str:
                self.hitParam_alg4_dr = data
                self.algInitDone = False
                if self.showPeaks:
                    self.updateClassification()
        ################################################
        # hit finder parameters
        ################################################
        if path[0] == spiParam_grp:
            if path[1] == spiParam_algorithm_str:
                #self.algInitDone = False
                #self.updateAlgorithm(data)
                self.spiAlgorithm = data
            elif path[1] == spiParam_outDir_str:
                self.spiParam_outDir = data
            elif path[1] == spiParam_tag_str:
                self.spiParam_tag = data
            elif path[1] == spiParam_runs_str:
                self.spiParam_runs = data
            elif path[1] == spiParam_queue_str:
                self.spiParam_queue = data
            elif path[1] == spiParam_cpu_str:
                self.spiParam_cpus = data
            elif path[1] == spiParam_noe_str:
                self.spiParam_noe = data
            elif path[2] == spiParam_alg1_pruneInterval_str and path[1] == spiParam_algorithm1_str:
                self.spiParam_alg1_pruneInterval = data
            elif path[2] == spiParam_alg2_threshold_str and path[1] == spiParam_algorithm2_str:
                self.spiParam_alg2_threshold = data
        ################################################
        # diffraction geometry parameters
        ################################################
        if path[0] == self.geom.geom_grp:
            self.geom.paramUpdate(path, change, data)
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
        ################################################
        # crystal indexing parameters
        ################################################
        if path[0] == self.index.index_grp:
            self.index.paramUpdate(path, change, data)
        elif path[0] == self.index.launch_grp:
            self.index.paramUpdate(path, change, data)

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
        if data == 0:
            self.runNumber = data
            self.hasRunNumber = False
        else:
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
        self.StreakMask = None
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
            # Check such a run exists
            import glob
            xtcs = glob.glob('/reg/d/psdm/'+self.experimentName[0:3]+'/'+self.experimentName+'/xtc/*-r'+str(self.runNumber).zfill(4)+'-*.xtc')
            #print "xtcs found: ", xtcs
            if len(xtcs) > 0:
                print "hasExpRunInfo: True"
                return True
            else:
                # reset run number
                if self.runNumber > 0:
                    print "No such run exists in: ", self.experimentName
                    self.runNumber = 0
                    self.updateRunNumber(self.runNumber)
                    self.p.param(exp_grp,exp_run_str).setValue(self.runNumber)
                    return False
        print "hasExpRunInfo: False"
        return False

    def hasExpRunDetInfo(self):
        if self.hasExperimentName and self.hasRunNumber and self.hasDetInfo:
            print "hasExpRunDetInfo: True ", self.runNumber
            return True
        else:
            print "hasExpRunDetInfo: False"
            return False

    def getUsername(self):
            process = subprocess.Popen('whoami', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out,err=process.communicate()
            self.username = out.strip()

    def setupPsocake(self):
        self.getUsername()
        self.loggerFile = self.psocakeDir+'/logger.data'
        if os.path.exists(self.psocakeDir) is False:
            os.mkdir(self.psocakeDir, 0774)
            # setup permissions
            process = subprocess.Popen('chgrp -R '+self.experimentName+' '+self.psocakeDir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out,err=process.communicate()
            process = subprocess.Popen('chmod -R u+rwx,g+rws,o+rx '+self.psocakeDir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out,err=process.communicate()
            #import stat
            #os.chmod(self.psocakeDir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_ISGID | stat.S_IROTH)
            # create logger
            with open(self.loggerFile,"w") as myfile:
                myfile.write(self.username)
        else:
            # check if I'm a logger
            with open(self.loggerFile,"r") as myfile:
                content = myfile.readlines()
                if content[0].strip() == self.username:
                    self.logger = True
                    print "I'm a logger"
                else:
                    self.logger = False
                    print "I'm not a logger"

    def setupExperiment(self):
        if self.hasExpRunInfo():
            # Set up psocake directory in scratch
            self.psocakeDir = '/reg/d/psdm/'+self.experimentName[:3]+'/'+self.experimentName+'/scratch/psocake'
            self.setupPsocake()

            if args.localCalib:
                print "Using local calib directory"
                psana.setOption('psana.calib-dir','./calib')
            try:
                print "Let's get Datasource"
                self.ds = psana.DataSource('exp='+str(self.experimentName)+':run='+str(self.runNumber)+':idx') # FIXME: psana crashes if runNumber is non-existent
            except:
                print "############# No such datasource exists ###############"
            self.run = self.ds.runs().next()
            self.times = self.run.times()
            self.eventTotal = len(self.times)
            self.spinBox.setMaximum(self.eventTotal-self.stackSize)
            self.p.param(exp_grp,exp_evt_str).setLimits((0,self.eventTotal-1))
            self.p.param(exp_grp,exp_evt_str,exp_numEvents_str).setValue(self.eventTotal)
            self.env = self.ds.env()

            self.evt = self.run.event(self.times[0])
            myAreaDetectors = []
            for k in self.evt.keys():
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
            self.epics = self.ds.env().epicsStore()
            # detector distance
            if 'cspad' in self.detInfo.lower():
                self.clenEpics = str(self.detInfo)+'_z'
                print "clenEpics: ", self.clenEpics
                self.clen = self.epics.value(self.clenEpics)
                print "clen: ", self.detectorDistance, self.clen
                self.coffset = self.detectorDistance - self.clen

            if 'cspad' in self.detInfo.lower(): # FIXME: increase pixel size list
                self.pixelSize = 110e-6
            elif 'pnccd' in self.detInfo.lower():
                self.pixelSize = 75e-6
            self.p1.param(self.geom.geom_grp,self.geom.geom_pixelSize_str).setValue(self.pixelSize)
            # photon energy
            self.ebeam = self.evt.get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
            self.photonEnergy = self.ebeam.ebeamPhotonEnergy()
            self.p1.param(self.geom.geom_grp,self.geom.geom_photonEnergy_str).setValue(self.photonEnergy)

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

    def updateDock42(self, data):
        a = ['a','b','c','d','e','k','m','n','r','s']
        myStr = a[5]+a[8]+a[0]+a[5]+a[4]+a[7]
        if myStr in data:
            self.d42 = Dock("Console", size=(100,100))
            # build an initial namespace for console commands to be executed in (this is optional;
            # the user can always import these modules manually)
            namespace = {'pg': pg, 'np': np, 'self': self}
            # initial text to display in the console
            text = "You have awoken the "+myStr+"\nWelcome to psocake IPython: dir(self)\n" \
                                                "Here are some commonly used variables:\n" \
                                                "unassembled detector: self.calib\n" \
                                                "assembled detector: self.data\n" \
                                                "user-defined mask: self.userMask\n" \
                                                "streak mask: self.streakMask\n" \
                                                "psana mask: self.psanaMask"
            self.w42 = pg.console.ConsoleWidget(parent=None,namespace=namespace, text=text)
            self.d42.addWidget(self.w42)
            self.area.addDock(self.d42, 'bottom')

    def updateCommonModeParam(self, data, ind):
        self.commonModeParams[ind] = data
        self.updateCommonMode(self.applyCommonMode)
        print "Done updateCommonModeParam: ", self.commonModeParams

    def updateCommonMode(self, data):
        self.applyCommonMode = data
        if self.applyCommonMode:
            self.commonMode = self.checkCommonMode(self.commonModeParams)
        if self.hasExpRunDetInfo():
            print "%%% Redraw image with new common mode: ", self.commonMode
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
        elif _alg == 5:
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
        self.streakMask = None
        self.initMask()
        self.updateClassification()
        print "Done updateStreakWidth: ", self.streak_width

    def updateStreakSigma(self, data):
        self.streak_sigma = data
        self.streakMask = None
        self.initMask()
        self.updateClassification()
        print "Done updateStreakSigma: ", self.streak_sigma

    def updatePsanaMask(self, data):
        self.psanaMaskOn = data
        self.updatePsanaMaskOn()
        print "Done updatePsanaMask: ", self.psanaMaskOn

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

            try:
                if self.quantifier_dataset[0] == '/': # dataset starts with "/"
                    self.quantifier_eventDataset = self.quantifier_dataset.split("/")[1] + "/event"
                else: # dataset does not start with "/"
                    self.quantifier_eventDataset = "/" + self.quantifier_dataset.split("/")[0] + "/event"
                self.quantifierEvent = self.quantifierFile[self.quantifier_eventDataset].value
            except:
                print "Couldn't find /event dataset"
                self.quantifierEvent = np.arange(len(self.quantifierMetric))

            print "Done reading metric"

    def updateQuantifierSort(self, data):
        self.quantifier_sort = data
        if self.quantifierHasData:
            if self.quantifier_sort is True:
                self.quantifierInd = np.argsort(self.quantifierFile[self.quantifier_dataset].value)
                self.quantifierMetric = self.quantifierFile[self.quantifier_dataset].value[self.quantifierInd]
                self.updateQuantifierPlot(self.quantifierInd,self.quantifierMetric)
            else:
                self.quantifierMetric = self.quantifierFile[self.quantifier_dataset].value
                self.quantifierInd = np.arange(len(self.quantifierMetric))
#                self.quantifierEvent = self.quantifierFile[self.quantifier_eventDataset].value
                self.updateQuantifierPlot(self.quantifierInd,self.quantifierMetric)
        print "metric: ", self.quantifierMetric
        print "ind: ", self.quantifierInd
        print "event: ", self.quantifierEvent

    def updateQuantifierPlot(self,ind,metric):
        self.w9.getPlotItem().clear()
        self.curve = self.w9.plot(metric, pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')
        self.w9.setLabel('left', "Small data")
        if self.quantifier_sort:
            self.w9.setLabel('bottom', "Sorted Event Index")
        else:
            self.w9.setLabel('bottom', "Event Index")
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
        print "%%%% event number: ", self.eventNumber
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
                self.roi_rect = pg.ROI(pos=[-300,0], size=[200, 200], snapSize=1.0, scaleSnap=True, translateSnap=True, pen={'color': 'c', 'width': 4})
                self.roi_rect.addScaleHandle([0.5, 1], [0.5, 0.5])
                self.roi_rect.addScaleHandle([0, 0.5], [0.5, 0.5])
                self.roi_rect.addRotateHandle([0.5, 0.5], [1, 1])
                # Circular mask
                self.roi_circle = pg.CircleROI([-300,300], size=[200, 200], snapSize=1.0, scaleSnap=True, translateSnap=True, pen={'color': 'c', 'width': 4})

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

class ABC(object):
    def __init__(self, parent = None):
        self.parent = parent
        self.t = 5

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
                  " -o .%J.log generatePowder exp="+self.experimentName+\
                  ":run="+str(run)+" -d "+self.detInfo+\
                  " -o "+str(self.parent.hitParam_outDir)
            if self.parent.hitParam_noe > 0:
                cmd += " -n "+str(self.parent.hitParam_noe)
            print "Submitting batch job: ", cmd
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            jobid = out.split("<")[1].split(">")[0]
            myLog = "."+jobid+".log"
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
              " -o .%J.log python /reg/neh/home/yoon82/ana-current/psocake/src/findPeaks.py -e "+self.experimentName+\
              " -r "+str(run)+" -d "+self.detInfo+\
              " --outDir "+str(self.parent.hitParam_outDir)+\
              " --algorithm "+str(self.parent.algorithm)

            if self.parent.algorithm == 1:
                cmd += " --alg_npix_min "+str(self.parent.hitParam_alg1_npix_min)+\
                       " --alg_npix_max "+str(self.parent.hitParam_alg1_npix_max)+\
                       " --alg_amax_thr "+str(self.parent.hitParam_alg1_amax_thr)+\
                       " --alg_atot_thr "+str(self.parent.hitParam_alg1_atot_thr)+\
                       " --alg_son_min "+str(self.parent.hitParam_alg1_son_min)+\
                       " --alg1_thr_low "+str(self.parent.hitParam_alg1_thr_low)+\
                       " --alg1_thr_high "+str(self.parent.hitParam_alg1_thr_high)+\
                       " --alg1_radius "+str(self.parent.hitParam_alg1_radius)+\
                       " --alg1_dr "+str(self.parent.hitParam_alg1_dr)
            elif self.parent.algorithm == 3:
                cmd += " --alg_npix_min "+str(self.parent.hitParam_alg3_npix_min)+\
                       " --alg_npix_max "+str(self.parent.hitParam_alg3_npix_max)+\
                       " --alg_amax_thr "+str(self.parent.hitParam_alg3_amax_thr)+\
                       " --alg_atot_thr "+str(self.parent.hitParam_alg3_atot_thr)+\
                       " --alg_son_min "+str(self.parent.hitParam_alg3_son_min)+\
                       " --alg3_rank "+str(self.parent.hitParam_alg3_rank)+\
                       " --alg3_r0 "+str(self.parent.hitParam_alg3_r0)+\
                       " --alg3_dr "+str(self.parent.hitParam_alg3_dr)
            elif self.parent.algorithm == 4:
                cmd += " --alg_npix_min "+str(self.parent.hitParam_alg4_npix_min)+\
                       " --alg_npix_max "+str(self.parent.hitParam_alg4_npix_max)+\
                       " --alg_amax_thr "+str(self.parent.hitParam_alg4_amax_thr)+\
                       " --alg_atot_thr "+str(self.parent.hitParam_alg4_atot_thr)+\
                       " --alg_son_min "+str(self.parent.hitParam_alg4_son_min)+\
                       " --alg4_thr_low "+str(self.parent.hitParam_alg4_thr_low)+\
                       " --alg4_thr_high "+str(self.parent.hitParam_alg4_thr_high)+\
                       " --alg4_rank "+str(self.parent.hitParam_alg4_rank)+\
                       " --alg4_r0 "+str(self.parent.hitParam_alg4_r0)+\
                       " --alg4_dr "+str(self.parent.hitParam_alg4_dr)
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
            myLog = "."+jobid+".log"
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

class HitFinder(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        print "HitFinder!!!!!!!!!!"
        self.parent = parent
        self.experimentName = None
        self.runNumber = None
        self.detInfo = None

    def __del__(self):
        print "del PeakFinder #$!@#$!#"
        self.exiting = True
        self.wait()

    def findHits(self,experimentName,runNumber,detInfo): # Pass in hit parameters
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
        print "Finding hits!!!!!!!!!!!!"
        # Digest the run list
        runsToDo = self.digestRunList(self.parent.spiParam_runs)
        print runsToDo

        for run in runsToDo:
            expRun = 'exp='+self.experimentName+':run='+str(run)
            cmd = "bsub -q "+self.parent.spiParam_queue+\
              " -a mympi -n "+str(self.parent.spiParam_cpus)+\
              " -o .%J.log litPixel_HitMetric"+\
              " "+expRun+\
              " -d "+self.detInfo+\
              " --outdir "+str(self.parent.spiParam_outDir)

            if self.parent.spiParam_tag is not None:
                cmd += " --tag "+str(self.parent.spiParam_tag)

            if self.parent.spiAlgorithm == 1:
                cmd += " --pruneInterval "+str(int(self.parent.spiParam_alg1_pruneInterval))
            elif self.parent.spiAlgorithm == 2:
                cmd += " --litPixelThreshold "+str(int(self.parent.spiParam_alg2_threshold))

            # Save user mask to a deterministic path
            if self.parent.userMaskOn:
                tempFilename = "tempUserMask.npy"
                np.save(tempFilename,self.parent.userMask) # TODO: save
                cmd += " --mask "+str(tempFilename)

            if self.parent.spiParam_noe > 0:
                cmd += " --noe "+str(self.parent.spiParam_noe)
            print "Submitting batch job: ", cmd
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            jobid = out.split("<")[1].split(">")[0]
            myLog = "."+jobid+".log"
            print "*******************"
            print "bsub log filename: ", myLog

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
              " -o .%J.log python /reg/neh/home/yoon82/ana-current/psocake/src/findPeaks.py -e "+self.experimentName+\
              " -r "+str(run)+" -d "+self.detInfo+\
              " --outDir "+str(self.parent.hitParam_outDir)+\
              " --algorithm "+str(self.parent.algorithm)

            if self.parent.algorithm == 1:
                cmd += " --alg_npix_min "+str(self.parent.hitParam_alg1_npix_min)+\
                       " --alg_npix_max "+str(self.parent.hitParam_alg1_npix_max)+\
                       " --alg_amax_thr "+str(self.parent.hitParam_alg1_amax_thr)+\
                       " --alg_atot_thr "+str(self.parent.hitParam_alg1_atot_thr)+\
                       " --alg_son_min "+str(self.parent.hitParam_alg1_son_min)+\
                       " --alg1_thr_low "+str(self.parent.hitParam_alg1_thr_low)+\
                       " --alg1_thr_high "+str(self.parent.hitParam_alg1_thr_high)+\
                       " --alg1_radius "+str(self.parent.hitParam_alg1_radius)+\
                       " --alg1_dr "+str(self.parent.hitParam_alg1_dr)
            elif self.parent.algorithm == 3:
                cmd += " --alg_npix_min "+str(self.parent.hitParam_alg3_npix_min)+\
                       " --alg_npix_max "+str(self.parent.hitParam_alg3_npix_max)+\
                       " --alg_amax_thr "+str(self.parent.hitParam_alg3_amax_thr)+\
                       " --alg_atot_thr "+str(self.parent.hitParam_alg3_atot_thr)+\
                       " --alg_son_min "+str(self.parent.hitParam_alg3_son_min)+\
                       " --alg3_rank "+str(self.parent.hitParam_alg3_rank)+\
                       " --alg3_r0 "+str(self.parent.hitParam_alg3_r0)+\
                       " --alg3_dr "+str(self.parent.hitParam_alg3_dr)
            elif self.parent.algorithm == 4:
                cmd += " --alg_npix_min "+str(self.parent.hitParam_alg4_npix_min)+\
                       " --alg_npix_max "+str(self.parent.hitParam_alg4_npix_max)+\
                       " --alg_amax_thr "+str(self.parent.hitParam_alg4_amax_thr)+\
                       " --alg_atot_thr "+str(self.parent.hitParam_alg4_atot_thr)+\
                       " --alg_son_min "+str(self.parent.hitParam_alg4_son_min)+\
                       " --alg4_thr_low "+str(self.parent.hitParam_alg4_thr_low)+\
                       " --alg4_thr_high "+str(self.parent.hitParam_alg4_thr_high)+\
                       " --alg4_rank "+str(self.parent.hitParam_alg4_rank)+\
                       " --alg4_r0 "+str(self.parent.hitParam_alg4_r0)+\
                       " --alg4_dr "+str(self.parent.hitParam_alg4_dr)
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
            myLog = "."+jobid+".log"
            print "*******************"

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtGui.QApplication(sys.argv)
    ex = MainFrame(sys.argv)
    sys.exit(app.exec_())

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    main()

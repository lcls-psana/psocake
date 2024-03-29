from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.dockarea import *
import pyqtgraph as pg
import numpy as np
import json
from psocake import LaunchHitFinder
from psocake import LaunchHitConverter
import psocake.HitFinder as alg

def writeStatus(fname, d):
    json.dump(d, open(fname, 'w'))

class HitFinder(object):
    def __init__(self, parent = None):
        self.parent = parent

        ## Dock: Hit finder
        self.dock = Dock("Hit Finder", size=(1, 1))
        self.win = ParameterTree()
        self.dock.addWidget(self.win)
        self.winL = pg.LayoutWidget()
        self.dock.addWidget(self.winL)

        # Hit finding
        self.spiParam_grp = 'Hit finder'
        self.spiParam_algorithm_str = 'Algorithm'
        # algorithm 0
        self.spiParam_algorithm0_str = 'None'
        # algorithm 1
        self.spiParam_algorithm1_str = 'chiSquared'
        self.spiParam_alg1_pruneInterval_str = 'prune interval'
        # algorithm 2
        self.spiParam_algorithm2_str = 'photonFinder'
        self.spiParam_alg2_threshold_str = 'ADUs per photon'
        self.spiParam_alg2_hitThreshold_str = 'Number of pixels for a hit'

        self.spiParam_outDir_str = 'Output directory'
        self.spiParam_runs_str = 'Run(s)'
        self.spiParam_queue_str = 'queue'
        self.spiParam_cpu_str = 'CPUs'
        self.spiParam_noe_str = 'Number of events to process'
        self.spiParam_launch_str = 'Launch hit finder'

        self.hitParam_grp = 'Find hits'
        self.hitParam_hitThresh_str = 'Hit threshold'
        self.hitParam_backgroundThresh_str = 'Background threshold'
        self.hitParam_sample_str = 'Sample name'
        self.hitParam_save_str = 'Save hits'
        self.tag_str = '.cxi tag'

        # Init hit finding
        self.nPixels = 0
        self.spiAlgorithm = 2
        self.spiParam_alg1_pruneInterval = -1
        self.spiParam_alg2_threshold = 30
        self.spiParam_outDir = self.parent.psocakeDir
        self.spiParam_outDir_overridden = False
        self.spiParam_runs = ''
        self.spiParam_queue = self.parent.pk.hitParam_psanaq_str
        self.spiParam_cpus = 24
        self.spiParam_noe = -1
        self.hitParam_hitThresh = '-1'
        self.hitParam_backgroundThresh = '-1'
        self.hitParam_sample = "sample"
        self.tag = ''


        self.params = [
            {'name': self.spiParam_grp, 'type': 'group', 'children': [
                {'name': self.spiParam_algorithm_str, 'type': 'list', 'values': {self.spiParam_algorithm2_str: 2,
                                                                                 self.spiParam_algorithm0_str: 0},
                                                                            'value': self.spiAlgorithm},
                #{'name': self.spiParam_algorithm1_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "",
                # 'readonly': True, 'children': [
                #    {'name': self.spiParam_alg1_pruneInterval_str, 'type': 'float', 'value': self.spiParam_alg1_pruneInterval,
                #     'tip': "update running background"},
                #]},
                {'name': self.spiParam_algorithm2_str, 'visible': True, 'expanded': False, 'type': 'str', 'value': "", 'readonly': True, 'children': [
                    {'name': self.spiParam_alg2_threshold_str, 'type': 'float', 'value': self.spiParam_alg2_threshold, 'tip': "search for pixels above ADU per photon"},
                ]},
                {'name': self.spiParam_outDir_str, 'type': 'str', 'value': self.spiParam_outDir},
                {'name': self.spiParam_runs_str, 'type': 'str', 'value': self.spiParam_runs, 'tip': "comma separated or use colon for a range, e.g. 1,3,5:7 = runs 1,3,5,6,7"},
                {'name': self.spiParam_queue_str, 'type': 'list', 'values': {self.parent.pk.hitParam_psfehhiprioq_str: 'psfehhiprioq',
                                                                           self.parent.pk.hitParam_psnehhiprioq_str: 'psnehhiprioq',
                                                                           self.parent.pk.hitParam_psfehprioq_str: 'psfehprioq',
                                                                           self.parent.pk.hitParam_psnehprioq_str: 'psnehprioq',
                                                                           self.parent.pk.hitParam_psfehq_str: 'psfehq',
                                                                           self.parent.pk.hitParam_psnehq_str: 'psnehq',
                                                                           self.parent.pk.hitParam_psanaq_str: 'psanaq',
                                                                           self.parent.pk.hitParam_psdebugq_str: 'psdebugq',
                                                                           self.parent.pk.hitParam_psanagpuq_str: 'psanagpuq',
                                                                           self.parent.pk.hitParam_psanaidleq_str: 'psanaidleq',
                                                                           self.parent.pk.hitParam_ffbh1q_str: 'ffbh1q',
                                                                           self.parent.pk.hitParam_ffbl1q_str: 'ffbl1q',
                                                                           self.parent.pk.hitParam_ffbh2q_str: 'ffbh2q',
                                                                           self.parent.pk.hitParam_ffbl2q_str: 'ffbl2q',
                                                                           self.parent.pk.hitParam_ffbh3q_str: 'ffbh3q',
                                                                           self.parent.pk.hitParam_ffbl3q_str: 'ffbl3q',
                                                                           self.parent.pk.hitParam_anaq_str: 'anaq'},
                 'value': self.spiParam_queue, 'tip': "Choose queue"},
                {'name': self.spiParam_cpu_str, 'type': 'int', 'value': self.spiParam_cpus},
                {'name': self.spiParam_noe_str, 'type': 'int', 'value': self.spiParam_noe, 'tip': "number of events to process, default=0 means process all events"},
                {'name': self.tag_str, 'type': 'str', 'value': self.tag},
                {'name': self.spiParam_launch_str, 'type': 'action'},
            ]},
            {'name': self.hitParam_grp, 'type': 'group', 'children': [
                {'name': self.hitParam_hitThresh_str, 'type': 'str', 'value': self.hitParam_hitThresh, 'tip': "Set as hit if number of pixels with photons above this value. Or specify a range using a colon, e.g. 100:10000."},
                {'name': self.hitParam_backgroundThresh_str, 'type': 'str', 'value': self.hitParam_backgroundThresh,
                 'tip': "Use as background if number of pixels with photons below this value. Or specify a range using a colon, e.g. 100:10000."},
                {'name': self.hitParam_sample_str, 'type': 'str', 'value': self.hitParam_sample},
                {'name': self.hitParam_save_str, 'type': 'action'},
            ]},
        ]
        self.p8 = Parameter.create(name='paramsHitFinder', type='group', \
                                   children=self.params, expanded=True)
        self.win.setParameters(self.p8, showTop=False)
        self.p8.sigTreeStateChanged.connect(self.change)

    # Launch hit finding
    def findHits(self):
        self.parent.thread.append(LaunchHitFinder.HitFinder(self.parent))  # send parent parameters with self
        self.parent.thread[self.parent.threadCounter].findHits(self.parent.experimentName, self.parent.runNumber, self.parent.detInfo)
        self.parent.threadCounter += 1

    def setThreshold(self):
        self.parent.thread.append(LaunchHitConverter.LaunchHitConverter(self.parent))  # send parent parameters with self
        self.parent.thread[self.parent.threadCounter].launch(self.parent.experimentName, self.parent.detInfo)
        self.parent.threadCounter += 1

    # If anything changes in the parameter tree, print a message
    def change(self, panel, changes):
        for param, change, data in changes:
            path = panel.childPath(param)
            if self.parent.args.v >= 1:
                print('  path: %s' % path)
                print('  change:    %s' % change)
                print('  data:      %s' % str(data))
                print('  ----------')
            self.paramUpdate(path, change, data)

    ##############################
    # Mandatory parameter update #
    ##############################
    def paramUpdate(self, path, change, data):
        if path[0] == self.spiParam_grp:
            if path[1] == self.spiParam_algorithm_str:
                self.spiAlgorithm = data
            elif path[1] == self.spiParam_outDir_str:
                self.spiParam_outDir = data
                self.spiParam_outDir_overridden = True
            elif path[1] == self.spiParam_runs_str:
                self.spiParam_runs = data
            elif path[1] == self.spiParam_queue_str:
                self.spiParam_queue = data
            elif path[1] == self.spiParam_cpu_str:
                self.spiParam_cpus = data
            elif path[1] == self.spiParam_noe_str:
                self.spiParam_noe = data
            elif path[1] == self.tag_str:
                self.updateTag(data)
            elif path[1] == self.spiParam_launch_str:
                self.findHits()
            elif path[2] == self.spiParam_alg1_pruneInterval_str and path[1] == self.spiParam_algorithm1_str:
                self.spiParam_alg1_pruneInterval = data
                self.updateHit()
            elif path[2] == self.spiParam_alg2_threshold_str and path[1] == self.spiParam_algorithm2_str:
                self.spiParam_alg2_threshold = data
                self.updateHit()
        elif path[0] == self.hitParam_grp:
            if path[1] == self.hitParam_hitThresh_str:
                self.hitParam_hitThresh = data
            elif path[1] == self.hitParam_backgroundThresh_str:
                self.hitParam_backgroundThresh = data
            elif path[1] == self.hitParam_sample_str:
                self.hitParam_sample = data
            elif path[1] == self.hitParam_save_str:
                self.setThreshold()

    def updateTag(self, data):
        if self.tag:
            fname = self.parent.small.quantifier_filename.split("_"+self.tag+".cxi")[0]
        else:
            fname = self.parent.small.quantifier_filename.split(".cxi")[0]
        if data:
            fname += "_" + data + ".cxi"
        else:
            fname += ".cxi"
        self.tag = data
        self.parent.small.pSmall.param(self.parent.small.quantifier_grp, self.parent.small.quantifier_filename_str).setValue(fname)
        self.parent.small.reloadQuantifier()

    def updateHit(self):
        # Save a temporary mask
        if self.parent.mk.userMask is None:
            userMask = None
        else:
            userMask = self.parent.psocakeDir + "/r" + str(self.parent.runNumber).zfill(4) + "/tempUserMask.npy"
            np.save(userMask, self.parent.mk.userMask)

        worker = alg.HitFinder(self.parent.experimentName,
                               self.parent.runNumber,
                               self.parent.detInfo,
                               self.parent.evt,
                               self.parent.det,
                               self.spiParam_alg2_threshold,
                               streakMask_on=str(self.parent.mk.streakMaskOn),
                               streakMask_sigma=self.parent.mk.streak_sigma,
                               streakMask_width=self.parent.mk.streak_width,
                               userMask_path=userMask,
                               psanaMask_on=str(self.parent.mk.psanaMaskOn),
                               psanaMask_calib=str(self.parent.mk.mask_calibOn),
                               psanaMask_status=str(self.parent.mk.mask_statusOn),
                               psanaMask_edges=str(self.parent.mk.mask_edgesOn),
                               psanaMask_central=str(self.parent.mk.mask_centralOn),
                               psanaMask_unbond=str(self.parent.mk.mask_unbondOn),
                               psanaMask_unbondnrs=str(self.parent.mk.mask_unbondnrsOn))
        worker.findHits(self.parent.calib, self.parent.evt)
        self.nPixels = worker.nPixels
        self.indicatePhotons()
        if self.parent.args.v >= 1: print("self.nPixels: ", self.nPixels)

    def indicatePhotons(self):
        self.parent.img.clearPeakMessage()
        self.parent.mk.displayMask()
        # Write number of pixels found containing photons
        xMargin = 5  # pixels
        yMargin = 0  # pixels
        maxX = np.max(self.parent.det.indexes_x(self.parent.evt)) + xMargin
        maxY = np.max(self.parent.det.indexes_y(self.parent.evt)) - yMargin
        myMessage = '<div style="text-align: center"><span style="color: cyan; font-size: 12pt;">Pixels=' + \
                    str(self.nPixels) + ' <br></span></div>'
        self.parent.img.peak_text = pg.TextItem(html=myMessage, anchor=(0, 0))
        self.parent.img.win.getView().addItem(self.parent.img.peak_text)
        self.parent.img.peak_text.setPos(maxY, maxX)

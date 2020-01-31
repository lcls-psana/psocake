from pyqtgraph.Qt import QtCore
import subprocess
import os
import numpy as np
import h5py
import json
from utils import *

class LaunchPeakFinder(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        self.parent = parent
        self.experimentName = None
        self.detInfo = None

    def __del__(self):
        self.exiting = True
        self.wait()

    def launch(self,experimentName,detInfo): # Pass in peak parameters
        self.experimentName = experimentName
        self.detInfo = detInfo
        self.start()

    def saveCheetahFormatMask(self, run, arg):
        if arg == self.parent.facilityLCLS:
            if 'cspad' in self.parent.detInfo.lower():# and 'cxi' in self.parent.experimentName:
                dim0 = 8 * 185
                dim1 = 4 * 388
            elif 'rayonix' in self.parent.detInfo.lower():# and 'mfx' in self.parent.experimentName:
                dim0 = 1920
                dim1 = 1920
            #elif 'rayonix' in self.parent.detInfo.lower() and 'xpp' in self.parent.experimentName:
            #    dim0 = 1920
            #    dim1 = 1920
            else:
                print "saveCheetahFormatMask not implemented"

            fname = self.parent.pk.hitParam_outDir+'/r'+str(run).zfill(4)+'/staticMask.h5'
            print "Saving static mask in Cheetah format: ", fname
            myHdf5 = h5py.File(fname, 'w')
            dset = myHdf5.create_dataset('/entry_1/data_1/mask', (dim0, dim1), dtype='int')

            # Convert calib image to cheetah image
            if self.parent.mk.combinedMask is None:
                img = np.ones((dim0, dim1))
            else:
                img = np.zeros((dim0, dim1))
                counter = 0
                if 'cspad' in self.parent.detInfo.lower():# and 'cxi' in self.parent.experimentName:
                    img = pct(self.parent.mk.combinedMask)
                elif 'rayonix' in self.parent.detInfo.lower():# and 'mfx' in self.parent.experimentName:
                    img = self.parent.mk.combinedMask[:, :] # psana format
                #elif 'rayonix' in self.parent.detInfo.lower() and 'xpp' in self.parent.experimentName:
                #    img = self.parent.mk.combinedMask[:, :]  # psana format
            dset[:,:] = img
            myHdf5.close()

    def run(self):
        # Digest the run list
        run = 99

        runDir = self.parent.pk.hitParam_outDir+"/r"+str(run).zfill(4)
        try:
            if os.path.exists(runDir) is False:
                os.makedirs(runDir, 0774)
        except:
            print "No write access to: ", runDir

        # Generate Cheetah mask
        self.saveCheetahFormatMask(run, self.parent.facility)

        # Update elog
        try:
            if self.parent.exp.logger == True:
                self.parent.exp.table.setValue(run,"Number of hits","#PeakFindingNow")
        except AttributeError:
            print "e-Log table does not exist"

        # Copy powder ring
        import shutil
        src = self.parent.psocakeDir+'/r'+str(self.parent.runNumber).zfill(4)+'/background.npy'
        dst = self.parent.psocakeDir+'/r'+str(run).zfill(4)+'/background.npy'
        if src != dst and os.path.exists(src):
            shutil.copyfile(src, dst)

        cmd = "bsub -q "+self.parent.pk.hitParam_queue + \
              " -n "+str(self.parent.pk.hitParam_cpus) + \
              " -o "+runDir+"/.%J.log mpirun findPeaks -e "+self.experimentName+\
              " -d "+self.detInfo+\
              " --outDir "+runDir+\
              " --algorithm "+str(self.parent.pk.algorithm)
        if self.parent.pk.algorithm == 1:
            cmd += " --alg_npix_min "+str(self.parent.pk.hitParam_alg1_npix_min)+\
                   " --alg_npix_max "+str(self.parent.pk.hitParam_alg1_npix_max)+\
                   " --alg_amax_thr "+str(self.parent.pk.hitParam_alg1_amax_thr)+\
                   " --alg_atot_thr "+str(self.parent.pk.hitParam_alg1_atot_thr)+\
                   " --alg_son_min "+str(self.parent.pk.hitParam_alg1_son_min)+\
                   " --alg1_thr_low "+str(self.parent.pk.hitParam_alg1_thr_low)+\
                   " --alg1_thr_high "+str(self.parent.pk.hitParam_alg1_thr_high)+ \
                   " --alg1_rank " + str(self.parent.pk.hitParam_alg1_rank) + \
                   " --alg1_radius "+str(self.parent.pk.hitParam_alg1_radius)+\
                   " --alg1_dr "+str(self.parent.pk.hitParam_alg1_dr)
        elif self.parent.pk.algorithm == 2:
            cmd += " --alg_npix_min "+str(self.parent.pk.hitParam_alg1_npix_min)+\
                   " --alg_npix_max "+str(self.parent.pk.hitParam_alg1_npix_max)+\
                   " --alg_amax_thr "+str(self.parent.pk.hitParam_alg1_amax_thr)+\
                   " --alg_atot_thr "+str(self.parent.pk.hitParam_alg1_atot_thr)+\
                   " --alg_son_min "+str(self.parent.pk.hitParam_alg1_son_min)+\
                   " --alg1_thr_low "+str(self.parent.pk.hitParam_alg1_thr_low)+\
                   " --alg1_thr_high "+str(self.parent.pk.hitParam_alg1_thr_high)+ \
                   " --alg1_rank " + str(self.parent.pk.hitParam_alg1_rank) + \
                   " --alg1_radius "+str(self.parent.pk.hitParam_alg1_radius)+\
                   " --alg1_dr "+str(self.parent.pk.hitParam_alg1_dr)
        # Save user mask to a deterministic path
        if self.parent.mk.userMaskOn:
            tempFilename = self.parent.psocakeDir+"/r"+str(run).zfill(4)+"/tempUserMask.npy"
            np.save(tempFilename,self.parent.mk.userMask) # TODO: save
            cmd += " --userMask_path "+str(tempFilename)
        if self.parent.mk.streakMaskOn:
            cmd += " --streakMask_on "+str(self.parent.mk.streakMaskOn)+\
                   " --streakMask_sigma "+str(self.parent.mk.streak_sigma)+\
                   " --streakMask_width "+str(self.parent.mk.streak_width)
        if self.parent.mk.psanaMaskOn:
            cmd += " --psanaMask_on "+str(self.parent.mk.psanaMaskOn) + \
                   " --psanaMask_calib "+str(self.parent.mk.mask_calibOn) + \
                   " --psanaMask_status "+str(self.parent.mk.mask_statusOn) + \
                   " --psanaMask_edges "+str(self.parent.mk.mask_edgesOn) + \
                   " --psanaMask_central "+str(self.parent.mk.mask_centralOn) + \
                   " --psanaMask_unbond "+str(self.parent.mk.mask_unbondOn) + \
                   " --psanaMask_unbondnrs "+str(self.parent.mk.mask_unbondnrsOn)

        cmd += " --mask "+self.parent.pk.hitParam_outDir+'/r'+str(run).zfill(4)+'/staticMask.h5'

        if self.parent.pk.hitParam_noe > 0:
            cmd += " --noe "+str(self.parent.pk.hitParam_noe)

        if self.parent.args.localCalib: cmd += " --localCalib"

        if self.parent.exp.image_property == self.parent.exp.disp_medianCorrection:
            cmd += " --medianBackground " + str(1) + \
                   " --medianRank " + str(self.parent.exp.medianFilterRank)
        elif self.parent.exp.image_property == self.parent.exp.disp_radialCorrection:
            cmd += " --radialBackground " + str(1) + \
                   " --detectorDistance " + str(self.parent.detectorDistance)

        cmd += " --clen " + str(self.parent.clenEpics)
        cmd += " --coffset " + str(self.parent.coffset)

        cmd += " --minPeaks " + str(self.parent.pk.minPeaks)
        cmd += " --maxPeaks " + str(self.parent.pk.maxPeaks)
        cmd += " --minRes " + str(self.parent.pk.minRes)
        cmd += " --sample " + str(self.parent.pk.sample)
        cmd += " --instrument " + str(self.parent.det.instrument())
        cmd += " --pixelSize " + str(self.parent.pixelSize)

        if self.parent.pk.hitParam_extra: cmd += " " + self.parent.pk.hitParam_extra

        cmd += " --auto " + str(self.parent.autoPeakFinding) + \
               " --detectorDistance " + str(self.parent.detectorDistance)

        cmd += " --access " + self.parent.access

        if self.parent.pk.tag: cmd += " --tag " + self.parent.pk.tag

        cmd += " -r " + str(run)
        # Launch peak finding
        print "Submitting batch job: ", cmd
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

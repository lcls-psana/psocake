from pyqtgraph.Qt import QtCore
import subprocess
import os
import numpy as np

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

    def digestRunList(self,runList):
        runsToDo = []
        if not runList:
            print "Run(s) is empty. Please type in the run number(s)."
            return runsToDo
        runLists = str(runList).split(",")
        for list in runLists:
            temp = list.split(":")
            if len(temp) == 2:
                for i in np.arange(int(temp[0]),int(temp[1])+1):
                    runsToDo.append(i)
            elif len(temp) == 1:
                runsToDo.append(int(temp[0]))
        return runsToDo

    def run(self):
        # Digest the run list
        runsToDo = self.digestRunList(self.parent.hitParam_runs)

        for run in runsToDo:
            runDir = self.parent.hitParam_outDir+"/r"+str(run).zfill(4)
            try:
                if os.path.exists(runDir) is False:
                    os.makedirs(runDir, 0774)
            except:
                print "No write access to: ", runDir

            # Update elog
            try:
                if self.parent.logger == True:
                    self.parent.table.setValue(run,"Number of hits","#PeakFindingNow")
            except AttributeError:
                print "e-Log table does not exist"

            cmd = "bsub -q "+self.parent.hitParam_queue + \
              " -n "+str(self.parent.hitParam_cpus) + \
              " -o "+runDir+"/.%J.log mpirun findPeaks -e "+self.experimentName+\
              " -d "+self.detInfo+\
              " --outDir "+runDir+\
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
                tempFilename = self.parent.psocakeDir+"/r"+str(run).zfill(4)+"/tempUserMask.npy"
                np.save(tempFilename,self.parent.userMask) # TODO: save
                cmd += " --userMask_path "+str(tempFilename)
            if self.parent.streakMaskOn:
                cmd += " --streakMask_on "+str(self.parent.streakMaskOn)+\
                    " --streakMask_sigma "+str(self.parent.streak_sigma)+\
                   " --streakMask_width "+str(self.parent.streak_width)
            if self.parent.psanaMaskOn:
                cmd += " --psanaMask_on "+str(self.parent.psanaMaskOn)+\
                   " --psanaMask_calib "+str(self.parent.mask_calibOn)+\
                   " --psanaMask_status "+str(self.parent.mask_statusOn)+\
                   " --psanaMask_edges "+str(self.parent.mask_edgesOn)+\
                   " --psanaMask_central "+str(self.parent.mask_centralOn)+\
                   " --psanaMask_unbond "+str(self.parent.mask_unbondOn)+\
                   " --psanaMask_unbondnrs "+str(self.parent.mask_unbondnrsOn)

            if self.parent.hitParam_noe > 0:
                cmd += " --noe "+str(self.parent.hitParam_noe)

            if self.parent.localCalib: cmd += " --localCalib"

            cmd += " -r " + str(run)
            print "Submitting batch job: ", cmd
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()

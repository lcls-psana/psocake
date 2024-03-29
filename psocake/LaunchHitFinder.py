from pyqtgraph.Qt import QtCore
import subprocess
import os
import numpy as np
from psocake.utils import batchSubmit

class HitFinder(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        self.parent = parent
        self.experimentName = None
        self.runNumber = None
        self.detInfo = None

    def __del__(self):
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
            print("Run(s) is empty. Please type in the run number(s).")
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
        runsToDo = self.digestRunList(self.parent.hf.spiParam_runs)

        for run in runsToDo:
            runDir = self.parent.hf.spiParam_outDir+"/r"+str(run).zfill(4)
            try:
                if os.path.exists(runDir) is False: os.makedirs(runDir, 0o0774)
                cmd = "mpirun --mca btl ^openib findHits"+\
                  " -e " + self.experimentName + \
                  " -d "+self.detInfo+\
                  " --outDir " + runDir

                cmd += " --algorithm " + str(self.parent.hf.spiAlgorithm)

                if self.parent.hf.spiAlgorithm == 1:
                    cmd += " --pruneInterval " + str(self.parent.hf.spiParam_alg1_pruneInterval)
                elif self.parent.hf.spiAlgorithm == 2:
                    cmd += " --litPixelThreshold " + str(self.parent.hf.spiParam_alg2_threshold)
                cmd += " --hitThreshold " + str(self.parent.hf.hitParam_hitThresh)

                # Save user mask to a deterministic path
                if self.parent.mk.userMaskOn:
                    tempFilename = self.parent.psocakeDir+"/r"+str(run).zfill(4)+"/tempUserMask.npy"
                    np.save(tempFilename,self.parent.mk.userMask) # TODO: save
                    cmd += " --userMask_path "+str(tempFilename)

                cmd += " --streakMask_on " + str(self.parent.mk.streakMaskOn)
                cmd += " --streakMask_sigma " + str(self.parent.mk.streak_sigma)
                cmd += " --streakMask_width " + str(self.parent.mk.streak_width)
                cmd += " --psanaMask_on " + str(self.parent.mk.psanaMaskOn)
                cmd += " --psanaMask_calib " + str(self.parent.mk.mask_calibOn)
                cmd += " --psanaMask_status " + str(self.parent.mk.mask_statusOn)
                cmd += " --psanaMask_edges " + str(self.parent.mk.mask_edgesOn)
                cmd += " --psanaMask_central " + str(self.parent.mk.mask_centralOn)
                cmd += " --psanaMask_unbond " + str(self.parent.mk.mask_unbondOn)
                cmd += " --psanaMask_unbondnrs " + str(self.parent.mk.mask_unbondnrsOn)

                if self.parent.hf.spiParam_noe > 0: cmd += " --noe "+str(self.parent.hf.spiParam_noe)

                if self.parent.hf.tag: cmd += " --tag " + self.parent.hf.tag

                if self.parent.args.localCalib: cmd += " --localCalib"

                cmd += " -r " + str(run)

                cmd = batchSubmit(cmd, self.parent.hf.spiParam_queue, self.parent.hf.spiParam_cpus, runDir + "/%J.log",
                                  "hit" + str(run), self.parent.batch)

                print("Submitting batch job: ", cmd)
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                out, err = process.communicate()
                if self.parent.batch == "lsf":
                    jobID = out.decode("utf-8").split("<")[1].split(">")[0]
                else:
                    jobID = out.decode("utf-8").split("job ")[1].split("\n")[0]
                myLog = self.parent.psocakeDir+"/r"+str(run).zfill(4)+"/"+jobID+".log"
                if self.parent.args.v >= 1: print("log filename: ", myLog)
            except:
                print("No write access to: ", runDir)

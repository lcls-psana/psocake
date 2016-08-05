from pyqtgraph.Qt import QtCore
import subprocess
import os
import numpy as np

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
        runsToDo = self.digestRunList(self.parent.hf.spiParam_runs)

        for run in runsToDo:
            runDir = self.parent.hf.spiParam_outDir+"/r"+str(run).zfill(4)
            try:
                if os.path.exists(runDir) is False: os.makedirs(runDir, 0774)
                #expRun = 'exp='+self.experimentName+':run='+str(run)
                cmd = "bsub -q "+self.parent.hf.spiParam_queue+\
                  " -n "+str(self.parent.hf.spiParam_cpus)+\
                  " -o "+runDir+"/.%J.log mpirun findHits"+\
                  " -e " + self.experimentName + \
                  " -d "+self.detInfo+\
                  " --outDir " + runDir

                if self.parent.hf.spiAlgorithm == 2:
                    cmd += " --litPixelThreshold " + str(self.parent.hf.spiParam_alg2_threshold)
                    #cmd += " --hitThreshold " + str(self.parent.hf.spiParam_alg2_hitThreshold)

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

                if self.parent.args.localCalib: cmd += " --localCalib"

                cmd += " -r " + str(run)

                print "Submitting batch job: ", cmd
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                out, err = process.communicate()
                jobid = out.split("<")[1].split(">")[0]
                myLog = self.parent.psocakeDir+"/r"+str(run).zfill(4)+"/."+jobid+".log"
                if self.parent.args.v >= 1: print "bsub log filename: ", myLog
            except:
                print "No write access to: ", runDir

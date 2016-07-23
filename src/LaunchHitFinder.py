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
        runsToDo = self.digestRunList(self.parent.spiParam_runs)

        for run in runsToDo:
            runDir = self.parent.spiParam_outDir+"/r"+str(run).zfill(4)
            try:
                if os.path.exists(runDir) is False:
                    os.makedirs(runDir, 0774)
                expRun = 'exp='+self.experimentName+':run='+str(run)
                cmd = "bsub -q "+self.parent.spiParam_queue+\
                  " -n "+str(self.parent.spiParam_cpus)+\
                  " -o "+runDir+"/.%J.log mpirun litPixels"+\
                  " "+expRun+\
                  " -d "+self.detInfo+\
                  " --outdir "+runDir

                if self.parent.spiParam_tag:
                    cmd += " --tag "+str(self.parent.spiParam_tag)

                if self.parent.spiAlgorithm == 1:
                    cmd += " --pruneInterval "+str(int(self.parent.spiParam_alg1_pruneInterval))
                elif self.parent.spiAlgorithm == 2:
                    cmd += " --litPixelThreshold "+str(int(self.parent.spiParam_alg2_threshold))

                # Save user mask to a deterministic path
                if self.parent.userMaskOn:
                    tempFilename = self.parent.psocakeDir+"/r"+str(run).zfill(4)+"/tempUserMask.npy"
                    np.save(tempFilename,self.parent.userMask) # TODO: save
                    cmd += " --mask "+str(tempFilename)

                if self.parent.spiParam_noe > 0:
                    cmd += " --noe "+str(self.parent.spiParam_noe)

                if self.parent.args.localCalib: cmd += " --localCalib"

                print "Submitting batch job: ", cmd
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                out, err = process.communicate()
                jobid = out.split("<")[1].split(">")[0]
                myLog = self.parent.psocakeDir+"/r"+str(run).zfill(4)+"/."+jobid+".log"
                if self.parent.args.v >= 1: print "bsub log filename: ", myLog
            except:
                print "No write access to: ", runDir

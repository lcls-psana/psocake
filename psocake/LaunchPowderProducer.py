from pyqtgraph.Qt import QtCore
import subprocess
import os
import numpy as np

class PowderProducer(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        self.parent = parent
        self.experimentName = None
        self.runNumber = None
        self.detInfo = None

    def __del__(self):
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
        runsToDo = self.digestRunList(self.parent.mk.powder_runs)
        for run in runsToDo:
            runDir = self.parent.mk.powder_outDir+"/r"+str(run).zfill(4)
            try:
                if os.path.exists(runDir) is False: os.makedirs(runDir, 0774)

                # Command for submitting to batch
                cmd = "bsub -q "+self.parent.mk.powder_queue+" -n "+str(self.parent.mk.powder_cpus)+\
                      " -o "+runDir+"/.%J.log mpirun generatePowder exp="+self.experimentName+\
                      ":run="+str(run)+" -d "+self.detInfo+\
                      " -o "+runDir
                if self.parent.mk.powder_noe > 0:
                    cmd += " -n "+str(self.parent.mk.powder_noe)
                if self.parent.mk.powder_threshold is not -1:
                    cmd += " -t " + str(self.parent.mk.powder_threshold)
                if self.parent.args.localCalib:
                    cmd += " --localCalib"
                print "Submitting batch job: ", cmd
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                out, err = process.communicate()
                jobid = out.split("<")[1].split(">")[0]
                myLog = self.parent.mk.powder_outDir+"/r"+str(run).zfill(4)+"/."+jobid+".log"
                if self.parent.args.v >= 1: print "bsub log filename: ", myLog
            except:
                print "No write access to: ", runDir

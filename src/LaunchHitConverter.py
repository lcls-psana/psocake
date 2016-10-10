from pyqtgraph.Qt import QtCore
import subprocess
import os
import numpy as np

class LaunchHitConverter(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        self.parent = parent
        self.experimentName = None
        self.detInfo = None

    def __del__(self):
        self.exiting = True
        self.wait()

    def launch(self, experimentName, detInfo): # Pass in peak parameters
        self.experimentName = experimentName
        self.detInfo = detInfo
        self.start()

    def digestRunList(self, runList):
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
                if os.path.exists(runDir) is False:
                    os.makedirs(runDir, 0774)
            except:
                print "No write access to: ", runDir

            # Update elog
            try:
                if self.parent.exp.logger == True:
                    self.parent.exp.table.setValue(run,"Number of hits","#ConvertingNow")
            except AttributeError:
                print "e-Log table does not exist"

            cmd = "bsub -q " + self.parent.hf.spiParam_queue + \
                  " -n " + str(self.parent.hf.spiParam_cpus) + \
                  " -o " + runDir + "/.%J.log mpirun xtc2cxi" + \
                  " -e " + self.experimentName + \
                  " -d " + self.detInfo + \
                  " -i " + runDir + \
                  " --sample " + self.parent.hf.hitParam_sample + \
                  " --instrument " + self.parent.det.instrument() + \
                  " --pixelSize " + str(self.parent.pixelSize) + \
                  " --detectorDistance " + str(self.parent.detectorDistance) + \
                  " --minPixels " + str(self.parent.hf.hitParam_threshMin) + \
                  " --backgroundThresh " + str(self.parent.hf.hitParam_background) + \
                  " --mode spi" + \
                  " --run " + str(run)
            print "Submitting batch job: ", cmd
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()

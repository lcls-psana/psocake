from pyqtgraph.Qt import QtCore
import subprocess
import os
import numpy as np

class LaunchIndexer(QtCore.QThread):
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
        runsToDo = self.digestRunList(self.parent.index.runs)

        for run in runsToDo:
            runDir = self.parent.psocakeDir+"/r"+str(run).zfill(4)
            try:
                if os.path.exists(runDir) is False:
                    os.makedirs(runDir, 0774)
            except:
                print "No write access to: ", runDir

            # Update elog
            try:
                if self.parent.logger == True:
                    self.parent.table.setValue(run,"Number of indexed","#IndexingNow")
            except AttributeError:
                print "e-Log table does not exist"

            cmd = "./indexCrystals.py" + \
                  " -e " + self.parent.experimentName + " -d " + self.parent.detInfo + \
                  " --geom " + self.parent.index.geom + \
                  " --peakMethod " + self.parent.index.peakMethod + \
                  " --integrationRadius " + self.parent.index.intRadius + \
                  " --indexingMethod " + self.parent.index.indexingMethod + \
                  " --minPeaks " + str(self.parent.index.minPeaks) + \
                  " --maxPeaks " + str(self.parent.index.maxPeaks) + \
                  " --minRes " + str(self.parent.index.minRes) + \
                  " --outDir " + self.parent.index.outDir + \
                  " --sample " + self.parent.index.sample + \
                  " --queue " + self.parent.index.queue + \
                  " --cpus " + str(self.parent.index.cpus) + \
                  " --noe " + str(self.parent.index.noe) + \
                  " --instrument " + self.parent.det.instrument() + \
                  " --pixelSize " + str(self.parent.pixelSize) + \
                  " --coffset " + str(self.parent.coffset) + \
                  " --clenEpics " + self.parent.clenEpics + \
                  " --logger " + str(self.parent.logger) + \
                  " --hitParam_threshold " + str(self.parent.hitParam_threshold) + \
                  " --keepData " + str(self.parent.index.keepData) + \
                  " -v " + str(self.parent.args.v)
            if self.parent.index.pdb: cmd += " --pdb " + self.parent.index.pdb
            cmd += " --run " + str(run)
            print "Submitting batch job: ", cmd
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)


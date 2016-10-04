from pyqtgraph.Qt import QtCore
import subprocess
import os, shlex
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
            runDir = self.parent.index.outDir+"/r"+str(run).zfill(4)
            try:
                if os.path.exists(runDir) is False:
                    os.makedirs(runDir, 0774)
            except:
                print "No write access to: ", runDir

            # Update elog
            try:
                if self.parent.exp.logger == True:
                    self.parent.exp.table.setValue(run,"Number of indexed","#IndexingNow")
            except AttributeError:
                print "e-Log table does not exist"

            cmd = "indexCrystals" + \
                  " -e " + self.parent.experimentName + \
                  " -d " + self.parent.detInfo + \
                  " --geom " + self.parent.index.geom + \
                  " --peakMethod " + self.parent.index.peakMethod + \
                  " --integrationRadius " + self.parent.index.intRadius + \
                  " --indexingMethod " + self.parent.index.indexingMethod + \
                  " --minPeaks " + str(self.parent.pk.minPeaks) + \
                  " --maxPeaks " + str(self.parent.pk.maxPeaks) + \
                  " --minRes " + str(self.parent.pk.minRes) + \
                  " --tolerance " + str(self.parent.index.tolerance) + \
                  " --outDir " + self.parent.index.outDir + \
                  " --sample " + self.parent.index.sample + \
                  " --queue " + self.parent.index.queue + \
                  " --chunkSize " + str(self.parent.index.chunkSize) + \
                  " --noe " + str(self.parent.index.noe) + \
                  " --instrument " + self.parent.det.instrument() + \
                  " --pixelSize " + str(self.parent.pixelSize) + \
                  " --coffset " + str(self.parent.coffset) + \
                  " --clenEpics " + self.parent.clenEpics + \
                  " --logger " + str(self.parent.exp.logger) + \
                  " --hitParam_threshold " + str(self.parent.pk.hitParam_threshold) + \
                  " --keepData " + str(self.parent.index.keepData) + \
                  " -v " + str(self.parent.args.v)
            if self.parent.index.pdb: cmd += " --pdb " + self.parent.index.pdb
            if self.parent.index.extra: cmd += " " + self.parent.index.extra
            cmd += " --run " + str(run)
            print "Launch indexing job: ", cmd
            p = subprocess.Popen(shlex.split(cmd))

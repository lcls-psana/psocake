from pyqtgraph.Qt import QtCore
import subprocess
import os
import numpy as np
import h5py

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

    def getCheetahMask(self, maskCalib):
        """Converts seg, row, col assuming (32,185,388)
           to cheetah 2-d table row and col (8*185, 4*388)
        """
        img = np.zeros((8 * 185, 4 * 388))
        counter = 0
        for quad in range(4):
            for seg in range(8):
                img[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = maskCalib[counter, :, :]
                counter += 1
        # Cheetah badpixels are 1's
        img = (img*-1)+1
        return img

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
        if self.parent.args.mode == 'spi':
            runsToDo = self.digestRunList(self.parent.hf.spiParam_runs)
        elif self.parent.args.mode == 'sfx':
            runsToDo = self.digestRunList(self.parent.pk.hitParam_runs)

        for run in runsToDo:
            if self.parent.args.mode == 'spi':
                runDir = self.parent.hf.spiParam_outDir + "/r" + str(run).zfill(4)
            elif self.parent.args.mode == 'sfx':
                runDir = self.parent.pk.hitParam_outDir + "/r" + str(run).zfill(4)

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

            if self.parent.args.mode == 'spi':
                cmd = "bsub -q " + self.parent.hf.spiParam_queue + \
                      " -n " + str(self.parent.hf.spiParam_cpus) + \
                      " -o " + runDir + "/.%J.log mpirun xtc2cxidb" + \
                      " -e " + self.experimentName + \
                      " -d " + self.detInfo + \
                      " -i " + runDir + \
                      " --sample " + self.parent.hf.hitParam_sample + \
                      " --instrument " + self.parent.det.instrument() + \
                      " --pixelSize " + str(self.parent.pixelSize) + \
                      " --detectorDistance " + str(self.parent.detectorDistance) + \
                      " --minPixels " + str(self.parent.hf.hitParam_threshMin) + \
                      " --maxBackground " + str(self.parent.hf.hitParam_backgroundMax) + \
                      " --mode spi" + \
                      " --run " + str(run)
            elif self.parent.args.mode == 'sfx':
                cmd = "bsub -q " + self.parent.pk.hitParam_queue + \
                      " -n " + str(self.parent.pk.hitParam_cpus) + \
                      " -o " + runDir + "/.%J.log mpirun xtc2cxidb" + \
                      " -e " + self.experimentName + \
                      " -d " + self.detInfo + \
                      " -i " + runDir + \
                      " --sample " + self.parent.pk.sample + \
                      " --instrument " + self.parent.det.instrument() + \
                      " --pixelSize " + str(self.parent.pixelSize) + \
                      " --coffset " + str(self.parent.coffset) + \
                      " --clen " + self.parent.clenEpics + \
                      " --minPeaks " + str(self.parent.pk.minPeaks) + \
                      " --maxPeaks " + str(self.parent.pk.maxPeaks) + \
                      " --minRes " + str(self.parent.pk.minRes) + \
                      " --mode sfx" + \
                      " --run " + str(run)
            print "Submitting batch job: ", cmd
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()

            # Save Cheetah mask
            if self.parent.args.mode == 'sfx':
                f = h5py.File(runDir+'/staticMask.h5','w')
                if self.parent.mk.combinedMask is not None:
                    cheetahMask = self.getCheetahMask(self.parent.mk.combinedMask)
                    f.create_dataset("/entry_1/data_1/mask", data=cheetahMask, dtype=int)
                f.close()



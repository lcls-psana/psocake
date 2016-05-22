import numpy as np
import re
from pyqtgraph.Qt import QtCore
import time
import subprocess
import os
import pandas as pd

class CrystalIndexing(object):
    def __init__(self, parent = None):
        self.parent = parent

        self.index_grp = 'Crystal indexing'
        self.index_on_str = 'Indexing on'

        #######################
        # Mandatory parameter #
        #######################
        self.params = [
            {'name': self.index_grp, 'type': 'group', 'children': [
                {'name': self.index_on_str, 'type': 'bool', 'value': False, 'tip': "Turn on indexing"}
            ]},
        ]

    ##############################
    # Mandatory parameter update #
    ##############################
    def paramUpdate(self, path, change, data):
        print "paramUpdate: ", path[1]
        if path[1] == self.index_on_str:
            print "Got here"
            self.updateIndexStatus(data)

    def updateIndexStatus(self, data):
        self.indexingOn = data
        print "indexing on: ", self.indexingOn
        if self.indexingOn:
            self.updateIndex()
        print "Done updateIndexStatus"

    def updateIndex(self):
        self.indexer = IndexHandler(parent=self.parent)
        self.indexer.computeIndex(self.parent.experimentName, self.parent.runNumber, self.parent.detInfo, self.parent.eventNumber)
        print "Done updateIndex"

class IndexHandler(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        print "WORKER!!!!!!!!!!"
        self.parent = parent
        self.experimentName = None
        self.runNumber = None
        self.detInfo = None
        self.eventNumber = None

    def __del__(self):
        print "del IndexHandler #$!@#$!#"
        self.exiting = True
        self.wait()

    def computeIndex(self,experimentName,runNumber,detInfo,eventNumber):
        self.experimentName = experimentName
        self.runNumber = runNumber
        self.detInfo = detInfo
        self.eventNumber = eventNumber
        self.start()

    def run(self):
        print "Running indexing!!!!!!!!!!!!"
        # Save image and peaks in cxidb format
        # Run crystfel dirax
        # Load indexed peak positions

        #cmd = "bsub -q psanaq -a mympi -o .%J.log " \
        cmd = "indexamajig -j 1 -i .temp.lst -g .temp.geom --peaks=cxi --int-radius=2,3,4 --pdb=lys.cell --indexing=dirax-axes-latt -o .temp.stream"

        print "Submitting batch job: ", cmd
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        print "out: ", out
        print "err: ", err
        #if "1 had crystals" in err:
        #    print "Done!!!!"
        #if "0 had crystals" in err:
        #    print "Damn!!!!"
        #jobid = out.split("<")[1].split(">")[0]
        #myLog = "."+jobid+".log"
        #print "bsub log filename: ", myLog

        #myKeyString = "The output (if any) is above this job summary."
        #mySuccessString = "Successfully completed."
        #notDone = 1
        #while notDone:
        #    if os.path.isfile(myLog):
        #        p = subprocess.Popen(["grep", myKeyString, myLog],stdout=subprocess.PIPE)
        #        output = p.communicate()[0]
        #        p.stdout.close()
        #        if myKeyString in output: # job has finished
        #            # check job was a success or a failure
        #            p = subprocess.Popen(["grep", mySuccessString, myLog], stdout=subprocess.PIPE)
        #            output = p.communicate()[0]
        #            p.stdout.close()
        #            if mySuccessString in output: # success
        #                print "successfully done"
        #                notDone = 0
        #            else:
        #                print "failed attempt"
        #                notDone = 0
        #        else:
        #            print "job hasn't finished yet"
        #            time.sleep(1)
        #    else:
        #        print "no such file yet"
        #        time.sleep(1)

        mySuccessString = "1 had crystals"
        # Read CrystFEL CSPAD geometry in stream
        if mySuccessString in err: # success
            print "Munging geometry file"
            f = open('.temp.stream')
            content = f.readlines()
            for i, val in enumerate(content):
                if '----- Begin geometry file -----' in val:
                    startLine = i
                elif   '----- End geometry file -----' in val:
                    endLine = i
                    break
            geom = content[startLine:endLine]
            numLines = endLine-startLine
            # Remove comments
            for i in np.arange(numLines-1,-1,-1): # Start from bottom
                if ';' in geom[i].lstrip(' ')[0]: geom.pop(i)

            print "### Geometry file: "
            print geom

            numQuads = 4
            numAsics = 16
            columns=['min_fs','min_ss','max_fs','max_ss','res','fs','ss','corner_x','corner_y']
            columnsScan=['fsx','fsy','ssx','ssy']
            indexScan=[]
            for i in np.arange(numQuads):
                for j in np.arange(numAsics):
                    indexScan.append('q'+str(i)+'a'+str(j))

            dfGeom = pd.DataFrame(np.empty((numQuads*numAsics,len(columns))), index=indexScan, columns=columns)
            dfScan = pd.DataFrame(np.empty((numQuads*numAsics,len(columnsScan))), index=indexScan, columns=columnsScan)
            counter = 0
            for i in np.arange(numQuads):
                for j in np.arange(numAsics):
                    myAsic = indexScan[counter]
                    for k in columns:
                        myLine = [s for s in geom if myAsic+'/'+k in s]
                        myVal = myLine[-1].split('=')[-1].rstrip().lstrip()
                        if k == 'fs' or k == 'ss':
                            dfGeom.loc[myAsic,k] = myVal
                        else:
                            dfGeom.loc[myAsic,k] = float(myVal)
                        if k == 'fs':
                            fsx = float(myVal.split('x')[0])
                            fsy = float(myVal.split('x')[-1].split('y')[0])
                            dfScan.loc[myAsic,'fsx'] = fsx
                            dfScan.loc[myAsic,'fsy'] = fsy
                        elif k == 'ss':
                            ssx = float(myVal.split('x')[0])
                            ssy = float(myVal.split('x')[-1].split('y')[0])
                            dfScan.loc[myAsic,'ssx'] = ssx
                            dfScan.loc[myAsic,'ssy'] = ssy
                    counter += 1
            print "#### GEOM: "
            print dfGeom
            print "#### SCAN: "
            print dfScan
            f.close()

        # Read CrystFEL indexed peaks
        if mySuccessString in err: # success
            f = open('.temp.stream')
            content = f.readlines()
            for i, val in enumerate(content):
                if   'num_peaks =' in val:
                    numPeaks = int(val.split('=')[-1])
                elif   'fs/px   ss/px (1/d)/nm^-1   Intensity  Panel' in val:
                    startLine = i+1
                    endLine = startLine+numPeaks
                    break
            print "### Peaks: "
            print content[startLine:endLine]

            columns=['fs','ss','res','intensity','asic']
            df = pd.DataFrame(np.empty((numPeaks,len(columns))), columns=columns)
            for i in np.arange(numPeaks):
                contentLine = startLine+i
                df['fs'][i] = float(content[contentLine][0:7])
                df['ss'][i] = float(content[contentLine][7:15])
                df['res'][i] = float(content[contentLine][15:26])
                df['intensity'][i] = float(content[contentLine][26:38])
                df['asic'][i] = str(content[contentLine][38:-1])
            print "### Stream"
            print df
            f.close()

            # Convert to CrystFEL coordinates
            columnsPeaks=['x','y','psocakeX','psocakeY']
            dfPeaks = pd.DataFrame(np.empty((numPeaks,len(columnsPeaks))), columns=columnsPeaks)
            for i in np.arange(numPeaks):
                myAsic = df['asic'][i].strip()
                x = (df['fs'][i] - dfGeom.loc[myAsic,'min_fs'])*dfScan.loc[myAsic,'fsx'] + (df['ss'][i] - dfGeom.loc[myAsic,'min_ss'])*dfScan.loc[myAsic,'ssx']
                x += dfGeom.loc[myAsic,'corner_x']
                y = (df['fs'][i] - dfGeom.loc[myAsic,'min_fs'])*dfScan.loc[myAsic,'fsy'] + (df['ss'][i] - dfGeom.loc[myAsic,'min_ss'])*dfScan.loc[myAsic,'ssy']
                y += dfGeom.loc[myAsic,'corner_y']
                dfPeaks['x'][i]=x
                dfPeaks['y'][i]=y

            # Convert to psocake coordinates
            for i in np.arange(numPeaks):
                dfPeaks['psocakeX'][i] = self.parent.cx - dfPeaks['x'][i]
                dfPeaks['psocakeY'][i] = self.parent.cy + dfPeaks['y'][i]

            print "dfPeaks: "
            print dfPeaks
            if self.parent.showIndexedPeaks:
                print "$$$$$$$$$$$$$$$$$ show indexed peaks"
                self.parent.numIndexedPeaksFound = numPeaks
                self.parent.indexedPeaks = dfPeaks[['psocakeX','psocakeY']].as_matrix()
                print self.parent.indexedPeaks.shape
                print np.array(self.parent.indexedPeaks[:,0],dtype=np.int64)
                self.parent.drawIndexedPeaks()



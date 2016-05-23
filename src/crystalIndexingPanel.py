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
        self.index_geom_str = 'CrystFEL geometry'
        self.index_intRadius_str = 'Integration radii'
        self.index_pdb_str = 'PDB'
        self.index_method_str = 'Indexing method'

        self.indexingOn = False
        self.geom = ''
        self.intRadius = '2,3,4'
        self.pdb = ''
        self.indexingMethod = 'dirax-axes-latt'

        #######################
        # Mandatory parameter #
        #######################
        self.params = [
            {'name': self.index_grp, 'type': 'group', 'children': [
                {'name': self.index_on_str, 'type': 'bool', 'value': self.indexingOn, 'tip': "Turn on indexing"},
                {'name': self.index_geom_str, 'type': 'str', 'value': self.geom, 'tip': "Turn on indexing"},
                {'name': self.index_intRadius_str, 'type': 'str', 'value': self.intRadius, 'tip': "Turn on indexing"},
                {'name': self.index_pdb_str, 'type': 'str', 'value': self.pdb, 'tip': "Turn on indexing"},
                {'name': self.index_method_str, 'type': 'str', 'value': self.indexingMethod, 'tip': "Turn on indexing"},
            ]},
        ]
#-g .temp.geom --peaks=cxi --int-radius=2,3,4 --pdb=lys.cell --indexing=dirax-axes-latt

    ##############################
    # Mandatory parameter update #
    ##############################
    def paramUpdate(self, path, change, data):
        print "paramUpdate: ", path[1]
        if path[1] == self.index_on_str:
            print "Got here"
            self.updateIndexStatus(data)
        elif path[1] == self.index_geom_str:
            self.updateGeom(data)
        elif path[1] == self.index_intRadius_str:
            self.updateIntegrationRadius(data)
        elif path[1] == self.index_pdb_str:
            self.updatePDB(data)
        elif path[1] == self.index_method_str:
            self.updateIndexingMethod(data)

    def updateIndexStatus(self, data):
        self.indexingOn = data
        print "indexing on: ", self.indexingOn
        if self.indexingOn:
            self.updateIndex()
        print "Done updateIndexStatus"

    def updateGeom(self, data):
        self.geom = data
        if self.indexingOn:
            self.updateIndex()

    def updateIntegrationRadius(self, data):
        self.intRadius = data
        if self.indexingOn:
            self.updateIndex()

    def updatePDB(self, data):
        self.pdb = data
        if self.indexingOn:
            self.updateIndex()

    def updateIndexingMethod(self, data):
        self.indexingMethod = data
        if self.indexingOn:
            self.updateIndex()

    def updateIndex(self):
        self.indexer = IndexHandler(parent=self.parent)
        self.indexer.computeIndex(self.parent.experimentName, self.parent.runNumber, self.parent.detInfo,
                                  self.parent.eventNumber, self.geom, self.intRadius, self.pdb, self.indexingMethod)
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
        self.geom = None
        self.intRadius = None
        self.pdb = None
        self.indexingMethod = None

    def __del__(self):
        print "del IndexHandler #$!@#$!#"
        self.exiting = True
        self.wait()

    def computeIndex(self, experimentName, runNumber, detInfo, eventNumber, geom, intRadius, pdb, indexingMethod):
        self.experimentName = experimentName
        self.runNumber = runNumber
        self.detInfo = detInfo
        self.eventNumber = eventNumber
        self.geom = geom
        self.intRadius = intRadius
        self.pdb = pdb
        self.indexingMethod = indexingMethod
        if self.geom is not '':
            self.start()

    def run(self):
        print "Running indexing!!!!!!!!!!!!"
        # Running indexing ...
        self.parent.numIndexedPeaksFound = 0
        self.parent.indexedPeaks = None
        self.parent.clearIndexedPeaks()

        # Write list
        with open(self.parent.hiddenCrystfelList, "w") as text_file:
            text_file.write("{} //0".format(self.parent.hiddenCXI))

        # FIXME: convert psana geom to crystfel geom
        cmd = "indexamajig -j 1 -i .temp.lst -g "+self.geom+" --peaks=cxi --int-radius="+self.intRadius+" --indexing="+self.indexingMethod+" -o .temp.stream"
        if self.pdb is not '':
            cmd += " --pdb="+self.pdb

        print "Submitting batch job: ", cmd
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        print "out: ", out
        print "err: ", err

        mySuccessString = "1 had crystals"
        # Read CrystFEL CSPAD geometry in stream
        if mySuccessString in err: # success
            print "Indexing successful"
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
        else:
            print "Indexing failed"
            self.parent.drawIndexedPeaks()

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
            if self.parent.showIndexedPeaks and self.eventNumber == self.parent.eventNumber:
                print "$$$$$$$$$$$$$$$$$ show indexed peaks"
                self.parent.numIndexedPeaksFound = numPeaks
                self.parent.indexedPeaks = dfPeaks[['psocakeX','psocakeY']].as_matrix()
                print self.parent.indexedPeaks.shape
                print np.array(self.parent.indexedPeaks[:,0],dtype=np.int64)
                self.parent.drawIndexedPeaks()



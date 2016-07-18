from pyqtgraph.Qt import QtCore
from LogBook.runtables import RunTables
import time, json

class LogbookCrawler(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        self.parent = parent
        self.experimentName = None
        self.outDir = None
        # Setup elog
        self.rt = RunTables(**{'web-service-url': 'https://pswww.slac.stanford.edu/ws-kerb'})

    def __del__(self):
        self.exiting = True
        self.wait()

    def updateLogbook(self, experimentName, outDir): # Pass in hit parameters
        self.experimentName = experimentName
        self.outDir = outDir
        self.table = self.rt.findUserTable(exper_name=self.experimentName, table_name='Run summary')
        self.start()

    def run(self):
        while 1:
            # Get number of runs
            lastRun = self.table.values(0)['last_run']
            for run in range(lastRun):
                # hit finder
                fname = self.outDir + '/r' + str(run).zfill(4) + '/status_hits.txt'
                try:
                    with open(fname) as infile:
                        d = json.load(infile)
                        numHits = d['numHits']
                        hitRate = d['hitRate']
                        fracDone = d['fracDone']
                        if fracDone == 100:
                            msg = '{0:.1f} hits / {1:.1f}% rate'.format(numHits, hitRate)
                        else:
                            msg = '{0:.1f} hits / {1:.1f}% rate / {2:.1f}% done'.format(numHits, hitRate, fracDone)
                        self.table.setValue(run, "Number of hits", msg)
                except:  # file may not exist yet
                    continue
                # indexing
                fname = self.outDir + '/r' + str(run).zfill(4) + '/status_index.txt'
                try:
                    with open(fname) as infile:
                        d = json.load(infile)
                        numIndexed = d['numIndexed']
                        indexingRate = d['indexRate']
                        fracDone = d['fracDone']
                        if fracDone == 100:
                            msg = '{0:.1f} indexed / {1:.1f}% rate'.format(numIndexed, indexingRate)
                        else:
                            msg = '{0:.1f} indexed / {1:.1f}% rate / {2:.1f}% done'.format(numIndexed, indexingRate, fracDone)
                        self.table.setValue(run, "Number of indexed", msg)
                except:  # file may not exist yet
                    continue
            time.sleep(10) # logbook update
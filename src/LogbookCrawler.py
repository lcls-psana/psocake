# Logbook crawler only update while the GUI is open
from pyqtgraph.Qt import QtCore
from LogBook.runtables import RunTables
import time, json, os

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
            try:
                # Get number of runs
                lastRun = self.table.values(0)['last_run']
                for run in range(lastRun):
                    # hit finder
                    fname = self.outDir + '/r' + str(run).zfill(4) + '/status_hits.txt'
                    if os.path.exists(fname):
                        with open(fname) as infile:
                            d = json.load(infile)
                            if 'message' in d:
                                msg = d['message']
                            elif 'numHits' in d:
                                numHits = d['numHits']
                                hitRate = d['hitRate']
                                fracDone = d['fracDone']
                                if fracDone == 100:
                                    msg = '{0:.1f} hits / {1:.1f}% rate'.format(numHits, hitRate)
                                else:
                                    msg = '{0:.1f} hits / {1:.1f}% rate / {2:.1f}% done'.format(numHits, hitRate,
                                                                                                fracDone)
                            else:
                                fracDone = d['fracDone']
                                msg = '{0:.1f}% done'.format(fracDone)
                            self.table.setValue(run, "Number of hits", msg)

                    # peak finder
                    fname = self.outDir + '/r' + str(run).zfill(4) + '/status_peaks.txt'
                    if os.path.exists(fname):
                        with open(fname) as infile:
                            d = json.load(infile)
                            if 'message' in d:
                                msg = d['message']
                            else:
                                numHits = d['numHits']
                                hitRate = d['hitRate']
                                fracDone = d['fracDone']
                                if fracDone == 100:
                                    msg = '{0:.1f} hits / {1:.1f}% rate'.format(numHits, hitRate)
                                else:
                                    msg = '{0:.1f} hits / {1:.1f}% rate / {2:.1f}% done'.format(numHits, hitRate, fracDone)
                            self.table.setValue(run, "Number of hits", msg)

                    # indexing
                    fname = self.outDir + '/r' + str(run).zfill(4) + '/status_index.txt'
                    if os.path.exists(fname):
                        with open(fname) as infile:
                            d = json.load(infile)
                            if 'message' in d:
                                msg = d['message']
                            elif 'convert' in d:
                                msg = "{0:.1f}% CXIDB ".format(d['fracDone'])
                            else:
                                numIndexed = d['numIndexed']
                                indexingRate = d['indexRate']
                                fracDone = d['fracDone']
                                if fracDone == 100:
                                    msg = '{0:.1f} indexed / {1:.1f}% rate'.format(numIndexed, indexingRate)
                                else:
                                    msg = '{0:.1f} indexed / {1:.1f}% rate / {2:.1f}% done'.format(numIndexed, indexingRate, fracDone)
                            self.table.setValue(run, "Number of indexed", msg)
                time.sleep(10)  # logbook update
            except:
                time.sleep(10)
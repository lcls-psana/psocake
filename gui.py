import sys, signal
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.console
import numpy as np
from pyqtgraph.dockarea import *
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import psana
from Detector.PyDetector import PyDetector

class MainFrame(QtGui.QWidget):
    """
    The main frame of the application
    """
    def __init__(self, arg_list):
        super(MainFrame, self).__init__()    
        self.ds = psana.DataSource('exp=amo86615:run=190:idx')
        self.src = psana.Source('DetInfo(Camp.0:pnCCD.1)')
        self.run = self.ds.runs().next()
        self.times = self.run.times()
        self.totalEvents = len(self.times)
        self.counter = 0
        self.env = self.ds.env()
        self.det = PyDetector(self.src, self.env, pbits=0)
        self.data = self.getEvt(self.counter)
        self.initUI()

    def getEvt(self,index):
        evt = self.run.event(self.times[index])
        calib = self.det.calib_data(evt)
        self.det.common_mode_apply(evt, calib)
        if calib is not None: 
            self.data = self.det.image(evt,calib)
            return self.data
        else:
            return None

    def initUI(self):
        ## Define a top-level widget to hold everything
        self.win = QtGui.QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1200,1200)
        self.win.setWindowTitle('psbrowser')

        ## Create docks, place them into the window one at a time.
        ## Note that size arguments are only a suggestion; docks will still have to
        ## fill the entire dock area and obey the limits of their internal widgets.
        self.d1 = Dock("Dock1", size=(1, 1))     ## give this dock the minimum possible size
        self.d2 = Dock("Dock2 - Console", size=(500,300), closable=True)
        self.d3 = Dock("Dock3", size=(500,400))
        self.d4 = Dock("Dock4 (tabbed) - Plot", size=(500,200))
        self.d5 = Dock("Dock5 - Image", size=(900,900))
        self.d6 = Dock("Dock6 (tabbed) - Plot", size=(500,200))
        self.d7 = Dock("Dock7 - Parameters", size=(500,200))
        self.area.addDock(self.d1, 'left')      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
        self.area.addDock(self.d2, 'right')     ## place d2 at right edge of dock area
        self.area.addDock(self.d3, 'bottom', self.d1)## place d3 at bottom edge of d1
        self.area.addDock(self.d4, 'right')     ## place d4 at right edge of dock area
        self.area.addDock(self.d5, 'left', self.d1)  ## place d5 at left edge of d1
        self.area.addDock(self.d6, 'top', self.d4)   ## place d5 at top edge of d4
        self.area.addDock(self.d7, 'bottom', self.d5)   ## place d7 at left edge of d5

        ## Add widgets into each dock
        self.w2 = pg.console.ConsoleWidget()
        self.d2.addWidget(self.w2)

        ## Hide title bar on dock 3
        self.d3.hideTitleBar()
        self.w3 = pg.PlotWidget(title="Plot inside dock with no title bar")
        self.w3.plot(np.random.normal(size=100))
        self.d3.addWidget(self.w3)

        ## Dock 4
        self.w4 = pg.PlotWidget(title="Dock 4 plot")
        self.w4.plot(np.random.normal(size=100))
        self.d4.addWidget(self.w4)

        ## Dock 5
        self.nextBtn = QtGui.QPushButton('Next evt')
        self.prevBtn = QtGui.QPushButton('Prev evt')
        self.wQ = pg.LayoutWidget()
        self.w5 = pg.ImageView()
        self.wQ.addWidget(self.w5, row=0, colspan=2)
        self.wQ.addWidget(self.prevBtn, row=1, col=0)
        self.wQ.addWidget(self.nextBtn, row=1, col=1)
        def next():
            self.counter += 1
            self.data = self.getEvt(self.counter)
            self.w5.setImage(self.data)
        def prev():
            self.counter -= 1
            self.data = self.getEvt(self.counter)
            self.w5.setImage(self.data)
        self.nextBtn.clicked.connect(next)
        self.prevBtn.clicked.connect(prev)
        self.d5.addWidget(self.wQ)

        ## Dock 6
        self.w6 = pg.PlotWidget(title="Dock 6 plot")
        self.w6.plot(np.random.normal(size=100))
        self.d6.addWidget(self.w6)

        self.win.show()

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)    
    app = QtGui.QApplication(sys.argv)
    ex = MainFrame(sys.argv)
    sys.exit(app.exec_())

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    main()
#    import sys
#    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#        QtGui.QApplication.instance().exec_()

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
        self.d2 = Dock("Dock2 - Parameters", size=(500,200))
        self.d3 = Dock("Dock3", size=(500,400))
        self.d4 = Dock("Dock4 (tabbed) - Plot", size=(500,200))
        self.d5 = Dock("Dock5 - Image", size=(900,900))
        self.d6 = Dock("Dock6 (tabbed) - Plot", size=(500,200))
        self.d7 = Dock("Dock7 - Console", size=(500,300), closable=True)

        self.area.addDock(self.d1, 'left')      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
        self.area.addDock(self.d2, 'right')     ## place d2 at right edge of dock area
        self.area.addDock(self.d3, 'bottom', self.d1)## place d3 at bottom edge of d1
        self.area.addDock(self.d4, 'right')     ## place d4 at right edge of dock area
        self.area.addDock(self.d5, 'left', self.d1)  ## place d5 at left edge of d1
        self.area.addDock(self.d6, 'top', self.d4)   ## place d5 at top edge of d4
        self.area.addDock(self.d7, 'bottom', self.d5)   ## place d7 at left edge of d5

        ## Dock 2: parameter
        self.w2 = ParameterTree()
        self.w2.setParameters(p, showTop=True)
        self.w2.setWindowTitle('Parameters')
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
        def jumpTo(evt):
            self.counter = evt
            self.data = self.getEvt(self.counter)
            self.w5.setImage(self.data)

        import h5py
        #f = h5py.File('/reg/d/psdm/amo/amo86615/scratch/yoon82/amo86615_89_139_class_v1.h5','r')
        #data = np.array(f['/hitClass/adu'])
        #f.close()
        #img = pg.gaussianFilter(np.random.normal(size=(200, 200)), (5, 5)) * 20 + 100
        #img = img[np.newaxis,:,:]
        #decay = np.exp(-np.linspace(0,0.3,100))[:,np.newaxis,np.newaxis]
        data = np.random.normal(size=(100, 200, 200))
        #data += img * decay
        #data += 2
        self.w5.setImage(data, xvals=np.linspace(1., data.shape[0], data.shape[0]))
        self.nextBtn.clicked.connect(next)
        self.prevBtn.clicked.connect(prev)
        self.d5.addWidget(self.wQ)

        ## Dock 6
        self.w6 = pg.PlotWidget(title="Dock 6 plot")
        self.w6.plot(np.random.normal(size=100))
        self.d6.addWidget(self.w6)

        ## Dock 7: console
        self.w7 = pg.console.ConsoleWidget()
        self.d7.addWidget(self.w7)

        self.win.show()

# Set up list of parameters
params = [
    {'name': 'Basic parameter data types', 'type': 'group', 'children': [
        {'name': 'Integer', 'type': 'int', 'value': 10},
        {'name': 'Float', 'type': 'float', 'value': 10.5, 'step': 0.1},
        {'name': 'String', 'type': 'str', 'value': "hi"},
        {'name': 'List', 'type': 'list', 'values': [1,2,3], 'value': 2},
        {'name': 'Named List', 'type': 'list', 'values': {"one": 1, "two": "twosies", "three": [3,3,3]}, 'value': 2},
        {'name': 'Boolean', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
        {'name': 'Color', 'type': 'color', 'value': "FF0", 'tip': "This is a color button"},
        {'name': 'Gradient', 'type': 'colormap'},
        {'name': 'Subgroup', 'type': 'group', 'children': [
            {'name': 'Sub-param 1', 'type': 'int', 'value': 10},
            {'name': 'Sub-param 2', 'type': 'float', 'value': 1.2e6},
        ]},
        {'name': 'Text Parameter', 'type': 'text', 'value': 'Some text...'},
        {'name': 'Action Parameter', 'type': 'action'},
    ]},
    {'name': 'Numerical Parameter Options', 'type': 'group', 'children': [
        {'name': 'Units + SI prefix', 'type': 'float', 'value': 1.2e-6, 'step': 1e-6, 'siPrefix': True, 'suffix': 'V'},
        {'name': 'Limits (min=7;max=15)', 'type': 'int', 'value': 11, 'limits': (7, 15), 'default': -6},
        {'name': 'DEC stepping', 'type': 'float', 'value': 1.2e6, 'dec': True, 'step': 1, 'siPrefix': True, 'suffix': 'Hz'},
        
    ]},
    {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
        {'name': 'Save State', 'type': 'action'},
        {'name': 'Restore State', 'type': 'action', 'children': [
            {'name': 'Add missing items', 'type': 'bool', 'value': True},
            {'name': 'Remove extra items', 'type': 'bool', 'value': True},
        ]},
    ]},
    {'name': 'Extra Parameter Options', 'type': 'group', 'children': [
        {'name': 'Read-only', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'readonly': True},
        {'name': 'Renamable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'renamable': True},
        {'name': 'Removable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'removable': True},
    ]},
]

## Create tree of Parameter objects
p = Parameter.create(name='params', type='group', children=params)

## If anything changes in the parameter tree, print a message
def change(param, changes):
    print("tree changes:")
    for param, change, data in changes:
        path = p.childPath(param)
        if path is not None:
            childName = '.'.join(path)
        else:
            childName = param.name()
        print('  parameter: %s'% childName)
        print('  change:    %s'% change)
        print('  data:      %s'% str(data))
        print('  ----------')

def save():
    global state
    state = p.saveState()
    print "call save"

## Listen for changes to parameters
p.sigTreeStateChanged.connect(change)
p.param('Basic parameter data types', 'Integer').sigNameChanged.connect(save)

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

from pyqtgraph.dockarea import *
import pyqtgraph as pg
import _colorScheme as color

class Mouse(object):
    def __init__(self, parent = None):
        self.parent = parent
        ## Dock 1 - mouse intensity display
        self.dock = Dock("Mouse", size=(500, 75), closable=False)
        self.dock.hideTitleBar()
        self.win = pg.GraphicsView(background=pg.mkColor(color.mouseBackground))
        self.dock.addWidget(self.win)

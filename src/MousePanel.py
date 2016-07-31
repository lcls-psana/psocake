from pyqtgraph.dockarea import *
import pyqtgraph as pg
import _colorScheme as color
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.pyplot as plt

class Mouse(object):
    def __init__(self, parent = None):
        self.parent = parent

        self.d5 = Dock("Mouse", size=(500, 75), closable=False)
        ## Dock 5 - mouse intensity display
        #self.d5.hideTitleBar()
        self.w5 = pg.GraphicsView(background=pg.mkColor(color.sandstone100_rgb))
        self.d5.addWidget(self.w5)
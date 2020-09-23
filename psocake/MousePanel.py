from pyqtgraph.dockarea import *
import pyqtgraph as pg
import _colorScheme as color
from pyqtgraph.Qt import QtCore, QtGui
from random import shuffle
import os

try:
    from PyQt5.QtWidgets import *
    using_pyqt4 = False
except ImportError:
    using_pyqt4 = True

def getGifPath(debug=False):
    if debug:
        path = os.path.dirname(__file__)
        gifDir = os.path.join(path, '../data/graphics')
        return gifDir
    else:
        paths = os.getenv('SIT_DATA')
        if paths:
            for path in paths.split(":"):
                gifDir = os.path.join(path, 'psocake/graphics')
                if os.path.exists(gifDir):
                    return gifDir
        else:
            print "Error: Couldn't find graphics folder"
            exit()

class Mouse(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)

        self.parent = parent
        ## Dock 1 - mouse intensity display
        self.dock = Dock("Mouse", size=(1, 1), closable=False)
        self.dock.hideTitleBar()
        self.win = pg.GraphicsView(background=pg.mkColor(color.mouseBackground))
        self.dock.addWidget(self.win)
        self.tm = ThreadsafeTimer(self.parent)

        gifFolder = getGifPath(debug=False)
        self.fnames = []
        for file in os.listdir(gifFolder):
            if file.endswith(".gif"):
                self.fnames.append(os.path.join(gifFolder, file))
        shuffle(self.fnames)

        self.movie_screen = None
        self.movie = None
        self.speed = 120
        self.index = 0

    def addGif(self):
        # Load the file into a QMovie
        fname = self.fnames[self.index % len(self.fnames)]
        self.index += 1
        if self.movie is None:
            # get a frame to get image size
            self.movie = QtGui.QMovie(fname, QtCore.QByteArray(), self)
            self.movie.jumpToFrame(0)
            movie_size = self.movie.currentImage().size()
            movie_aspect = movie_size.width() * 1.0 / movie_size.height()
            # create a new movie with correct aspect ratio
            self.movie = QtGui.QMovie(fname, QtCore.QByteArray(), self)
            size = self.movie.scaledSize()
            self.setGeometry(100, 100, size.width(), size.height())
            self.movie.setScaledSize(QtCore.QSize(int(150*movie_aspect),150)) # pixels

            # Create the layout
            if self.layout is None:
                main_layout = QtGui.QVBoxLayout()
                self.setLayout(main_layout)

            self.movie_screen = QtGui.QLabel()
            # Make label fit the gif
            self.movie_screen.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
            self.movie_screen.setAlignment(QtCore.Qt.AlignCenter)
            self.dock.addWidget(self.movie_screen)

            # Add the QMovie object to the label
            self.movie.setCacheMode(QtGui.QMovie.CacheAll)
            self.movie.setSpeed(self.speed)
            self.movie_screen.setMovie(self.movie)
            self.movie.start()
        else:
            pass

    def removeGif(self):
        try:
            if self.movie_screen:
                self.movie.stop()
                self.movie = None #self.movie.deleteLater()
                self.movie_screen.setParent(None)
                self.movie_screen = None #self.movie_screen.deleteLater()
        except:
            pass

class ThreadsafeTimer(QtCore.QObject):
    """
    Thread-safe replacement for QTimer.
    """
    timeout = QtCore.Signal()
    sigTimerStopRequested = QtCore.Signal()
    sigTimerStartRequested = QtCore.Signal(object)

    def __init__(self, parent = None):
        QtCore.QObject.__init__(self, parent)
        self.parent = parent
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timerFinished)
        self.timer.moveToThread(QtCore.QCoreApplication.instance().thread())
        self.moveToThread(QtCore.QCoreApplication.instance().thread())
        self.sigTimerStopRequested.connect(self.stop, QtCore.Qt.QueuedConnection)
        self.sigTimerStartRequested.connect(self.start, QtCore.Qt.QueuedConnection)

    def start(self, timeout):
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if isGuiThread:
            #print "start timer", self, "from gui thread"
            self.timer.start(timeout)
            self.parent.mouse.addGif()
        else:
            #print "start timer", self, "from remote thread"
            self.sigTimerStartRequested.emit(timeout)

    def stop(self):
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if isGuiThread:
            #print "stop timer", self, "from gui thread"
            self.timer.stop()
            self.parent.mouse.removeGif()
        else:
            #print "stop timer", self, "from remote thread"
            self.sigTimerStopRequested.emit()

    def timerFinished(self):
        #print "emit!!!!"
        self.timeout.emit()
        self.stop()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import psana
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
import sys
import pyqtgraph as pg
from matplotlib.widgets import Button
import pandas as pd
import os
import h5py
import argparse
import threading
import copy
from sklearn.neighbors import NearestNeighbors
import subprocess

# Parse user input
parser = argparse.ArgumentParser()
parser.add_argument("-e","--embedding",help="full filename to embed_<exp>.h5", type=str)
parser.add_argument("-f","--filepath",help="full path to <exp>_<run>.cxi", type=str)
parser.add_argument("-t","--tag",help="tag", default=None, type=str)
parser.add_argument("-v","--verbose",help="print detail", default=0, type=int)
args = parser.parse_args()
fname = args.embedding
path = args.filepath

# Get username
p = subprocess.Popen(['whoami'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = p.communicate()
user = out[:len(out)-1]

global minIntensity, maxIntensity
minIntensity = "-1"
maxIntensity = "-1"

global autoPropagate
autoPropagate = True

global maxEvents
maxEvents = "8"

global dialogVisible
dialogVisible = True
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

# Scroll area
class Scroll(QtGui.QScrollArea):

    def __init__(self, w, parent=None):
        QtGui.QScrollArea.__init__(self,parent)
        self.setWidget(w)
        self.setGeometry(0,0,w.width(),600)
        self.window = w

    def resizeEvent(self,resizeEvent):
        #self.setGeometry(self.x(), self.y(), self.width(), self.height())
        self.window.gridLayoutWidget.setGeometry(self.window.gridLayoutWidget.x(), self.window.gridLayoutWidget.y(), self.width()-30, (self.width()/self.window.gridLayout.columnCount())*self.window.gridLayout.rowCount())
        #self.window.gridLayout.setGeometry(self.window.gridLayoutWidget.geometry())
        self.window.setGeometry(self.window.x(), self.window.y(), self.width(), self.window.gridLayoutWidget.height()+30*self.window.gridLayout.rowCount())
        self.window.centralwidget.setGeometry(self.window.geometry())
        #QtGui.QMessageBox.information(self,"Information!","Window has been resized...")

class Window(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self,parent)
        self.centralwidget = QtGui.QWidget(self)
        # Read label types
        labelTypesFile = args.filepath+'/labels.txt'
        try:
            self.readLabelTypes(labelTypesFile)
        except:
            self.createLabelTypes(labelTypesFile)
            self.readLabelTypes(labelTypesFile)
        py = 10
        dh = 25
        # Min text
        self.l1 = QtGui.QLabel(self)
        self.l1.setText("Min:")
        self.l1.setGeometry(QtCore.QRect(10, py, 30, dh))
        self.l1.setObjectName(_fromUtf8("l1"))
        # Min 
        self.lineEdit_2 = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(40, py, 60, dh))
        self.lineEdit_2.setObjectName(_fromUtf8("lineEdit_2"))
        self.lineEdit_2.setText(minIntensity)
        self.lineEdit_2.textChanged[str].connect(self.onMinChanged)
        # Max text
        self.l2 = QtGui.QLabel(self)
        self.l2.setText("Max:")
        self.l2.setGeometry(QtCore.QRect(110, py, 30, dh))
        self.l2.setObjectName(_fromUtf8("l2"))
        # Max
        self.lineEdit_3 = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(140, py, 60, dh))
        self.lineEdit_3.setObjectName(_fromUtf8("lineEdit_3"))
        self.lineEdit_3.setText(maxIntensity)
        self.lineEdit_3.textChanged[str].connect(self.onMaxChanged)
        # Change contrast button
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(210, py, 120, dh))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton.clicked.connect(self.adjust)
        # Selection
        self.comboBox = QtGui.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(400, py, 120, dh))
        self.comboBox.setInsertPolicy(QtGui.QComboBox.NoInsert)
        self.comboBox.setObjectName(_fromUtf8("comboBox"))
        for i in range(self.numLabelTypes):
            self.comboBox.addItem(_fromUtf8(""))
        # Save label
        self.pushButton_2 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(530, py, 100, dh))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_2.clicked.connect(self.updateLabels)
        # Unselect
        self.pushButton_3 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(640, py, 100, dh))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.pushButton_3.clicked.connect(self.unselectAll)
        # Select
        self.pushButton_4 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(750, py, 100, dh))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.pushButton_4.clicked.connect(self.selectAll)
        # Maximum events text
        self.l3 = QtGui.QLabel(self)
        self.l3.setText("Max. Events:")
        self.l3.setGeometry(QtCore.QRect(1000, py, 80, dh))
        self.l3.setObjectName(_fromUtf8("l3"))
        # Maximum events to display
        self.lineEdit_4 = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(1080, py, 80, dh))
        self.lineEdit_4.setObjectName(_fromUtf8("lineEdit_4"))
        self.lineEdit_4.setText(maxEvents)
        self.lineEdit_4.textChanged[str].connect(self.onChanged)
        # Auto propagate
        self.cb = QtGui.QCheckBox('Auto propagate', self)
        self.cb.stateChanged.connect(self.changeTitle)
        if autoPropagate: self.cb.toggle()
        self.cb.setGeometry(QtCore.QRect(1170, py, 80, dh))

    def changeTitle(self, state):
        global autoPropagate
        if state == QtCore.Qt.Checked:
            autoPropagate = True
        else:
            autoPropagate = False

    def readLabelTypes(self, fname):
        f = open(fname, 'r')
        _types = f.readlines()
        f.close()
        self.labelTypes = []
        for i in _types:
            if i.strip(): self.labelTypes.append(i.strip())
        self.numLabelTypes = len(self.labelTypes)

    def createLabelTypes(self, fname):
        f = open(fname, 'w')
        f.write('A\n')
        f.write('B\n')
        f.write('None\n')
        f.close()

    def onMinChanged(self, text):
        global minIntensity
        self.lineEdit_2.setText(text)
        minIntensity = self.lineEdit_2.text()

    def onMaxChanged(self, text):
        global maxIntensity
        self.lineEdit_3.setText(text)
        maxIntensity = self.lineEdit_3.text()

    def onChanged(self, text):
        global maxEvents  
        self.lineEdit_4.setText(text)
        maxEvents = self.lineEdit_4.text()

    # Check user's scratch folder for labels.h5
    def updateLabels(self):
        global userLabels
        global propLabels
        global autoPropagate
        files = {}
        filePath = "/reg/d/psdm/" + expName[:3] + "/" + expName + "/scratch/" + user + "/psocake"
        for run in ir.runs:
            if not os.path.exists(filePath + '/r' + str(run).zfill(4)):
                os.makedirs(filePath + '/r' + str(run).zfill(4))
            f = h5py.File(filePath+ '/r' + str(run).zfill(4)+'/'+expName+'_'+str(run).zfill(4) + '_labels.h5', 'a')
            files[run] = f
        for idx in self.selected:
            evtNum = ir.eventInd[idx]
            if self.comboBox.currentIndex() != 3:
                userLabels[idx] = self.comboBox.currentIndex() + 1
                propLabels[idx] = self.comboBox.currentIndex() + 1
            else:
                userLabels[idx] = 0
                propLabels[idx] = 0
            try:
                files[ir.runList[idx]]['labels'][ir.eventInd[idx]] = userLabels[idx]
            except:
                if args.tag is not None:
                    eventsTotal = h5py.File(path+ '/r' + str(int(ir.runList[idx])).zfill(4)+'/'+expName+'_'+str(int(ir.runList[idx])).zfill(4) + '_' + args.tag + '.cxi', 'r+')['/entry_1/result_1/nHitsAll'].size
                else:
                    eventsTotal = h5py.File(path+ '/r' + str(int(ir.runList[idx])).zfill(4)+'/'+expName+'_'+str(int(ir.runList[idx])).zfill(4) + '.cxi', 'r+')['/entry_1/result_1/nHitsAll'].size
                dset = files[ir.runList[idx]].create_dataset("labels", (eventsTotal, 1))
                dset[ir.eventInd[idx]] = userLabels[idx]
            try:
                files[ir.runList[idx]]['propLabels'][ir.eventInd[idx]] = propLabels[idx]
            except:
                dset = files[ir.runList[idx]].create_dataset("propLabels", (files[ir.runList[idx]]['labels'].size, 1))
                dset[ir.eventInd[idx]] = propLabels[idx]
        for run, f in files.iteritems():
            f.close()
        propLabels = np.copy(userLabels)
        if autoPropagate: fillProp()
        self.unselectAll()
        fracUserLabels = getUserLabelFraction(userLabels)
        print "Provided Labels: ", round(fracUserLabels, 3), " %"

    def defaultBackground(self, idx):
        global propLabels
        global userLabels
        userColor = "#FFFFFF"
        propColor = "#FFFFFF"
        if propLabels[idx] == 1:
            propColor = "#DC143C"
        elif propLabels[idx] == 2:
            propColor = "#7FFF00"
        elif propLabels[idx] == 3:
            propColor = "#00BBFF"
        if userLabels[idx] == 1:
            userColor = "#DC143C"
        elif userLabels[idx] == 2:
            userColor = "#7FFF00"
        elif userLabels[idx] == 3:
            userColor = "#00BBFF"
        return userColor, propColor

    def textLabels(self, idx):
        global propLabels
        global userLabels
        if userLabels[idx] == 0:
            userLabel = "None"
        else:
            userLabel = self.labelTypes[int(userLabels[idx]-1)]
        if propLabels[idx] == 0:
            propLabel = "None"
        else:
            propLabel = self.labelTypes[int(propLabels[idx]-1)]
        return userLabel, propLabel

    # Select all the images
    def selectAll(self):
        self.selected = []
        for i in np.arange(self.gridLayout.count()):
            vb = self.gridLayout.itemAt(i).widget().getItem(0,0)
            vb.setBackgroundColor('w')
            self.selected.append(self.ind[i])

    # Refresh the colors on the manifold
    def unselectAll(self):
        global ax
        global X
        for idx, val in enumerate(self.ind):
            if val in self.selected: self.selected.remove(val)
            self.gridLayout.itemAt(idx).widget().getItem(0,0).setBackgroundColor("000")
            userLabel, propLabel = self.textLabels(val)
            userColor, propColor = self.defaultBackground(val)
            self.textBoxes[idx][0].setHtml('<font size="4" color="' + userColor + '"><b>User Label: ' + userLabel + '</b></font>')
            self.textBoxes[idx][1].setHtml('<font size="4" color="' + propColor + '"><b>Propagated Label: ' + propLabel + '</b></font>')
        ax.clear()
        colors = ['red' if propLabels[idx] == 1 else 'green' if propLabels[idx] == 2 else 'blue' if propLabels[idx] == 3 else 'black' for idx, val in enumerate(X[:,1])]
        ax.scatter(X[:,0],X[:,1],X[:,2], color = colors, picker=5, alpha=0.1)

    # Adjust image display
    def adjust(self):
        global minIntensity, maxIntensity
        for val in self.selected:
            idx = self.ind.index(val)
            img = self.gridLayout.itemAt(idx).widget().getItem(0,0).allChildren()[1]
            if int(minIntensity) is not -1 and int(maxIntensity) is not -1:
                img.setLevels([float(self.lineEdit_2.text()), float(self.lineEdit_3.text())])

    def pop(self, arr, h, w, ind):
        self.gridLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 39, h, w))
        self.gridLayoutWidget.setObjectName(_fromUtf8("gridLayoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.ind = ind
        self.selected = []
        self.textBoxes = []
        self.images = arr
        num = float(len(arr))/4.0
        if int(num) != num: num = int(num + 1)
        else: num = int(num)
        #self.layout.setSpacing(25)
        global ir, minIntensity, maxIntensity
        for row in range(num):
            for column in range(4):
                if(row*4+column < len(arr)):
                    #img = pg.ImageView()
                    #img.setImage(arr[row*4+column])
                    #img.ui.histogram.hide()
                    imageWidget = pg.GraphicsLayoutWidget()
                    vb = imageWidget.addViewBox()
                    #text.setText()
                    idx = self.ind[row*4+column]
                    userLabel, propLabel = self.textLabels(idx)
                    userColor, propColor = self.defaultBackground(idx)
                    img = pg.ImageItem()
                    img.setImage(arr[row*4+column])

                    if int(minIntensity) is not -1 and int(maxIntensity) is not -1:
                        img.setLevels([float(minIntensity), float(maxIntensity)])
                    #else:
                    #    img.setLevels([0,55])

                    vb.addItem(img)
                    origX = img.image.shape[0]/10.0
                    origY = img.image.shape[1]-img.image.shape[1]/10.0
                    # Display user label
                    text = pg.TextItem()
                    text.setHtml('<font size="4" color="' + userColor + '"><b>User Label: ' + userLabel + '</b></font>')
                    vb.addItem(text)
                    text.setPos(origX, origY)
                    # Display propagated label
                    text1 = pg.TextItem()
                    text1.setHtml('<font size="4" color="' + propColor + '"><b>Propagated Label: ' + propLabel + '</b></font>')
                    vb.addItem(text1)
                    text1.setPos(origX, origY-60)
                    # Display event number
                    text2 = pg.TextItem()
                    text2.setHtml('<font size="4" color=white><b>Run,evt: ' + str(int(ir.runList[idx])) + "," + str(int(ir.eventInd[idx])) + '</b></font>')
                    vb.addItem(text2)
                    text2.setPos(origX, origY-120)
                    # Lock aspect ratio
                    vb.setAspectLocked(True)
                    imageWidget.scene().sigMouseClicked.connect(self.buttonClicked)
                    self.gridLayout.addWidget(imageWidget, row, column)
                    texts = [text, text1]
                    self.textBoxes.append(texts)

    def buttonClicked(self, event):
        button = self.sender().items()
        vb = button[len(button)-2]
        idx = -1
        for i in np.arange(self.gridLayout.count()):
            if vb == self.gridLayout.itemAt(i).widget().getItem(0,0):
                idx = i
                break
        if self.ind[idx] in self.selected:
            vb.setBackgroundColor("000")
            self.selected.remove(self.ind[idx])
        else:
            vb.setBackgroundColor('w')
            self.selected.append(self.ind[idx])

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.pushButton.setText(_translate("MainWindow", "Change Contrast", None))
        for i, label in enumerate(self.labelTypes):
            self.comboBox.setItemText(i, _translate("MainWindow", label, None))
        self.pushButton_2.setText(_translate("MainWindow", "Save Label", None))
        self.pushButton_3.setText(_translate("MainWindow", "Unselect all", None))
        self.pushButton_4.setText(_translate("MainWindow", "Select all", None))

class Dialog(QtGui.QDialog):
    def __init__(self, num, col, images, ind, parent=None):
        global dialogVisible
        QtGui.QDialog.__init__(self,parent)
        self.w = Window(self)
        self.w.retranslateUi(self.w)
        self.w.setGeometry(0, 300, col, num*400)
        self.w.pop(images, col, num*400, ind)
        self.setGeometry(0,0,col,num*400)
        self.s = Scroll(self.w, self)
        self.s.resize(self.width(), self.height())
        self.s = Scroll(self.w, self)
        self.s.show()
        self.w.show()
        dialogVisible = True

    def resizeEvent(self, resizeEvent):
        self.s.resize(self.width(), self.height())

    def refresh(self, num, col, images, ind):
        self.s.hide()
        self.w.hide()
        self.w = Window(self)
        self.w.retranslateUi(self.w)
        self.w.setGeometry(0, 300, col, num*400)
        self.w.pop(images, col, num*400, ind)
        self.setGeometry(0,0,col,num*400)
        self.s.resize(self.width(), self.height())
        self.s = Scroll(self.w, self)
        self.s.show()
        self.w.show()
        self.s.resize(self.width(), self.height())

    def closeEvent(self, closeEvent):
        global dialogVisible
        dialogVisible = False

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.d = Dialog(2, 1600, [],[])
        self.d.show()

    def test(self, num, col, images,ind):
        global dialogVisible
        if dialogVisible: self.d.refresh(num, col, images, ind)
        else: 
            self.d = Dialog(2, 1600, [],[])
            self.d.show()
            self.d.refresh(num, col, images, ind)

class imageRetriever:
    def __init__(self,expName,runList,detInfo,filepath):
        self.expName = expName
        self.runs = []
        self.runList = runList
        self.detInfo = detInfo
        self.filepath = filepath
        self.myRuns = self.digestRunList(self.runList)
        self.fileList = []
        self.numHitsPerFile = []
        self.allHitInd = np.array([])
        self.eventInd = np.array([])
        self.runList = np.array([])
        self.run, self.times, self.det, self.evt = self.setup(self.expName,self.myRuns[0],self.detInfo)
        self.lastRun = 0

        for r in self.myRuns:
            if args.tag is not None:
                filename = filepath+ '/r' + str(r).zfill(4)+'/'+expName+'_'+str(r).zfill(4) + '_' + args.tag +'.cxi'
            else:
                filename = filepath+ '/r' + str(r).zfill(4)+'/'+expName+'_'+str(r).zfill(4) + '.cxi'
            if os.path.exists(filename):
                self.runs.append(r)
                f1 = h5py.File(filename,'r')

                numHits = f1['/entry_1/result_1/nHits'].attrs['numEvents']
                hitInd = np.arange(0,numHits)
                hitEvent = f1['/LCLS/eventNumber'].value
                runInd = np.ones((numHits,),dtype=int)*r

                if numHits > 0:
                    self.allHitInd = np.append(self.allHitInd,hitInd)
                    self.eventInd = np.append(self.eventInd,hitEvent)
                    self.runList = np.append(self.runList,runInd)
                    self.fileList.append(f1)
                    self.numHitsPerFile.append(numHits)
                    f1.close()

        self.totalHits = np.sum(self.numHitsPerFile)
        self.numFiles = len(self.fileList)
        self.accumHits = np.zeros(self.numFiles,)
        for i,val in enumerate(self.numHitsPerFile):
            self.accumHits[i] = np.sum(self.numHitsPerFile[0:i+1])
        print "totalHits: ", self.totalHits

    def digestRunList(self,runList):
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

    def setup(self,experimentName,runNumber,detInfo):
        ds = psana.DataSource('exp='+str(experimentName)+':run='+str(runNumber)+':idx')
        run = ds.runs().next()
        times = run.times()
        env = ds.env()
        evt = run.event(times[0])
        det = psana.Detector(str(detInfo), env)
        return run,times,det,evt

def setup(experimentName,runNumber,detInfo):
    ds = psana.DataSource('exp='+str(experimentName)+':run='+str(runNumber)+':idx')
    run = ds.runs().next()
    times = run.times()
    env = ds.env()
    evt = run.event(times[0])
    det = psana.Detector(str(detInfo), env)
    return run, times, det, evt

def diffusionKernel(X, eps, knn, D=None):
    nbrs = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree').fit(X)    
    D = nbrs.kneighbors_graph(X, mode='distance')
    term = D.multiply(D)/-eps
    G = np.exp(term.toarray())
    G[np.where(G==1)]=0
    G = G + np.eye(G.shape[0])
    deg = np.sum(G,axis=0)
    P = G/deg
    return P, D

def propagate(matrix, list):
    for i in np.arange(X[:,0].size):
        label = np.amax(matrix[:,i])
        if label != 0:
            idx = np.argmax(matrix[:,i])
            if propLabels[list[idx]] != 0:
                propLabels[i] = propLabels[list[idx]]
            else:
                propLabels[i] = userLabels[list[idx]]

def fillProp():
    global propLabels
    global userLabels
    num = 0
    while 1:
        list = []
        for i in np.arange(X[:,0].size):
            if propLabels[i] != 0 :
                list.append(i)
        matrix = P[list,:]
        if num == matrix.shape[0]:
            num = matrix.shape[0]
            break
        num = matrix.shape[0]
        propagate(matrix, list)
    #print "###: ", userLabels, len(userLabels)
    #propagate(P,np.arange(X[:,0].size))
    ##Update the dataset
    files = {}
    filePath = "/reg/d/psdm/" + expName[:3] + "/" + expName + "/scratch/" + user + "/psocake"
    for run in ir.runs:
        if not os.path.exists(filePath + '/r' + str(run).zfill(4)):
            os.makedirs(filePath + '/r' + str(run).zfill(4))
        f = h5py.File(filePath+ '/r' + str(run).zfill(4)+'/'+expName+'_'+str(run).zfill(4) + '_labels.h5', 'a')
        files[run] = f
    for idx, label in enumerate(propLabels):
        evtNum = ir.eventInd[idx]
        try:
            files[ir.runList[idx]]['propLabels'][ir.eventInd[idx]] = label
        except:
            if args.tag is not None:
                eventsTotal = h5py.File(filePath+ '/r' + str(int(ir.runList[idx])).zfill(4)+'/'+expName+'_'+str(int(ir.runList[idx])).zfill(4) + '_' + args.tag + '.cxi', 'r+')['/entry_1/result_1/nHitsAll'].size
            else:
                eventsTotal = h5py.File(filePath+ '/r' + str(int(ir.runList[idx])).zfill(4)+'/'+expName+'_'+str(int(ir.runList[idx])).zfill(4) + '.cxi', 'r+')['/entry_1/result_1/nHitsAll'].size
            dset = files[ir.runList[idx]].create_dataset("propLabels", (eventsTotal, 1))
            dset[ir.eventInd[idx]] = label
    for run, f in files.iteritems():
        f.close()
    print "Done"

## Propagate Button ##
def onClick(event):
    list = []
    for i in np.arange(X[:,0].size):
        if propLabels[i] != 0 :
            list.append(i)
    matrix = P[list,:]
    #print matrix.shape
    propagate(matrix, list)
    #propagate(P,np.arange(X[:,0].size))
    ax.clear()
    colors = ['red' if propLabels[idx] == 1 else 'green' if propLabels[idx] == 2 else 'blue' if propLabels[idx] == 3 else 'black' for idx, val in enumerate(X[:,1])]
    ax.scatter(X[:,0],X[:,1],X[:,2], color = colors, picker=5, alpha=0.1)
    files = {}
    filePath = "/reg/d/psdm/" + expName[:3] + "/" + expName + "/scratch/" + user + "/psocake"
    for run in ir.runs:
        if not os.path.exists(filePath + '/r' + str(run).zfill(4)):
            os.makedirs(filePath + '/r' + str(run).zfill(4))
        f = h5py.File(filePath+ '/r' + str(run).zfill(4)+'/'+expName+'_'+str(run).zfill(4) + '_labels.h5', 'a')
        files[run] = f
    for idx, label in enumerate(propLabels):
        evtNum = ir.eventInd[idx]
        try:
            files[ir.runList[idx]]['propLabels'][ir.eventInd[idx]] = label
        except:
            if args.tag is not None:
                eventsTotal = h5py.File(filePath+ '/r' + str(int(ir.runList[idx])).zfill(4)+'/'+expName+'_'+str(int(ir.runList[idx])).zfill(4) + '_' + args.tag + '.cxi', 'r+')['/entry_1/result_1/nHitsAll'].size
            else:
                eventsTotal = h5py.File(filePath+ '/r' + str(int(ir.runList[idx])).zfill(4)+'/'+expName+'_'+str(int(ir.runList[idx])).zfill(4) + '.cxi', 'r+')['/entry_1/result_1/nHitsAll'].size
            dset = files[ir.runList[idx]].create_dataset("propLabels", (eventsTotal, 1))
            dset[ir.eventInd[idx]] = label
    for run, f in files.iteritems():
        f.close()
    print "Done"

## Refresh Button ##
def refresh():
    ax.clear()
    colors = ['red' if propLabels[idx] == 1 else 'green' if propLabels[idx] == 2 else 'blue' if propLabels[idx] == 3 else 'black' for idx, val in enumerate(X[:,1])]
    ax.scatter(X[:,0],X[:,1],X[:,2], color = colors, picker=5, alpha=0.1)
    print "Done"

## Show user labels ##
def showUserLabels(event):
    ax.clear()
    colors = ['red' if userLabels[idx] == 1 else 'green' if userLabels[idx] == 2 else 'blue' if userLabels[idx] == 3 else 'black' for idx, val in enumerate(X[:,1])]
    ax.scatter(X[:,0],X[:,1],X[:,2], color = colors, picker=5, alpha=0.1)
    print "Done"

## Showing Images ##
def getAssemImage(globalIndex):
    runInd = ir.runs.index(ir.runList[globalIndex])
    eventInd = int(ir.eventInd[globalIndex])
    evt = run[runInd].event(times[runInd][eventInd])
    assemImage = det[runInd].image(evt)
    return assemImage

## Assembeles images with the specified events and shows them ## 
def eventsToShow(events):
    global run, times, det, evt, maxEvents
    if len(events) == 0: return
    if len(events) > int(maxEvents):
        events = events[:int(maxEvents)]
    images = []
    for dataind in events:
        arr = getAssemImage(dataind)
        images.append(arr)
    # Calculate display size: (num, col)
    num = float(len(images))/4.0
    if int(num) != num: 
        num = int(num + 1)
    else: 
        num = int(num)
    col = 1600
    if len(images) < 4: col = len(images) * 400
    win.test(num, col, images, events)
    print "Done"

## Find the most confusing events ##
def mostConfusing(event):
    #Loop through columns
        #Get the column sorted
        #Sum the probabilities of nearest neighbors with different label
        #Save it as confusion
    global maxEvents
    knn = 10
    num = 0
    events = []
    (numEvents, numLabels) = X.shape
    # X.shape = (numEvents, numEigs)
    myList = np.where(propLabels != 0)[0]
    if len(myList) < 2: return
    # P is non-symmetric
    matrix = P[:,myList]

    if 0:
        ind = 1868 # amo86615 197
        probs = np.copy(matrix[:,ind])
        sorted = np.sort(probs)#, axis=None)
        argsorted = np.argsort(probs)#, axis=None)
        print "sorted: ", sorted[0], sorted[1], sorted[-2], sorted[-1]
        print "argsorted: ", argsorted[0], argsorted[1], argsorted[-2], argsorted[-1]
        fig = plt.figure()
        plt.subplot(341)
        plt.plot(matrix[:,ind],'x-')
        print "sum: ", np.sum(matrix[ind,:]), np.sum(matrix[:,ind])
        plt.subplot(342)
        plt.imshow(getAssemImage(ind),interpolation='none',vmin=0, vmax=300)
        plt.subplot(345)
        maxInd = argsorted[-2]
        plt.imshow(getAssemImage(maxInd),interpolation='none',vmin=0, vmax=300)
        plt.title(matrix[maxInd,ind])
        plt.subplot(346)
        maxInd = argsorted[-3]
        plt.imshow(getAssemImage(maxInd),interpolation='none',vmin=0, vmax=300)
        plt.title(matrix[maxInd,ind])
        plt.subplot(347)
        maxInd = argsorted[-4]
        plt.imshow(getAssemImage(maxInd),interpolation='none',vmin=0, vmax=300)
        plt.title(matrix[maxInd,ind])
        plt.subplot(348)
        maxInd = argsorted[-5]
        plt.imshow(getAssemImage(maxInd),interpolation='none',vmin=0, vmax=300)
        plt.title(matrix[maxInd,ind])
        plt.subplot(349)
        minInd = argsorted[0]
        plt.imshow(getAssemImage(minInd),interpolation='none',vmin=0, vmax=300)
        plt.title(matrix[minInd,ind])
        plt.subplot(3,4,10)
        minInd = argsorted[1]
        plt.imshow(getAssemImage(minInd),interpolation='none',vmin=0, vmax=300)
        plt.title(matrix[minInd,ind])
        plt.subplot(3,4,11)
        minInd = argsorted[2]
        plt.imshow(getAssemImage(minInd),interpolation='none',vmin=0, vmax=300)
        plt.title(matrix[minInd,ind])
        plt.subplot(3,4,12)
        minInd = argsorted[3]
        plt.imshow(getAssemImage(minInd),interpolation='none',vmin=0, vmax=300)
        plt.title(matrix[minInd,ind])
        plt.show()

    import time
    tic = time.time()
    confusion = np.zeros(numEvents) # confusion
    for i in np.arange(numEvents):
        probs = np.copy(matrix[:,i])
        sorted = np.argsort(probs)
        _label = propLabels[sorted[-knn:]]
        _currentLabel = _label[-1]
        _prob = probs[sorted[-knn:]]
        confusion[i] = np.sum(_prob[np.where(_label!=_currentLabel)])
        #print i, _label, _prob, np.sum(_prob[np.where(_label==1)]), np.sum(_prob[np.where(_label!=1)]), _currentLabel, confusion[i]
    toc = time.time()
    print "time: ", toc-tic

    events = np.argsort(confusion)
    if len(events) > maxEvents: events = events[-maxEvents:]
    
    # Only for debug
    for i in events:
        probs = np.copy(matrix[:,i])
        sorted = np.argsort(probs)
        _label = propLabels[sorted[-knn:]]
        _currentLabel = _label[-1]
        _prob = probs[sorted[-knn:]]
        confusion[i] = np.sum(_prob[np.where(_label!=_currentLabel)])
        if args.verbose >= 1: print "most confusing: ", i, _label, _currentLabel, _prob, confusion[i]
    eventsToShow(events)

    #tic = time.time()
    #conf = np.zeros(numEvents) # confusion
    #for i in np.arange(numEvents):
    #    probs = np.copy(matrix[:,i]) # column of probabilities for ith event
    #    sorted = np.sort(probs, axis=None) # ascending
    #    idx = myList[np.where(probs == sorted[-1])[0][0]] # highest correlation
    #    labelTemp = propLabels[idx] # most likely label for ith event
    #    counter = -1
    #    # Get index of next likely label
    #    while propLabels[idx] == labelTemp and counter*-1 != len(myList):
    #        idx = myList[np.where(probs == sorted[counter - 1])[0][0]]
    #        counter -= 1
    #    # confusion is the difference between probabilities
    #    if propLabels[idx] != labelTemp:
    #        conf[i] = probs[-1] - probs[counter]
    #        num += 1
    #    else: 
    #        conf[i] = probs[-1]
    #confSorted = np.copy(conf)
    #confSorted.sort() # ascending
    #print "confusion sort: ", confSorted[:10]
    #counter = 0
    #while len(events) < maxEventLimit and num > 0:
    #    label = np.where(conf==confSorted[counter])[0][0]
    #    counter += 1
    #    if userLabels[label] == 0:
    #        events.append(label)
    #        num -= 1
    #if len(events) == 0: return
    #toc = time.time()
    #print "time: ", toc-tic
    #eventsToShow(events)

def notLabeled():
    events = []
    count = 0
    for dataind, val in enumerate(ir.eventInd):
        if propLabels[dataind] == 0:
            if count == 20: break
            events.append(dataind)
            count += 1
    eventsToShow(events)

## Shows up to 20 unlabeled events ##
def unlabeled(event):
    notLabeled()

## Shows the clicked events ##
def onpick(event):
    print "Pick event"
    eventsToShow(event.ind)

## Finds users and their runs ##
def setupRuns(runs, runsToUsers):
    children = next(os.walk('/reg/d/psdm/' + expName[:3] + '/' + expName + '/scratch'))[1]
    for child in children:
        if os.path.exists('/reg/d/psdm/' + expName[:3] + '/' + expName + '/scratch/' + child + '/psocake'):
            possibleRuns = next(os.walk('/reg/d/psdm/' + expName[:3] + '/' + expName + '/scratch/' + child + '/psocake'))[1]
            for run in ir.runs:
                if not os.path.exists('/reg/d/psdm/' + expName[:3] + '/' + expName + '/scratch/' + child + '/psocake/r' + str(run).zfill(4) + '/' + expName + '_' + str(run).zfill(4) + '_labels.h5'): continue
                if run not in runsToUsers:
                    runsToUsers[run] = [child]
                else:
                    runsToUsers[run].append(child)

## Turns datasets from how they saved to numpy arrays ##
def getDset(user, run):
    f = h5py.File('/reg/d/psdm/' + expName[:3] + '/' + expName + '/scratch/' + user + '/psocake/r' + str(run).zfill(4) + '/' + expName + '_' + str(run).zfill(4) + '_labels.h5', 'r+')
    try:
        dset = f['labels']
    except:
        if args.tag is not None:
            eventsTotal = h5py.File(path+ '/r' + str(run).zfill(4)+'/'+expName+'_'+str(run).zfill(4) + '_' + args.tag + '.cxi', 'r+')['/entry_1/result_1/nHitsAll'].size
        else:
            eventsTotal = h5py.File(path+ '/r' + str(run).zfill(4)+'/'+expName+'_'+str(run).zfill(4) + '.cxi', 'r+')['/entry_1/result_1/nHitsAll'].size
        dset = f.create_dataset("labels", (eventsTotal, 1))
    evts = np.where(ir.runList == run)[0]
    idxs = ir.eventInd[evts].astype(int)
    subDset = dset[idxs,:]
    f.close()
    return subDset

## Figures out labels and conflicts ##
def labelsAndConflicts(totalEvts, data, users, labels, conflicts, run):
    firstRunIdx = int(np.where(ir.runList == run)[0][0])
    for evt in np.arange(data.shape[1]):
        un = np.unique(data[:,evt])
        label = np.nonzero(un)[0]
        if label.size > 1:
            confEntry = {}
            for i in np.arange(data[:,evt].size):
               confEntry[users[i]] = data[i,evt]
            df = pd.DataFrame.from_dict(confEntry, orient='index')
            df.columns = [ir.eventInd[firstRunIdx + evt]]
            if run not in conflicts:
                conflicts[run] = df
            else:
                conflicts[run] = pd.concat([conflicts[run], df], axis=1)
            labels.append(un[label[0]])
        elif label.size == 0: labels.append(0)
        elif label.size == 1: labels.append(un[label[0]])
    return labels, conflicts

## Does the merging after setting up the info ##
def merge(runs, labels, conflicts, runsToUsers):
    for run in runs:
        if run not in runsToUsers: 
            labels.append([0]*np.where(ir.runList == run)[0].size)
            continue
        users = runsToUsers[run]
        runDsets = []
        totalEvts = 0
        for user in users:
            dset = getDset(user,run)
            runDsets.append(dset)
            totalEvts = dset.size
        data = np.array(runDsets)
        print "Started Merging: ", run
        labels, conflicts = labelsAndConflicts(totalEvts, data, users, labels, conflicts, run)
    return labels, conflicts

def showMessage():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText("Conflicts between users were found!")
    msg.setWindowTitle("Conlicting Labels")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()

def promptConflicts(conflicts):
    if len(conflicts) == 0: return
    idxs = []
    for key, value in conflicts.iteritems():
        evts = value.columns.values
        for idx in evts:
            idx = np.where(ir.eventInd == idx)
            run = np.where(ir.runList == key)
            print idx, run
            idxs.append(np.intersect1d(idx,run)[0])
            print idxs
    eventsToShow(idxs)
    showMessage()

## The setting up + merging code ##
def backgroundCollecting(userLabels, conflicts):
    runsToUsers = {}
    #Modify what's above when you do threading
    setupRuns(ir.runs, runsToUsers)
    labels = []
    conflicts = {}
    if len(runsToUsers) != 0:
        labels, conflicts = merge(ir.runs, userLabels, conflicts, runsToUsers)
    if len(labels) == 0:
        labels = np.zeros(ir.totalHits)
    return labels, conflicts

## Calculate fraction of user labels
def getUserLabelFraction(userLabels):
    counter = 0
    for val in userLabels: 
        if val != 0: counter += 1
    return float(float(counter)/len(userLabels))*100

##################
###    Main    ###
##################

global userLabels
global propLabels
global ax
global X

# Open diffusion map manifold
f = h5py.File(fname, 'r+')
grpNameDM = '/diffusionMap'
dset_indices = '/D_indices'
dset_indptr = '/D_indptr'
dset_data = '/D_data'
X = f[grpNameDM + '/eigvec'].value
expName = f[grpNameDM + '/eigvec'].attrs['exp']
runs = f[grpNameDM + '/eigvec'].attrs['runs']
detInfo = f[grpNameDM + '/eigvec'].attrs['detectorName']
eps = f[grpNameDM + '/eigvec'].attrs['sigma']
f.close()

# Set up image retriever
ir = imageRetriever(expName=expName, runList=runs, detInfo=detInfo, filepath=path)

# Set up a list of all runs, times, detector, and events
run = []
times = []
det = []
evt = []
for runNumber in ir.runs:
    print "runNumber: ", runNumber
    r, t, d, e = setup(expName, runNumber, detInfo)
    run.append(r)
    times.append(t)
    det.append(d)
    evt.append(e)

# Set up user labels and conflicts
userLabels = []
propLabels = []
conflicts = {}
userLabels, conflicts = backgroundCollecting(userLabels, conflicts)
fracUserLabels = getUserLabelFraction(userLabels)
#counter = 0
#for val in userLabels:
#    if val != 0: counter += 1
print "Provided Labels: ", round(fracUserLabels, 3), " %"
propLabels = np.copy(userLabels)

# Calculate diffusion kernel
P,_ = diffusionKernel(X, eps=eps, knn=len(propLabels))

# Launch GUI app
app = QtGui.QApplication(sys.argv)
win = MainWindow()
fig = plt.figure()
canvas = FigureCanvas(fig)
canvas.setParent(win)
promptConflicts(conflicts)
if set(userLabels) != set([0] * len(userLabels)):
    fillProp()
    notLabeled()
ax = fig.add_subplot(111, projection='3d')
colors = ['red' if propLabels[idx] == 1 else 'green' if propLabels[idx] == 2 else 'blue' if propLabels[idx] == 3 else 'black' for idx, val in enumerate(X[:,1])]
ax.scatter(X[:,0],X[:,1],X[:,2], color = colors, picker=5, alpha=0.1)
ax.set_xlabel('$\psi_1$')
ax.set_ylabel('$\psi_2$')
ax.set_zlabel('$\psi_3$')
fig.canvas.mpl_connect('pick_event', onpick)
userPosition = plt.axes([0.1, 0.03, 0.25, 0.055])
userBtn = Button(userPosition, 'User Labels')
userBtn.on_clicked(showUserLabels)
propPosition = plt.axes([0.4, 0.03, 0.25, 0.055])
prop = Button(propPosition, 'Propagate')
prop.on_clicked(onClick)
confPosition = plt.axes([0.7, 0.03, 0.25, 0.055])
confBtn = Button(confPosition, 'Confusion')
confBtn.on_clicked(mostConfusing)
win.setGeometry(win.x(), win.y(),fig.get_size_inches()[0]*fig.dpi, fig.get_size_inches()[1]*fig.dpi)
win.show()
sys.exit(app.exec_())


#Test
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#import skimage.measure as sm
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

parser = argparse.ArgumentParser()
parser.add_argument("-f","--filepath",help="full path to hdf5", type=str)
parser.add_argument("-exp","--expName", help="psana experiment (e.g. amo86615)")
parser.add_argument("-s","--startRun",help="number of events, all events=0", type=int)
parser.add_argument("-e","--endRun",help="append tag to end of filename", type=int)
parser.add_argument("-ver","--version",help="version",default=3, type=int)
parser.add_argument("-det","--detector",help="detector name",default='pnccdFront', type=str)

args = parser.parse_args()
path = args.filepath
#user = 'abdallah'
startRun = args.startRun
endRun = args.endRun
expName = args.expName
version = args.version
detInfo = args.detector #User entered

p = subprocess.Popen(['whoami'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = p.communicate()
user = out[:len(out)-1]
print "Hi"
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
class Scroll(QtGui.QScrollArea):
    def __init__(self, w, parent=None):
        QtGui.QScrollArea.__init__(self,parent)
        self.setWidget(w)
        self.setGeometry(0,0,w.width(),600)
        self.window = w
    def resizeEvent(self,resizeEvent):
        #self.setGeometry(self.x(), self.y(), self.width(), self.height())
        self.window.gridLayoutWidget.setGeometry(self.window.gridLayoutWidget.x(), self.window.gridLayoutWidget.y(), self.width()-20, (self.width()/4.0)*self.window.gridLayout.rowCount())
        #self.window.gridLayout.setGeometry(self.window.gridLayoutWidget.geometry())
        self.window.setGeometry(self.window.x(), self.window.y(), self.width(), self.window.gridLayoutWidget.height()+20*self.window.gridLayout.rowCount())
        self.window.centralwidget.setGeometry(self.window.geometry())
        #QtGui.QMessageBox.information(self,"Information!","Window has been resized...")
class Window(QtGui.QWidget):
    def updateLabels(self):
        global userLabels
        global propLabels
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
        fillProp()
        self.refresh()
    def defaultBackground(self, idx):
        global propLabels
        color = "000"
        if propLabels[idx] == 1:
            color = "F00"
        elif propLabels[idx] == 2:
            color = "0F0"
        elif propLabels[idx] == 3:
            color = "00F"
        return color
    def refresh(self):
        global ax
        global X
        for idx, val in enumerate(self.ind):
            if val in self.selected: self.selected.remove(val)
            self.gridLayout.itemAt(idx).widget().getItem(0,0).setBackgroundColor(self.defaultBackground(val))
        ax.clear()
        colors = ['red' if propLabels[idx] == 1 else 'green' if propLabels[idx] == 2 else 'blue' if propLabels[idx] == 3 else 'black' for idx, val in enumerate(X[:,1])]
        ax.scatter(X[:,0],X[:,1],X[:,2], color = colors, picker=5)
    def adjust(self):
        print self.selected
        print self.ind
        for val in self.selected:
            idx = self.ind.index(val)
            img = self.gridLayout.itemAt(idx).widget().getItem(0,0).allChildren()[1]
            img.setLevels([float(self.lineEdit_2.text()),float(self.lineEdit_3.text())])
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self,parent)
        self.centralwidget = QtGui.QWidget(self)
        self.lineEdit_2 = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(130, 10, 113, 22))
        self.lineEdit_2.setObjectName(_fromUtf8("lineEdit_2"))
        self.lineEdit_3 = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(250, 10, 121, 22))
        self.lineEdit_3.setObjectName(_fromUtf8("lineEdit_3"))
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(380, 10, 101, 26))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.comboBox = QtGui.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(543, 10, 141, 22))
        self.comboBox.setInsertPolicy(QtGui.QComboBox.NoInsert)
        self.comboBox.setObjectName(_fromUtf8("comboBox"))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.pushButton_2 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(790, 10, 101, 26))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_3 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(690, 10, 101, 26))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.pushButton.clicked.connect(self.adjust)
        self.pushButton_2.clicked.connect(self.refresh)
        self.pushButton_3.clicked.connect(self.updateLabels)
    def pop(self, arr, h, w, ind):
        self.gridLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 39, h, w))
        self.gridLayoutWidget.setObjectName(_fromUtf8("gridLayoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.ind = ind
        self.selected = []
        self.images = arr
        num = float(len(arr))/4.0
        if int(num) != num: num = int(num + 1)
        else: num = int(num)
        #self.layout.setSpacing(25)
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
                    color = self.defaultBackground(idx)
                    img = pg.ImageItem()
                    img.setImage(arr[row*4+column])
                    img.setLevels([0,3.5])
                    vb.addItem(img)
                    text = pg.TextItem(anchor=(1,1))
                    text.setHtml('<font size="4"><b>Hi</b></font>')
                    vb.addItem(text)
                    text1 = pg.TextItem()
                    text1.setHtml('<font size="4"><b>Hello</b></font>')
                    vb.addItem(text1)
                    vb.setAspectLocked(True)
                    vb.setBackgroundColor(color)
                    imageWidget.scene().sigMouseClicked.connect(self.buttonClicked)
                    self.gridLayout.addWidget(imageWidget, row, column)
    def buttonClicked(self, event):
        button = self.sender().items()
        vb = button[len(button)-2]
        idx = -1
        for i in np.arange(self.gridLayout.count()):
            #print "Iteration: ", i
            #print "Button: ", button
            #print "Widget: ", self.gridLayout.itemAt(i).widget().getItem(1,1)
            if vb == self.gridLayout.itemAt(i).widget().getItem(0,0):
                idx = i
                break
        if self.ind[idx] in self.selected:
            vb.setBackgroundColor(self.defaultBackground(self.ind[idx]))
            self.selected.remove(self.ind[idx])
        else:
            vb.setBackgroundColor('w')
            self.selected.append(self.ind[idx])
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.pushButton.setText(_translate("MainWindow", "Change Contrast", None))
        self.comboBox.setItemText(0, _translate("MainWindow", "Single", None))
        self.comboBox.setItemText(1, _translate("MainWindow", "Multi", None))
        self.comboBox.setItemText(2, _translate("MainWindow", "Dunno", None))
        self.comboBox.setItemText(3, _translate("MainWindow", "None", None))
        self.pushButton_2.setText(_translate("MainWindow", "Refresh", None))
        self.pushButton_3.setText(_translate("MainWindow", "Save", None))
class Dialog(QtGui.QDialog):
    def __init__(self, num, col, images, ind, parent=None):
        QtGui.QDialog.__init__(self,parent)
        self.w = Window(self)
        self.w.retranslateUi(self.w)
        self.w.setGeometry(0, 0, col, num*400)
        self.w.pop(images, col, num*400, ind)
        self.setGeometry(0,0,col,num*400)
        self.s = Scroll(self.w, self)
        self.s.resize(self.width(), self.height())
        self.s = Scroll(self.w, self)
        self.s.show()
        self.w.show()
    def resizeEvent(self, resizeEvent):
        self.s.resize(self.width(), self.height())
    def refresh(self, num, col, images, ind):
        self.s.hide()
        self.w.hide()
        self.w = Window(self)
        self.w.retranslateUi(self.w)
        self.w.setGeometry(0, 0, col, num*400)
        self.w.pop(images, col, num*400, ind)
        self.setGeometry(0,0,col,num*400)
        self.s.resize(self.width(), self.height())
        self.s = Scroll(self.w, self)
        self.s.show()
        self.w.show()
        self.s.resize(self.width(), self.height())
    def closeEvent(self, closeEvent):
        refresh()
class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.d = Dialog(2, 1600, [],[])
        self.d.show()
    def test(self, num, col, images,ind):
        self.d.refresh(num, col, images, ind)

class imageRetriever:
    def __init__(self,filepath,expName,startRun,endRun,detInfo):
        self.runs = []
        self.filepath = filepath
        self.expName = expName
        self.startRun = startRun
        self.endRun = endRun
        self.detInfo = detInfo
        self.fileList = []
        self.numHitsPerFile = []
        self.allHitInd = np.array([])
        self.eventInd = np.array([])
        self.runList = np.array([])
        self.downsampleRows = None
        self.downsampleCols = None
        self.run,self.times,self.det,self.evt = self.setup(self.expName,self.startRun,self.detInfo)
        self.lastRun = 0
        for r in np.arange(startRun,endRun+1):
            filename = filepath+ '/r' + str(r).zfill(4)+'/'+expName+'_'+str(r).zfill(4) + '.cxi'
            if os.path.exists(filename):
                print "filename :", filename
                f1 = h5py.File(filename,'r')
                self.runs.append(r)
                numHits = f1['/entry_1/result_1/nHits'].attrs['numEvents']
                hitInd = np.arange(0,numHits)
                hitEvent = f1['/LCLS/eventNumber'].value
                runInd = np.ones((numHits,),dtype=int)*r

                if numHits > 0:
                    self.allHitInd = np.append(self.allHitInd,hitInd)
                    self.eventInd = np.append(self.eventInd,hitEvent)
                    self.runList = np.append(self.runList,runInd)
                    self.fileList.append(f1)
                    print "numHits: ", numHits
                    self.numHitsPerFile.append(numHits)
                    #self.downsampleRows = f1[dataset+'/photonCount'].attrs['downsampleRows']
                    #self.downsampleCols = f1[dataset+'/photonCount'].attrs['downsampleCols']
                f1.close()
        self.totalHits = np.sum(self.numHitsPerFile)
        self.numFiles = len(self.fileList)
        self.accumHits = np.zeros(self.numFiles,)
        for i,val in enumerate(self.numHitsPerFile):
            self.accumHits[i] = np.sum(self.numHitsPerFile[0:i+1])
        print "totalHits: ", self.totalHits, self.allHitInd

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
def diffusionKernel(X,eps,knn,D=None):
    nbrs = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree').fit(X)    
    D=nbrs.kneighbors_graph(X,mode='distance')
    term = D.multiply(D)/-eps
    G = np.exp(term.toarray())
    G[np.where(G==1)]=0
    G = G + np.eye(G.shape[0])
    deg = np.sum(G,axis=0)
    term1 = G/deg
    P = term1#np.matrix(term1)
    return P,D
def propagate(matrix, list):
    #while 0 in propLabels:
    for i in np.arange(X[:,0].size):
        label = np.amax(matrix[:,i])
        if label != 0:
            idx = np.argmax(matrix[:,i])
                #print matrix.shape
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
    print matrix.shape
    propagate(matrix, list)

    ax.clear()
    colors = ['red' if propLabels[idx] == 1 else 'green' if propLabels[idx] == 2 else 'blue' if propLabels[idx] == 3 else 'black' for idx, val in enumerate(X[:,1])]
    ax.scatter(X[:,0],X[:,1],X[:,2], color = colors, picker=5)
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
    ax.scatter(X[:,0],X[:,1],X[:,2], color = colors, picker=5)
    print "Done"

## Reset Button ##
#def reset(event):
#    propLabels = copy.copy(userLabels)
#    ax.clear()
#    colors = ['red' if propLabels[idx] == 1 else 'green' if propLabels[idx] == 2 else 'blue' if propLabels[idx] == 3 else 'black' for idx, val in enumerate(X[:,1])]
#    ax.scatter(X[:,0],X[:,1],X[:,2], color = colors, picker=5)
#    print "Done"

## Showing Images ##
def getAssemImage(globalIndex):
    evt = run.event(times[int(ir.eventInd[globalIndex])])
    print "Run, Index: ", ir.eventInd[globalIndex], ir.runList[globalIndex]
    img = det.calib(evt) * det.gain(evt)
    assemImageOrig = det.image(evt,img)
    assemImage = assemImageOrig
    return assemImage

## Assembeles images with the specified events and shows them ## 
def eventsToShow(events):
    if len(events) == 0: return
    images = []
    for dataind in events:
        arr = np.log10(abs(getAssemImage(dataind))+1e-7)
        images.append(arr)
    num = float(len(images))/4.0
    if int(num) != num: num = int(num + 1)
    else: num = int(num)
    col = 1600
    if len(images) < 4: col = len(images) * 400
    print num
    win.test(num,col,images,events)
    print "Done"

## Finds the 5 most confusing events that are not labeled ##
def mostConfusing(event):
    #Loop through columns
        #Get the column sorted
        #Start from the highest value. Keep looking for 2 different labels
        #Subtract the values
        #Save it as confusion
    num = 0
    events = []
    list = []
    for i in np.arange(X[:,0].size):
        if propLabels[i] != 0 :
            list.append(i)
    if len(list) < 2: return
    matrix = P[list,:]
    conf = np.zeros(X[:,0].size)
    for i in np.arange(X[:,0].size):
        probs = np.copy(matrix[:,i])
        sorted = np.sort(probs, axis=None)
        idx = list[np.where(probs == sorted[-1])[0][0]]
        labelTemp = propLabels[idx]
        counter = -1
        while propLabels[idx] == labelTemp and counter*-1 != len(list):
            idx = list[np.where(probs == sorted[counter - 1])[0][0]]
            counter -= 1
        if propLabels[idx] != labelTemp:
            print "propLabels, labelTemp", propLabels[idx], labelTemp
            conf[i] = probs[-1] - probs[counter]
            num += 1
        else: conf[i] = probs[-1]
    confSorted = np.copy(conf)
    confSorted.sort()
    counter = 0
    while len(events) < 10 and num > 0:
        label = np.where(conf==confSorted[counter])[0][0]
        counter += 1
        if userLabels[label] == 0:
            events.append(label)
            num -= 1
    if len(events) == 0: return
    eventsToShow(events)

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
    print "###### ind: ", event.ind
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
        users = runsToUsers[run]
        runDsets = []
        totalEvts = 0
        for user in users:
            dset = getDset(user,run)
            runDsets.append(dset)
            totalEvts = dset.size
            print totalEvts
        #print "dsets: ", runDsets
        print "Building the matrix"
        data = np.array(runDsets)
        print "Got the matrix"
        #print "shape: ", data.shape
        #print "Started Merging"
        labels, conflicts = labelsAndConflicts(totalEvts, data, users, labels, conflicts, run)
        #print labels.keys()
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
    #print "Runs To Users: ", runsToUsers
    labels = []
    conflicts = {}
    if len(runsToUsers) != 0:
        labels, conflicts = merge(ir.runs,userLabels,conflicts, runsToUsers)
    if len(labels) == 0:
        labels = np.zeros(ir.totalHits)
    return labels, conflicts
        #print "merged"
##################
### Kinda main ###
##################

global userLabels
global propLabels
global ax
global X

run, times, det, evt = setup(expName,startRun,detInfo)
f = h5py.File(path + expName + '_' + str(startRun) + '_' + str(endRun) + '_class_v' + str(version) + '.h5', 'r+')
grpNameDM = '/diffusionMap'
dset_indices = '/D_indices'
dset_indptr = '/D_indptr'
dset_data = '/D_data'
X = f[grpNameDM + '/eigvec']

ir = imageRetriever(filepath=path,expName=expName,startRun=startRun,
                    endRun=endRun,detInfo=detInfo)


userLabels = []
propLabels = []
conflicts = {}
userLabels, conflicts = backgroundCollecting(userLabels, conflicts)
#print userLabels
propLabels = np.copy(userLabels)
P,_ = diffusionKernel(X,eps=1e5,knn=70)
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
ax.scatter(X[:,0],X[:,1],X[:,2], color = colors, picker=5)
fig.canvas.mpl_connect('pick_event', onpick)
propPosition = plt.axes([0.2, 0.03, 0.2, 0.055])
prop = Button(propPosition, 'Propagate')
prop.on_clicked(onClick)
unlabeledPosition = plt.axes([0.4, 0.03, 0.2, 0.055])
unlabeledBtn = Button(unlabeledPosition, 'Unlabeled')
unlabeledBtn.on_clicked(unlabeled)
confPosition = plt.axes([0.6, 0.03, 0.2, 0.055])
confBtn = Button(confPosition, 'Confusion')
confBtn.on_clicked(mostConfusing)
win.setGeometry(win.x(), win.y(),fig.get_size_inches()[0]*fig.dpi, fig.get_size_inches()[1]*fig.dpi)
win.show()
sys.exit(app.exec_())

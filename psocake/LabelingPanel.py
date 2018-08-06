import numpy as np
import pyqtgraph as pg
from pyqtgraph.dockarea import *
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
import LaunchPeakFinder
import json, os, time
from database import LabelDatabase
import runAlgorithm

if 'LCLS' in os.environ['PSOCAKE_FACILITY'].upper():
    pass
elif 'PAL' in os.environ['PSOCAKE_FACILITY'].upper():
    pass

class Labeling(object):
    def __init__(self, parent = None):
        self.parent = parent
        self.setupRunDir()
        ## Dock: Labeler
        self.dock = Dock("Labeling", size=(1, 1))
        self.win = ParameterTree()
        self.dock.addWidget(self.win)

        #String Names for Buttons and Menus
        self.labelParam_grp = 'Labeler'
        self.labelParam_shapes_str = 'Shape'
        self.labelParam_pluginDir_str = 'Plugin directory'
        self.labelParam_outDir_str = 'Output directory'
        self.labelParam_runs_str = 'Run(s)'
        self.labelParam_queue_str = 'queue'
        self.labelParam_cpu_str = 'CPUs'
        self.labelParam_psanaq_str = 'psanaq'
        self.labelParam_psnehq_str = 'psnehq'
        self.labelParam_psfehq_str = 'psfehq'
        self.labelParam_psnehprioq_str = 'psnehprioq'
        self.labelParam_psfehprioq_str = 'psfehprioq'
        self.labelParam_psnehhiprioq_str = 'psnehhiprioq'
        self.labelParam_psfehhiprioq_str = 'psfehhiprioq'
        self.labelParam_psdebugq_str = 'psdebugq'
        self.labelParam_noQueue_str = 'N/A'
        self.labelParam_noe_str = 'Number of events to process'
        self.labelParam_launch_str = 'Launch labeler'
        self.labelParam_pluginParam_str = 'Plugin parameters'
        self.labelParam_mode_str = 'Mode'
        self.labelParam_add_str = 'Add'
        self.labelParam_remove_str = 'Remove'
        self.labelParam_loadName_str = 'Label Load Name'
        self.labelParam_saveName_str = 'Label Save Name'
        self.labelParam_load_str = 'Load Labels'
        self.labelParam_save_str = 'Save Labels'
        self.tag_str = 'Tag'
        self.labelParam_showPeaks_str = 'Show labels from Plug-in'


        self.labelParam_poly_str = 'Polygon'
        self.labelParam_circ_str = 'Circle'
        self.labelParam_rect_str = 'Rectangle'
        self.labelParam_none_str = 'None'

        self.shape = None
        self.mode = self.labelParam_add_str
        self.labelParam_pluginParam = "{\"npix_min\": 2,\"npix_max\":30,\"amax_thr\":300, \"atot_thr\":600,\"son_min\":10, \"rank\":3, \"r0\":3, \"dr\":2, \"nsigm\":5 }"
        self.tag = ''
        self.labelParam_pluginDir = '' #"adaptiveAlgorithm"
        self.labelParam_loadName = ''
        self.labelParam_saveName = None
        self.showPeaks = False
        self.numPeaksFound = 0
        self.algorithm_name = 0

        if self.parent.facility == self.parent.facilityLCLS:
            self.labelParam_outDir = self.parent.psocakeDir
            self.labelParam_outDir_overridden = False
            self.labelParam_runs = ''
            self.labelParam_queue = self.labelParam_psanaq_str
            self.labelParam_cpus = 24
            self.labelParam_noe = -1
        elif self.parent.facility == self.parent.facilityPAL:
            pass

        self.rectRois = []
        self.circleRois = []
        self.polyRois = []
        self.algRois = []

        self.rectAttributes = []
        self.circleAttributes = []
        self.polyAttributes = []

        self.db = LabelDatabase()

        if self.parent.facility == self.parent.facilityLCLS:
            self.params = [
                {'name': self.labelParam_grp, 'type': 'group', 'children': [
                    {'name': self.labelParam_mode_str, 'type': 'list', 'values': {self.labelParam_add_str: 'Add',
                                                                                    self.labelParam_remove_str: 'Remove'},
                     'value': self.mode, 'tip': "Choose labelling mode"},
                    {'name': self.labelParam_shapes_str, 'type': 'list', 'values': {self.labelParam_poly_str: 'Polygon',
                                                                                    self.labelParam_circ_str: 'Circle',
                                                                                    self.labelParam_rect_str: 'Rectangle',
                                                                                    self.labelParam_none_str: 'None'},
                     'value': self.shape, 'tip': "Choose label shape"},
                    {'name': self.labelParam_showPeaks_str, 'type': 'bool', 'value': self.showPeaks,
                     'tip': "Show peaks found shot-to-shot"},
                    {'name': self.labelParam_pluginDir_str, 'type': 'str', 'value': self.labelParam_pluginDir, 
                     'tip': "Input your algorithm directory, e.g. \"adaptiveAlgorithm\""},
                    {'name': self.labelParam_pluginParam_str, 'type': 'str', 'value': self.labelParam_pluginParam,
                     'tip': "Dictionary/kwargs of parameters for your algorithm -- use double quotes"},
                    {'name': self.tag_str, 'type': 'str', 'value': self.tag,
                     'tip': "attach tag to stream, e.g. cxitut13_0010_tag.stream"},
                    {'name': self.labelParam_outDir_str, 'type': 'str', 'value': self.labelParam_outDir,
                     'tip': "Output Directory for save files"},
                    {'name': self.labelParam_runs_str, 'type': 'str', 'value': self.labelParam_runs},
                    {'name': self.labelParam_queue_str, 'type': 'list', 'values': {self.labelParam_psfehhiprioq_str: 'psfehhiprioq',
                                                                                 self.labelParam_psnehhiprioq_str: 'psnehhiprioq',
                                                                                 self.labelParam_psfehprioq_str: 'psfehprioq',
                                                                                 self.labelParam_psnehprioq_str: 'psnehprioq',
                                                                                 self.labelParam_psfehq_str: 'psfehq',
                                                                                 self.labelParam_psnehq_str: 'psnehq',
                                                                                 self.labelParam_psanaq_str: 'psanaq',
                                                                                 self.labelParam_psdebugq_str: 'psdebugq'},
                     'value': self.labelParam_queue, 'tip': "Choose queue"},
                    {'name': self.labelParam_cpu_str, 'type': 'int', 'decimals': 7, 'value': self.labelParam_cpus},
                    {'name': self.labelParam_noe_str, 'type': 'int', 'decimals': 7, 'value': self.labelParam_noe,
                     'tip': "number of events to process, default=-1 means process all events"},
                    {'name': self.labelParam_saveName_str, 'type': 'str', 'value': self.labelParam_saveName, 
                     'tip': "Input the name you want to save these labels as"},
                    {'name': self.labelParam_save_str, 'type': 'action'},
                    {'name': self.labelParam_loadName_str, 'type': 'str', 'value': self.labelParam_loadName, 
                     'tip': "Input the name of the label save post you want to load"},
                    {'name': self.labelParam_load_str, 'type': 'action'},
                    {'name': self.labelParam_launch_str, 'type': 'action'},
                ]},
            ]
        elif self.parent.facility == self.parent.facilityPAL:
            pass

        self.paramWidget = Parameter.create(name='paramsLabel', type='group', \
                                   children=self.params, expanded=True)
        self.win.setParameters(self.paramWidget, showTop=False)
        self.paramWidget.sigTreeStateChanged.connect(self.change)

    def digestRunList(self, runList):
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

    def updateParam(self):
        if self.userUpdate is None:
            if self.parent.psocakeRunDir is not None:
                labelParamFname = self.parent.psocakeRunDir + '/pluginParam'
                if self.tag: labelParamFname += '_' + self.tag
                labelParamFname += '.json'
                if os.path.exists(labelParamFname):
                    with open(labelParamFname) as infile:
                        d = json.load(infile)
                        self.labelParam_pluginParam = d[self.labelParam_pluginParam_str]

    def writeStatus(self, fname, d):
        json.dump(d, open(fname, 'w'))

    # Launch labeling
    def findLabels(self):
        self.parent.thread.append(LaunchPlugin.LaunchPlugin(self.parent)) # send parent parameters with self
        self.parent.thread[self.parent.threadCounter].launch(self.parent.experimentName, self.parent.detInfo)
        self.parent.threadCounter+=1
        # Save peak finding parameters
        runsToDo = self.digestRunList(self.labelParam_runs)
        for run in runsToDo:
            pluginParamFname = self.parent.psocakeDir+'/r'+str(run).zfill(4)+'/pluginParam'
            if self.tag: pluginParamFname += '_'+self.tag
            pluginParamFname += '.json'
            if self.parent.facility == self.parent.facilityLCLS:
                d = {self.labelParam_pluginDir_str: self.labelParam_pluginParam,
                     self.labelParam_pluginParam_str: self.labelParam_pluginParam}
            elif self.parent.facility == self.parent.facilityPAL:
                pass
            if not os.path.exists(self.parent.psocakeDir+'/r'+str(run).zfill(4)):
                os.mkdir(self.parent.psocakeDir+'/r'+str(run).zfill(4))
            self.writeStatus(pluginParamFname, d)

    # If anything changes in the parameter tree, print a message
    def change(self, panel, changes):
        for param, change, data in changes:
            path = panel.childPath(param)
            if self.parent.args.v >= 1:
                print('  path: %s' % path)
                print('  change:    %s' % change)
                print('  data:      %s' % str(data))
                print('  ----------')
            self.paramUpdate(path, change, data)

    ##############################
    # Mandatory parameter update #
    ##############################
    def paramUpdate(self, path, change, data):
        if path[0] == self.labelParam_grp:
            if path[1] == self.labelParam_outDir_str:
                self.labelParam_outDir = data
                self.labelParam_outDir_overridden = True
            elif path[1] == self.labelParam_pluginDir_str:
                self.algorithm_name = data
            elif path[1] == self.labelParam_runs_str:
                self.labelParam_runs = data
            elif path[1] == self.labelParam_queue_str:
                self.labelParam_queue = data
            elif path[1] == self.labelParam_cpu_str:
                self.labelParam_cpus = data
            elif path[1] == self.labelParam_noe_str:
                self.labelParam_noe = data
            elif path[1] == self.labelParam_launch_str:
                self.findLabels()
            elif path[1] == self.labelParam_save_str:
                self.postLabels()
            elif path[1] == self.tag_str:
                self.updateTag(data)
            elif path[1] == self.labelParam_pluginParam_str:
                self.labelParam_pluginParam = data
            elif path[1] == self.labelParam_shapes_str:
                self.shapes = data
            elif path[1] == self.labelParam_mode_str:
                self.mode = data
            elif path[1] == self.labelParam_loadName_str:
                self.labelParam_loadName = data
            elif path[1] == self.labelParam_saveName_str:
                self.labelParam_saveName = data
            elif path[1] == self.labelParam_load_str:
                self.loadLabels(self.labelParam_loadName)
            elif path[1] == self.labelParam_showPeaks_str:
                self.showPeaks = data
                self.updateAlgorithm()
                self.drawPeaks()

    def updateAlgorithm(self):
        self.algInitDone = False
        self.updateLabel()
        if self.parent.args.v >= 1: print "##### Done updateAlgorithm: ", self.algorithm_name

    def updateTag(self, data):
        self.tag = data

    def updateLabel(self):
        if self.parent.calib is not None:
            if self.parent.mk.streakMaskOn:
                self.parent.mk.initMask()
                self.parent.mk.streakMask = self.parent.mk.StreakMask.getStreakMaskCalib(self.parent.evt)
                if self.parent.mk.streakMask is None:
                    self.parent.mk.streakMaskAssem = None
                else:
                    self.parent.mk.streakMaskAssem = self.parent.det.image(self.parent.evt, self.parent.mk.streakMask)
                self.algInitDone = False

            self.parent.mk.displayMask()

            # update combined mask
            self.parent.mk.combinedMask = np.ones_like(self.parent.calib)
            if self.parent.mk.streakMask is not None and self.parent.mk.streakMaskOn is True:
                self.parent.mk.combinedMask *= self.parent.mk.streakMask
            if self.parent.mk.userMask is not None and self.parent.mk.userMaskOn is True:
                self.parent.mk.combinedMask *= self.parent.mk.userMask
            if self.parent.mk.psanaMask is not None and self.parent.mk.psanaMaskOn is True:
                self.parent.mk.combinedMask *= self.parent.mk.psanaMask

            # Compute
            if self.algorithm_name == 0: # No algorithm
                self.labels = None
                self.drawLabels()
            else:
                if self.parent.facility == self.parent.facilityLCLS:
                    # Only initialize the hit finder algorithm once
                    if self.algInitDone is False:
                        if (self.labelParam_pluginParam is not None):
                            print("Loading %s!" % self.algorithm_name)
                            kwargs = json.loads(self.labelParam_pluginParam)
                            self.peakRadius = kwargs["r0"]
                            self.peaks = runAlgorithm.invoke_model(self.algorithm_name, self.parent.calib,self.parent.mk.combinedMask.astype(np.uint16), **kwargs)
                            self.numPeaksFound = self.peaks.shape[0]
                        else:
                            print("Enter plug-in parameters")
                        self.algInitDone = True
                elif self.parent.facility == self.parent.facilityPAL:
                    pass
                if self.parent.args.v >= 1: print "Labels found: ", self.labels
                self.drawLabels()
            if self.parent.args.v >= 1: print "Done updateLabel"

    def drawLabels(self):
        if self.parent.args.v >= 1: print "Done drawLabels"
        self.parent.geom.drawCentre()

    def action(self, x, y, coords, w, h, d):
        if(self.mode == "Add"):
            self.clickCreateShape(x,y, coords, w,h,d)
        elif(self.mode == "Remove"):
            pass

    def setupRunDir(self):
        # Set up psocake directory in scratch
        if self.parent.args.outDir is None:
            self.parent.rootDir = self.parent.dir + '/' + self.parent.experimentName[:3] + '/' + self.parent.experimentName
            self.parent.elogDir = self.parent.rootDir + '/scratch/psocake'
            self.parent.psocakeDir = self.parent.rootDir + '/scratch/' + self.parent.username + '/psocake'
        else:
            self.parent.rootDir = self.parent.args.outDir
            self.parent.elogDir = self.parent.rootDir + '/psocake'
            self.parent.psocakeDir = self.parent.rootDir + '/' + self.parent.experimentName + '/' + self.parent.username + '/psocake'
        self.parent.psocakeRunDir = self.parent.psocakeDir + '/r' + str(self.parent.runNumber).zfill(4)

    def clickCreateShape(self,x,y,coords = [], w = 8, h = 8, d = 9, algorithm = False, color = 'g'):
        try:
            if((algorithm == True) or (self.shapes == "Rectangle")):
                width = w
                height = h
                roiRect = pg.ROI(pos=[x-(width/2), y-(height/2)], size=[width, height], snapSize=1.0, scaleSnap=True, translateSnap=True,
                          pen={'color': color, 'width': 4, 'style': QtCore.Qt.DashLine}, removable = True)
                roiRect.addScaleHandle([1, 0.5], [0.5, 0.5])
                roiRect.addScaleHandle([0.5, 0], [0.5, 0.5])
                roiRect.addScaleHandle([0.5, 1], [0.5, 0.5])
                roiRect.addScaleHandle([0, 0.5], [0.5, 0.5])
                roiRect.addScaleHandle([0, 0], [1, 1]) # bottom,left handles scaling both vertically and horizontally
                roiRect.addScaleHandle([1, 1], [0, 0])  # top,right handles scaling both vertically and horizontally
                roiRect.addScaleHandle([1, 0], [0, 1])  # bottom,right handles scaling both vertically and horizontally
                roiRect.addScaleHandle([0, 1], [1, 0])
                roiRect.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
                roiRect.sigClicked.connect(self.update)
                self.rectRois.append(roiRect)
                if (algorithm == True):
                    self.algRois.append(roiRect)
                self.parent.img.win.getView().addItem(roiRect)
                print("Rectangle added at x = %d, y = %d" % (x, y))
            elif(self.shapes == "Circle"):
                xd = d
                yd = d
                roiCircle = pg.CircleROI([x- (xd/2), y - (yd/2)], size=[xd, yd], snapSize=0.1, scaleSnap=False, translateSnap=False,
                                        pen={'color': color, 'width': 4, 'style': QtCore.Qt.DashLine}, removable = True)
                roiCircle.addScaleHandle([0.1415, 0.707*1.2], [0.5, 0.5])
                roiCircle.addScaleHandle([0.707 * 1.2, 0.1415], [0.5, 0.5])
                roiCircle.addScaleHandle([0.1415, 0.1415], [0.5, 0.5])
                roiCircle.addScaleHandle([0.5, 0.0], [0.5, 0.5]) # south
                roiCircle.addScaleHandle([0.5, 1.0], [0.5, 0.5]) # north
                roiCircle.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
                roiCircle.sigClicked.connect(self.update)
                self.circleRois.append(roiCircle)
                self.parent.img.win.getView().addItem(roiCircle)
                print("Circle added at x = %d, y = %d" % (x, y))
            elif(self.shapes == "Polygon"):
                #roiPoly = pg.PolyLineROI([[x-75, y-100], [x-75,y+100], [x+125,y+100], [x+125,y], [x,y], [x,y-100]], pos = [0,0],
                                      #closed=True, snapSize=1.0, scaleSnap=True, translateSnap=True,
                                      #pen={'color': color, 'width': 4, 'style': QtCore.Qt.DashLine}, removable = True)
                roiPoly = pg.PolyLineROI(coords, pos = [x-375,y+150],
                                      closed=True, snapSize=1.0, scaleSnap=True, translateSnap=True,
                                      pen={'color': color, 'width': 4, 'style': QtCore.Qt.DashLine}, removable = True)
                roiPoly.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
                roiPoly.sigClicked.connect(self.update)
                roiPoly.sigHoverEvent.connect(self.update)
                roiPoly.sigRegionChanged.connect(self.update)
                self.polyRois.append(roiPoly)
                self.parent.img.win.getView().addItem(roiPoly)
                print("Polygon added at x = %d, y = %d" % (x, y))
            else:
                print("Choose a Shape.")
        except AttributeError:
            pass

    def update(self, roi):
        if(self.mode == "Remove"):
            self.parent.img.win.getView().removeItem(roi)
            try:
                self.polyRois.remove(roi)
                print(roi, "removed.")
            except ValueError:
                pass
            try:
                self.circleRois.remove(roi)
                print(roi, "removed.")
            except ValueError:
                pass
            try:
                self.rectRois.remove(roi)
                print(roi, "removed.")
            except ValueError:
                pass
        elif(self.mode == "Add"):
            pass
        else:
            pass
    
    def getPosition(self, roi):
        return [roi.pos()[0],roi.pos()[1]]

    def getSize(self, roi):
        return [roi.size()[0],roi.size()[1]]

    def getPoints(self, roi):
            coords = []
            for point in roi.getState()["points"]:
                coords.append([point[0],point[1]])
            return coords

    def getAttributesRectangle(self, roi):
        return {"Position" : self.getPosition(roi), "Size" : self.getSize(roi)}

    def getAttributesCircle(self, roi):
        return {"Position" : self.getPosition(roi), "Diameter" : self.getSize(roi)[0]} ##only return radius

    def getAttributesPolygon(self, roi):
        return {"Position" : self.getPosition(roi), "Coordinates" : self.getPoints(roi)}

    def saveLabels(self):
        self.rectAttributes = []
        self.circleAttributes = []
        self.polyAttributes = []
        for roi in self.polyRois:
            self.polyAttributes.append(self.getAttributesPolygon(roi))
        for roi in self.rectRois:
            self.rectAttributes.append(self.getAttributesRectangle(roi))
        for roi in self.circleRois:
            self.circleAttributes.append(self.getAttributesCircle(roi))
        return {"Polygons" : self.polyAttributes, "Circles" : self.circleAttributes, "Rectangles" : self.rectAttributes}

    def postLabels(self):
        self.db.post(self.labelParam_saveName, self.saveLabels())
        self.db.printDatabase()

    def loadLabels(self, loadName):
        try:
            shapes = self.db.findPost(loadName)[loadName]
        except TypeError:
            print("Invalid Load Name")
        circles = shapes["Circles"]
        rectangles = shapes["Rectangles"]
        polygons = shapes["Polygons"]
        for circle in circles:
            self.loadCircle(circle["Position"][0],circle["Position"][1],circle["Diameter"])
        for rectangle in rectangles:
            self.loadRectangle(rectangle["Position"][0],rectangle["Position"][1],rectangle["Size"][0],rectangle["Size"][1])
        for polygon in polygons:
            self.loadPolygon(polygon["Position"], polygon["Coordinates"])

    def loadRectangle(self, x, y, w, h):
        roiRect = pg.ROI(pos = [x,y], size=[w, h], snapSize=1.0, scaleSnap=True, translateSnap=True,
                         pen={'color': 'g', 'width': 4, 'style': QtCore.Qt.DashLine}, removable = True)
        roiRect.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        roiRect.sigClicked.connect(self.update)
        self.rectRois.append(roiRect)
        self.parent.img.win.getView().addItem(roiRect)
        print("Rectangle added at x = %d, y = %d" % (x, y))

    def loadCircle(self, x, y, r):
        roiCircle = pg.CircleROI(pos = [x, y], size=[r, r], snapSize=0.1, scaleSnap=False, translateSnap=False,
                                 pen={'color': 'g', 'width': 4, 'style': QtCore.Qt.DashLine}, removable = True)
        roiCircle.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        roiCircle.sigClicked.connect(self.update)
        self.circleRois.append(roiCircle)
        self.parent.img.win.getView().addItem(roiCircle)
        print("Circle added at x = %d, y = %d" % (x, y))

    def loadPolygon(self, pos, coords):
        roiPoly = pg.PolyLineROI(coords, pos=pos,
                                 closed=True, snapSize=1.0, scaleSnap=True, translateSnap=True,
                                 pen={'color': 'g', 'width': 4, 'style': QtCore.Qt.DashLine}, removable = True)
        roiPoly.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        roiPoly.sigClicked.connect(self.update)
        self.polyRois.append(roiPoly)
        self.parent.img.win.getView().addItem(roiPoly)
        print("Polygon added at x = %d, y = %d" % (coords[0][0], coords[0][1]))

    def assemblePeakPos(self, peaks):
        self.ix = self.parent.det.indexes_x(self.parent.evt)
        self.iy = self.parent.det.indexes_y(self.parent.evt)
        if self.ix is None:
            (_, dim0, dim1) = self.parent.calib.shape
            self.iy = np.tile(np.arange(dim0), [dim1, 1])
            self.ix = np.transpose(self.iy)
        self.iX = np.array(self.ix, dtype=np.int64)
        self.iY = np.array(self.iy, dtype=np.int64)
        if len(self.iX.shape) == 2:
            self.iX = np.expand_dims(self.iX, axis=0)
            self.iY = np.expand_dims(self.iY, axis=0)
        cenX = self.iX[np.array(peaks[:, 0], dtype=np.int64), np.array(peaks[:, 1], dtype=np.int64), np.array(
            peaks[:, 2], dtype=np.int64)] + 0.5
        cenY = self.iY[np.array(peaks[:, 0], dtype=np.int64), np.array(peaks[:, 1], dtype=np.int64), np.array(
            peaks[:, 2], dtype=np.int64)] + 0.5
        return cenX, cenY

    def drawPeaks(self):
        self.parent.img.clearPeakMessage()
        if self.showPeaks:
            if self.peaks is not None and self.numPeaksFound > 0:
                if self.parent.facility == self.parent.facilityLCLS:
                    cenX, cenY = self.assemblePeakPos(self.peaks)
                elif self.parent.facility == self.parent.facilityPAL:
                    (dim0, dim1) = self.parent.calib.shape
                    self.iy = np.tile(np.arange(dim0), [dim1, 1])
                    self.ix = np.transpose(self.iy)
                    self.iX = np.array(self.ix, dtype=np.int64)
                    self.iY = np.array(self.iy, dtype=np.int64)
                    cenX = self.iX[np.array(self.peaks[:, 1], dtype=np.int64), np.array(self.peaks[:, 2], dtype=np.int64)] + 0.5
                    cenY = self.iY[np.array(self.peaks[:, 1], dtype=np.int64), np.array(self.peaks[:, 2], dtype=np.int64)] + 0.5
                diameter = self.peakRadius*2+1
                for i,x in enumerate(cenX):
                        self.clickCreateShape(cenX[i],cenY[i],w=diameter, h=diameter, algorithm=True, color = 'b')
            else:
                self.parent.img.peak_feature.setData([], [], pxMode=False)
                self.parent.img.peak_text = pg.TextItem(html='', anchor=(0, 0))
                self.parent.img.win.getView().addItem(self.parent.img.peak_text)
                self.parent.img.peak_text.setPos(0,0)
        else:
            self.removePeaks()
        if self.parent.args.v >= 1: print "Done drawPeaks"
        self.parent.geom.drawCentre()

    def removePeaks(self):
        for roi in self.algRois:
            self.parent.img.win.getView().removeItem(roi)
        self.algRois = []

#TODO: save peaks for each event, so that if a user places labels down, and then changes the event, the labels from the first event are saved and can be returned to.

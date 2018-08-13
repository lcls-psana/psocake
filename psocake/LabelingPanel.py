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

    ##############################
    # Initialize Menu Parameters #
    ##############################

    def __init__(self, parent = None):
        self.parent = parent
        self.setupRunDir()
        ## Dock: Labeler
        self.dock = Dock("Labeling", size=(1, 1))
        self.win = ParameterTree()
        self.dock.addWidget(self.win)

        #String Names for Buttons and Menus
        self.labelParam_labeler = 'Labeler'
        self.labelParam_classifier = 'Classifier'
        self.labelParam_saveload = 'Save or Load Work'
        self.labelParam_shapes_str = 'Shape'
        self.labelParam_algorithm_name_str = 'Plugin directory'
        self.labelParam_classificationOptions_str = 'Classifications'
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
        self.labelParam_load_str = 'Load Labels'
        self.labelParam_save_str = 'Save Labels'
        self.tag_str = 'Tag'


        self.labelParam_poly_str = 'Polygon'
        self.labelParam_circ_str = 'Circle'
        self.labelParam_rect_str = 'Rectangle'
        self.labelParam_none_str = 'None'

        self.shapes = None
        self.mode = self.labelParam_add_str
        self.labelParam_pluginParam = None
        self.tag = ''
        self.labelParam_algorithm_name = '' #"adaptiveAlgorithm"
        self.labelParam_classificationOptions_display = '' 
        self.labelParam_classificationOptions_memory = ''
        self.labelParam_loadName = ''
        self.labelParam_saveName = '%s_%d'%(self.parent.experimentName, self.parent.runNumber)
        self.numLabelsFound = 0
        self.lastEventNumber = 0
        self.algInitDone = False

        self.algorithm = None

        if self.parent.facility == self.parent.facilityLCLS:
            self.labelParam_outDir = self.parent.psocakeDir
            self.labelParam_outDir_overridden = False
            self.labelParam_runs = ''
            self.labelParam_queue = self.labelParam_psanaq_str
            self.labelParam_cpus = 24
            self.labelParam_noe = -1
        elif self.parent.facility == self.parent.facilityPAL:
            pass

        self.loadName = None

        self.rectRois = []
        self.circleRois = []
        self.polyRois = []
        self.algRois = []

        self.rectAttributes = []
        self.circleAttributes = []
        self.polyAttributes = []

        self.eventLabels = {}
        self.algorithmEvaluated = {}

        self.db = LabelDatabase()

        self.eventClassifications = {}

        self.eventsSeen = []

        self.updateMenu()

    ##############################
    # Parameter Update Functions #
    ##############################

    def updateMenu(self):
        if self.parent.facility == self.parent.facilityLCLS:
            self.params = [
                {'name': self.labelParam_labeler, 'type': 'group', 'children': [
                    {'name': self.labelParam_mode_str, 'type': 'list', 'values': {self.labelParam_add_str: 'Add',
                                                                                    self.labelParam_remove_str: 'Remove'},
                     'value': self.mode, 'tip': "Choose labelling mode"},
                    {'name': self.labelParam_shapes_str, 'type': 'list', 'values': {self.labelParam_poly_str: 'Polygon',
                                                                                    self.labelParam_circ_str: 'Circle',
                                                                                    self.labelParam_rect_str: 'Rectangle',
                                                                                    self.labelParam_none_str: 'None'},
                     'value': self.shapes, 'tip': "Choose label shape"},
                    {'name': self.labelParam_algorithm_name_str, 'type': 'str', 'value': self.labelParam_algorithm_name, 
                     'tip': "Input your algorithm directory, e.g. \"adaptiveAlgorithm\""},
                    {'name': self.labelParam_pluginParam_str, 'type': 'str', 'value': self.updateParametersOnMenu(),
                     'tip': "Dictionary/kwargs of parameters for your algorithm -- use double quotes"},
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
                    {'name': self.labelParam_launch_str, 'type': 'action'},
                ]},
                {'name': self.labelParam_classifier, 'type': 'group', 'children': [
                    {'name': self.labelParam_classificationOptions_str, 'type': 'str', 'value': self.labelParam_classificationOptions_display, 
                     'tip': "Type a few classifications you would like to use for each event, separated by spaces \n Use number keys as shortcuts to classify an event"},
                ]},
                {'name': self.labelParam_saveload, 'type': 'group', 'children': [
                    {'name': self.tag_str, 'type': 'str', 'value': self.tag,
                     'tip': "Labels are saved with name 'exp_run', adding a tag will save labels as 'exp_run_tag'"},
                    {'name': self.labelParam_save_str, 'type': 'action'},
                    {'name': self.labelParam_loadName_str, 'type': 'str', 'value': self.labelParam_loadName, 
                     'tip': "Input the name of the label save post you want to load"},
                    {'name': self.labelParam_load_str, 'type': 'action'},

                ]},
            ]
        elif self.parent.facility == self.parent.facilityPAL:
            pass
        self.paramWidget = Parameter.create(name='paramsLabel', type='group', \
                                   children=self.params, expanded=True)
        self.win.setParameters(self.paramWidget, showTop=False)
        self.paramWidget.sigTreeStateChanged.connect(self.change)

    # If anything changes in the parameter tree, print a message
    def change(self, panel, changes):
        self.updateParametersOnMenu()
        for param, change, data in changes:
            path = panel.childPath(param)
            if self.parent.args.v >= 1:
                print('  path: %s' % path)
                print('  change:    %s' % change)
                print('  data:      %s' % str(data))
                print('  ----------')
            self.paramUpdate(path, change, data)


    def paramUpdate(self, path, change, data):
        if path[0] == self.labelParam_labeler:
            if path[1] == self.labelParam_mode_str:
                self.mode = data
            elif path[1] == self.labelParam_shapes_str:
                self.shapes = data
            elif path[1] == self.labelParam_algorithm_name_str:
                self.removeLabels(clearAll= False)
                self.labelParam_algorithm_name = data
                self.updateAlgorithm()
                self.drawLabels()
                if(self.labelParam_pluginParam == None):
                    self.labelParam_pluginParam = self.algorithm.getDefaultParams()
            elif path[1] == self.labelParam_pluginParam_str:
                self.removeLabels(clearAll= False)
                self.labelParam_pluginParam = data
                self.updateAlgorithm()
                self.drawLabels()
            elif path[1] == self.labelParam_runs_str:
                self.labelParam_runs = data
            elif path[1] == self.labelParam_queue_str:
                self.labelParam_queue = data
            elif path[1] == self.labelParam_cpu_str:
                self.labelParam_cpus = data
            elif path[1] == self.labelParam_noe_str:
                self.labelParam_noe = data
            elif path[1] == self.labelParam_launch_str:
                pass
                #self.findLabels()
        elif path[0] == self.labelParam_classifier:
            if path[1] == self.labelParam_classificationOptions_str:
                self.updateClassificationOptions(data)
        elif path[0] == self.labelParam_saveload:
            if path[1] == self.tag_str:
                self.updateTag(data)
            elif path[1] == self.labelParam_save_str:
                self.saveLabelsFromEvent(self.parent.eventNumber)
                self.postLabels()
                self.postClassifications()
            elif path[1] == self.labelParam_loadName_str:
                self.labelParam_loadName = data
            elif path[1] == self.labelParam_load_str:
                self.loadLabelsFromDatabase(self.labelParam_loadName)
                self.loadClassificationsFromDatabase(self.labelParam_loadName)
        self.updateMenu()

    def updateClassificationOptions(self, data):
        self.labelParam_classificationOptions_display = data
        self.labelParam_classificationOptions_memory = self.splitWords(data, " ")

    def updateTag(self, data):
        """ updates Tag for directory
  
        Arguments:
        data - tag name
        """
        self.tag = data
        print(self.labelParam_saveName + "_" + data) #TODO: TATE when tag grabbed, this is not saved properly

    def updateParametersOnMenu(self):
        return self.labelParam_pluginParam

    ##############################
    #      Launch Functions      #
    ##############################

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
                d = {self.labelParam_algorithm_name_str: self.labelParam_pluginParam,
                     self.labelParam_pluginParam_str: self.labelParam_pluginParam}
            elif self.parent.facility == self.parent.facilityPAL:
                pass
            if not os.path.exists(self.parent.psocakeDir+'/r'+str(run).zfill(4)):
                os.mkdir(self.parent.psocakeDir+'/r'+str(run).zfill(4))
            self.writeStatus(pluginParamFname, d)

    def setupRunDir(self):
        """ Set up directory to run in
        """
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

    ##############################
    #      Shape  Functions      #
    ##############################

    def action(self, x, y, coords, w, h, d):
        """ When mouse is clicked in Add mode, an ROI is created

        Arguments:
        x - x position
        y - y position
        coords - coordinates of polygon corners
        w - width of rectangle
        h - height of rectangle
        d - diameter of circle
        """
        if(self.mode == "Add"):
            self.createROI(x,y, coords, w,h,d)
        elif(self.mode == "Remove"):
            pass

    def createROI(self,x,y,coords = [], w = 8, h = 8, d = 9, algorithm = False, color = 'm'):
        """ creates a ROI shape/label based on the input set of parameters

        Arguments:
        x - x position
        y - y position
        **coords - coordinates for a polygon
        **w - width of rectangle
        **h - height of rectangle
        **d - diameter of circle
        **algorithm - Boolean value, True if these labels are loaded
                      from an algorithm, and not from a click event
        **color - color of ROI, green if click event, blue if 
                  loaded from algorithm
        """
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
                roiRect.sigClicked.connect(self.removeROI)
                if (algorithm == True):
                    self.algRois.append(roiRect)
                else:
                    self.rectRois.append(roiRect)
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
                roiCircle.sigClicked.connect(self.removeROI)
                self.circleRois.append(roiCircle)
                self.parent.img.win.getView().addItem(roiCircle)
                print("Circle added at x = %d, y = %d" % (x, y))
            elif(self.shapes == "Polygon"):
                roiPoly = pg.PolyLineROI(coords, pos = [x-375,y+150],
                                      closed=True, snapSize=1.0, scaleSnap=True, translateSnap=True,
                                      pen={'color': color, 'width': 4, 'style': QtCore.Qt.DashLine}, removable = True)
                roiPoly.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
                roiPoly.sigClicked.connect(self.removeROI)
                roiPoly.sigHoverEvent.connect(self.removeROI)
                roiPoly.sigRegionChanged.connect(self.removeROI)
                self.polyRois.append(roiPoly)
                self.parent.img.win.getView().addItem(roiPoly)
                print("Polygon added at x = %d, y = %d" % (x, y))
            else:
                print("Choose a Shape.")
        except AttributeError:
            pass

    def removeROI(self, roi):
        """ signal called when an ROI is clicked on in remove mode, will remove
        roi from screen and from family array

        Arguments:
        roi - ROI variable that was clicked on
        """
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
            try:
                self.algRois.remove(roi)
                print(roi, "removed.")
            except ValueError:
                pass
        elif(self.mode == "Add"):
            pass
        else:
            pass
    
    def getPosition(self, roi):
        """ returns the position of a circle or a rectangle

        Arguments:
        roi - circle or rectangle variable
        """
        return [roi.pos()[0],roi.pos()[1]]

    def getSize(self, roi):
        """ returns the size of a circle or a rectangle

        Arguments:
        roi - circle or rectangle variable
        """
        return [roi.size()[0],roi.size()[1]]

    def getPoints(self, roi):
        """ returns the coordinates of the corners of a polygon

        Arguments:
        roi - polygon variable
        """
        coords = []
        for point in roi.getState()["points"]:
            coords.append([point[0],point[1]])
        return coords

    def getAttributesRectangle(self, roi):
        """ returns the Position and Size of a Rectangle

        Arguments:
        roi - rectangle variable
        """
        return {"Position" : self.getPosition(roi), "Size" : self.getSize(roi)}

    def getAttributesCircle(self, roi):
        """ returns the Position and Size of a Circle

        Arguments:
        roi - circle variable
        """
        return {"Position" : self.getPosition(roi), "Diameter" : self.getSize(roi)[0]} ##only return radius

    def getAttributesPolygon(self, roi):
        """ returns the Position and Cooridinates of a Polygon

        Arguments:
        roi - polygon variable
        """
        return {"Position" : self.getPosition(roi), "Coordinates" : self.getPoints(roi)}

    ##############################
    #      Plugin Functions      #
    ##############################

    def updateAlgorithm(self): #TODO: merge with setAlgorithm
        """updates the Algorithm based on Plugin Parameters
        """
        self.algInitDone = False
        self.setAlgorithm()
        if self.parent.args.v >= 1: print "##### Done updateAlgorithm: ", self.labelParam_algorithm_name

    def setAlgorithm(self):
        """ sets the algorithm based on Plugin Paramaters
        """
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
            if self.labelParam_algorithm_name == 0: # No algorithm
                self.centers = None
                self.drawCenters()
            else:
                if self.parent.facility == self.parent.facilityLCLS:
                    # Only initialize the hit finder algorithm once
                    if self.algInitDone is False:
                        print("Loading %s!" % self.labelParam_algorithm_name)
                        if self.labelParam_pluginParam == None:
                            kw = None
                            self.labelRadius = 1 #TODO: Fix this!
                        else:
                            kw = json.loads(self.labelParam_pluginParam)
                            self.labelRadius = kw["r0"]
                        self.algorithm = runAlgorithm.invoke_model(self.labelParam_algorithm_name)
                        self.labels = self.algorithm.algorithm(self.parent.calib, self.parent.mk.combinedMask.astype(np.uint16), kw)
                        self.numLabelsFound = self.labels.shape[0]
                        self.algInitDone = True
                        self.algorithmEvaluated[self.parent.eventNumber] = True
                elif self.parent.facility == self.parent.facilityPAL:
                    pass
                if self.parent.args.v >= 1: print "Labels found: ", self.centers
                self.drawCenters()
            if self.parent.args.v >= 1: print "Done setAlgorithm"

    def drawCenters(self):
        if self.parent.args.v >= 1: print "Done drawCenters"
        self.parent.geom.drawCentre()

    def assembleLabelPos(self, label):
        """ Determine position of labels from an algorithm
        
        Arguments:
        label - the array of label found in an image
        """
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
        cenX = self.iX[np.array(label[:, 0], dtype=np.int64), np.array(label[:, 1], dtype=np.int64), np.array(
            label[:, 2], dtype=np.int64)] + 0.5
        cenY = self.iY[np.array(label[:, 0], dtype=np.int64), np.array(label[:, 1], dtype=np.int64), np.array(
            label[:, 2], dtype=np.int64)] + 0.5
        return cenX, cenY

    def drawLabels(self):
        """ Draw Labels from an algorithm.
        """
        if self.labels is not None and self.numLabelsFound > 0:
            if self.parent.facility == self.parent.facilityLCLS:
                cenX, cenY = self.assembleLabelPos(self.labels)
            elif self.parent.facility == self.parent.facilityPAL:
                (dim0, dim1) = self.parent.calib.shape
                self.iy = np.tile(np.arange(dim0), [dim1, 1])
                self.ix = np.transpose(self.iy)
                self.iX = np.array(self.ix, dtype=np.int64)
                self.iY = np.array(self.iy, dtype=np.int64)
                cenX = self.iX[np.array(self.labels[:, 1], dtype=np.int64), np.array(self.labels[:, 2], dtype=np.int64)] + 0.5
                cenY = self.iY[np.array(self.labels[:, 1], dtype=np.int64), np.array(self.labels[:, 2], dtype=np.int64)] + 0.5
            diameter = self.labelRadius*2+1
            for i,x in enumerate(cenX):
                self.createROI(cenX[i],cenY[i],w=diameter, h=diameter, algorithm=True, color = 'b')
        else:
            #self.parent.img.peak_feature.setData([], [], pxMode=False)
            self.parent.img.peak_text = pg.TextItem(html='', anchor=(0, 0))
            self.parent.img.win.getView().addItem(self.parent.img.peak_text)
            self.parent.img.peak_text.setPos(0,0)
        if self.parent.args.v >= 1: print "Done drawLabels"
        self.parent.geom.drawCentre()

    ##############################
    #  Label Database Functions  #
    ##############################

    def grabTag(self,loadName):
        self.tag = self.splitWords(loadName, "_")[2]

    def attachTag(self):
        return self.labelParam_saveName + "_" + self.tag

    def saveLabelsToDictionary(self):
        unseenEvents = self.checkLoadEvents()
        translatedEventLabels = {}
        for event in self.eventLabels:
            polyAttributes = []
            circleAttributes = []
            rectAttributes = []
            translatedEventLabels["%s"%event] = {}
            for roi in self.eventLabels[event]["Algorithm"]:
                print(roi)
                if type(roi) is pg.graphicsItems.ROI.ROI:
                    rectAttributes.append(self.getAttributesRectangle(roi))
                elif type(roi) is pg.graphicsItems.ROI.CircleROI:
                    circleAttributes.append(self.getAttributesCircle(roi))
                elif type(roi) is pg.graphicsItems.ROI.PolyLineROI:
                    polyAttributes.append(self.getAttributesPolygon(roi))
                else:
                    print("Cant use this type: %s" % type(roi))
            translatedEventLabels["%s"%event]["Algorithm"] = {"Polygons" : polyAttributes, "Circles" : circleAttributes, "Rectangles" : rectAttributes}
            polyAttributes = []
            circleAttributes = []
            rectAttributes = []
            for roi in self.eventLabels[event]["User"]:
                print(roi)
                if type(roi) is pg.graphicsItems.ROI.ROI:
                    rectAttributes.append(self.getAttributesRectangle(roi))
                elif type(roi) is pg.graphicsItems.ROI.CircleROI:
                    circleAttributes.append(self.getAttributesCircle(roi))
                elif type(roi) is pg.graphicsItems.ROI.PolyLineROI:
                    polyAttributes.append(self.getAttributesPolygon(roi))
                else:
                    print("Cant use this type: %s" % type(roi))
            translatedEventLabels["%s"%event]["User"] = {"Polygons" : polyAttributes, "Circles" : circleAttributes, "Rectangles" : rectAttributes}
        translatedEventLabels.update(unseenEvents)
        return translatedEventLabels

    def postLabels(self):
        """ Post a set of labels to MongoDB
        """
        string = "Label"
        self.db.post(self.attachTag(), string, self.saveLabelsToDictionary())

    def checkLoadEvents(self):
        unseenEvents = {}
        if(self.loadName is not None):
            try:
                allShapes = self.db.findPost(self.loadName)[self.loadName]["Label"]
                for event in allShapes:
                    if event in self.eventsSeen:
                        continue
                    else:
                        unseenEvents[event] = allShapes[event]
            except TypeError:
                pass
        return unseenEvents

    def loadLabelsFromDatabase(self, loadName):
        """Load a saved set of labels from MongoDB

        Arguments:
        loadName - name of the saved set of labels
        """
        try:
            self.loadName = loadName
            self.grabTag(loadName)
            shapes = self.db.findPost(loadName)[loadName]["Label"]["%d"%self.parent.eventNumber]
            self.eventsSeen.append("%d"%self.parent.eventNumber)
            for shapeType in shapes:
                color = None
                if shapeType == "Algorithm":
                    color = 'b'
                elif shapeType == "User":
                    color = 'm'
                circles = shapes[shapeType]["Circles"]
                rectangles = shapes[shapeType]["Rectangles"]
                polygons = shapes[shapeType]["Polygons"]
                for circle in circles:
                    self.loadCircleFromDatabase(circle["Position"][0],circle["Position"][1],circle["Diameter"], color)
                for rectangle in rectangles:
                    self.loadRectangleFromDatabase(rectangle["Position"][0],rectangle["Position"][1],rectangle["Size"][0],rectangle["Size"][1], color)
                for polygon in polygons:
                    self.loadPolygonFromDatabase(polygon["Position"], polygon["Coordinates"], color)
        except TypeError:
            print("Invalid Load Name")
        except KeyError:
            print("Labels Do Not Exist For This Event")
    

    def loadRectangleFromDatabase(self, x, y, w, h, color):
        """ Used to draw labels on an image based on locations saved
        in a database.

        Arguments:
        x - x axis center position
        y - y axis center position
        w - width
        h - height
        """
        roiRect = pg.ROI(pos = [x,y], size=[w, h], snapSize=1.0, scaleSnap=True, translateSnap=True,
                         pen={'color': color, 'width': 4, 'style': QtCore.Qt.DashLine}, removable = True)
        roiRect.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        roiRect.sigClicked.connect(self.removeROI)
        roiRect.addScaleHandle([0, 0], [1, 1]) # bottom,left handles scaling both vertically and horizontally
        roiRect.addScaleHandle([1, 1], [0, 0])  # top,right handles scaling both vertically and horizontally
        roiRect.addScaleHandle([1, 0], [0, 1])  # bottom,right handles scaling both vertically and horizontally
        roiRect.addScaleHandle([0, 1], [1, 0])
        self.rectRois.append(roiRect)
        self.parent.img.win.getView().addItem(roiRect)
        print("Rectangle added at x = %d, y = %d" % (x, y))

    def loadCircleFromDatabase(self, x, y, d, color):
        """ Used to draw labels on an image based on locations saved
        in a database.

        Arguments:
        x - x axis center position
        y - y axis center position
        d - diameter
        """
        roiCircle = pg.CircleROI(pos = [x, y], size=[d, d], snapSize=0.1, scaleSnap=False, translateSnap=False,
                                 pen={'color': color, 'width': 4, 'style': QtCore.Qt.DashLine}, removable = True)
        roiCircle.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        roiCircle.sigClicked.connect(self.removeROI)
        roiCircle.addScaleHandle([0.5, 0.0], [0.5, 0.5]) # south
        roiCircle.addScaleHandle([0.5, 1.0], [0.5, 0.5]) # north
        self.circleRois.append(roiCircle)
        self.parent.img.win.getView().addItem(roiCircle)
        print("Circle added at x = %d, y = %d" % (x, y))

    def loadPolygonFromDatabase(self, pos, coords, color):
        """ Used to draw labels on an image based on locations saved
        in a database.

        Arguments:
        pos - position of the polygon
        coords - coordinates of the corners
        """
        roiPoly = pg.PolyLineROI(coords, pos=pos,
                                 closed=True, snapSize=1.0, scaleSnap=True, translateSnap=True,
                                 pen={'color': color, 'width': 4, 'style': QtCore.Qt.DashLine}, removable = True)
        roiPoly.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        roiPoly.sigClicked.connect(self.removeROI)
        self.polyRois.append(roiPoly)
        self.parent.img.win.getView().addItem(roiPoly)
        print("Polygon added at x = %d, y = %d" % (coords[0][0], coords[0][1]))

    ##############################
    #     Save/Remove Labels     #
    ##############################

    def removeLabels(self, clearAll = True):
        """ Remove the labels from the last event from the screen.
        """
        if(clearAll == True):
            for roi in self.algRois:
                self.parent.img.win.getView().removeItem(roi)
            self.algRois = []
            for roi in self.rectRois:
                self.parent.img.win.getView().removeItem(roi)
            self.rectRois = []
            for roi in self.circleRois:
                self.parent.img.win.getView().removeItem(roi)
            self.circleRois = []
            for roi in self.polyRois:
                self.parent.img.win.getView().removeItem(roi)
            self.polyRois = []
        else:
            for roi in self.algRois:
                self.parent.img.win.getView().removeItem(roi)
            self.algRois = []


    def saveLabelsFromEvent(self, eventNum):
        """ Save the labels from the last event.
        """
        self.eventLabels["%d"%eventNum] = {"Algorithm" : [], "User": []}
        
        for roi in self.algRois:
            self.eventLabels["%d"%eventNum]["Algorithm"].append(roi)
        for roi in self.rectRois:
            self.eventLabels["%d"%eventNum]["User"].append(roi)
        for roi in self.circleRois:
            self.eventLabels["%d"%eventNum]["User"].append(roi)
        for roi in self.polyRois:
            self.eventLabels["%d"%eventNum]["User"].append(roi)
        self.eventLabels["%d"%eventNum]["Algorithm"] = list(tuple(set(self.eventLabels["%d"%eventNum]["Algorithm"])))
        self.eventLabels["%d"%eventNum]["User"] = list(tuple(set(self.eventLabels["%d"%eventNum]["User"])))

    def checkLabels(self):
        """ If an algorithm has been used to load labels for the current
        event, then checkLabels returns True, otherwise, it returns
        false.
        """
        if self.parent.eventNumber in self.algorithmEvaluated:
            if self.algorithmEvaluated[self.parent.eventNumber] == True:
                return True
        else:
            return False

    def loadLabelsEventChange(self):
        """ Either load labels from a previous event or use an algorithm
        to load new labels.
        """
        if self.algInitDone == True:
            if self.checkLabels():
                self.loadLabelsFromPreviousEvent()
            else:
                self.updateAlgorithm()
                self.drawLabels()
        else:
            try:
                self.loadLabelsFromPreviousEvent()
            except KeyError:
                pass

    def loadLabelsFromPreviousEvent(self):
        """ Show the labels for an event that has already been evaluated
        with an algorithm.
        """
        for roi in self.eventLabels["%d"%self.parent.eventNumber]["Algorithm"]:
            self.parent.img.win.getView().addItem(roi)
            self.algRois.append(roi)
        for roi in self.eventLabels["%d"%self.parent.eventNumber]["User"]:
            self.parent.img.win.getView().addItem(roi)
            if type(roi) is pg.graphicsItems.ROI.ROI:
                self.rectRois.append(roi)
            elif type(roi) is pg.graphicsItems.ROI.CircleROI:
                self.circleRois.append(roi)
            elif type(roi) is pg.graphicsItems.ROI.PolyLineROI:
                self.polyRois.append(roi)
            else:
                print("Type Error While Reloading Labels")

    def saveEventNumber(self):
        """ Save the last event's number
        """
        self.lastEventNumber = self.parent.eventNumber

    def actionEventChange(self):
        """ When an event changes, First save the labels from the previous
        event, remove them from the screen, then load the next labels.
        """
        self.saveLabelsFromEvent(self.lastEventNumber)
        self.removeLabels()
        self.loadLabelsEventChange()
        self.saveEventNumber()
        self.updateText()

    ##############################
    #       Classification       #
    #    Database  Functions     #
    ##############################

    def postClassifications(self):
        """ Post a set of labels to MongoDB
        """
        string = "Classification"
        self.db.post(self.labelParam_saveName, string, self.returnClassificationsDictionary())
        self.db.printDatabase()

    def returnClassificationsDictionary(self):
        self.eventClassifications["Options"] = self.labelParam_classificationOptions_memory
        return self.eventClassifications

    def returnClassificationOptionsForDisplay(self):
        return self.labelParam_classificationOptions_display

    def loadClassificationsFromDatabase(self, loadName):
        try:
            self.eventClassifications = self.db.findPost(loadName)[loadName]["Classification"]
            self.labelParam_classificationOptions_display = ' '.join(self.eventClassifications["Options"])
        except TypeError:
            print("Invalid Load Name")
        except KeyError:
            print("Classifications Do Not Exist For This Event")
        self.updateText()

    ##############################
    #   Additional  Functions    #
    ##############################

    def splitWords(self, string, delimiter):
        """ Splits the words in a string and returns an array where each index
        corresponds to words in the original string.
        For ex:
        input  ---> string = "Here is my sentence"
        output ---> stringarray = ["Here", "is", "my", "sentence"]
        stringarray[1] = "is"

        Arguments:
        string - string of words with spaces separating each word
        """
        stringarray = []
        beginningLetter = 0
        endingLetter = 0
        for i,ascii in enumerate(string):
            if (ascii == delimiter):
                endingLetter = i
                stringarray.append(string[beginningLetter:endingLetter])
                beginningLetter = i+1
            elif i == len(string)-1:
                endingLetter = i+1
                stringarray.append(string[beginningLetter:endingLetter])
            else:
                pass
        return stringarray

    def keyPressed(self, val):
        if("%d"%self.parent.eventNumber) in self.eventClassifications:
            pass
        else:
            self.eventClassifications["%d"%self.parent.eventNumber] = []
        if val in self.eventClassifications["%d"%self.parent.eventNumber]:
            self.eventClassifications["%d"%self.parent.eventNumber].remove(val)
        else:
            self.eventClassifications["%d"%self.parent.eventNumber].append(val)
        self.eventClassifications["%d"%self.parent.eventNumber] = list(tuple(set(self.eventClassifications["%d"%self.parent.eventNumber])))
        self.updateText()

    def updateText(self):
        self.clearText()
        self.displayText()

    def displayText(self):
        try:
            self.ix = self.parent.det.indexes_x(self.parent.evt)
            self.iy = self.parent.det.indexes_y(self.parent.evt)
            xMargin = 5 # pixels
            yMargin = 0 # pixels
            maxX = np.max(self.ix) + xMargin
            maxY = np.max(self.iy) - yMargin
            if(("%d"%self.parent.eventNumber) in self.eventClassifications):
                myMessage = '<div style="text-align: center"><span style="color: cyan; font-size: 12pt;">Classifications=' + \
                            ' <br>' + (' '.join(self.eventClassifications["%d"%self.parent.eventNumber])) + \
                            '<br></span></div>'
            else:
                myMessage = '<div style="text-align: center"><span style="color: cyan; font-size: 12pt;">Classifications='+ \
                            '<br></span></div>'
            self.parent.img.peak_text = pg.TextItem(html=myMessage, anchor=(0, 0))
            self.parent.img.win.getView().addItem(self.parent.img.peak_text)
            self.parent.img.peak_text.setPos(maxX, maxY)
        except AttributeError:
            pass

    def clearText(self):
        self.parent.img.clearPeakMessage()
        #self.parent.img.peak_text = pg.TextItem(html='', anchor=(0, 0))
        #self.parent.img.win.getView().addItem(self.parent.img.peak_text)

#BUGS TO FIX:
#TODO: Fetch button to allow multiple users to get the next event to label (so 
     # that multiple users to not double label an event/ overlap work)
#TODO: Fix bug to always show text with event changes -- tricky, not sure why this isnt working...

import numpy as np
import fileinput
import pyqtgraph as pg
import pyqtgraph.console as console
import h5py
import os
from pyqtgraph.dockarea import *
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtCore, QtGui
import subprocess
import scipy.spatial.distance as sd
import glob
try:
    from PyQt5.QtWidgets import *
    using_pyqt4 = False
except ImportError:
    using_pyqt4 = True
    pass
import Detector.PyDetector
import PSCalib.GlobalUtils as gu
from PSCalib.CalibFileFinder import deploy_calib_file
from psocake.utils import highlight

# Return two equal sized halves of the input image
# If axis is None, halve along the first axis
def getTwoHalves(I,centre,axis=None):
    if axis is None or axis == 0:
        A = I[:centre,:]
        B = np.flipud(I[centre:,:])

        (numRowUpper,_) = A.shape
        (numRowLower,_) = B.shape
        if numRowUpper >= numRowLower:
            numRow = numRowLower
            A = A[-numRow:,:]
        else:
            numRow = numRowUpper
            B = B[-numRow:,:]
    else:
        A = I[:,:centre]
        B = np.fliplr(I[:,centre:])

        (_,numColLeft) = A.shape
        (_,numColRight) = B.shape
        if numColLeft >= numColRight:
            numCol = numColRight
            A = A[:,-numCol:]
        else:
            numCol = numColLeft
            B = B[:,-numCol:]
    return A, B

def getScore(A,B):
    ind = (A>0) & (B>0)
    dist = sd.euclidean(A[ind].ravel(),B[ind].ravel())
    numPix = len(ind[np.where(ind==True)])
    return dist/numPix

def findDetectorCentre(I,guessRow=None,guessCol=None,range=0):
    """
    :param I: assembled image
    :param guessRow: best guess for centre row position (optional)
    :param guessCol: best guess for centre col position (optional)
    :param range: range of pixels to search either side of the current guess of the centre
    :return:
    """
    range = int(range)
    # Search for optimum column centre
    if guessCol is None:
        startCol = 1 # search everything
        endCol = I.shape[1]
    else:
        startCol = guessCol - range
        if startCol < 1: startCol = 1
        endCol = guessCol + range
        if endCol > I.shape[1]: endCol = I.shape[1]
    searchArray = np.arange(startCol,endCol)
    scoreCol = np.zeros(searchArray.shape)
    for i, centreCol in enumerate(searchArray):
        A,B = getTwoHalves(I,centreCol,axis=0)
        scoreCol[i] = getScore(A,B)
    centreCol = searchArray[np.argmin(scoreCol)]
    # Search for optimum row centre
    if guessRow is None:
        startRow = 1 # search everything
        endRow = I.shape[0]
    else:
        startRow = guessRow - range
        if startRow < 1: startRow = 1
        endRow = guessRow + range
        if endRow > I.shape[0]: endRow = I.shape[0]
    searchArray = np.arange(startRow,endRow)
    scoreRow = np.zeros(searchArray.shape)
    for i, centreRow in enumerate(searchArray):
        A,B = getTwoHalves(I,centreRow,axis=1)
        scoreRow[i] = getScore(A,B)
    centreRow = searchArray[np.argmin(scoreRow)]
    return centreCol,centreRow

class DiffractionGeometry(object):
    def __init__(self, parent = None):
        self.parent = parent

        #############################
        ## Dock: Diffraction geometry
        #############################
        self.dock = Dock("Diffraction Geometry", size=(1, 1))
        self.win = ParameterTree()
        self.win.setWindowTitle('Diffraction geometry')
        self.dock.addWidget(self.win)
        self.winL = pg.LayoutWidget()
        self.deployGeomBtn = QtGui.QPushButton('Deploy manually centred geometry')
        self.winL.addWidget(self.deployGeomBtn, row=0, col=0)
        self.deployAutoGeomBtn = QtGui.QPushButton('Deploy automatically centred geometry')
        self.winL.addWidget(self.deployAutoGeomBtn, row=1, col=0)
        self.dock.addWidget(self.winL)

        self.resolutionRingList = np.array([100.,300.,500.,700.,900.,1100.])
        self.resolutionText = []
        self.hasUserDefinedResolution = False

        self.geom_grp = 'Diffraction geometry'
        self.geom_detectorDistance_str = 'Detector distance'
        self.geom_clen_str = 'Home to Detector (clen)'
        self.geom_coffset_str = 'Sample to Home (coffset)'
        self.geom_photonEnergy_str = 'Photon energy'
        self.geom_wavelength_str = "Wavelength"
        self.geom_pixelSize_str = 'Pixel size'
        self.geom_resolutionRings_str = 'Resolution rings'
        self.geom_resolution_str = 'Resolution (pixels)'
        self.geom_resolutionUnits_str = 'Units'
        self.geom_unitA_crystal_str = 'Crystallography (Angstrom)'
        self.geom_unitNm_crystal_str = 'Crystallography (Nanometre)'
        self.geom_unitQ_crystal_str = 'Crystallography Reciprocal Space (q)'
        self.geom_unitA_physics_str = 'Physics (Angstrom)'
        self.geom_unitNm_physics_str = 'Physics (Nanometre)'
        self.geom_unitQ_physics_str = 'Physics Reciprocal Space (q)'
        self.geom_unitTwoTheta_str = 'Scattering Angle 2Theta'
        (self.unitA_c,self.unitNm_c,self.unitQ_c,self.unitA_p,self.unitNm_p,self.unitQ_p,self.unitTwoTheta) = (0,1,2,3,4,5,6)

        #######################
        # Mandatory parameter #
        #######################
        self.params = [
            {'name': self.geom_grp, 'type': 'group', 'children': [
                {'name': self.geom_detectorDistance_str, 'type': 'float', 'value': 0.0, 'precision': 6, 'minVal': 0.0001, 'siFormat': (6,6), 'siPrefix': True, 'suffix': 'mm'},
                {'name': self.geom_clen_str, 'type': 'float', 'value': 0.0, 'step': 1e-6, 'siPrefix': True,
                 'suffix': 'm', 'readonly': True},
                {'name': self.geom_coffset_str, 'type': 'float', 'value': 0.0, 'step': 1e-6, 'siPrefix': True,
                 'suffix': 'm', 'readonly': True},
                {'name': self.geom_photonEnergy_str, 'type': 'float', 'value': 0.0, 'step': 1e-6, 'siPrefix': True, 'suffix': 'eV'},
                {'name': self.geom_wavelength_str, 'type': 'float', 'value': 0.0, 'step': 1e-6, 'siPrefix': True, 'suffix': 'm', 'readonly': True},
                {'name': self.geom_pixelSize_str, 'type': 'float', 'value': 0.0, 'precision': 12, 'minVal': 1e-6, 'siPrefix': True, 'suffix': 'm'},
                {'name': self.geom_resolutionRings_str, 'type': 'bool', 'value': False, 'tip': "Display resolution rings", 'children': [
                    {'name': self.geom_resolution_str, 'type': 'str', 'value': None},
                    {'name': self.geom_resolutionUnits_str, 'type': 'list', 'values': {self.geom_unitA_crystal_str: self.unitA_c,
                                                                                  self.geom_unitNm_crystal_str: self.unitNm_c,
                                                                                  self.geom_unitQ_crystal_str: self.unitQ_c,
                                                                                  self.geom_unitA_physics_str: self.unitA_p,
                                                                                  self.geom_unitNm_physics_str: self.unitNm_p,
                                                                                  self.geom_unitQ_physics_str: self.unitQ_p,
                                                                                  self.geom_unitTwoTheta_str: self.unitTwoTheta},
                     'value': self.unitA_c},
                ]},
            ]},
        ]

        self.p1 = Parameter.create(name='paramsDiffractionGeometry', type='group', \
                                   children=self.params, expanded=True)
        self.p1.sigTreeStateChanged.connect(self.change)
        self.win.setParameters(self.p1, showTop=False)

        if using_pyqt4:
            self.parent.connect(self.deployGeomBtn, QtCore.SIGNAL("clicked()"), self.deploy)
            self.parent.connect(self.deployAutoGeomBtn, QtCore.SIGNAL("clicked()"), self.autoDeploy)
        else:
            self.deployGeomBtn.clicked.connect(self.deploy)
            self.deployAutoGeomBtn.clicked.connect(self.autoDeploy)

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
        if path[1] == self.geom_detectorDistance_str:
            self.updateDetectorDistance(data)
        elif path[1] == self.geom_clen_str:
            pass
        elif path[1] == self.geom_coffset_str:
            pass
        elif path[1] == self.geom_photonEnergy_str:
            self.updatePhotonEnergy(data)
        elif path[1] == self.geom_pixelSize_str:
            self.updatePixelSize(data)
        elif path[1] == self.geom_wavelength_str:
            pass
        elif path[1] == self.geom_resolutionRings_str and len(path) == 2:
            self.updateResolutionRings(data)
        elif path[2] == self.geom_resolution_str:
            self.updateResolution(data)
        elif path[2] == self.geom_resolutionUnits_str:
            self.updateResolutionUnits(data)

    def findPsanaGeometry(self):
        try:
            self.source = Detector.PyDetector.map_alias_to_source(self.parent.detInfo,
                                                                  self.parent.exp.ds.env())  # 'DetInfo(CxiDs2.0:Cspad.0)'
            self.calibSource = self.source.split('(')[-1].split(')')[0]  # 'CxiDs2.0:Cspad.0'
            self.detectorType = gu.det_type_from_source(self.source)  # 1
            self.calibGroup = gu.dic_det_type_to_calib_group[self.detectorType]  # 'CsPad::CalibV1'
            self.detectorName = gu.dic_det_type_to_name[self.detectorType].upper()  # 'CSPAD'

            if self.parent.args.localCalib:
                self.calibPath = "./calib/" + self.calibGroup + "/" + self.calibSource + "/geometry"
            else:
                self.calibPath = self.parent.dir + '/' + self.parent.experimentName[:3] + '/' + \
                                 self.parent.experimentName + "/calib/" + self.calibGroup + '/' + \
                                 self.calibSource + "/geometry"
            if self.parent.args.v >= 1: print("### calibPath: ", self.calibPath)

            # Determine which calib file to use
            geometryFiles = os.listdir(self.calibPath)
            if self.parent.args.v >= 1: print("geom: ", geometryFiles)
            self.calibFile = None
            minDiff = -1e6
            for fname in geometryFiles:
                if fname.endswith('.data'):
                    endValid = False
                    try:
                        startNum = int(fname.split('-')[0])
                    except:
                        continue
                    endNum = fname.split('-')[-1].split('.data')[0]
                    diff = startNum - self.parent.runNumber
                    # Make sure it's end number is valid too
                    if 'end' in endNum:
                        endValid = True
                    else:
                        try:
                            if self.parent.runNumber <= int(endNum):
                                endValid = True
                        except:
                            continue
                    if diff <= 0 and diff > minDiff and endValid is True:
                        minDiff = diff
                        self.calibFile = fname
        except:
            if self.parent.args.v >= 1: print("Couldn't find psana geometry")
            self.calibFile = None

    def deployCrystfelGeometry(self, arg):
        self.findPsanaGeometry()
        if self.calibFile is not None and self.parent.writeAccess:
            # Convert psana geometry to crystfel geom
            if '.temp.geom' in self.parent.index.geom:
                self.parent.index.p9.param(self.parent.index.index_grp,
                                           self.parent.index.index_geom_str).setValue(
                    self.parent.psocakeRunDir + '/.temp.geom')
                cmd = ["psana2crystfel", self.calibPath + '/' + self.calibFile,
                       self.parent.psocakeRunDir + "/.temp.geom", str(self.parent.coffset)]
                if self.parent.args.v >= 1: print("cmd: ", cmd)
                try:
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                    p.communicate()[0]
                    p.stdout.close()
                except:
                    print(highlight("Warning! deployCrystfelGeometry() failed.", 'r', 1))
                # FIXME: Configure crystfel geom file to read in a mask (crystfel 'mask_file=' broken?)
                with open(self.parent.psocakeRunDir + '/.temp.geom', 'r') as f: lines = f.readlines()
                newGeom = []
                for line in lines: # remove commented out lines
                    if '; mask =' in line:
                        newGeom.append(line.split('; ')[-1])
                        _cxiname = self.parent.psocakeRunDir + '/' \
                                 + self.parent.experimentName + '_' \
                                 + str(self.parent.runNumber).zfill(4)
                        if self.parent.pk.tag:
                            _cxiname += '_' + self.parent.pk.tag
                        _cxiname += '.cxi\n'
                        _line = 'mask_file =' + _cxiname
                        newGeom.append("; mask_file =\n")
                    elif '; mask_good =' in line:
                        newGeom.append(line.split('; ')[-1])
                    elif '; mask_bad =' in line:
                        newGeom.append(line.split('; ')[-1])
                    elif '; clen =' in line:
                        newGeom.append(line.split('; ')[-1])
                    elif '; photon_energy =' in line:
                        newGeom.append(line.split('; ')[-1])
                    elif '; adu_per_eV =' in line:
                        if 'epix10k' in self.parent.detInfo.lower() or \
                           'jungfrau4m' in self.parent.detInfo.lower():
                            newGeom.append(line.split('; ')[-1].split('0.1')[0]+"0.001") # override
                        else:
                            newGeom.append(line.split('; ')[-1])
                    else:
                        newGeom.append(line)
                with open(self.parent.psocakeRunDir + '/.temp.geom', 'w') as f:
                    f.writelines(newGeom)
        if self.parent.args.v >= 1: print("Done deployCrystfelGeometry")

    def updateClen(self, arg):
        self.p1.param(self.geom_grp, self.geom_clen_str).setValue(self.parent.clen)
        self.parent.coffset = self.parent.detectorDistance - self.parent.clen
        self.p1.param(self.geom_grp, self.geom_coffset_str).setValue(self.parent.coffset)
        if self.parent.args.v >= 1: print("Done updateClen: ", self.parent.coffset, self.parent.detectorDistance, self.parent.clen)

    def updateDetectorDistance(self, data):
        self.parent.detectorDistance = data / 1000.  # mm to metres
        self.updateClen(self.parent.facility)
        if self.parent.args.v >= 1: print("coffset (m), detectorDistance (m), clen (m): ", self.parent.coffset, self.parent.detectorDistance, self.parent.clen)
        self.writeCrystfelGeom(self.parent.facility)
        if self.hasGeometryInfo():
            if self.parent.args.v >= 1: print("has geometry info")
            self.updateGeometry()
        self.parent.img.updatePolarizationFactor()
        if self.parent.exp.image_property == self.parent.exp.disp_radialCorrection:
            self.parent.img.updateImage()
        if self.parent.pk.showPeaks: self.parent.pk.updateClassification()
        if self.parent.args.v >= 1: print("Done updateDetectorDistance")

    def updatePhotonEnergy(self, data):
        if data > 0:
            self.parent.photonEnergy = data
            # E = hc/lambda
            h = 6.626070e-34 # J.m
            c = 2.99792458e8 # m/s
            joulesPerEv = 1.602176621e-19 #J/eV
            if self.parent.photonEnergy > 0:
                self.parent.wavelength = (h/joulesPerEv*c)/self.parent.photonEnergy
            else:
                self.parent.wavelength = 0
        self.p1.param(self.geom_grp,self.geom_wavelength_str).setValue(self.parent.wavelength)
        if self.hasGeometryInfo():
            self.updateGeometry()

    def updatePixelSize(self, data):
        self.parent.pixelSize = data
        if self.hasGeometryInfo():
            self.updateGeometry()

    def hasGeometryInfo(self):
        if self.parent.detectorDistance is not None \
           and self.parent.photonEnergy is not None \
           and self.parent.pixelSize is not None:
            return True
        else:
            return False

    def writeCrystfelGeom(self, arg):
        if self.parent.index.hiddenCXI is not None:
            if os.path.isfile(self.parent.index.hiddenCXI):
                f = h5py.File(self.parent.index.hiddenCXI,'r')
                encoderVal = f['/LCLS/detector_1/EncoderValue'][0] / 1000. # metres
                f.close()
            else:
                encoderVal = self.parent.clen # metres

            coffset = self.parent.detectorDistance - encoderVal
            if self.parent.args.v >= 1: print("coffset (m),detectorDistance (m) ,encoderVal (m): ", coffset, self.parent.detectorDistance, encoderVal)
            # Replace coffset value in geometry file
            if '.temp.geom' in self.parent.index.geom and os.path.exists(self.parent.index.geom):
                for line in fileinput.FileInput(self.parent.index.geom, inplace=True):
                    if 'coffset' in line and line.strip()[0] is not ';':
                        coffsetStr = line.split('=')[0]+"= "+str(coffset)+"\n"
                        print(coffsetStr.rstrip()) # FIXME: check whether comma is required
                    elif 'clen' in line and line.strip()[0] is ';': #FIXME: hack for mfxc00318
                        _c =  line.split('; ')[-1]
                        print(_c.rstrip())  # comma is required
                    elif 'photon_energy' in line and line.strip()[0] is ';': #FIXME: hack for mfxc00318
                        _c =  line.split('; ')[-1]
                        print(_c.rstrip())  # comma is required
                    elif 'adu_per_eV' in line and line.strip()[0] is ';': #FIXME: hack for mfxc00318
                        _c =  line.split('; ')[-1]
                        print(_c.rstrip())  # comma is required
                    else:
                        print(line.rstrip()) # comma is required
        if self.parent.args.v >= 1: print("Done writeCrystfelGeom")

    def getClosestGeom(self):
        # Search for the correct geom file to use
        calibDir = self.parent.rootDir + '/calib/' + self.parent.detInfo + '/geometry'
        _geomFiles = glob.glob(calibDir + '/*.geom')
        _runWithGeom = np.array([int(a.split('/')[-1].split('-')[0]) for a in _geomFiles])
        diff = _runWithGeom - self.parent.runNumber
        geomFile = _geomFiles[ int(np.where(diff == np.max(diff[np.where(diff <= 0)[0]]))[0]) ]
        if self.parent.args.v >= 1: print("getClosestGeom::Choosing this geom file: ", geomFile)
        return geomFile

    def updateGeometry(self):
        if self.hasUserDefinedResolution:
            self.myResolutionRingList = self.parent.resolution
        else:
            self.myResolutionRingList = self.resolutionRingList
        self.thetaMax = np.zeros_like(self.myResolutionRingList)
        self.dMin_crystal = np.zeros_like(self.myResolutionRingList)
        self.qMax_crystal = np.zeros_like(self.myResolutionRingList)
        self.dMin_physics = np.zeros_like(self.myResolutionRingList)
        self.qMax_physics = np.zeros_like(self.myResolutionRingList)
        for i, pix in enumerate(self.myResolutionRingList):
            if self.parent.detectorDistance > 0 and self.parent.wavelength is not None:
                self.thetaMax[i] = np.arctan(pix*self.parent.pixelSize/self.parent.detectorDistance)
                self.qMax_crystal[i] = 2/self.parent.wavelength*np.sin(self.thetaMax[i]/2)
                self.dMin_crystal[i] = 1/self.qMax_crystal[i]
                self.qMax_physics[i] = 4*np.pi/self.parent.wavelength*np.sin(self.thetaMax[i]/2)
                self.dMin_physics[i] = np.pi/self.qMax_physics[i]
            if self.parent.args.v >= 1:
                print("updateGeometry: ", i, self.thetaMax[i], self.dMin_crystal[i], self.dMin_physics[i])
            if self.parent.resolutionRingsOn:
                self.updateRings()
        if self.parent.args.v >= 1: print("Done updateGeometry")

    def updateDock42(self, data):
        a = ['a','b','c','d','e','k','m','n','r','s']
        myStr = a[5]+a[8]+a[0]+a[5]+a[4]+a[7]
        if myStr in data:
            self.d42 = Dock("Console", size=(100,100))
            # build an initial namespace for console commands to be executed in (this is optional;
            # the user can always import these modules manually)
            namespace = {'pg': pg, 'np': np, 'self': self}
            # initial text to display in the console
            text = "You have awoken the "+myStr+"\nWelcome to psocake IPython: dir(self)\n" \
                                                "Here are some commonly used variables:\n" \
                                                "unassembled detector: self.parent.calib\n" \
                                                "assembled detector: self.parent.data\n" \
                                                "user-defined mask: self.parent.mk.userMask\n" \
                                                "streak mask: self.parent.mk.streakMask\n" \
                                                "psana mask: self.parent.mk.psanaMask"
            self.w42 = console.ConsoleWidget(parent=None,namespace=namespace, text=text)
            self.d42.addWidget(self.w42)
            self.parent.area.addDock(self.d42, 'bottom')

    def updateResolutionRings(self, data):
        self.parent.resolutionRingsOn = data
        if self.parent.exp.hasExpRunDetInfo():
            self.updateRings()
        if self.parent.args.v >= 1:
            print("Done updateResolutionRings")

    def updateResolution(self, data):
        # convert to array of floats
        _resolution = data.split(',')
        self.parent.resolution = np.zeros((len(_resolution,)))

        self.updateDock42(data)

        if data != '':
            for i in range(len(_resolution)):
                self.parent.resolution[i] = float(_resolution[i])

        if data != '':
            self.hasUserDefinedResolution = True
        else:
            self.hasUserDefinedResolution = False

        self.myResolutionRingList = self.parent.resolution
        self.dMin = np.zeros_like(self.myResolutionRingList)
        if self.hasGeometryInfo():
            self.updateGeometry()
        if self.parent.exp.hasExpRunDetInfo():
            self.updateRings()
        if self.parent.args.v >= 1:
            print("Done updateResolution")

    def updateResolutionUnits(self, data):
        # convert to array of floats
        self.parent.resolutionUnits = data
        if self.hasGeometryInfo():
            self.updateGeometry()
        if self.parent.exp.hasExpRunDetInfo():
            self.updateRings()
        if self.parent.args.v >= 1:
            print("Done updateResolutionUnits")

    def updateRings(self):
        if self.parent.resolutionRingsOn:
            self.clearRings()

            cenx = np.ones_like(self.myResolutionRingList)*self.parent.cx
            ceny = np.ones_like(self.myResolutionRingList)*self.parent.cy
            diameter = 2*self.myResolutionRingList

            self.parent.img.ring_feature.setData(cenx, ceny, symbol='o', \
                                      size=diameter, brush=(255,255,255,0), \
                                      pen='r', pxMode=False)

            for i,val in enumerate(self.dMin_crystal):
                if self.parent.resolutionUnits == self.unitA_c:
                    self.resolutionText.append(pg.TextItem(text='%s A' % float('%.3g' % (val*1e10)), border='w', fill=(0, 0, 255, 100)))
                elif self.parent.resolutionUnits == self.unitNm_c:
                    self.resolutionText.append(pg.TextItem(text='%s nm' % float('%.3g' % (val*1e9)), border='w', fill=(0, 0, 255, 100)))
                elif self.parent.resolutionUnits == self.unitQ_c:
                    self.resolutionText.append(pg.TextItem(text='%s m^-1' % float('%.3g' % (self.qMax_crystal[i])), border='w', fill=(0, 0, 255, 100)))
                elif self.parent.resolutionUnits == self.unitA_p:
                    self.resolutionText.append(pg.TextItem(text='%s A' % float('%.3g' % (self.dMin_physics[i]*1e10)), border='w', fill=(0, 0, 255, 100)))
                elif self.parent.resolutionUnits == self.unitNm_p:
                    self.resolutionText.append(pg.TextItem(text='%s nm' % float('%.3g' % (self.dMin_physics[i]*1e9)), border='w', fill=(0, 0, 255, 100)))
                elif self.parent.resolutionUnits == self.unitQ_p:
                    self.resolutionText.append(pg.TextItem(text='%s m^-1' % float('%.3g' % (self.qMax_physics[i])), border='w', fill=(0, 0, 255, 100)))
                elif self.parent.resolutionUnits == self.unitTwoTheta:
                    self.resolutionText.append(pg.TextItem(text='%s degrees' % float('%.3g' % (self.thetaMax[i]*180/np.pi)), border='w', fill=(0, 0, 255, 100)))
                self.parent.img.win.getView().addItem(self.resolutionText[i])
                self.resolutionText[i].setPos(self.myResolutionRingList[i]+self.parent.cx, self.parent.cy)
        else:
            self.clearRings()
        if self.parent.args.v >= 1: print("Done updateRings")

    def drawCentre(self):
        # Always indicate centre of detector
        try:
            self.parent.img.centre_feature.setData(np.array([self.parent.cx]), np.array([self.parent.cy]), symbol='o', \
                                               size=6, brush=(255, 255, 255, 0), pen='r', pxMode=False)
            if self.parent.args.v >= 1: print("Done drawCentre")
        except:
            pass

    def clearRings(self):
        if self.resolutionText:
            cen = [0,]
            self.parent.img.ring_feature.setData(cen, cen, size=0)
            for i,val in enumerate(self.resolutionText):
                self.parent.img.win.getView().removeItem(self.resolutionText[i])
            self.resolutionText = []

    def deploy(self):
        with pg.BusyCursor():
            # Calculate detector translation required in x and y
            dx = self.parent.pixelSize * 1e6 * (self.parent.roi.centreX - self.parent.cx)  # microns
            dy = self.parent.pixelSize * 1e6 * (self.parent.roi.centreY - self.parent.cy)  # microns
            dz = np.mean(-self.parent.det.coords_z(self.parent.evt)) - self.parent.detectorDistance * 1e6 # microns
            geo = self.parent.det.geometry(self.parent.evt)
            top = geo.get_top_geo()
            children = top.get_list_of_children()[0]
            geo.move_geo(children.oname, 0, dx=-dy, dy=-dx, dz=dz)
            fname =  self.parent.psocakeRunDir + "/"+str(self.parent.runNumber)+'-end.data'
            geo.save_pars_in_file(fname)
            print("#################################################")
            print("Deploying psana detector geometry: ", fname)
            print("#################################################")
            cmts = {'exp': self.parent.experimentName, 'app': 'psocake', 'comment': 'recentred geometry'}
            if self.parent.args.localCalib:
                calibDir = './calib'
            elif self.parent.args.outDir is None:
                calibDir = self.parent.rootDir + '/calib'
            else:
                calibDir = self.parent.dir + '/' + self.parent.experimentName[:3] + '/' + \
                           self.parent.experimentName + '/calib'
            deploy_calib_file(cdir=calibDir, src=str(self.parent.det.name), type='geometry',
                              run_start=self.parent.runNumber, run_end=None, ifname=fname, dcmts=cmts, pbits=0)
            # Reload new psana geometry
            self.parent.exp.setupExperiment()
            self.parent.img.getDetImage(self.parent.eventNumber)
            self.updateRings()
            self.parent.index.updateIndex()
            self.drawCentre()
            # Show mask
            self.parent.mk.updatePsanaMaskOn()

    def autoDeploy(self): #FIXME: yet to verify this works correctly on new lab coordinate
        with pg.BusyCursor():
            powderHits = np.load(self.parent.psocakeRunDir + '/' + self.parent.experimentName + '_' + str(self.parent.runNumber).zfill(4) + '_maxHits.npy')
            powderMisses = np.load(self.parent.psocakeRunDir + '/' + self.parent.experimentName + '_' + str(self.parent.runNumber).zfill(4) + '_maxMisses.npy')
            powderImg = self.parent.det.image(self.parent.evt, np.maximum(powderHits,powderMisses))
            centreRow, centreCol = findDetectorCentre(np.log(abs(powderImg)), self.parent.cx, self.parent.cy, range=200)
            print("Current centre along row,centre along column: ", self.parent.cx, self.parent.cy)
            print("Optimum centre along row,centre along column: ", centreRow, centreCol)
            allowedDeviation = 175 # pixels
            if abs(self.parent.cx - centreRow) <= allowedDeviation and \
                abs(self.parent.cy - centreCol) <= allowedDeviation:
                deploy = True
            else:
                deploy = False
                print("Too far away from current centre. I will not deploy the auto centred geometry.")
            if deploy:
                # Calculate detector translation in x and y
                dx = self.parent.pixelSize * 1e6 * (self.parent.cx - centreRow)  # microns
                dy = self.parent.pixelSize * 1e6 * (self.parent.cy - centreCol)  # microns
                dz = np.mean(-self.parent.det.coords_z(self.parent.evt)) - self.parent.detectorDistance * 1e6  # microns

                dx = self.parent.pixelSize * 1e6 * (self.parent.roi.centreX - self.parent.cx)  # microns
                dy = self.parent.pixelSize * 1e6 * (self.parent.roi.centreY - self.parent.cy)  # microns
                dz = np.mean(-self.parent.det.coords_z(self.parent.evt)) - self.parent.detectorDistance * 1e6  # microns

                geo = self.parent.det.geometry(self.parent.evt)
                top = geo.get_top_geo()
                children = top.get_list_of_children()[0]
                geo.move_geo(children.oname, 0, dx=-dy, dy=-dx, dz=dz)
                fname = self.parent.psocakeRunDir + "/" + str(self.parent.runNumber) + '-end.data'
                geo.save_pars_in_file(fname)
                print("#################################################")
                print("Deploying psana detector geometry: ", fname)
                print("#################################################")
                cmts = {'exp': self.parent.experimentName, 'app': 'psocake', 'comment': 'auto recentred geometry'}
                if self.parent.args.localCalib:
                    calibDir = './calib'
                elif self.parent.args.outDir is None:
                    calibDir = self.parent.rootDir + '/calib'
                else:
                    calibDir = self.parent.dir + '/' + self.parent.experimentName[:3] + '/' + self.parent.experimentName + \
                               '/calib'
                deploy_calib_file(cdir=calibDir, src=str(self.parent.det.name), type='geometry',
                                  run_start=self.parent.runNumber, run_end=None, ifname=fname, dcmts=cmts, pbits=0)
                # Reload new psana geometry
                self.parent.exp.setupExperiment()
                self.parent.img.getDetImage(self.parent.eventNumber)
                self.updateRings()
                self.parent.index.updateIndex()
                self.drawCentre()
                # Show mask
                self.parent.mk.updatePsanaMaskOn()

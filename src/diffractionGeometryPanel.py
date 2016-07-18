import numpy as np
import fileinput
import pyqtgraph as pg
import h5py
import os

class DiffractionGeometry(object):
    def __init__(self, parent = None):
        self.parent = parent

        self.resolutionRingList = np.array([100.,300.,500.,700.,900.,1100.])
        self.resolutionText = []
        self.geom_grp = 'Diffraction geometry'
        self.geom_detectorDistance_str = 'Detector distance'
        self.geom_clen_str = 'Home to Detector'
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

    ##############################
    # Mandatory parameter update #
    ##############################
    def paramUpdate(self, path, change, data):
        if path[1] == self.geom_detectorDistance_str:
            self.updateDetectorDistance(data)
        elif path[1] == self.geom_clen_str:
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

    def updateDetectorDistance(self, data):
        print "!updateDetectorDistance (mm): ", data
        self.parent.detectorDistance = data / 1000. # mm to metres
        self.parent.coffset = self.parent.detectorDistance - self.parent.clen
        print "!coffset (m), detectorDistance (m), clen (m): ", self.parent.coffset, self.parent.detectorDistance, self.parent.clen
        self.writeCrystfelGeom()
        self.parent.updateClassification()
        if self.hasGeometryInfo():
            if self.parent.args.v >= 1:
                print "has geometry info"
            self.updateGeometry()
        if self.parent.args.v >= 1:
            print "Done updateDetectorDistance"

    def updatePhotonEnergy(self, data):
        self.parent.photonEnergy = data
        # E = hc/lambda
        h = 6.626070e-34 # J.m
        c = 2.99792458e8 # m/s
        joulesPerEv = 1.602176621e-19 #J/eV
        self.parent.wavelength = (h/joulesPerEv*c)/self.parent.photonEnergy
        self.parent.p1.param(self.geom_grp,self.geom_wavelength_str).setValue(self.parent.wavelength)
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

    def writeCrystfelGeom(self):
        print "writeCrystfelGeom: ", self.parent.index.geom, self.parent.hiddenCXI
        if os.path.isfile(self.parent.hiddenCXI):
            f = h5py.File(self.parent.hiddenCXI,'r')
            encoderVal = f['/LCLS/detector_1/EncoderValue'][0] / 1000. # metres
            f.close()
            coffset = self.parent.detectorDistance - encoderVal
            if 1:#self.parent.args.v >= 1:
                print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
                print "&&& coffset (m),detectorDistance (m) ,encoderVal (m): ",coffset, self.parent.detectorDistance, encoderVal
            coffsetStr = "coffset = "+str(coffset)+"\n"
            print "coffsetStr: ", coffsetStr

            # Replace coffset value in geometry file
            for line in fileinput.input(self.parent.index.geom, inplace=True):
                if 'coffset' in line and line.strip()[0] is not ';':
                    coffsetStr = line.split('=')[0]+"= "+str(coffset)+"\n"
                    print coffsetStr, # comma is required
                else:
                    print line, # comma is required

    def updateGeometry(self):
        if self.parent.hasUserDefinedResolution:
            self.myResolutionRingList = self.parent.resolution
        else:
            self.myResolutionRingList = self.resolutionRingList
        self.thetaMax = np.zeros_like(self.myResolutionRingList)
        self.dMin_crystal = np.zeros_like(self.myResolutionRingList)
        self.qMax_crystal = np.zeros_like(self.myResolutionRingList)
        self.dMin_physics = np.zeros_like(self.myResolutionRingList)
        self.qMax_physics = np.zeros_like(self.myResolutionRingList)
        for i, pix in enumerate(self.myResolutionRingList):
            self.thetaMax[i] = np.arctan(pix*self.parent.pixelSize/self.parent.detectorDistance)
            self.qMax_crystal[i] = 2/self.parent.wavelength*np.sin(self.thetaMax[i]/2)
            self.dMin_crystal[i] = 1/self.qMax_crystal[i]
            self.qMax_physics[i] = 4*np.pi/self.parent.wavelength*np.sin(self.thetaMax[i]/2)
            self.dMin_physics[i] = np.pi/self.qMax_physics[i]
            if self.parent.args.v >= 1:
                print "updateGeometry: ", i, self.thetaMax[i], self.dMin_crystal[i], self.dMin_physics[i]
            if self.parent.resolutionRingsOn:
                self.updateRings()
        if self.parent.args.v >= 1:
            print "Done updateGeometry"

    def updateResolutionRings(self, data):
        self.parent.resolutionRingsOn = data
        if self.parent.hasExpRunDetInfo():
            self.updateRings()
        if self.parent.args.v >= 1:
            print "Done updateResolutionRings"

    def updateResolution(self, data):
        # convert to array of floats
        _resolution = data.split(',')
        self.parent.resolution = np.zeros((len(_resolution,)))

        self.parent.updateDock42(data)

        if data != '':
            for i in range(len(_resolution)):
                self.parent.resolution[i] = float(_resolution[i])

        if data != '':
            self.parent.hasUserDefinedResolution = True
        else:
            self.parent.hasUserDefinedResolution = False

        self.myResolutionRingList = self.parent.resolution
        self.dMin = np.zeros_like(self.myResolutionRingList)
        if self.hasGeometryInfo():
            self.updateGeometry()
        if self.parent.hasExpRunDetInfo():
            self.updateRings()
        if self.parent.args.v >= 1:
            print "Done updateResolution"

    def updateResolutionUnits(self, data):
        # convert to array of floats
        self.parent.resolutionUnits = data
        if self.hasGeometryInfo():
            self.updateGeometry()
        if self.parent.hasExpRunDetInfo():
            self.updateRings()
        if self.parent.args.v >= 1:
            print "Done updateResolutionUnits"

    def updateRings(self):
        if self.parent.resolutionRingsOn:
            self.clearRings()

            cenx = np.ones_like(self.myResolutionRingList)*self.parent.cx
            ceny = np.ones_like(self.myResolutionRingList)*self.parent.cy
            diameter = 2*self.myResolutionRingList

            self.parent.ring_feature.setData(cenx, ceny, symbol='o', \
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
                self.parent.w1.getView().addItem(self.resolutionText[i])
                self.resolutionText[i].setPos(self.myResolutionRingList[i]+self.parent.cx, self.parent.cy)
        else:
            self.clearRings()
        if self.parent.args.v >= 1:
            print "Done updateRings"

    def clearRings(self):
        if self.resolutionText:
            cen = [0,]
            self.parent.ring_feature.setData(cen, cen, size=0)
            for i,val in enumerate(self.resolutionText):
                self.parent.w1.getView().removeItem(self.resolutionText[i])
            self.resolutionText = []

    def deploy(self):
        print "Hello"
import time
import psana, os
import numpy as np
import PSCalib.GlobalUtils as gu
from PSCalib.GeometryAccess import GeometryAccess
from pyimgalgos.RadialBkgd import RadialBkgd, polarization_factor
import Detector.PyDetector

class psanaWhisperer():
    def __init__(self, experimentName, runNumber, detInfo, clen='', aduPerPhoton=1, localCalib=False):
        self.experimentName = experimentName
        self.runNumber = runNumber
        self.detInfo = detInfo
        self.clen = clen
        self.aduPerPhoton = aduPerPhoton
        self.localCalib = localCalib

    def setupExperiment(self):
        self.ds = psana.DataSource('exp=' + str(self.experimentName) + ':run=' + str(self.runNumber) + ':idx')
        self.run = self.ds.runs().next()
        self.times = self.run.times()
        self.eventTotal = len(self.times)
        self.env = self.ds.env()
        self.evt = self.run.event(self.times[0])
        self.det = psana.Detector(str(self.detInfo), self.env)
        self.det.do_reshape_2d_to_3d(flag=True)
        self.getDetInfoList()
        self.detAlias = self.getDetectorAlias(str(self.detInfo))
        self.updateClen() # Get epics variable, clen

    def updateClen(self):
        if 'cspad' in self.detInfo.lower() and 'cxi' in self.experimentName:
            self.epics = self.ds.env().epicsStore()
            self.clen = self.epics.value(self.clen)
        elif 'rayonix' in self.detInfo.lower() and 'mfx' in self.experimentName:
            self.clen = 0

    def getDetectorAlias(self, srcOrAlias):
        for i in self.detInfoList:
            src, alias, _ = i
            if srcOrAlias.lower() == src.lower() or srcOrAlias.lower() == alias.lower():
                return alias

    def getDetInfoList(self):
        myAreaDetectors = []
        self.detnames = psana.DetNames()
        for k in self.detnames:
            try:
                if Detector.PyDetector.dettype(str(k[0]), self.env) == Detector.AreaDetector.AreaDetector:
                    myAreaDetectors.append(k)
            except ValueError:
                continue
        self.detInfoList = list(set(myAreaDetectors))
        print "detInfoList: ", self.detInfoList

    def getEvent(self, number):
        self.evt = self.run.event(self.times[number])

    def getImg(self, number):
        self.getEvent(number)
        img = self.det.image(self.evt, self.det.calib(self.evt))
        return img

    def getImg(self):
        if self.evt is not None:
            img = self.det.image(self.evt, self.det.calib(self.evt))
            return img
        return None

    def getCheetahImg(self, calib=None):
        """Converts seg, row, col assuming (32,185,388)
           to cheetah 2-d table row and col (8*185, 4*388)
        """
        if 'cspad2x2' in self.detInfo.lower():
            print "Not implemented yet: cspad2x2"
        elif 'cspad' in self.detInfo.lower():
            if calib is None: calib = self.det.calib(self.evt) # (32,185,388)
            img = np.zeros((8 * 185, 4 * 388))
            counter = 0
            for quad in range(4):
                for seg in range(8):
                    img[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = calib[counter, :, :]
                    counter += 1
        elif 'rayonix' in self.detInfo.lower():
            if calib is None:
                img = np.squeeze(self.det.calib(self.evt))  # (1920,1920)
            else:
                img = np.squeeze(calib)
        return img

    def getCleanAssembledImg(self, backgroundEvent):
        """Returns psana assembled image
        """
        backgroundEvt = self.run.event(self.times[backgroundEvent])
        backgroundCalib = self.det.calib(backgroundEvt)
        calib = self.det.calib(self.evt)
        cleanCalib = calib - backgroundCalib
        img = self.det.image(self.evt, cleanCalib)
        return img

    def getAssembledImg(self):
        """Returns psana assembled image
        """
        img = self.det.image(self.evt)
        return img

    def getCalibImg(self):
        """Returns psana assembled image
        """
        img = self.det.calib(self.evt)
        return img

    def getCleanAssembledPhotons(self, backgroundEvent):
        """Returns psana assembled image in photon counts
        """
        backgroundEvt = self.run.event(self.times[backgroundEvent])
        backgroundCalib = self.det.calib(backgroundEvt)
        calib = self.det.calib(self.evt)
        cleanCalib = calib - backgroundCalib
        img = self.det.photons(self.evt, nda_calib=cleanCalib, adu_per_photon=self.aduPerPhoton)
        phot = self.det.image(self.evt, img)
        return phot

    def getAssembledPhotons(self):
        """Returns psana assembled image in photon counts
        """
        img = self.det.photons(self.evt, adu_per_photon=self.aduPerPhoton)
        phot = self.det.image(self.evt, img)
        return phot

    def getPsanaEvent(self, cheetahFilename):
        # Gets psana event given cheetahFilename, e.g. LCLS_2015_Jul26_r0014_035035_e820.h5
        hrsMinSec = cheetahFilename.split('_')[-2]
        fid = int(cheetahFilename.split('_')[-1].split('.')[0], 16)
        for t in self.times:
            if t.fiducial() == fid:
                localtime = time.strftime('%H:%M:%S', time.localtime(t.seconds()))
                localtime = localtime.replace(':', '')
                if localtime[0:3] == hrsMinSec[0:3]:
                    self.evt = self.run.event(t)
                else:
                    self.evt = None

    def getStartTime(self):
        self.evt = self.run.event(self.times[0])
        evtId = self.evt.get(psana.EventId)
        sec = evtId.time()[0]
        nsec = evtId.time()[1]
        fid = evtId.fiducials()
        return time.strftime('%FT%H:%M:%S-0800', time.localtime(sec))  # Hard-coded pacific time

    #####################################################################
    # TODO: Functions below are not being used yet
    #####################################################################
    def findPsanaGeometry(self):
        try:
            self.source = psana.Detector.PyDetector.map_alias_to_source(self.detInfo,
                                                                  self.ds.env())  # 'DetInfo(CxiDs2.0:Cspad.0)'
            self.calibSource = self.source.split('(')[-1].split(')')[0]  # 'CxiDs2.0:Cspad.0'
            self.detectorType = gu.det_type_from_source(self.source)  # 1
            self.calibGroup = gu.dic_det_type_to_calib_group[self.detectorType]  # 'CsPad::CalibV1'
            self.detectorName = gu.dic_det_type_to_name[self.detectorType].upper()  # 'CSPAD'
            if self.localCalib:
                self.calibPath = "./calib/" + self.calibGroup + "/" + self.calibSource + "/geometry"
            else:
                self.calibPath = "/reg/d/psdm/" + self.parent.experimentName[0:3] + \
                                 "/" + self.parent.experimentName + "/calib/" + \
                                 self.calibGroup + "/" + self.calibSource + "/geometry"

            # Determine which calib file to use
            geometryFiles = os.listdir(self.calibPath)
            self.calibFile = None
            minDiff = -1e6
            for fname in geometryFiles:
                if fname.endswith('.data'):
                    endValid = False
                    startNum = int(fname.split('-')[0])
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
            print "Couldn't find psana geometry"
            self.calibFile = None

    def setupRadialBackground(self):
        self.findPsanaGeometry()
        if self.calibFile is not None:
            self.geo = GeometryAccess(self.calibPath+'/'+self.calibFile)
            self.xarr, self.yarr, self.zarr = self.geo.get_pixel_coords()
            self.iX, self.iY = self.geo.get_pixel_coord_indexes()
            self.mask = self.geo.get_pixel_mask(mbits=0377)  # mask for 2x1 edges, two central columns, and unbound pixels with their neighbours
            self.rb = RadialBkgd(self.xarr, self.yarr, mask=self.mask, radedges=None, nradbins=100, phiedges=(0, 360), nphibins=1)
        else:
            self.rb = None

    def updatePolarizationFactor(self, detectorDistance_in_m):
        if self.rb is not None:
            self.pf = polarization_factor(self.rb.pixel_rad(), self.rb.pixel_phi(), detectorDistance_in_m*1e6) # convert to um

    def getCalib(self, evtNumber):
        if self.run is not None:
            self.evt = self.getEvent(evtNumber)
            if self.applyCommonMode: # play with different common mode
                if self.commonMode[0] == 5: # Algorithm 5
                    calib = self.det.calib(self.evt,cmpars=(self.commonMode[0], self.commonMode[1]))
                else: # Algorithms 1 to 4
                    print "### Overriding common mode: ", self.commonMode
                    calib = self.det.calib(self.evt,cmpars=(self.commonMode[0], self.commonMode[1],
                                                          self.commonMode[2], self.commonMode[3]))
            else:
                calib = self.det.calib(self.evt)
            return calib
        else:
            return None

    def getPreprocessedImage(self, evtNumber, image_property):
        disp_medianCorrection = 19
        disp_radialCorrection = 18
        disp_gainMask = 17
        disp_coordy= 16
        disp_coordx= 15
        disp_col= 14
        disp_row= 13
        disp_seg= 12
        disp_quad= 11
        disp_gain= 10
        disp_commonMode= 9
        disp_rms= 8
        disp_status= 7
        disp_pedestal= 6
        disp_photons= 5
        disp_raw= 4
        disp_pedestalCorrected= 3
        disp_commonModeCorrected= 2
        disp_adu= 1

        if image_property == disp_medianCorrection:  # median subtraction
            print "Sorry, this feature isn't available yet"
        elif image_property == disp_radialCorrection:  # radial subtraction + polarization corrected
            self.getEvent(evtNumber)
            calib = self.getCalib(evtNumber)
            if calib:
                self.pf.shape = self.parent.calib.shape
                calib = self.rb.subtract_bkgd(calib * self.pf)
        elif image_property == disp_adu:  # gain and hybrid gain corrected
            calib = self.getCalib(evtNumber)
        elif image_property == disp_commonModeCorrected:  # common mode corrected
            calib = self.getCommonModeCorrected(evtNumber)
        elif image_property == disp_pedestalCorrected:  # pedestal corrected
            calib = self.det.raw(self.evt).astype('float32')
            if calib: calib -= self.det.pedestals(self.evt)
        elif image_property == disp_raw:  # raw
            calib = self.det.raw(self.evt)
        elif image_property == disp_photons:  # photon counts
            calib = self.det.photons(self.evt, mask=self.parent.mk.userMask,
                                            adu_per_photon=self.parent.exp.aduPerPhoton)
            if calib is None:
                calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='int32')
        elif image_property == disp_pedestal:  # pedestal
            calib = self.parent.det.pedestals(self.parent.evt)
        elif image_property == disp_status:  # status
            calib = self.parent.det.status(self.parent.evt)
        elif image_property == disp_rms:  # rms
            calib = self.parent.det.rms(self.parent.evt)
        elif image_property == disp_commonMode:  # common mode
            calib = self.getCommonMode(evtNumber)
        elif image_property == disp_gain:  # gain
            calib = self.parent.det.gain(self.parent.evt)
        elif image_property == disp_gainMask:  # gain_mask
            calib = self.parent.det.gain_mask(self.parent.evt)
        elif image_property == disp_coordx:  # coords_x
            calib = self.parent.det.coords_x(self.parent.evt)
        elif image_property == disp_coordy:  # coords_y
            calib = self.parent.det.coords_y(self.parent.evt)

        shape = self.parent.det.shape(self.parent.evt)
        if len(shape) == 3:
            if image_property == disp_quad:  # quad ind
                calib = np.zeros(shape)
                for i in range(shape[0]):
                    # FIXME: handle detectors properly
                    if shape[0] == 32:  # cspad
                        calib[i, :, :] = int(i) % 8
                    elif shape[0] == 2:  # cspad2x2
                        calib[i, :, :] = int(i) % 2
                    elif shape[0] == 4:  # pnccd
                        calib[i, :, :] = int(i) % 4
            elif image_property == disp_seg:  # seg ind
                calib = np.zeros(shape)
                if shape[0] == 32:  # cspad
                    for i in range(32):
                        calib[i, :, :] = int(i) / 8
                elif shape[0] == 2:  # cspad2x2
                    for i in range(2):
                        calib[i, :, :] = int(i)
                elif shape[0] == 4:  # pnccd
                    for i in range(4):
                        calib[i, :, :] = int(i)
            elif image_property == disp_row:  # row ind
                calib = np.zeros(shape)
                if shape[0] == 32:  # cspad
                    for i in range(185):
                        calib[:, i, :] = i
                elif shape[0] == 2:  # cspad2x2
                    for i in range(185):
                        calib[:, i, :] = i
                elif shape[0] == 4:  # pnccd
                    for i in range(512):
                        calib[:, i, :] = i
            elif image_property == disp_col:  # col ind
                calib = np.zeros(shape)
                if shape[0] == 32:  # cspad
                    for i in range(388):
                        calib[:, :, i] = i
                elif shape[0] == 2:  # cspad2x2
                    for i in range(388):
                        calib[:, :, i] = i
                elif shape[0] == 4:  # pnccd
                    for i in range(512):
                        calib[:, :, i] = i

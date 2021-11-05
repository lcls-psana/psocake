from pyqtgraph.dockarea import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import time
import PSCalib.GlobalUtils as gu
import psana
from pyimgalgos.RadialBkgd import RadialBkgd, polarization_factor
from pyimgalgos.MedianFilter import median_filter_ndarr

from psocake.utils import *

class FriedelSym(object):
    def __init__(self, dim, centre):
        self.dim = dim
        self.centre = centre
        # Centre by zeropadding
        pdim = np.zeros_like(centre)  # padded dimension
        for i, val in enumerate(centre):
            pdim[i] = 2 * max(val, dim[i] - val + 1) + 1
        shift = (np.floor(pdim / 2.) + 1 - centre).astype('i4')
        endGap = pdim - dim
        self.pad = []
        for i in zip(shift, endGap - 1):
            self.pad.append(i)

    def __zeropad(self, img):
        zeropad = np.lib.pad(img, (self.pad), 'constant')
        return zeropad

    def applyFriedel(self, img, mask=None, mode='same'):
        zimg = self.__zeropad(img)
        if mask is not None:
            zmask = self.__zeropad(mask)
            zimg[np.where(zmask == 0)] = 0
        imgSym = zimg.ravel() + zimg.ravel()[::-1]
        imgSym.shape = zimg.shape
        if mask is None:
            imgSym /= 2.
        else:
            maskSym = zmask.ravel() + zmask.ravel()[::-1]
            maskSym.shape = zimg.shape
            a = np.zeros_like(imgSym)
            a[np.where(maskSym > 0)] = imgSym[np.where(maskSym > 0)] / maskSym[np.where(maskSym > 0)]
            imgSym = a
        if mode == 'same':
            slices = [slice(a, imgSym.shape[i] - b) for i, (a, b) in enumerate(self.pad)]
            cropImg = imgSym[slices]
            return cropImg
        elif mode == 'full':
            return imgSym

class ImageViewer(object):
    def __init__(self, parent = None):
        self.parent = parent

        self.maxPercentile = 0
        self.minPercentile = 0
        self.displayMaxPercentile = 99.8
        self.displayMinPercentile = 1.0

        self.rb = None
        pg.setConfigOptions(imageAxisOrder="row-major")
        ## Dock 2: Image Panel
        self.dock = Dock("Image Panel", size=(500, 400))
        self.win = pg.ImageView(view=pg.PlotItem())
        self.win.getView().invertY(True)
        self.img_feature = pg.ImageItem()
        # Set color map white -> black
        #colors = [
        #    (255, 255, 255),
        #    (0, 0, 0)
        #]
        #self.cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        #self.win.setColorMap(self.cmap)

        self.win.getView().addItem(self.img_feature)
        self.ring_feature = pg.ScatterPlotItem()
        self.centre_feature = pg.ScatterPlotItem()
        self.peak_feature = pg.ScatterPlotItem()
        self.indexedPeak_feature = pg.ScatterPlotItem()
        self.filePeak_feature = pg.ScatterPlotItem()
        self.z_direction = pg.ScatterPlotItem()
        self.z_direction1 = pg.ScatterPlotItem()
        self.win.getView().addItem(self.ring_feature)
        self.win.getView().addItem(self.centre_feature)
        self.win.getView().addItem(self.peak_feature)
        self.win.getView().addItem(self.indexedPeak_feature)
        self.win.getView().addItem(self.filePeak_feature)
        self.win.getView().addItem(self.z_direction)
        self.win.getView().addItem(self.z_direction1)
        self.abc_text = pg.TextItem(html='', anchor=(0,0)) # unit cell display
        self.win.getView().addItem(self.abc_text)
        self.peak_text = pg.TextItem(html='', anchor=(0,0)) # peak display
        self.win.getView().addItem(self.peak_text)
        self.dock.addWidget(self.win)
        self.drawLabCoordinates()

    def clearPeakMessage(self):
        self.win.getView().removeItem(self.peak_text)
        self.peak_feature.setData([], [], pxMode=False)
        if self.parent.args.v >= 1: print("Done clearPeakMessage")

    def clearIndexingMessage(self):
        self.win.getView().removeItem(self.abc_text)
        self.indexedPeak_feature.setData([], [], pxMode=False)
        if self.parent.args.v >= 1: print("Done clearIndexingMessage")

    def updateIndexingMessage(self, abc_text):
        self.win.getView().addItem(abc_text)

    def drawLabCoordinates(self):
        (cenX,cenY) = (-20,-20) # no offset
        # Draw xy arrows
        symbolSize = 40
        cutoff=symbolSize/2
        headLen=-30
        tailLen=-30+cutoff
        xArrow = pg.ArrowItem(angle=180, tipAngle=30, baseAngle=20, headLen=headLen, tailLen=tailLen, tailWidth=8, pen=None, brush='b', pxMode=False)
        xArrow.setPos(2*headLen+cenY, 0+cenX)
        self.win.getView().addItem(xArrow)
        yArrow = pg.ArrowItem(angle=-90, tipAngle=30, baseAngle=20, headLen=headLen, tailLen=tailLen, tailWidth=8, pen=None, brush='r', pxMode=False)
        yArrow.setPos(0+cenY, 2*headLen+cenX)
        self.win.getView().addItem(yArrow)

        # Lab coordinates: Add z-direction
        self.z_direction.setData([0+cenY], [0+cenX], symbol='o', \
                                 size=symbolSize, brush='w', \
                                 pen={'color': 'k', 'width': 4}, pxMode=False)
        self.z_direction1.setData([0+cenY], [0+cenX], symbol='x', \
                                 size=symbolSize, brush='k', \
                                 pen={'color': 'k', 'width': 4}, pxMode=False)
        # Lab coordinates: Add xyz text
        self.x_text = pg.TextItem(html='<div style="text-align: left"><span style="color: #0000FF; font-size: 16pt;">x</span></div>', anchor=(0,0))
        self.win.getView().addItem(self.x_text)
        self.x_text.setPos(2*headLen+cenY, 0+cenX)
        self.y_text = pg.TextItem(html='<div style="text-align: left"><span style="color: #FF0000; font-size: 16pt;">y</span></div>', anchor=(1,1))
        self.win.getView().addItem(self.y_text)
        self.y_text.setPos(0+cenY, 2*headLen+cenX)
        self.z_text = pg.TextItem(html='<div style="text-align: left"><span style="color: #000000; font-size: 16pt;">z</span></div>', anchor=(1,0))
        self.win.getView().addItem(self.z_text)
        self.z_text.setPos(-headLen+cenY, 0+cenX)

        # Label xy axes
        self.x_axis = self.win.getView().getAxis('bottom')
        self.x_axis.setLabel('X-axis (pixels)')
        self.y_axis = self.win.getView().getAxis('left')
        self.y_axis.setLabel('Y-axis (pixels)')

    def updateImage(self, calib=None):
        if self.parent.hasExperimentName and self.parent.hasRunNumber and self.parent.hasDetInfo:
            if calib is None:
                self.parent.calib, self.parent.data = self.getDetImage(self.parent.eventNumber)
            else:
                _, self.parent.data = self.getDetImage(self.parent.eventNumber, calib=calib)

            if self.parent.firstUpdate:
                if self.parent.exp.logscaleOn:
                    self.win.setImage(np.log10(abs(self.parent.data) + self.parent.eps))
                    self.win.view.setRange(xRange=[0, self.parent.data.shape[0]],
                                           yRange=[0, self.parent.data.shape[1]],
                                           padding=0)
                    self.parent.firstUpdate = False
                else:
                    self.minPercentile = np.percentile(self.parent.data, self.displayMinPercentile)
                    if self.minPercentile < 0: self.minPercentile = 0
                    self.maxPercentile = np.percentile(self.parent.data, self.displayMaxPercentile)
                    self.win.setImage(self.parent.data, levels=(self.minPercentile, self.maxPercentile))
                    # properly resize the image to fit on screen
                    self.win.view.setRange(xRange=[0, self.parent.data.shape[0]],
                                           yRange=[0, self.parent.data.shape[1]],
                                           padding=0)
                    self.parent.firstUpdate = False
            else:
                if self.parent.exp.logscaleOn:
                    self.win.setImage(np.log10(abs(self.parent.data) + self.parent.eps),
                                     autoRange=False, autoLevels=False, autoHistogramRange=False)
                else:
                    if self.minPercentile == 0 and self.maxPercentile == 0:
                        self.minPercentile = np.percentile(self.parent.data, self.displayMinPercentile)
                        if self.minPercentile < 0: self.minPercentile = 0
                        self.maxPercentile = np.percentile(self.parent.data, self.displayMaxPercentile)
                        self.win.setImage(self.parent.data, levels=(self.minPercentile, self.maxPercentile))
                    else:
                        self.win.setImage(self.parent.data, autoRange=False, autoLevels=False,
                                         autoHistogramRange=False)

        # Load peak parameters if exists
        if 'sfx' in self.parent.args.mode and self.parent.pk.userUpdate is None:
            self.parent.pk.updateParam()
        else:
            self.parent.mk.displayMask()

        if 'label' in self.parent.args.mode:
            self.parent.labeling.updateText()

        if self.parent.args.v >= 1: print("Done updateImage")

    def getCalib(self, evtNumber):
        if self.parent.exp.run is not None:
            self.parent.evt = self.parent.exp.getEvt(evtNumber)
            if self.parent.exp.applyCommonMode: # play with different common mode
                if self.parent.exp.commonMode[0] == 5: # Algorithm 5
                    calib = self.parent.det.calib(self.parent.evt,
                                                  cmpars=(self.parent.exp.commonMode[0], self.parent.exp.commonMode[1]))
                else: # Algorithms 1 to 4
                    print("### Overriding common mode: ", self.parent.exp.commonMode)
                    calib = self.parent.det.calib(self.parent.evt,
                                                  cmpars=(self.parent.exp.commonMode[0], self.parent.exp.commonMode[1],
                                                          self.parent.exp.commonMode[2], self.parent.exp.commonMode[3]))
            else:
                if not self.parent.inputImages:
                    calib = self.parent.det.calib(self.parent.evt)
                else:
                    f = h5py.File(self.parent.inputImages)
                    ind = np.where(f['eventNumber'][()] == evtNumber)[0][0]
                    if len(f['/data/data'].shape) == 3:
                        calib = self.parent.detDesc.ipct(f['data/data'][ind, :, :])
                    else:
                        calib = f['data/data'][ind, :, :, :]
                    f.close()
            return calib
        else:
            return None

    def getCommonModeCorrected(self, evtNumber):
        if self.parent.exp.run is not None:
            try:
                self.parent.evt = self.parent.exp.getEvt(evtNumber)
                pedestalCorrected = self.parent.det.raw(self.parent.evt) - self.parent.det.pedestals(self.parent.evt)
                if self.parent.exp.applyCommonMode:  # play with different common mode
                    if self.parent.exp.commonMode[0] == 5:  # Algorithm 5
                        commonMode = self.parent.det.common_mode_correction(self.parent.evt, pedestalCorrected,
                                                                            cmpars=(self.parent.exp.commonMode[0],
                                                                                    self.parent.exp.commonMode[1]))
                        commonModeCorrected = pedestalCorrected - commonMode
                    else:  # Algorithms 1 to 4
                        print("### Overriding common mode: ", self.parent.exp.commonMode)
                        commonMode = self.parent.det.common_mode_correction(self.parent.evt, pedestalCorrected,
                                                                     cmpars=(self.parent.exp.commonMode[0], self.parent.exp.commonMode[1],
                                                                             self.parent.exp.commonMode[2], self.parent.exp.commonMode[3]))
                        commonModeCorrected = pedestalCorrected - commonMode
                else:
                    commonMode = self.parent.det.common_mode_correction(self.parent.evt, pedestalCorrected)
                    commonModeCorrected = pedestalCorrected + commonMode # WHAT! You need to ADD common mode?!!
                return commonModeCorrected
            except:
                return None
        else:
            return None

    def getCommonMode(self, evtNumber):
        if self.parent.exp.run is not None:
            self.parent.evt = self.parent.exp.getEvt(evtNumber)
            pedestalCorrected = self.parent.det.raw(self.parent.evt) - self.parent.det.pedestals(self.parent.evt)
            if self.parent.exp.applyCommonMode: # play with different common mode
                print("### Overriding common mode: ", self.parent.exp.commonMode)
                if self.parent.exp.commonMode[0] == 5: # Algorithm 5
                    cm = self.parent.det.common_mode_correction(self.parent.evt, pedestalCorrected,
                                                                cmpars=(self.parent.exp.commonMode[0],
                                                                        self.parent.exp.commonMode[1]))
                else: # Algorithms 1 to 4
                    cm = self.parent.det.common_mode_correction(self.parent.evt, pedestalCorrected,
                                                                cmpars=(self.parent.exp.commonMode[0],
                                                                        self.parent.exp.commonMode[1],
                                                                        self.parent.exp.commonMode[2],
                                                                        self.parent.exp.commonMode[3]))
            else:
                cm = self.parent.det.common_mode_correction(self.parent.evt, pedestalCorrected)
            return cm
        else:
            return None

    def getAssembledImage(self, arg, calib):
        _calib = calib.copy() # this is important
        tic = time.time()
        if self.parent.exp.applyFriedel: # Apply Friedel symmetry
            print("Apply Friedel symmetry")
            try:
                centre = self.parent.det.point_indexes(self.parent.evt, pxy_um=(0, 0),
                                                   pix_scale_size_um=None,
                                                   xy0_off_pix=None,
                                                   cframe=gu.CFRAME_PSANA, fract=True)
            except AttributeError:
                centre = self.parent.det.point_indexes(self.parent.evt)
            self.fs = FriedelSym(self.parent.exp.detGuaranteedData.shape, centre)
            data = self.parent.det.image(self.parent.evt, _calib)
            if self.parent.mk.combinedMask is not None:
                data = self.fs.applyFriedel(data, mask=self.parent.det.image(self.parent.evt, self.parent.mk.combinedMask), mode='same')
            else:
                data = self.fs.applyFriedel(data, mask=None, mode='same')
        else:
            data = self.parent.det.image(self.parent.evt, _calib)
        if data is None: data = _calib
        toc = time.time()
        if self.parent.args.v >= 1: print("time assemble: ", toc-tic)
        return data

    def setupRadialBackground(self):
        self.parent.geom.findPsanaGeometry()
        if self.parent.geom.calibFile is not None:
            if self.parent.args.v >= 1: print("calibFile: ", self.parent.geom.calibPath+'/'+self.parent.geom.calibFile)
            self.geo = self.parent.det.geometry(self.parent.runNumber) #self.geo = GeometryAccess(self.parent.geom.calibPath+'/'+self.parent.geom.calibFile)
            self.xarr, self.yarr, self.zarr = self.geo.get_pixel_coords()
            self.iX, self.iY = self.geo.get_pixel_coord_indexes()
            self.mask = self.geo.get_pixel_mask(mbits=0o0377)
            # mask for 2x1 edges, two central columns, and unbound pixels with their neighbours
            self.rb = RadialBkgd(self.xarr, self.yarr, mask=self.mask, radedges=None, nradbins=100, phiedges=(0, 360), nphibins=1)
            if self.parent.args.v >= 1: print("Done setupRadialBackground")
        else:
            self.rb = None

    def updatePolarizationFactor(self): # FIXME: change to vertical polarization after July 2020
        if self.rb is not None:
            self.pf = polarization_factor(self.rb.pixel_rad(), self.rb.pixel_phi()+90, self.parent.detectorDistance*1e6) # convert to um
            if self.parent.args.v >= 1: print("Done updatePolarizationFactor")

    def updateDetectorCentre(self):
        try:
            self.parent.cy, self.parent.cx = self.parent.det.point_indexes(self.parent.evt, pxy_um=(0, 0),
                                                                           pix_scale_size_um=None,
                                                                           xy0_off_pix=None,
                                                                           cframe=gu.CFRAME_PSANA, fract=True)
        except AttributeError:
            self.parent.cy, self.parent.cx = self.parent.det.point_indexes(self.parent.evt, pxy_um=(0, 0))
        if self.parent.cx is None:
            print("#######################################")
            print("WARNING: Unable to get detector center position. Check detector geometry is deployed.")
            print("#######################################")
            data = self.parent.det.image(self.parent.evt, self.parent.exp.detGuaranteed)
            self.parent.cx, self.parent.cy = self.getCentre(data.shape)
        if self.parent.args.v >= 1: print("cx, cy: ", self.parent.cx, self.parent.cy)


    def getDetImage(self, evtNumber, calib=None):
        if calib is None:
            if self.parent.exp.image_property == self.parent.exp.disp_medianCorrection:  # median subtraction
                calib = self.getCalib(evtNumber)
                if calib is None: calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='float32')
                calib -= median_filter_ndarr(calib, self.parent.exp.medianFilterRank)
            elif self.parent.exp.image_property == self.parent.exp.disp_radialCorrection:  # radial subtraction + polarization corrected
                calib = self.getCalib(evtNumber)
                if calib is None: calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='float32')
                self.pf.shape = calib.shape # FIXME: shape is 1d
                calib = self.rb.subtract_bkgd(calib * self.pf)
                calib.shape = self.parent.calib.shape # FIXME: shape is 1d
            elif self.parent.exp.image_property == self.parent.exp.disp_adu: # gain and hybrid gain corrected
                calib = self.getCalib(evtNumber)
                if calib is None: calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='float32')
            elif self.parent.exp.image_property == self.parent.exp.disp_commonModeCorrected: # common mode corrected
                calib = self.getCommonModeCorrected(evtNumber)
                if calib is None: calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='float32')
            elif self.parent.exp.image_property == self.parent.exp.disp_pedestalCorrected: # pedestal corrected
                calib = self.parent.det.raw(self.parent.evt).astype('float32')
                if calib is None:
                    calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='float32')
                else:
                    calib -= self.parent.det.pedestals(self.parent.evt)
            elif self.parent.exp.image_property == self.parent.exp.disp_raw: # raw
                calib = self.parent.det.raw(self.parent.evt)
                if calib is None:
                    calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='float32')
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == self.parent.exp.disp_photons: # photon counts
                calib = self.parent.det.photons(self.parent.evt, mask=self.parent.mk.userMask, adu_per_photon=self.parent.exp.aduPerPhoton)
                if calib is None:
                    calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='int32')
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == self.parent.exp.disp_pedestal: # pedestal
                calib = self.parent.det.pedestals(self.parent.evt)
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == self.parent.exp.disp_status: # status
                calib = self.parent.det.status(self.parent.evt)
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == self.parent.exp.disp_rms: # rms
                calib = self.parent.det.rms(self.parent.evt)
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == self.parent.exp.disp_commonMode: # common mode
                calib = self.getCommonMode(evtNumber)
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == self.parent.exp.disp_gain: # gain
                calib = self.parent.det.gain(self.parent.evt)
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == self.parent.exp.disp_gainMask: # gain_mask
                calib = self.parent.det.gain_mask(self.parent.evt)
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == self.parent.exp.disp_coordx: # coords_x
                try:
                    calib = self.parent.det.coords_x(self.parent.evt, cframe=1) # lab coords
                except:
                    calib = self.parent.det.coords_x(self.parent.evt) # matrix coords
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == self.parent.exp.disp_coordy: # coords_y
                try:
                    calib = self.parent.det.coords_y(self.parent.evt, cframe=1) # lab coords
                except:
                    calib = self.parent.det.coords_y(self.parent.evt) # matrix coords
                self.parent.firstUpdate = True

            shape = self.parent.det.shape(self.parent.evt)
            if len(shape) == 3:
                if self.parent.exp.image_property == self.parent.exp.disp_seg: # seg ind
                    calib = np.zeros(shape)
                    for i in range(shape[0]):
                        calib[i, :, :] = i
                    self.parent.firstUpdate = True
                elif self.parent.exp.image_property == self.parent.exp.disp_row: # row ind
                    calib = np.zeros(shape)
                    for i in range(shape[1]):
                        calib[:, i, :] = i
                    self.parent.firstUpdate = True
                elif self.parent.exp.image_property == self.parent.exp.disp_col: # col ind
                    calib = np.zeros(shape)
                    for i in range(shape[2]):
                        calib[:,:,i] = i
                    self.parent.firstUpdate = True

        # Update photon energy
        self.parent.exp.updatePhotonEnergy(self.parent.facility)

        # Update clen
        self.parent.geom.updateClen(self.parent.facility)

        # Write a temporary geom file
        #self.parent.geom.deployCrystfelGeometry(self.parent.facility)
        #self.parent.geom.writeCrystfelGeom(self.parent.facility) # Hack to override coffset

        # Get assembled image
        if calib is not None:
            data = self.getAssembledImage(self.parent.facility, calib)
        else:
            calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='float32')
            data = self.getAssembledImage(self.parent.facility, calib)

        # Update ROI histogram
        if self.parent.roi.roiCurrent == 'rect':
            self.parent.roi.updateRoi(self.parent.roi.roi)
        elif self.parent.roi.roiCurrent == 'poly':
            self.parent.roi.updateRoi(self.parent.roi.roiPoly)
        elif self.parent.roi.roiCurrent == 'circ':
            self.parent.roi.updateRoi(self.parent.roi.roiCircle)

        return calib, data

    def getCentre(self,shape):
        cx = shape[1]/2
        cy = shape[0]/2
        return cx,cy

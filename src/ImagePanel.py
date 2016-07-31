from pyqtgraph.dockarea import *
import pyqtgraph as pg
import numpy as np
import time, psana, datetime

class ImageViewer(object):
    def __init__(self, parent = None):
        self.parent = parent

        self.maxPercentile = 0
        self.minPercentile = 0
        self.displayMaxPercentile = 99.0
        self.displayMinPercentile = 1.0

        ## Dock 1: Image Panel
        self.d1 = Dock("Image Panel", size=(500, 400))
        self.w1 = pg.ImageView(view=pg.PlotItem())
        self.w1.getView().invertY(False)
        self.img_feature = pg.ImageItem()
        self.w1.getView().addItem(self.img_feature)
        self.ring_feature = pg.ScatterPlotItem()
        self.centre_feature = pg.ScatterPlotItem()
        self.peak_feature = pg.ScatterPlotItem()
        self.indexedPeak_feature = pg.ScatterPlotItem()
        self.z_direction = pg.ScatterPlotItem()
        self.z_direction1 = pg.ScatterPlotItem()
        self.w1.getView().addItem(self.ring_feature)
        self.w1.getView().addItem(self.centre_feature)
        self.w1.getView().addItem(self.peak_feature)
        self.w1.getView().addItem(self.indexedPeak_feature)
        self.w1.getView().addItem(self.z_direction)
        self.w1.getView().addItem(self.z_direction1)
        self.abc_text = pg.TextItem(html='', anchor=(0,0))
        self.w1.getView().addItem(self.abc_text)
        self.peak_text = pg.TextItem(html='', anchor=(0,0))
        self.w1.getView().addItem(self.peak_text)
        self.d1.addWidget(self.w1)

        self.drawLabCoordinates()  # FIXME: This does not match the lab coordinates yet!

    def drawLabCoordinates(self):
        (cenX,cenY) = (0,0) # no offset
        # Draw xy arrows
        symbolSize = 40
        cutoff=symbolSize/2
        headLen=30
        tailLen=30-cutoff
        xArrow = pg.ArrowItem(angle=180, tipAngle=30, baseAngle=20, headLen=headLen, tailLen=tailLen, tailWidth=8, pen=None, brush='b', pxMode=False)
        xArrow.setPos(2*headLen+cenX,0+cenY)
        self.w1.getView().addItem(xArrow)
        yArrow = pg.ArrowItem(angle=-90, tipAngle=30, baseAngle=20, headLen=headLen, tailLen=tailLen, tailWidth=8, pen=None, brush='r', pxMode=False)
        yArrow.setPos(0+cenX,2*headLen+cenY)
        self.w1.getView().addItem(yArrow)

        # Lab coordinates: Add z-direction
        self.z_direction.setData([0+cenX], [0+cenY], symbol='o', \
                                 size=symbolSize, brush='w', \
                                 pen={'color': 'k', 'width': 4}, pxMode=False)
        self.z_direction1.setData([0+cenX], [0+cenY], symbol='o', \
                                 size=symbolSize/6, brush='k', \
                                 pen={'color': 'k', 'width': 4}, pxMode=False)
        # Lab coordinates: Add xyz text
        self.x_text = pg.TextItem(html='<div style="text-align: center"><span style="color: #0000FF; font-size: 16pt;">x</span></div>', anchor=(0,0))
        self.w1.getView().addItem(self.x_text)
        self.x_text.setPos(2*headLen+cenX, 0+cenY)
        self.y_text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FF0000; font-size: 16pt;">y</span></div>', anchor=(1,1))
        self.w1.getView().addItem(self.y_text)
        self.y_text.setPos(0+cenX, 2*headLen+cenY)
        self.z_text = pg.TextItem(html='<div style="text-align: center"><span style="color: #FFFFFF; font-size: 16pt;">z</span></div>', anchor=(1,0))
        self.w1.getView().addItem(self.z_text)
        self.z_text.setPos(-headLen+cenX, 0+cenY)

        # Label xy axes
        self.x_axis = self.w1.getView().getAxis('bottom')
        self.x_axis.setLabel('X-axis (pixels)')
        self.y_axis = self.w1.getView().getAxis('left')
        self.y_axis.setLabel('Y-axis (pixels)')

    def updateImage(self,calib=None):
        if self.parent.hasExperimentName and self.parent.hasRunNumber and self.parent.hasDetInfo:
            if calib is None:
                self.parent.calib, self.parent.data = self.getDetImage(self.parent.eventNumber)
            else:
                _, self.parent.data = self.getDetImage(self.parent.eventNumber, calib=calib)

            if self.parent.firstUpdate:
                if self.parent.exp.logscaleOn:
                    self.w1.setImage(np.log10(abs(self.parent.data) + self.parent.eps))
                    self.parent.firstUpdate = False
                else:
                    self.minPercentile = np.percentile(self.parent.data, self.displayMinPercentile)
                    self.maxPercentile = np.percentile(self.parent.data, self.displayMaxPercentile)
                    self.w1.setImage(self.parent.data, levels=(self.minPercentile, self.maxPercentile))
                    self.parent.firstUpdate = False
            else:
                if self.parent.exp.logscaleOn:
                    self.w1.setImage(np.log10(abs(self.parent.data) + self.parent.eps),
                                     autoRange=False, autoLevels=False, autoHistogramRange=False)
                else:
                    if self.minPercentile == 0 and self.maxPercentile == 0:
                        self.minPercentile = np.percentile(self.parent.data, self.displayMinPercentile)
                        self.maxPercentile = np.percentile(self.parent.data, self.displayMaxPercentile)
                        self.w1.setImage(self.parent.data, levels=(self.minPercentile, self.maxPercentile))
                    else:
                        self.w1.setImage(self.parent.data, autoRange=False, autoLevels=False, autoHistogramRange=False)

            try:
                fname = "/reg/d/psdm/cxi/cxitut13/res/psocake/log/" + \
                        self.parent.exp.username + "_" + self.parent.experimentName + "_" \
                        + str(self.parent.runNumber) + "_" + self.parent.detInfo + ".txt"
                with open(fname, "a") as myfile:
                    date = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
                    myStr = date + ": " + str(self.parent.eventNumber) + '\n'
                    myfile.write(myStr)
            except:
                pass

        if self.parent.args.v >= 1: print "Done updateImage"

    def getCalib(self, evtNumber):
        if self.parent.exp.run is not None:
            self.parent.evt = self.parent.exp.getEvt(evtNumber)
            if self.parent.exp.applyCommonMode: # play with different common mode
                if self.parent.exp.commonMode[0] == 5: # Algorithm 5
                    calib = self.parent.det.calib(self.parent.evt,
                                                  cmpars=(self.parent.exp.commonMode[0], self.parent.exp.commonMode[1]))
                else: # Algorithms 1 to 4
                    print "### Overriding common mode: ", self.parent.exp.commonMode
                    calib = self.parent.det.calib(self.parent.evt,
                                                  cmpars=(self.parent.exp.commonMode[0], self.parent.exp.commonMode[1],
                                                          self.parent.exp.commonMode[2], self.parent.exp.commonMode[3]))
            else:
                calib = self.parent.det.calib(self.parent.evt)
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
                        print "### Overriding common mode: ", self.parent.exp.commonMode
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
                print "### Overriding common mode: ", self.parent.exp.commonMode
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

    def getAssembledImage(self, calib):
        _calib = calib.copy() # this is important
        tic = time.time()
        data = self.parent.det.image(self.parent.evt, _calib)
        if data is None: data = _calib
        toc = time.time()
        if self.parent.args.v >= 1: print "time assemble: ", toc-tic
        return data

    def getDetImage(self, evtNumber, calib=None):
        if calib is None:
            if self.parent.exp.image_property == 1: # gain and hybrid gain corrected
                calib = self.getCalib(evtNumber)
                if calib is None: calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='float32')
            elif self.parent.exp.image_property == 2: # common mode corrected
                if self.parent.args.v >= 1: print "common mode corrected"
                calib = self.getCommonModeCorrected(evtNumber)
                if calib is None: calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='float32')
            elif self.parent.exp.image_property == 3: # pedestal corrected
                calib = self.parent.det.raw(self.parent.evt).astype('float32')
                if calib is None:
                    calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='float32')
                else:
                    calib -= self.parent.det.pedestals(self.parent.evt)
            elif self.parent.exp.image_property == 4: # raw
                calib = self.parent.det.raw(self.parent.evt)
                if calib is None:
                    calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='float32')
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == 5: # photon counts
                print "Sorry, this feature is not available"
            elif self.parent.exp.image_property == 6: # pedestal
                calib = self.parent.det.pedestals(self.parent.evt)
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == 7: # status
                calib = self.parent.det.status(self.parent.evt)
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == 8: # rms
                calib = self.parent.det.rms(self.parent.evt)
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == 9: # common mode
                calib = self.getCommonMode(evtNumber)
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == 10: # gain
                calib = self.parent.det.gain(self.parent.evt)
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == 17: # gain_mask
                calib = self.parent.det.gain_mask(self.parent.evt)
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == 15: # coords_x
                calib = self.parent.det.coords_x(self.parent.evt)
                self.parent.firstUpdate = True
            elif self.parent.exp.image_property == 16: # coords_y
                calib = self.parent.det.coords_y(self.parent.evt)
                self.parent.firstUpdate = True

            shape = self.parent.det.shape(self.parent.evt)
            if len(shape) == 3:
                if self.parent.exp.image_property == 11: # quad ind
                    calib = np.zeros(shape)
                    for i in range(shape[0]):
                        # TODO: handle detectors properly
                        if shape[0] == 32: # cspad
                            calib[i,:,:] = int(i)%8
                        elif shape[0] == 2: # cspad2x2
                            calib[i,:,:] = int(i)%2
                        elif shape[0] == 4: # pnccd
                            calib[i,:,:] = int(i)%4
                    self.parent.firstUpdate = True
                elif self.parent.exp.image_property == 12: # seg ind
                    calib = np.zeros(shape)
                    if shape[0] == 32: # cspad
                        for i in range(32):
                            calib[i,:,:] = int(i)/8
                    elif shape[0] == 2: # cspad2x2
                        for i in range(2):
                            calib[i,:,:] = int(i)
                    elif shape[0] == 4: # pnccd
                        for i in range(4):
                            calib[i,:,:] = int(i)
                    self.parent.firstUpdate = True
                elif self.parent.exp.image_property == 13: # row ind
                    calib = np.zeros(shape)
                    if shape[0] == 32: # cspad
                        for i in range(185):
                            calib[:,i,:] = i
                    elif shape[0] == 2: # cspad2x2
                        for i in range(185):
                            calib[:,i,:] = i
                    elif shape[0] == 4: # pnccd
                        for i in range(512):
                            calib[:,i,:] = i
                    self.parent.firstUpdate = True
                elif self.parent.exp.image_property == 14: # col ind
                    calib = np.zeros(shape)
                    if shape[0] == 32: # cspad
                        for i in range(388):
                            calib[:,:,i] = i
                    elif shape[0] == 2: # cspad2x2
                        for i in range(388):
                            calib[:,:,i] = i
                    elif shape[0] == 4: # pnccd
                        for i in range(512):
                            calib[:,:,i] = i
                    self.parent.firstUpdate = True

        # Update photon energy
        self.ebeam = self.parent.evt.get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
        if self.ebeam:
            self.parent.photonEnergy = self.ebeam.ebeamPhotonEnergy()
        else:
            self.parent.photonEnergy = 0
        self.parent.geom.p1.param(self.parent.geom.geom_grp,
                             self.parent.geom.geom_photonEnergy_str).setValue(self.parent.photonEnergy)
        # Update clen
        if 'cspad' in self.parent.detInfo.lower() and 'cxi' in self.parent.experimentName:
            self.parent.geom.p1.param(self.parent.geom.geom_grp,
                                 self.parent.geom.geom_clen_str).setValue(self.parent.clen)

        if calib is not None:
            # assemble image
            data = self.getAssembledImage(calib)
            self.parent.cx, self.parent.cy = self.parent.det.point_indexes(self.parent.evt, pxy_um=(0,0))
            if self.parent.cx is None:
                self.parent.cx, self.parent.cy = self.getCentre(data.shape)
            if self.parent.args.v >= 1: print "cx, cy: ", self.parent.cx, self.parent.cy
            return calib, data
        else:
            calib = np.zeros_like(self.parent.exp.detGuaranteed, dtype='float32')
            data = self.getAssembledImage(calib)
            self.parent.cx, self.parent.cy = self.parent.det.point_indexes(self.parent.evt, pxy_um=(0, 0))
            if self.parent.cx is None:
                self.parent.cx, self.parent.cy = self.getCentre(data.shape)
            return calib, data

    def getCentre(self,shape):
        cx = shape[1]/2
        cy = shape[0]/2
        return cx,cy
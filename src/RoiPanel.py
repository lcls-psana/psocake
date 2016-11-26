from pyqtgraph.dockarea import *
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.pyplot as plt

class RoiHistogram(object):
    def __init__(self, parent = None):
        self.parent = parent

        #############################
        ## Dock 4: ROI histogram
        #############################
        self.d4 = Dock("ROI Histogram", size=(1, 1))
        self.w4 = pg.PlotWidget(title="ROI histogram")
        hist, bin = np.histogram(np.random.random(1000), bins=1000)
        self.w4.plot(bin, hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150), clear=True)
        self.d4.addWidget(self.w4)
        self.roiCheckbox = QtGui.QCheckBox('Update ROI')
        self.roiCheckbox.setCheckState(True)
        self.roiCheckbox.setTristate(False)
        self.roiCheckbox.stateChanged.connect(self.updateRoiStatus)
        # Layout
        self.w4a = pg.LayoutWidget()
        self.w4a.addWidget(self.roiCheckbox, row=0, col=0)
        self.d4.addWidget(self.w4a)

        #############################
        # Local variables
        #############################
        self.updateRoiStatus = True

        self.roiCurrent = None
        # Custom ROI for selecting an image region
        self.roi = pg.ROI(pos=[0, -250], size=[200, 200], snapSize=1.0, scaleSnap=True, translateSnap=True,
                          pen={'color': 'g', 'width': 4, 'style': QtCore.Qt.DashLine})
        self.roi.addScaleHandle([1, 0.5], [0.5, 0.5])
        self.roi.addScaleHandle([0.5, 0], [0.5, 0.5])
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0], [1, 1]) # bottom,left handles scaling both vertically and horizontally
        self.roi.addScaleHandle([1, 1], [0, 0])  # top,right handles scaling both vertically and horizontally
        self.roi.addScaleHandle([1, 0], [0, 1])  # bottom,right handles scaling both vertically and horizontally
        self.roi.addScaleHandle([0, 1], [1, 0])
        self.roi.name = 'rect'
        self.parent.img.w1.getView().addItem(self.roi)
        self.roiPoly = pg.PolyLineROI([[300, -250], [300,-50], [500,-50], [500,-150], [375,-150], [375,-250]],
                                      closed=True, snapSize=1.0, scaleSnap=True, translateSnap=True,
                                      pen={'color': 'g', 'width': 4, 'style': QtCore.Qt.DashLine})
        self.roiPoly.name = 'poly'
        self.parent.img.w1.getView().addItem(self.roiPoly)
        self.roiCircle = pg.CircleROI([600, -250], size=[200, 200], snapSize=0.1, scaleSnap=False, translateSnap=False,
                                        pen={'color': 'g', 'width': 4, 'style': QtCore.Qt.DashLine})
        self.roiCircle.addScaleHandle([0.1415, 0.707*1.2], [0.5, 0.5])
        self.roiCircle.addScaleHandle([0.707 * 1.2, 0.1415], [0.5, 0.5])
        self.roiCircle.addScaleHandle([0.1415, 0.1415], [0.5, 0.5])
        self.roiCircle.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.roiCircle.addScaleHandle([0.5, 0.0], [0.5, 0.5])
        self.roiCircle.addScaleHandle([0.5, 1.0], [0.5, 0.5])
        self.roiCircle.addScaleHandle([1.0, 0.5], [0.5, 0.5])
        self.roiCircle.name = 'circ'
        self.parent.img.w1.getView().addItem(self.roiCircle)

        self.rois = []
        self.rois.append(self.roi)
        self.rois.append(self.roiPoly)
        self.rois.append(self.roiCircle)
        for roi in self.rois:
            roi.sigRegionChangeFinished.connect(self.updateRoi)

    # Callbacks for handling user interaction
    def updateRoiHistogram(self):
        if self.parent.data is not None:
            selected, coord = self.roi.getArrayRegion(self.parent.data, self.parent.img.w1.getImageItem(), returnMappedCoords=True)
            hist,bin = np.histogram(selected.flatten(), bins=1000)
            self.w4.plot(bin, hist, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True)

    def updateRoi(self, roi):
        self.roiCurrent = roi.name
        if self.parent.data is not None:
            if self.updateRoiStatus == True:
                calib = np.ones_like(self.parent.calib)
                img = self.parent.det.image(self.parent.evt, calib)
                pixelsExist = roi.getArrayRegion(img, self.parent.img.w1.getImageItem())
                if roi.name == 'poly':
                    self.ret = roi.getArrayRegion(self.parent.data, self.parent.img.w1.getImageItem(), returnMappedCoords=True)
                    self.ret = self.ret[np.where(pixelsExist == 1)]
                elif roi.name == 'circ':
                    try:
                        self.ret = roi.getArrayRegion(self.parent.data, self.parent.img.w1.getImageItem())
                        self.ret = self.ret[np.where(pixelsExist == 1)]
                    except:
                        pass
                else:
                    self.ret = roi.getArrayRegion(self.parent.data, self.parent.img.w1.getImageItem(), returnMappedCoords=True)

                if roi.name == 'rect':  # isinstance(self.ret,tuple): # rectangle
                    selected, coord = self.ret
                    self.x0 = int(coord[0][0][0])
                    self.x1 = int(coord[0][-1][0]) + 1
                    self.y0 = int(coord[1][0][0])
                    self.y1 = int(coord[1][0][-1]) + 1

                    # Limit coordinates to inside the assembled image
                    if self.x0 < 0: self.x0 = 0
                    if self.y0 < 0: self.y0 = 0
                    if self.x1 > self.parent.data.shape[0]: self.x1 = self.parent.data.shape[0]
                    if self.y1 > self.parent.data.shape[1]: self.y1 = self.parent.data.shape[1]
                    print "######################################################"
                    print "Assembled ROI: [" + str(self.x0) + ":" + str(self.x1) + "," + str(self.y0) + ":" + str(
                        self.y1) + "]"  # Note: self.parent.data[x0:x1,y0:y1]
                    selected = selected[np.where(pixelsExist == 1)]

                    mask_roi = np.zeros_like(self.parent.data)
                    mask_roi[self.x0:self.x1, self.y0:self.y1] = 1
                    self.nda = self.parent.det.ndarray_from_image(self.parent.evt, mask_roi, pix_scale_size_um=None,
                                                           xy0_off_pix=None)
                    for itile, tile in enumerate(self.nda):
                        if tile.sum() > 0:
                            ax0 = np.arange(0, tile.sum(axis=0).shape[0])[tile.sum(axis=0) > 0]
                            ax1 = np.arange(0, tile.sum(axis=1).shape[0])[tile.sum(axis=1) > 0]
                            print 'Unassembled ROI: [[%i,%i], [%i,%i], [%i,%i]]' % (
                                itile, itile + 1, ax1.min(), ax1.max(), ax0.min(), ax0.max())
                            if self.parent.args.v >= 1:
                                fig = plt.figure(figsize=(6, 6))
                                plt.imshow(self.parent.calib[itile, ax1.min():ax1.max(), ax0.min():ax0.max()],
                                           interpolation='none')
                                plt.show()
                    print "######################################################"
                elif roi.name == 'circ':
                    selected = self.ret
                    self.centreX = roi.x() + roi.size().x() / 2
                    self.centreY = roi.y() + roi.size().y() / 2
                    print "###########################################"
                    print "Centre: [" + str(self.centreX) + "," + str(self.centreY) + "]"
                    print "###########################################"

                else:
                    selected = self.ret
                hist, bin = np.histogram(selected.flatten(), bins=1000)
                self.w4.plot(bin, hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150), clear=True)
            else: # update ROI off
                if roi.name == 'rect':  # isinstance(self.ret,tuple): # rectangle
                    self.ret = roi.getArrayRegion(self.parent.data, self.parent.img.w1.getImageItem(), returnMappedCoords=True)
                    selected, coord = self.ret
                    self.x0 = int(coord[0][0][0])
                    self.x1 = int(coord[0][-1][0]) + 1
                    self.y0 = int(coord[1][0][0])
                    self.y1 = int(coord[1][0][-1]) + 1

                    # Limit coordinates to inside the assembled image
                    if self.x0 < 0: self.x0 = 0
                    if self.y0 < 0: self.y0 = 0
                    if self.x1 > self.parent.data.shape[0]: self.x1 = self.parent.data.shape[0]
                    if self.y1 > self.parent.data.shape[1]: self.y1 = self.parent.data.shape[1]
                    print "######################################################"
                    print "Assembled ROI: [" + str(self.x0) + ":" + str(self.x1) + "," + str(self.y0) + ":" + str(
                        self.y1) + "]"  # Note: self.parent.data[x0:x1,y0:y1]
                    mask_roi = np.zeros_like(self.parent.data)
                    mask_roi[self.x0:self.x1, self.y0:self.y1] = 1
                    self.nda = self.parent.det.ndarray_from_image(self.parent.evt, mask_roi, pix_scale_size_um=None,
                                                           xy0_off_pix=None)
                    for itile, tile in enumerate(self.nda):
                        if tile.sum() > 0:
                            ax0 = np.arange(0, tile.sum(axis=0).shape[0])[tile.sum(axis=0) > 0]
                            ax1 = np.arange(0, tile.sum(axis=1).shape[0])[tile.sum(axis=1) > 0]
                            print 'Unassembled ROI: [[%i,%i], [%i,%i], [%i,%i]]' % (
                                itile, itile + 1, ax1.min(), ax1.max(), ax0.min(), ax0.max())
                    print "######################################################"
                elif roi.name == 'circ':
                    self.centreX = roi.x() + roi.size().x() / 2
                    self.centreY = roi.y() + roi.size().y() / 2
                    print "###########################################"
                    print "Centre: [" + str(self.centreX) + "," + str(self.centreY) + "]"
                    print "###########################################"

    def updateRoiStatus(self):
        if self.roiCheckbox.checkState() == 0:
            self.updateRoiStatus = False
        else:
            self.updateRoiStatus = True



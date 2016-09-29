from scipy import signal as sg
import numpy as np
from skimage.measure import label
import time

def getStreakMask(det,evt):
    calib = det.calib(evt)
    img = det.image(evt,calib)

    # Set edge pixels to val
    # Other pixels are zeros
    edgePixels = np.zeros_like(calib)
    for i in range(edgePixels.shape[0]):
        edgePixels[i,0,:] = 1
        edgePixels[i,-1,:] = 1
        edgePixels[i,:,0] = 1
        edgePixels[i,:,-1] = 1
    imgEdges = det.image(evt,edgePixels)

    # Crop centre of image
    (ix,iy) = det.point_indexes(evt)
    halfWidth = 150
    imgCrop = img[ix-halfWidth:ix+halfWidth,iy-halfWidth:iy+halfWidth]
    imgEdges = imgEdges[ix-halfWidth:ix+halfWidth,iy-halfWidth:iy+halfWidth]
    myInd = np.where(imgEdges==1)

    # Blur image
    imgBlur=sg.convolve(imgCrop,np.ones((2,2)),mode='same')
    mean = imgBlur[imgBlur>0].mean()
    std = imgBlur[imgBlur>0].std()

    # Mask out pixels above 2 sigma
    #val = filters.threshold_otsu(imgBlur)
    mask = imgBlur > mean+2*std
    mask = mask.astype(int) + imgEdges

    # Connected components
    myLabel = label(mask, connectivity=2)

    # All pixels connected to edge pixels is masked out
    myMask = np.ones_like(mask)
    myParts = np.unique(myLabel[myInd])
    for i in myParts:
        myMask[np.where(myLabel == i)] = 0

    fullMask = np.ones_like(img)
    fullMask[ix-halfWidth:ix+halfWidth,iy-halfWidth:iy+halfWidth] = myMask

    return fullMask

class StreakMask:
    def __init__(self, det, evt, width=300, sigma=1):
        self.det = det
        self.evt = evt
        self.width = width
        self.sigma = sigma
        calib = det.calib(evt)
        self.calibShape = calib.shape
        self.calibSize = calib.size
        # Edge pixels
        edgePixels = np.zeros_like(calib)
        #print "calib: ", calib.shape, edgePixels.shape
        for i in range(edgePixels.shape[0]):
            edgePixels[i,0,:] = 1
            edgePixels[i,-1,:] = 1
            edgePixels[i,:,0] = 1
            edgePixels[i,:,-1] = 1
        imgEdges = det.image(evt,edgePixels)
        # Centre of image
        (self.ix,self.iy) = det.point_indexes(evt)
        if self.ix is not None:
            self.halfWidth = int(width/2) # pixels
            self.imgEdges = imgEdges[self.ix-self.halfWidth:self.ix+self.halfWidth,self.iy-self.halfWidth:self.iy+self.halfWidth]
            self.myInd = np.where(self.imgEdges==1)
            # Pixel indices
            a=np.arange(calib.size)+1
            a=a.reshape(calib.shape)
            self.assem=det.image(evt,a)
        else:
            self.assem = None

    def getStreakMaskCalib(self,evt):
        if self.assem is not None:
            tic = time.time()

            tic1 = time.time()
            img = self.det.image(evt)

            tic2 = time.time()

            tic3 = time.time()

            # Crop centre of image
            imgCrop = img[self.ix-self.halfWidth:self.ix+self.halfWidth,self.iy-self.halfWidth:self.iy+self.halfWidth]
            tic4 = time.time()

            # Blur image
            imgBlur=sg.convolve(imgCrop,np.ones((2,2)),mode='same')
            mean = imgBlur[imgBlur>0].mean()
            std = imgBlur[imgBlur>0].std()
            tic5 = time.time()

            # Mask out pixels above 1 sigma
            mask = imgBlur > mean+self.sigma*std
            mask = mask.astype(int)
            signalOnEdge = mask * self.imgEdges
            mySigInd = np.where(signalOnEdge==1)
            mask[self.myInd[0].ravel(),self.myInd[1].ravel()] = 1
            tic6 = time.time()

            # Connected components
            myLabel = label(mask, neighbors=4, connectivity=1, background=0)
            # All pixels connected to edge pixels is masked out
            myMask = np.ones_like(mask)
            myParts = np.unique(myLabel[self.myInd])
            for i in myParts:
                myMask[np.where(myLabel == i)] = 0
            tic7 = time.time()

            # Delete edges
            myMask[self.myInd]=1
            myMask[mySigInd]=0

            # Convert assembled to unassembled
            wholeMask = np.ones_like(self.assem)
            wholeMask[self.ix-self.halfWidth:self.ix+self.halfWidth,self.iy-self.halfWidth:self.iy+self.halfWidth] = myMask
            pixInd = self.assem[np.where(wholeMask==0)]
            pixInd = pixInd[np.nonzero(pixInd)]-1
            calibMask=np.ones((self.calibSize,))
            calibMask[pixInd.astype(int)] = 0
            calibMask=calibMask.reshape(self.calibShape)
            tic8 = time.time()

            #print "calib, image, edge, crop, blur, mask, connect, convert: ", tic1-tic, tic2-tic1, tic3-tic2, tic4-tic3, tic5-tic4, tic6-tic5, tic7-tic6, tic8-tic7

            return calibMask
        else:
            return None

def getStreakMaskCalib(det,evt,width=300,sigma=1):
    tic = time.time()
    calib = det.calib(evt)

    tic1 = time.time()
    img = det.image(evt,calib)

    tic2 = time.time()
    # Edge pixels
    edgePixels = np.zeros_like(calib)
    for i in range(edgePixels.shape[0]):
        edgePixels[i,0,:] = 1
        edgePixels[i,-1,:] = 1
        edgePixels[i,:,0] = 1
        edgePixels[i,:,-1] = 1
    imgEdges = det.image(evt,edgePixels)
    tic3 = time.time()

    # Crop centre of image
    (ix,iy) = det.point_indexes(evt)
    halfWidth = int(width/2) # pixels
    imgCrop = img[ix-halfWidth:ix+halfWidth,iy-halfWidth:iy+halfWidth]
    imgEdges = imgEdges[ix-halfWidth:ix+halfWidth,iy-halfWidth:iy+halfWidth]
    myInd = np.where(imgEdges==1)
    tic4 = time.time()

    # Blur image
    imgBlur=sg.convolve(imgCrop,np.ones((2,2)),mode='same')
    mean = imgBlur[imgBlur>0].mean()
    std = imgBlur[imgBlur>0].std()
    tic5 = time.time()

    # Mask out pixels above 1 sigma
    mask = imgBlur > mean+sigma*std
    mask = mask.astype(int)
    signalOnEdge = mask * imgEdges
    mySigInd = np.where(signalOnEdge==1)
    mask[myInd[0].ravel(),myInd[1].ravel()] = 1
    tic6 = time.time()

    # Connected components
    myLabel = label(mask, neighbors=4, connectivity=1, background=0)
    # All pixels connected to edge pixels is masked out
    myMask = np.ones_like(mask)
    myParts = np.unique(myLabel[myInd])
    for i in myParts:
        myMask[np.where(myLabel == i)] = 0
    tic7 = time.time()

    # Delete edges
    myMask[myInd]=1
    myMask[mySigInd]=0

    # Convert assembled to unassembled
    a=np.arange(calib.size)+1
    a=a.reshape(calib.shape)
    assem=det.image(evt,a)
    wholeMask = np.ones_like(assem)
    wholeMask[ix-halfWidth:ix+halfWidth,iy-halfWidth:iy+halfWidth] = myMask
    pixInd = assem[np.where(wholeMask==0)]
    pixInd = pixInd[np.nonzero(pixInd)]-1
    calibMask=np.ones((calib.size,))
    calibMask[pixInd.astype(int)] = 0
    calibMask=calibMask.reshape(calib.shape)
    tic8 = time.time()

    print "calib, image, edge, crop, blur, mask, connect, convert: ", tic1-tic, tic2-tic1, tic3-tic2, tic4-tic3, tic5-tic4, tic6-tic5, tic7-tic6, tic8-tic7

    return calibMask
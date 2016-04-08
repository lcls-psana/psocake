import skimage
from skimage import filters
from scipy import signal as sg
import numpy as np
import psana
from skimage.measure import label

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

def getStreakMaskCalib(det,evt,width=300,sigma=1):
    calib = det.calib(evt)
    img = det.image(evt,calib)
    # Edge pixels
    edgePixels = np.zeros_like(calib)
    for i in range(edgePixels.shape[0]):
        edgePixels[i,0,:] = 1
        edgePixels[i,-1,:] = 1
        edgePixels[i,:,0] = 1
        edgePixels[i,:,-1] = 1
    imgEdges = det.image(evt,edgePixels)

    # Crop centre of image
    (ix,iy) = det.point_indexes(evt)
    halfWidth = int(width/2) # pixels
    imgCrop = img[ix-halfWidth:ix+halfWidth,iy-halfWidth:iy+halfWidth]
    imgEdges = imgEdges[ix-halfWidth:ix+halfWidth,iy-halfWidth:iy+halfWidth]
    myInd = np.where(imgEdges==1)

    # Blur image
    imgBlur=sg.convolve(imgCrop,np.ones((2,2)),mode='same')
    mean = imgBlur[imgBlur>0].mean()
    std = imgBlur[imgBlur>0].std()

    # Mask out pixels above 1 sigma
    mask = imgBlur > mean+sigma*std
    mask = mask.astype(int)
    signalOnEdge = mask * imgEdges
    mySigInd = np.where(signalOnEdge==1)
    mask[myInd[0].ravel(),myInd[1].ravel()] = 1

    # Connected components
    myLabel = label(mask, neighbors=4, connectivity=1, background=0)
    # All pixels connected to edge pixels is masked out
    myMask = np.ones_like(mask)
    myParts = np.unique(myLabel[myInd])
    for i in myParts:
        myMask[np.where(myLabel == i)] = 0

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

    return calibMask
from scipy import signal as sg
import numpy as np
from skimage.morphology import h_maxima
from skimage.measure import label, regionprops
import time
import PSCalib.GlobalUtils as gu

# Donut mask
# Returns a donut mask with inner radius r and outer radius R in NxM image
# Good = 1
# Background = 0
def donutMask(N,M,R,r,centreRow=0,centreCol=0):
    """
    Calculate a donut mask.

    N,M - The height and width of the image
    R   - Maximum radius from center
    r   - Minimum radius from center
    centreRow  - center in y
    centreCol  - center in x
    """
    if centreRow == 0:
        centerY = N/2.-0.5
    if centreCol == 0:
        centerX = M/2.-0.5
    mask = np.zeros((N,M))
    radialDistRow = np.zeros((N,M))
    radialDistCol = np.zeros((N,M))
    myN = np.zeros((N,M))
    myM = np.zeros((N,M))
    for i in range(N):
        xDist = i-centerY
        xDistSq = xDist**2
        for j in range(M):
            yDist = j-centerX
            yDistSq = yDist**2
            rSq = xDistSq+yDistSq
            if rSq < R**2 and rSq >= r**2:
                mask[i,j] = 1
                radialDistRow[i,j] = xDist
                radialDistCol[i,j] = yDist
                myN[i,j] = i
                myM[i,j] = j
    return mask, radialDistRow, radialDistCol, myN, myM

def findPeaks_hdome(calib, npix_min=0, npix_max=0, atot_thr=0,
              son_min=0, hvalue=0, r0=0, dr=0, mask=None):
    hmax = h_maxima(calib, hvalue)
    if mask is not None: hmax = np.multiply(hmax, mask)

    ll = label(hmax)
    regions = regionprops(ll)

    numPeaks = len(regions)
    x = np.zeros((numPeaks,))
    y = np.zeros((numPeaks,))
    for i, p in enumerate(regions):
        x[i], y[i] = p.centroid

    innerRing = r0
    outerRing = dr
    width = int(outerRing * 2 + 1)

    outer, rr, rc, n, m = donutMask(width, width, outerRing, innerRing, centreRow=0, centreCol=0)
    inner, rr, rc, n, m = donutMask(width, width, innerRing, 0, centreRow=0, centreCol=0)
    if width % 2 == 0:
        lh = rh = int(width / 2)
    else:
        lh = int(width / 2)
        rh = int(width / 2) + 1

    snr = np.zeros_like(x)
    tot = np.zeros_like(x)
    numPix = np.zeros_like(x)
    numInner = len(np.where(inner == 1)[0])
    for i in range(numPeaks):
        d = calib[int(x[i]) - lh:int(x[i]) + rh, int(y[i]) - lh:int(y[i]) + rh]
        try:
            meanSig = np.mean(d[np.where(inner == 1)])
            stdNoise = np.std(d[np.where(outer == 1)])
            meanBackground = np.mean(d[np.where(outer == 1)])
            tot[i] = np.sum(d[np.where(inner == 1)]) - numInner * meanBackground
            numPix[i] = len(np.where(d[np.where(inner == 1)] >= hvalue)[0])
            if stdNoise == 0:
                snr[i] = -1
            else:
                snr[i] = meanSig / stdNoise
        except:
            snr[i] = 0

    ind= (snr >= son_min) & (tot >= atot_thr) & (numPix >= npix_min) & (numPix < npix_max)
    x = x[ind]
    y = y[ind]
    npix = numPix[ind]
    atot = tot[ind]
    son = snr[ind]

    peaks = np.zeros((len(x), 5))
    if len(x) > 0:
        peaks[:,0] = x.T
        peaks[:,1] = y.T
        peaks[:,2] = npix.T
        peaks[:,3] = atot.T
        peaks[:,4] = son.T

    return peaks

class DropletA:
    def __init__(self, innerRing, outerRing):
        self.innerRing = innerRing
        self.outerRing = outerRing
        width = int(self.outerRing * 2 + 1)

        outer, rr, rc, n, m = donutMask(width, width, self.outerRing, self.innerRing, centreRow=0, centreCol=0)
        inner, rr, rc, n, m = donutMask(width, width, self.innerRing, 0, centreRow=0, centreCol=0)
        if width % 2 == 0:
            self.lh = self.rh = int(width / 2)
        else:
            self.lh = int(width / 2)
            self.rh = int(width / 2) + 1

        self.fgInd = np.where(inner == 1)
        self.bgInd = np.where(outer == 1)

    def findPeaks(self, calib, npix_min=0, npix_max=0, atot_thr=0, son_min=0, thr_low=0, thr_high=0, r0=0, dr=0, mask=None):
        seg = []
        posX = []
        posY = []
        npix = []
        atot = []
        son = []
        hmax = np.zeros_like(calib)
        hmax[np.where(calib>=thr_high)] = 1
        if mask is not None: hmax = np.multiply(hmax, mask)

        numDim = len(calib.shape)
        if numDim == 2:
            numAsic = 1
        elif numDim == 3:
            numAsic = calib.shape[0]

        for j in range(numAsic):
            if numDim == 2:
                ll = label(hmax)
            elif numDim == 3:
                ll = label(hmax[j])
            regions = regionprops(ll)
            numPeaks = len(regions)
            x = np.zeros((numPeaks,))
            y = np.zeros((numPeaks,))
            for i, p in enumerate(regions):
                x[i], y[i] = p.centroid

            snr = np.zeros_like(x)
            tot = np.zeros_like(x)
            numPix = np.zeros_like(x)
            numInner = len(self.fgInd[0])
            for i in range(numPeaks):
                try:
                    if numDim == 2:
                        d = calib[int(x[i]) - self.lh:int(x[i]) + self.rh, int(y[i]) - self.lh:int(y[i]) + self.rh]
                    elif numDim == 3:
                        d = calib[j, int(x[i]) - self.lh:int(x[i]) + self.rh, int(y[i]) - self.lh:int(y[i]) + self.rh]
                    meanSig = np.mean(d[self.fgInd])
                    stdNoise = np.std(d[self.bgInd])
                    meanBackground = np.mean(d[self.bgInd])
                    tot[i] = np.sum(d[self.fgInd]) - numInner * meanBackground
                    numPix[i] = len(np.where(d[self.fgInd] >= thr_low)[0])
                    if stdNoise == 0:
                        snr[i] = -1
                    else:
                        snr[i] = meanSig / stdNoise
                except:
                    snr[i] = 0

            ind= (snr >= son_min) & (tot >= atot_thr) & (numPix >= npix_min) & (numPix < npix_max)
            seg.append(np.ones_like(x[ind])*j)
            posX.append(x[ind])
            posY.append(y[ind])
            npix.append(numPix[ind])
            atot.append(tot[ind])
            son.append(snr[ind])

        seg = np.concatenate(seg)
        posX = np.concatenate(posX)
        posY = np.concatenate(posY)
        npix = np.concatenate(npix)
        atot = np.concatenate(atot)
        son = np.concatenate(son)

        peaks = np.zeros((len(posX), 6))
        if len(posX) > 0:
            peaks[:,0] = seg.T
            peaks[:,1] = posX.T
            peaks[:,2] = posY.T
            peaks[:,3] = npix.T
            peaks[:,4] = atot.T
            peaks[:,5] = son.T
        return peaks

class Droplet:
    def __init__(self, innerRing, outerRing):
        self.innerRing = innerRing
        self.outerRing = outerRing
        width = int(self.outerRing * 2 + 1)

        outer, rr, rc, n, m = donutMask(width, width, self.outerRing, self.innerRing, centreRow=0, centreCol=0)
        inner, rr, rc, n, m = donutMask(width, width, self.innerRing, 0, centreRow=0, centreCol=0)
        if width % 2 == 0:
            self.lh = self.rh = int(width / 2)
        else:
            self.lh = int(width / 2)
            self.rh = int(width / 2) + 1

        self.fgInd = np.where(inner == 1)
        self.bgInd = np.where(outer == 1)

    def findPeaks(self, calib, npix_min=0, npix_max=0, atot_thr=0, son_min=0, thr_low=0, thr_high=0, r0=0, dr=0, mask=None):
        hmax = np.zeros_like(calib)
        hmax[np.where(calib>=thr_high)] = 1
        if mask is not None: hmax = np.multiply(hmax, mask)

        ll = label(hmax)
        regions = regionprops(ll)

        numPeaks = len(regions)
        x = np.zeros((numPeaks,))
        y = np.zeros((numPeaks,))
        for i, p in enumerate(regions):
            x[i], y[i] = p.centroid

        snr = np.zeros_like(x)
        tot = np.zeros_like(x)
        numPix = np.zeros_like(x)
        numInner = len(self.fgInd[0])
        for i in range(numPeaks):
            try:
                d = calib[int(x[i]) - self.lh:int(x[i]) + self.rh, int(y[i]) - self.lh:int(y[i]) + self.rh]
                meanSig = np.mean(d[self.fgInd])
                stdNoise = np.std(d[self.bgInd])
                meanBackground = np.mean(d[self.bgInd])
                tot[i] = np.sum(d[self.fgInd]) - numInner * meanBackground
                numPix[i] = len(np.where(d[self.fgInd] >= thr_low)[0])
                if stdNoise == 0:
                    snr[i] = -1
                else:
                    snr[i] = meanSig / stdNoise
            except:
                snr[i] = 0

        ind= (snr >= son_min) & (tot >= atot_thr) & (numPix >= npix_min) & (numPix < npix_max)
        x = x[ind]
        y = y[ind]
        npix = numPix[ind]
        atot = tot[ind]
        son = snr[ind]

        peaks = np.zeros((len(x), 5))
        if len(x) > 0:
            peaks[:,0] = x.T
            peaks[:,1] = y.T
            peaks[:,2] = npix.T
            peaks[:,3] = atot.T
            peaks[:,4] = son.T
        return peaks

def findPeaks(calib, npix_min=0, npix_max=0, atot_thr=0,
              son_min=0, pmax=0, pmin=0, r0=0, dr=0, mask=None):
    hmax = np.zeros_like(calib)
    hmax[np.where(calib>=pmax)] = 1
    if mask is not None: hmax = np.multiply(hmax, mask)

    ll = label(hmax)
    regions = regionprops(ll)

    numPeaks = len(regions)
    x = np.zeros((numPeaks,))
    y = np.zeros((numPeaks,))
    for i, p in enumerate(regions):
        x[i], y[i] = p.centroid

    innerRing = r0
    outerRing = dr
    width = int(outerRing * 2 + 1)

    outer, rr, rc, n, m = donutMask(width, width, outerRing, innerRing, centreRow=0, centreCol=0)
    inner, rr, rc, n, m = donutMask(width, width, innerRing, 0, centreRow=0, centreCol=0)
    if width % 2 == 0:
        lh = rh = int(width / 2)
    else:
        lh = int(width / 2)
        rh = int(width / 2) + 1

    snr = np.zeros_like(x)
    tot = np.zeros_like(x)
    numPix = np.zeros_like(x)
    numInner = len(np.where(inner == 1)[0])
    for i in range(numPeaks):
        d = calib[int(x[i]) - lh:int(x[i]) + rh, int(y[i]) - lh:int(y[i]) + rh]
        try:
            meanSig = np.mean(d[np.where(inner == 1)])
            stdNoise = np.std(d[np.where(outer == 1)])
            meanBackground = np.mean(d[np.where(outer == 1)])
            tot[i] = np.sum(d[np.where(inner == 1)]) - numInner * meanBackground
            numPix[i] = len(np.where(d[np.where(inner == 1)] >= pmin)[0])
            if stdNoise == 0:
                snr[i] = -1
            else:
                snr[i] = meanSig / stdNoise
        except:
            snr[i] = 0

    ind= (snr >= son_min) & (tot >= atot_thr) & (numPix >= npix_min) & (numPix < npix_max)
    x = x[ind]
    y = y[ind]
    npix = numPix[ind]
    atot = tot[ind]
    son = snr[ind]

    peaks = np.zeros((len(x), 5))
    if len(x) > 0:
        peaks[:,0] = x.T
        peaks[:,1] = y.T
        peaks[:,2] = npix.T
        peaks[:,3] = atot.T
        peaks[:,4] = son.T

    return peaks

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
    #(ix,iy) = det.point_indexes(evt)
    (ix,iy) = det.point_indexes(evt, pxy_um=(0, 0),
                                pix_scale_size_um=None,
                                xy0_off_pix=None,
                                cframe=gu.PS, fract=False)
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
        if calib is not None:
            self.calibShape = calib.shape
            self.calibSize = calib.size
            # Edge pixels
            edgePixels = np.zeros_like(calib)
            for i in range(edgePixels.shape[0]):
                edgePixels[i,0,:] = 1
                edgePixels[i,-1,:] = 1
                edgePixels[i,:,0] = 1
                edgePixels[i,:,-1] = 1
            imgEdges = det.image(evt,edgePixels)
            # Centre of image
            #(self.ix,self.iy) = det.point_indexes(evt)
            try:
                (self.ix, self.iy) = det.point_indexes(evt, pxy_um=(0, 0),
                                         pix_scale_size_um=None,
                                         xy0_off_pix=None,
                                         cframe=gu.CFRAME_PSANA, fract=False)
            except AttributeError:
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
        else:
            self.assem = None

    def getStreakMaskCalib(self, evt, calib=None):
        if self.assem is not None:

            if calib is not None:
                img = self.det.image(evt, calib)
            else:
                img = self.det.image(evt)

            # Crop centre of image
            imgCrop = img[self.ix-self.halfWidth:self.ix+self.halfWidth,self.iy-self.halfWidth:self.iy+self.halfWidth]

            # Blur image
            imgBlur=sg.convolve(imgCrop,np.ones((2,2)),mode='same')
            mean = imgBlur[imgBlur>0].mean()
            std = imgBlur[imgBlur>0].std()

            # Mask out pixels above 1 sigma
            mask = imgBlur > mean+self.sigma*std
            mask = mask.astype(int)
            signalOnEdge = mask * self.imgEdges
            mySigInd = np.where(signalOnEdge==1)
            mask[self.myInd[0].ravel(),self.myInd[1].ravel()] = 1

            # Connected components
            myLabel = label(mask, neighbors=4, connectivity=1, background=0)
            # All pixels connected to edge pixels is masked out
            myMask = np.ones_like(mask)
            myParts = np.unique(myLabel[self.myInd])
            for i in myParts:
                myMask[np.where(myLabel == i)] = 0

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
    #(ix,iy) = det.point_indexes(evt)
    try:
        (ix, iy) = det.point_indexes(evt, pxy_um=(0, 0),
                                 pix_scale_size_um=None,
                                 xy0_off_pix=None,
                                 cframe=gu.CFRAME_PSANA, fract=False)
    except AttributeError:
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

    print("calib, image, edge, crop, blur, mask, connect, convert: ", tic1-tic, tic2-tic1, tic3-tic2, tic4-tic3, tic5-tic4, tic6-tic5, tic7-tic6, tic8-tic7)

    return calibMask

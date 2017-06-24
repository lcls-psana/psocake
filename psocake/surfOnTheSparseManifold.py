# usage: python surfOnTheSparseManifold.py /reg/d/psdm/amo/amo87215/scratch/yoon82/data/ amo87215 64 78 PR772 /photonConverter/pnccdFront 3
# amo86615
# python surfOnTheSparseManifold.py /reg/d/psdm/amo/amo86615/res/yoon82/data/ amo86615 191 191 PR772 /photonConverter/pnccdBack 3
# Program used to identify clusters
import numpy as np
import matplotlib.pyplot as plt
import h5py
import  sys
from numpy.matlib import repmat
from scipy import stats
import os
#from IPython import embed
import argparse
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigs, eigsh
import scipy
import time
import psana
import skimage.measure as sm
from sklearn.semi_supervised import label_propagation

parser = argparse.ArgumentParser()
parser.add_argument("-f","--filepath",help="full path to hdf5", type=str)
#parser.add_argument("-exp","--expName", help="psana experiment (e.g. amo86615)")
parser.add_argument("-e","--embed", type=str)
#parser.add_argument("-s","--startRun",help="number of events, all events=0", type=int)
#parser.add_argument("-e","--endRun",help="append tag to end of filename", type=int)
parser.add_argument("-t","--tag",help="tag", type=str)
parser.add_argument("-ver","--version",help="version",default=3, type=int)
parser.add_argument("-d","--dataset",help="hdf5 dataset which contains the image", type=str)
parser.add_argument("--test",help="test mode where the images are replaced with test images",action='store_true')
parser.add_argument("--sigma",help="kernel sigma",default=0, type=float)
parser.add_argument("--condition",help="comparator operation for choosing input data from an hdf5 dataset."
                                       "Must be comma separated for now."
                                       "Available comparators are: gt(>=), ge(>), eq(==), le(<=), lt(<)"
                                       "e.g. /particleSize/corrCoef,ge,0.85 ",default='', type=str)
args = parser.parse_args()

filepath = args.filepath
tag = args.tag
dataset = args.dataset
version = args.version

def digestRunList(runList):
    print "runList: ", runList
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

def onpick(event):
    plotDim = 2
    maxPlot = plotDim**2 # max subplots
    figi.clf()
    N = len(event.ind)
    if not N: return True
    if N > maxPlot:
        event.ind = event.ind[0:maxPlot]
        N = len(event.ind)
    for subplotnum, dataind in enumerate(event.ind):
        print "onpick: ", subplotnum
        ax2 = figi.add_subplot(plotDim,plotDim,subplotnum+1)
        if args.test:
            ax2.imshow(ir.getAssemImage(dataind))
        else:
            ax2.imshow(ir.getAssemImage(dataind),vmax=400,vmin=0)
        ax2.set_axis_off()
    figi.show()
    return True

def particleSize(img,numBins,angInd,my1DModel):
    # get radial average
    radAvg = getRadialAverage(img,numBins,angInd)
    # get best fit
    maxInd,maxCorr = getBestRadialFit(radAvg,my1DModel)
    diameter = radii[maxInd]*2
    if 0 and maxCorr > 0.95:
        plt.imshow(np.log10(img))
        plt.show()
    return diameter, maxCorr

def getRadialAverage(aduImg,numBins,angInd):
    # radially average data
    myData = aduImg[0:numBins,:].flatten()[angInd]
    radProf = np.zeros(numBins,)
    for i in range(numBins):
        radProf[i] = np.mean(myData[binInd==i])
    radProf[np.isnan(radProf)]=0
    return radProf

def getBestRadialFit(radProf,my1DModel):
    numTemplates = my1DModel.shape[0]
    myCorr = np.zeros((numTemplates,))
    for i in range(numTemplates):
        radProfModel = my1DModel[i,:]
        localMinInd = (np.diff(np.sign(np.diff(radProfModel))) > 0).nonzero()[0] + 1
        if radProfModel[0] == 0: # mask was applied, so use next minima
            startInd = localMinInd[1]
        else:
            startInd = localMinInd[0]
        corrcoeff = stats.pearsonr(radProf[startInd:-1],radProfModel[startInd:-1])
        myCorr[i] = corrcoeff[0]
    maxInd = np.argmax(myCorr)
    maxCorr = np.max(myCorr)
    return maxInd, maxCorr

def getDiffractionAngles(distance_x,distance_y,num_pixels_along_x,num_pixels_along_y):
    vertical = np.linspace(-distance_y,distance_y,num_pixels_along_y)
    horizontal = np.linspace(-distance_x,distance_x,num_pixels_along_x)
    xv, yv = np.meshgrid(horizontal,vertical)
    distance_from_center = np.sqrt(xv**2+yv**2)
    thetas = np.arctan2(distance_from_center,detector_distance)
    return thetas

def intensity_from_sphere(R,wavelength,scatteringAngle,model=0):
    """Compute the embedding vectors for data X

    Parameters
    ----------
    R : float
        radius of the sphere (m)

    wavelength : float
        wavelength of incident radiation (m)

    scatteringAngle : float
        numpy array of angles for every pixel in the detector, a.k.a twoTheta (radians)

    model : int
        diffraction model to use.
        0: Spherically symmetric body model: Feigin & Svergun 1987
        1: Bessel approximation of a large sphere: http://www.philiplaven.com/p8c.html

    Returns
    -------
    I : numpy array of diffraction intensities, shape_like(theta).
    """
    eps = np.finfo("float64").eps # Add a small number to denominator
    if model == 0:
        s = 4*np.pi*np.sin(scatteringAngle/2)/wavelength
        sR = s*R
        I = ( 3*(np.sin(sR)-sR*np.cos(sR))/(sR**3+eps) )**2
    elif model == 1:
        x = 2*np.pi*R/wavelength
        term1 = (1+np.cos(scatteringAngle))/2
        term2 = special.jv(1, x*np.sin(scatteringAngle))/np.sin(scatteringAngle)
        I = (x*term1*term2)**2
    else:
        print "Unknown model: ", model
    return I

def normalizedGraphLaplacian(D_data,D_indices,D_indptr,dim,sigmaK):
    print "sparse laplacian"
    yVal = np.exp(-np.square(D_data/sigmaK))
    D2 = scipy.sparse.csc_matrix((yVal,D_indices,D_indptr),shape=(dim,dim)).tocoo()

    d = D2.sum(axis=0)
    d = np.transpose(d)

    denom = np.squeeze(np.asarray(d[D2.row]))
    denom1 = np.squeeze(np.asarray(d[D2.col]))
    del D2

    D3 = yVal / (denom * denom1)

    thresh = 1e-7
    M = np.max(D3)
    D3[D3<thresh*M]=0

    D2 = scipy.sparse.csc_matrix((D3,D_indices,D_indptr),shape=(dim,dim)).tocoo()
    return D2

def normalizedGraphLaplacianDense(D_data,D_indices,D_indptr,dim,sigmaK,alpha=1):
    D = scipy.sparse.csc_matrix((D_data,D_indices,D_indptr),shape=(dim,dim)) #.tocoo()
    K = np.exp(-np.square(D.todense()/sigmaK)) # kernel
    p = np.matrix(np.sum(K,axis=0))
    P = np.multiply(p.transpose(),p)
    if alpha == 1:
        K1 = K/P
    else:
        K1 = K/np.power(P,alpha)
    v = np.sqrt(np.sum(K1,axis=0))
    A = K1/np.multiply(v.transpose(),v)
    return A

def diffusionMap(sigmaK, alpha=1, numEigs=6):

    myHdf5 = h5py.File(fname, 'r+')

    D_data = myHdf5[grpNameDM+dset_data].value
    dim = myHdf5[grpNameDM+dset_data].attrs['numHits']
    D_indices = myHdf5[grpNameDM+dset_indices]
    D_indptr = myHdf5[grpNameDM+dset_indptr]

    P = normalizedGraphLaplacian(D_data,D_indices,D_indptr,dim,sigmaK)

    tic = time.time()
    s, u = eigsh(P,k=numEigs+1,which='LM')
    u = np.real(u)
    u = np.fliplr(u)
    toc = time.time()
    print "%%%%: ",toc-tic, u, u.shape, s, s.shape

    U = u/repmat(np.matrix(u[:,0]).transpose(),1,numEigs+1)
    Y = U[:,1:numEigs+1]

    myHdf5.close()
    return Y,s

class imageRetriever:
    def __init__(self,filepath,expName,runs,tag,dataset,detInfo):
        self.filepath = filepath
        self.expName = expName
        self.startRun = runs[0]
        self.tag = tag
        self.dataset = dataset
        self.detInfo = detInfo
        self.fileList = []
        self.numHitsPerFile = []
        self.allHitInd = np.array([])
        self.eventInd = np.array([])
        self.runList = np.array([])
        self.downsampleRows = None
        self.downsampleCols = None
        self.run,self.times,self.det,self.evt = self.setup(self.expName,self.startRun,self.detInfo)
        self.lastRun = 0
        for r in runs:
            if tag is None:
                filename = filepath+ '/r' + str(r).zfill(4)+'/'+expName+'_'+str(r).zfill(4) + '.cxi'
            else:
                filename = filepath+ '/r' + str(r).zfill(4)+'/'+expName+'_'+str(r).zfill(4) + '_' + self.tag + '.cxi'
            if os.path.exists(filename):
                print "filename :", filename
                f1 = h5py.File(filename,'r')
                numHits = f1['/entry_1/result_1/nHits'].attrs['numEvents']
                hitInd = np.arange(0,numHits)
                hitEvent = f1['/LCLS/eventNumber'].value
                runInd = np.ones((numHits,),dtype=int)*r

                if numHits > 0:
                    self.allHitInd = np.append(self.allHitInd,hitInd)
                    self.eventInd = np.append(self.eventInd,hitEvent)
                    self.runList = np.append(self.runList,runInd)
                    self.fileList.append(f1)
                    self.numHitsPerFile.append(numHits)
                    #self.downsampleRows = f1[dataset+'/photonCount'].attrs['downsampleRows']
                    #self.downsampleCols = f1[dataset+'/photonCount'].attrs['downsampleCols']
                else:
                    f1.close()
        self.totalHits = np.sum(self.numHitsPerFile)
        self.numFiles = len(self.fileList)
        self.accumHits = np.zeros(self.numFiles,)
        for i,val in enumerate(self.numHitsPerFile):
            self.accumHits[i] = np.sum(self.numHitsPerFile[0:i+1])
        print "totalHits: ", self.totalHits, self.allHitInd

    def setup(self,experimentName,runNumber,detInfo):
        ds = psana.DataSource('exp='+str(experimentName)+':run='+str(runNumber)+':idx')
        run = ds.runs().next()
        times = run.times()
        env = ds.env()
        evt = run.event(times[0])
        det = psana.Detector(str(detInfo), env)
        return run,times,det,evt

    def getFileIndex(self,globalIndex):
        for i, val in reversed(list(enumerate(self.accumHits))):
            if globalIndex < val and globalIndex >= val-self.numHitsPerFile[i]:
                localInd = globalIndex-(val-self.numHitsPerFile[i])
                if i == 0:
                    skip = 0
                else:
                    skip = self.accumHits[i-1]
                return i, int(self.allHitInd[localInd+skip])

    def getAssemImage(self,globalIndex):
        assert(globalIndex < self.totalHits)

        if self.lastRun is not int(self.runList[globalIndex]):
            self.lastRun = int(self.runList[globalIndex])
            self.run,self.times,self.det,self.evt = self.setup(self.expName,self.lastRun,self.detInfo)

        print "#### run number, event: ", int(self.runList[globalIndex]), int(self.eventInd[globalIndex])
        evt = self.run.event(self.times[int(self.eventInd[globalIndex])])
        assemImage = self.det.image(evt)
        #assemImage = sm.block_reduce(assemImageOrig,block_size=(8,8),func=np.sum)

        if args.test:
            assemImage = f[grpNameDM+'/testData'][globalIndex,:,:]

        return assemImage

# Open manifold hdf5 for clustering
f = h5py.File(args.embed)
myRuns = f['diffusionMap/eigvec'].attrs['runs']
runs = digestRunList(myRuns)
expName = f['diffusionMap/eigvec'].attrs['exp']
detName = f['diffusionMap/eigvec'].attrs['detectorName']
grpNameDM = '/diffusionMap'
dset_indices = '/D_indices'
dset_indptr = '/D_indptr'
dset_data = '/D_data'
if args.sigma == 0:
    print "reading eigvec"
    eigvec = f[grpNameDM+'/eigvec'] # manifold position
    numFrames = eigvec.attrs['numImages']
    sigma = eigvec.attrs['sigma']
else:
    print "calculating eigvec"
    eigvec,eigval = diffusionMap(args.sigma, alpha=1, numEigs=6)
    numFrames = len(eigval)
    sigma = args.sigma
print "eigs: ", eigvec, eigvec.dtype, eigvec.shape

mask = f[grpNameDM+'/mask'].value
hasSigmas = True
try:
    sigmas = f[grpNameDM+'/L'].value
except:
    hasSigmas = False

ir = imageRetriever(filepath=filepath, expName=expName, runs=runs, tag=tag, dataset=dataset, detInfo=detName)

import matplotlib
colors = matplotlib.cm.rainbow(np.linspace(0, 1, numFrames))

# Transition probability from centre of cluster
if hasSigmas:
    figx = plt.figure()
    axx = figx.add_subplot(111)
    logEps = np.linspace(-10.0, 20.0, num=20)
    axx.plot(logEps, sigmas, 'x-')
    #axx.xlabel('log(Epsilon)')
    #axx.ylabel('log(sum(Wij))')
    axx.set_title('optimum sigma: '+str(np.log(sigma)))

fig = plt.figure()
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
figi = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(eigvec[:,0], eigvec[:,1], picker=2, alpha=0.1)
ax.set_xlabel('eig1')
ax.set_ylabel('eig2')
ax1 = fig1.add_subplot(111)

ax1.scatter(eigvec[:,0], eigvec[:,2], marker='o', picker=2, alpha=0.1)
ax1.set_xlabel('eig1')
ax1.set_ylabel('eig3')
ax2 = fig2.add_subplot(111)

ax2.scatter(eigvec[:,1], eigvec[:,2], marker='o', picker=2, alpha=0.1)
ax2.set_xlabel('eig2')
ax2.set_ylabel('eig3')
from mpl_toolkits.mplot3d import Axes3D
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(eigvec[:,0], eigvec[:,1], eigvec[:,2], picker=2, alpha=0.1)
ax3.set_xlabel('eig1')
ax3.set_ylabel('eig2')
ax3.set_zlabel('eig3')
fig.canvas.mpl_connect('pick_event', onpick)
fig1.canvas.mpl_connect('pick_event', onpick)
fig2.canvas.mpl_connect('pick_event', onpick)
fig3.canvas.mpl_connect('pick_event', onpick)
plt.show()
f.close()

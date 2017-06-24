# usage: python manifoldROI.py -f /reg/d/psdm/amo/amo86615/res/yoon82/data --expName amo86615 -s 182 -e 197 -t PR772 -d /photonConverter/pnccdBack -v 3 --eigs 1,2
# usage: python manifoldROI.py -f /reg/d/psdm/amo/amo86615/res/yoon82/data --expName amo86615 -s 182 -e 197 -t PR772_v1 -d /photonConverter/pnccdBack -v 3 --eigs 0,1
import numpy as np
import matplotlib.pyplot as plt
import h5py
import  sys
from numpy.matlib import repmat
from scipy import stats
import os
from pylab import *
from matplotlib.path import Path
import matplotlib.patches as patches
import argparse
from shutil import copyfile, move
import psana

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

def coords(s):
    try:
        eig0,eig1 = s.split(',')
        return int(eig0), int(eig1)
    except:
        raise argparse.ArgumentTypeError("Eigenvectors must be comma separated")

parser = argparse.ArgumentParser()
parser.add_argument("-f","--filepath",help="full path to hdf5", type=str)
parser.add_argument("-e","--embed", type=str)
#parser.add_argument("-exp","--expName", help="psana experiment (e.g. amo86615)")
#parser.add_argument("-s","--startRun",help="number of events, all events=0", type=int)
#parser.add_argument("-e","--endRun",help="append tag to end of filename", type=int)
parser.add_argument("-t","--tag",help="tag", type=str)
parser.add_argument("--newTag",help="new tag attached after tag (default=v1)", default='v1', type=str)
parser.add_argument("-ver","--version",help="version",default=1, type=int)
parser.add_argument("-d","--dataset",help="hdf5 dataset which contains the image", type=str)
parser.add_argument("--test",help="test mode where the images are replaced with test images",action='store_true')
parser.add_argument('--eigs', help="specify which eigenvectors to plot (default=0,1)", default='0,1', dest="eigs", type=coords, nargs=1)
parser.add_argument("--override",help="override the existing hdf5. WARNING: don't use this if you don't know what you are doing",action='store_true')
parser.add_argument("--keepROI",help="keep indices inside ROI",default=0, type=int)
args = parser.parse_args()

# FIXME: change to argparse
filepath = args.filepath
embed = args.embed
#expName = args.expName
#startRun = args.startRun
#endRun = args.endRun
tag = args.tag
dataset = args.dataset
version = args.version
eigs = args.eigs[0]

import matplotlib.pyplot as plt

class Canvas(object):
    def __init__(self,ax):
        self.ax = ax

        self.path, = ax.plot([],[],'ro-',lw=1)
        self.vert = []
        self.ax.set_title('LEFT: new point, MIDDLE or x: delete last point, RIGHT or c: close polygon')

        self.x = [] 
        self.y = []

        self.mouse_button = {1: self._add_point, 2: self._delete_point, 3: self._close_polygon}
        self.key_button = {'x': self._delete_point, 'c': self._close_polygon}

    def set_location(self,event):
        if event.inaxes:
            self.x = event.xdata
            self.y = event.ydata
               
    def _add_point(self):
        self.vert.append((self.x,self.y))

    def _delete_point(self):
        if len(self.vert)>0:
            self.vert.pop()

    def _close_polygon(self):
        self.vert.append(self.vert[0])
        print self.vert

    def update_path_mouse(self,event):
        print('update path')
        # If the mouse pointer is not on the canvas, ignore buttons
        if not event.inaxes: return

        # Do whichever action correspond to the mouse button clicked
        self.mouse_button[event.button]()
        
        x = [self.vert[k][0] for k in range(len(self.vert))]
        y = [self.vert[k][1] for k in range(len(self.vert))]
        self.path.set_data(x,y)
        plt.draw()

    def update_path_key(self,event):
        # If the mouse pointer is not on the canvas, ignore buttons
        if not event.inaxes: return

        # Do whichever action correspond to the mouse button clicked
        self.key_button[event.key]()
        
        x = [self.vert[k][0] for k in range(len(self.vert))]
        y = [self.vert[k][1] for k in range(len(self.vert))]
        self.path.set_data(x,y)
        plt.draw()

class imageRetriever:
    def __init__(self,filepath,expName,runs,tag,dataset,detInfo):
        self.filepath = filepath
        self.expName = expName
        self.startRun = runs[0]
        #self.endRun = endRun
        self.tag = tag
        self.dataset = dataset
        self.detInfo = detInfo
        self.fileList = []
        self.filenames = []
        self.numHitsPerFile = []
        self.allHitInd = np.array([])
        self.eventInd = np.array([])
        self.runList = np.array([])
        self.downsampleRows = None
        self.downsampleCols = None
        self.run,self.times,self.det,self.evt = self.setup(self.expName,self.startRun,self.detInfo)
        self.lastRun = 0
        for r in runs:
            filename = filepath+ '/r' + str(r).zfill(4)+'/'+expName+'_'+str(r).zfill(4) + '.cxi'
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
                    self.filenames.append(filename)
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
                #print "### global, fileInd, local", globalIndex, i, localInd, skip
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

print "I'm here"

if __name__ == '__main__':
    # Open manifold hdf5 for clustering
    #fname = filepath+'/'+expName+'_'+str(startRun)+'_'+str(endRun)+'_class_v'+str(version)+'.h5'
    #print "Opening filename: ", fname
    f1 = h5py.File(args.embed, 'a')
    grpNameDM = '/diffusionMap'
    eigvec = f1[grpNameDM+'/eigvec'] # manifold position
    numFrames = eigvec.attrs['numImages']
    mask = f1[grpNameDM+'/mask']
    expName = eigvec.attrs['exp']
    myRuns = eigvec.attrs['runs']
    detName = eigvec.attrs['detectorName']
    runs = digestRunList(myRuns)

    ir = imageRetriever(filepath=filepath,expName=expName,runs=runs,tag=tag,dataset=dataset,detInfo=detName)

    fig = plt.figure(1,(10,10))
    ax = fig.add_subplot(111)
    print eigs[0],eigs[1]
    plt.plot(eigvec[:,eigs[0]],eigvec[:,eigs[1]], 'b.')
    cnv = Canvas(ax)

    plt.connect('button_press_event',cnv.update_path_mouse)
    plt.connect('motion_notify_event',cnv.set_location)
    plt.connect('key_press_event', cnv.update_path_key)

    if args.keepROI:
        print "Keep points inside ROI"
    else:
        print "Throw away points inside ROI"
    plt.show()

    myVerts = cnv.vert
    eigvec.attrs['/outlierROI'] = myVerts
    print "myVerts: ", myVerts

    path1 = Path(myVerts)
    index = path1.contains_points(eigvec[:,np.array([eigs[0],eigs[1]])])

    if args.keepROI:
        myRejects = np.sort(np.asarray(np.where(index==0)[0])) # reject indices outside ROI
        myKeeps = np.asarray(np.where(index==1)[0]) # keep indices inside ROI
    else:
        myRejects = np.sort(np.asarray(np.where(index==1)[0])) # reject indices inside ROI
        myKeeps = np.asarray(np.where(index==0)[0]) # keep indices outside ROI

    numRejects = len(myRejects)
    print "myRejects: ",numRejects

    numView = 9
    if numRejects <= numView:
        numView = numRejects
    for i in range(numView):
        plt.subplot(3,3,i)
        img = ir.getAssemImage(myRejects[i])
        plt.imshow(np.log10(img))
        plt.title('rejected image'+str(i))
    plt.show()

    # Populate which are kept
    myFileInd = np.zeros_like(myKeeps) # list of file indices
    myLocalInd = np.zeros_like(myKeeps) # list of local image indices
    for i,val in enumerate(myKeeps):
        fileInd,localInd = ir.getFileIndex(val)
        myFileInd[i] = fileInd
        myLocalInd[i] = localInd

    def modify_attrs(name, obj):
        for name1, value in obj.attrs.iteritems():
            newF[name].attrs[name1] = value
            try:
                if value == numHits:
                    print "*** Overriding: ", name
                    newF[name].attrs[name1] = newNumHits # override attr value
            except:
                continue

    for i in np.unique(myFileInd):
        oldFilename = ir.filenames[i]
        print "old: ", oldFilename
        deleteLocalInd = myLocalInd[np.where(myFileInd==i)]
        newFilename = ir.filenames[i].split(".cxi")[0]+"_"+args.newTag+".cxi"
        numHits = ir.numHitsPerFile[i]
        newNumHits = len(deleteLocalInd)

        print "New filename, numHits: ", newFilename, newNumHits
        if numHits == newNumHits:
            if os.path.isfile(newFilename):
                os.remove(newFilename)
            os.symlink(oldFilename,newFilename)
            continue
        else:
            copyfile(oldFilename, newFilename)

        oldF = h5py.File(oldFilename,'r')
        newF = h5py.File(newFilename,'r+')

        members = []
        oldF.visit(members.append)
        for j in range(len(members)):
            try:
                if oldF[members[j]].shape[0] == numHits:
                    print "+++ Overriding: ", str(members[j])
                    val = oldF[members[j]].value
                    val = val[deleteLocalInd]
                    oldAttr = oldF[str(members[j])].attrs
                    oldChunks = oldF[str(members[j])].chunks
                    del newF[str(members[j])]
                    if oldChunks:
                        ds = newF.create_dataset( str(members[j]), (val.shape), chunks=oldChunks, dtype=val.dtype )
                    else:
                        ds = newF.create_dataset( str(members[j]), (val.shape), dtype=val.dtype )
                    print "copying: ", str(members[j])
                    ds[...] = val
                    #if oldAttr:
                    #    ds[...].attrs.update(oldAttr)
                    print "Done"
            except:
                continue

        oldF.visititems(modify_attrs)

        newF.close()
        oldF.close()
        if args.override: move(newFilename,oldFilename)
        print "Finished"

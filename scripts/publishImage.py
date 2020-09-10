import psana
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

experimentName = 'mfxp17318'
runNumber = '13'
detInfo = 'epix10k2M'
evtNum = 626

"""
######## PSOCAKE #########
np.save('index_%s_%s.npy'%(self.parent.runNumber,self.parent.eventNumber),self.parent.index.indexedPeaks); np.save('peaks_%s_%s.npy'%(self.parent.runNumber,self.parent.eventNumber),self.parent.pk.peaks); 
np.save('ix_%s_%s.npy'%(self.parent.runNumber,self.parent.eventNumber),self.parent.det.indexes_x(self.parent.evt)); 
np.save('iy_%s_%s.npy'%(self.parent.runNumber,self.parent.eventNumber),self.parent.det.indexes_y(self.parent.evt));
np.save('cx_%s_%s.npy'%(self.parent.runNumber,self.parent.eventNumber),self.parent.cx);
np.save('cy_%s_%s.npy'%(self.parent.runNumber,self.parent.eventNumber),self.parent.cy)
"""

peaks = np.load('peaks_%s_%s.npy'%(runNumber,evtNum))
indexedPeaks = np.load('index_%s_%s.npy'%(runNumber,evtNum))
ix = np.load('ix_%s_%s.npy'%(runNumber,evtNum))
iy = np.load('iy_%s_%s.npy'%(runNumber,evtNum))
cx = np.load('cx_%s_%s.npy'%(runNumber,evtNum))
cy = np.load('cy_%s_%s.npy'%(runNumber,evtNum))

vmax=1000 # intensity maximum
vmin=0  # intensity minimum
distEdge=820 # pixels to edge
res='2.1A'
resColor='#0497cb'
textSize=24
move=3*textSize # move text to the left

def assemblePeakPos(peaks):
    iX = np.array(ix, dtype=np.int64)
    iY = np.array(iy, dtype=np.int64)
    if len(iX.shape) == 2:
        iX = np.expand_dims(iX, axis=0)
        iY = np.expand_dims(iY, axis=0)
    cenX = iX[np.array(peaks[:, 0], dtype=np.int64), np.array(peaks[:, 1], dtype=np.int64), np.array(
            peaks[:, 2], dtype=np.int64)] + 0.5
    cenY = iY[np.array(peaks[:, 0], dtype=np.int64), np.array(peaks[:, 1], dtype=np.int64), np.array(
            peaks[:, 2], dtype=np.int64)] + 0.5
    return cenX, cenY

ds = psana.DataSource('exp='+experimentName+':run='+runNumber+':idx')
run = ds.runs().next()
det = psana.Detector(detInfo)

times = run.times()
env = ds.env()
evt = run.event(times[evtNum])

psanaMask = det.mask(evt, calib=1, status=1, edges=1, central=1, unbond=1, unbondnbrs=1)

hitParam_alg1_radius = 3
cenX, cenY = assemblePeakPos(peaks)
diameter = int(hitParam_alg1_radius) * 2 + 1

numIndexedPeaksFound = indexedPeaks.shape[0]
intRadius='4,5,6'
cenX1 = indexedPeaks[:,0]+0.5
cenY1 = indexedPeaks[:,1]+0.5
#cenX1 = np.concatenate((cenX1,cenX1,cenX1))
#cenY1 = np.concatenate((cenY1,cenY1,cenY1))
diameter0 = np.ones_like(cenX1)
diameter1 = np.ones_like(cenX1)
diameter2 = np.ones_like(cenX1)
diameter0[0:numIndexedPeaksFound] = float(intRadius.split(',')[0])*2
diameter1[0:numIndexedPeaksFound] = float(intRadius.split(',')[1])*2
diameter2[0:numIndexedPeaksFound] = float(intRadius.split(',')[2])*2

img = det.image(evt, psanaMask*det.calib(evt))

fix,ax = plt.subplots(1,figsize=(10,10))
plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
ax.imshow(img,cmap='binary',vmax=vmax,vmin=vmin)
plt.savefig('%s_%s_%s_img.png'%(experimentName,runNumber,evtNum),dpi=300)
plt.show()

#
#
#

fix,ax = plt.subplots(1,figsize=(10,10))
plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
ax.imshow(img,cmap='binary',vmax=vmax,vmin=vmin)
for i in range(peaks.shape[0]):
    rect = patches.Rectangle((cenY[i]-round(7/2.),cenX[i]-round(7/2.)),7,7,linewidth=0.5,edgecolor='#0497cb',facecolor='none')
    ax.add_patch(rect)

# Resolution ring
circ = patches.Circle((cy,cx),radius=distEdge,linewidth=1,edgecolor=resColor,facecolor='none')
ax.add_patch(circ)

plt.text(cy-textSize/2-move,cx+distEdge,res,size=textSize,color=resColor)

plt.savefig('%s_%s_%s_pks.png'%(experimentName,runNumber,evtNum),dpi=300)
#plt.show()

#
#
#

fix,ax = plt.subplots(1,figsize=(10,10))
plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
ax.imshow(img,cmap='binary',vmax=vmax,vmin=vmin)

for i in range(numIndexedPeaksFound):
    #circ = patches.Circle((cenY1[i],cenX1[i]),radius=diameter0[i]/2.,linewidth=1,edgecolor='#c84c6b',facecolor='none')
    #ax.add_patch(circ)
    circ = patches.Circle((cenY1[i],cenX1[i]),radius=diameter1[i]/2.,linewidth=0.5,edgecolor='#c84c6b',facecolor='none')
    ax.add_patch(circ)
    #circ = patches.Circle((cenY1[i],cenX1[i]),radius=diameter2[i]/2.,linewidth=1,edgecolor='#6897bb',facecolor='none')
    #ax.add_patch(circ)

# Resolution ring
circ = patches.Circle((cy,cx),radius=distEdge,linewidth=1,edgecolor=resColor,facecolor='none')
ax.add_patch(circ)

plt.text(cy-textSize/2-move,cx+distEdge,res,size=textSize,color=resColor)

plt.savefig('%s_%s_%s_idx.png'%(experimentName,runNumber,evtNum),dpi=300)
#plt.show()

















import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='', help="cxidb file path")
parser.add_argument('-e', type=str, help="experiment name")
parser.add_argument('-r', type=str, help="run number")
args = parser.parse_args()

minBraggPeaks = 15

fname = args.p + '/' + args.e + '_' + args.r.zfill(4) + '.cxi'

foundPeaks = False
foundIndex = False

f=h5py.File(fname,'r')
if '/entry_1/result_1/nPeaksAll' in f: 
    foundPeaks = True
    hitRate = f['/entry_1/result_1/nPeaksAll'].value
if '/entry_1/result_1/index' in f: 
    foundIndex = True
    indexRate = f['/entry_1/result_1/index'].value
f.close()

fig, ax = plt.subplots()

if foundPeaks:
    numEvents = len(hitRate)
    numSec = numEvents/120+1
    hits = np.zeros((numSec*120))
    hits[np.where(hitRate>=minBraggPeaks)]=1
    hitsPerSec=np.reshape(hits,(-1,120))
    hitsPerSec=np.sum(hitsPerSec,axis=1)
    ax.plot(hitsPerSec, label='Hit Rate')
    #plt.ylabel('Number of hits')
    #plt.xlabel('time (s)')
    #plt.title('Hit rate exp='+args.e+':run='+args.r)
    #plt.show()

# Plot indexing rates if exists
if foundIndex:
    numEvents = len(indexRate)
    numSec = numEvents/120+1
    indexed = np.zeros((numSec*120))
    indexed[np.where(indexRate>=1)]=1
    indexedPerSec=np.reshape(indexed,(-1,120))
    indexedPerSec=np.sum(indexedPerSec,axis=1)
    ax.plot(indexedPerSec, 'r', label='Indexing Rate')

# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=True)

plt.ylabel('Number of hits/indexed')
plt.xlabel('time (s)')
plt.title('Hit/Indexing rate exp='+args.e+':run='+args.r)
plt.show()


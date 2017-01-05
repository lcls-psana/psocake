import h5py
import sys
f=h5py.File(sys.argv[1])
st = f['/entry_1/result_1/saveTime']
ct = f['/entry_1/result_1/calibTime']
pt = f['/entry_1/result_1/peakTime']
rt = f['/entry_1/result_1/reshapeTime']

import matplotlib.pyplot as plt
import numpy as np
plt.figure()
plt.subplot(221)
plt.plot(st,'b.')
plt.ylabel('time (s)')
plt.title('save time: '+str(round(np.mean(st),3))+' '+str(round(np.std(st),2)))
plt.subplot(222)
plt.plot(ct,'rx')
plt.ylabel('time (s)')
plt.title('calib time: '+str(round(np.mean(ct),3))+' '+str(round(np.std(ct),2)))
plt.subplot(223)
plt.plot(pt,'go')
plt.ylabel('time (s)')
plt.title('peak finding time: '+str(round(np.mean(pt),3))+' '+str(round(np.std(pt),2)))
plt.subplot(224)
plt.plot(rt,'ms')
plt.ylabel('time (s)')
plt.title('reshape time: '+str(round(np.mean(rt),3))+' '+str(round(np.std(rt),2)))
plt.show()

f.close()


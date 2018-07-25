import psana
from psalgos.pypsalgos import PyAlgos
from psana import *
import numpy as np
from pprint import pprint


alg = PyAlgos(mask = None, pbits = 0)
alg.set_peak_selection_pars(npix_min=2, npix_max=30, amax_thr=300, atot_thr=600, son_min=10)
exp = "mfxlp4815"

pixelList = []

for runnum in range(3, 140):
    if (runnum == 4) or (runnum == 5) :
        continue
    ds = psana.DataSource('exp=%s:run=%d:idx'%(exp, runnum))
    print(runnum)
    d = psana.Detector("Rayonix")
    d.do_reshape_2d_to_3d(flag=True)
    run = ds.runs().next()
    times = run.times()
    env = ds.env()
    numEvents = len(times)
    mask = d.mask(runnum,calib=True,status=True,edges=True,central=True,unbond=True,unbondnbrs=True)
    for j in range(numEvents):     
        print("Event #:", j+1)   
        evt = run.event(times[j])
        try:
            nda = d.calib(evt) * mask
        except TypeError:
            nda = d.calib(evt)
        if (nda is not None):
            print(nda.shape[1])
            peaks = alg.peak_finder_v3r3(nda, rank=3, r0=3, dr=2, nsigm =5)
            numPeaksFound = len(peaks)
            if (numPeaksFound >= 15):
                for x in range(nda.shape[1]):
                    for y in range(nda.shape[2]):
                        if nda[0][x][y] > 5000:
                            print(exp, runnum, j+1, x, y)
                            pixelList.append([exp,runnum, j+1,x,y])

print("done")

pprint(pixelList)

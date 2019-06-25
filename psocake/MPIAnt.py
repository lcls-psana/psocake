import psana
import os
import numpy as np
import PeakFinderAnt as pf
import glob
from psalgos.pypsalgos import PyAlgos
from scipy.spatial import distance
import random
import re
import time
import subprocess

def checkXtcSize(exp, runnum):
    minBytes = 100
    realpath = os.path.realpath('/reg/d/psdm/cxi/' + exp + '/xtc')
    if '/reg/data/ana01' in realpath:  # FIXME: ana01 is down temporarily
        return False
    runList = glob.glob('/reg/d/psdm/cxi/' + exp + '/xtc/*-r%04d*' % (runnum))
    idxList = glob.glob('/reg/d/psdm/cxi/' + exp + '/xtc/index/*-r%04d*' % (runnum))
    smdList = glob.glob('/reg/d/psdm/cxi/' + exp + '/xtc/smalldata/*-r%04d*' % (runnum))
    if runList and (len(runList) == len(idxList)) and (len(runList) == len(smdList)):
        for f in runList + idxList:
            if os.stat(f).st_size <= minBytes:  # bytes
                return False
        return True
    else:
        return False

def safeDataSource(exp, runnum):
    if checkXtcSize(exp, runnum):
        ds = psana.DataSource('exp=%s:run=%s:smd' % (exp, runnum))
        evt = None
        for _evt in ds.events():
            evt = _evt
            break
        if evt is not None:
            ds = psana.DataSource('exp=%s:run=%s:idx' % (exp, runnum))
            return ds
    return None

class MPIAnt:

    def __init__(self):
        self.outdir = '/reg/d/psdm/cxi/cxitut13/res/autosfx/output'
        self.maxJobs = 15
        self.numCores = 48
        self.maxCores = self.numCores * self.maxJobs

    def updateClen(self, exp, detname, clenStr):
        if 'cspad' in detname.lower() and 'cxi' in exp:
            self.epics = self.ds.env().epicsStore()
            self.clen = self.epics.value(clenStr)
        elif 'rayonix' in detname.lower() and 'mfx' in exp:
            self.epics = self.ds.env().epicsStore()
            self.clen = self.epics.value(clenStr)
        elif 'rayonix' in detname.lower() and 'xpp' in exp:
            self.epics = self.ds.env().epicsStore()
            self.clen = self.epics.value(clenStr)
        return self.clen

    def launchJob(self, exp, runnum, detname):
        self.ds = safeDataSource(exp, runnum)
        run = self.ds.runs().next()
        times = run.times()
        env = self.ds.env()
        evt = run.event(times[0])
        det = psana.Detector(str(detname), env)

        instrument = det.instrument()
        clenEpics = detname+"_z"
        clen = self.updateClen(exp, detname, clenEpics)
        if clen is None:
            clen = 0
        try:
            detectorDistance = np.mean(det.coords_z(evt)) * 1e-6
        except:
            detectorDistance = 1
        coffset = detectorDistance - clen * 1e-3
        try:
            pixelSize = det.pixel_size(runnum) * 1e-6 # metres
        except:
            print "Couldn't determine pixel size"
            return

        hitParam_queue = 'psanaq'
        hitParam_cpus = self.numCores
        algorithm = 2
        hitParam_alg1_npix_min = 2
        hitParam_alg1_npix_max = 30
        hitParam_alg1_amax_thr = 300
        hitParam_alg1_atot_thr = 600
        hitParam_alg1_son_min = 10
        hitParam_alg1_thr_low = 0
        hitParam_alg1_thr_high = 0
        hitParam_alg1_rank = 3
        hitParam_alg1_radius = 3
        hitParam_alg1_dr = 2
        psanaMaskOn = True
        mask_calibOn = True
        mask_statusOn = True
        mask_edgesOn = True
        mask_centralOn = True
        mask_unbondOn = True
        mask_unbondnrsOn = True
        minPeaks = 10
        maxPeaks = 2048
        minRes = -1
        sample = 'sample'
        autoPeakFinding = False
        access = 'ana'
        tag = None

        cmd = "bsub -q " + hitParam_queue + \
              " -n " + str(hitParam_cpus) + \
              " -o " + self.outdir + "/." + exp + "_" + str(runnum) + ".log mpirun findPeaksAnt -e " + exp + \
              " -d " + detname + \
              " --outDir " + self.outdir + \
              " --algorithm " + str(algorithm)
        if algorithm == 1:
            cmd += " --alg_npix_min " + str(hitParam_alg1_npix_min) + \
                   " --alg_npix_max " + str(hitParam_alg1_npix_max) + \
                   " --alg_amax_thr " + str(hitParam_alg1_amax_thr) + \
                   " --alg_atot_thr " + str(hitParam_alg1_atot_thr) + \
                   " --alg_son_min " + str(hitParam_alg1_son_min) + \
                   " --alg1_thr_low " + str(hitParam_alg1_thr_low) + \
                   " --alg1_thr_high " + str(hitParam_alg1_thr_high) + \
                   " --alg1_rank " + str(hitParam_alg1_rank) + \
                   " --alg1_radius " + str(hitParam_alg1_radius) + \
                   " --alg1_dr " + str(hitParam_alg1_dr)
        elif algorithm == 2:
            cmd += " --alg_npix_min " + str(hitParam_alg1_npix_min) + \
                   " --alg_npix_max " + str(hitParam_alg1_npix_max) + \
                   " --alg_amax_thr " + str(hitParam_alg1_amax_thr) + \
                   " --alg_atot_thr " + str(hitParam_alg1_atot_thr) + \
                   " --alg_son_min " + str(hitParam_alg1_son_min) + \
                   " --alg1_thr_low " + str(hitParam_alg1_thr_low) + \
                   " --alg1_thr_high " + str(hitParam_alg1_thr_high) + \
                   " --alg1_rank " + str(hitParam_alg1_rank) + \
                   " --alg1_radius " + str(hitParam_alg1_radius) + \
                   " --alg1_dr " + str(hitParam_alg1_dr)
        if psanaMaskOn:
            cmd += " --psanaMask_on " + str(psanaMaskOn) + \
                   " --psanaMask_calib " + str(mask_calibOn) + \
                   " --psanaMask_status " + str(mask_statusOn) + \
                   " --psanaMask_edges " + str(mask_edgesOn) + \
                   " --psanaMask_central " + str(mask_centralOn) + \
                   " --psanaMask_unbond " + str(mask_unbondOn) + \
                   " --psanaMask_unbondnrs " + str(mask_unbondnrsOn)

        cmd += " --clen " + str(clenEpics)
        cmd += " --coffset " + str(coffset)

        cmd += " --minPeaks " + str(minPeaks)
        cmd += " --maxPeaks " + str(maxPeaks)
        cmd += " --minRes " + str(minRes)
        cmd += " --sample " + str(sample)
        cmd += " --instrument " + str(instrument)
        cmd += " --pixelSize " + str(pixelSize)

        cmd += " --auto " + str(autoPeakFinding) + \
               " --detectorDistance " + str(detectorDistance)

        cmd += " --access " + access

        if tag: cmd += " --tag " + tag

        cmd += " -r " + str(runnum)
        # Launch peak finding
        print "Submitting batch job: ", cmd
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    def run(self):
        from collections import deque
        dq = deque([]) # this queue is kept filled up to maxJobs
        while True:
            fname = os.path.join(self.outdir, 'todo.txt')
            outname = os.path.join(self.outdir, 'done.txt')
            with open(fname, 'r') as f, open(outname, 'a+') as o:
                todo = f.readlines()
                done = o.readlines()
                avail = len(todo)
                ind = len(done)
                if avail == ind:
                    time.sleep(10)
                    continue
                nextTask = todo[ind]
                if self.maxCores-self.numCores >= 0:
                    # Launch next task
                    exp, runnum, detname = nextTask.rstrip().split(" ")
                    self.launchJob(exp, int(runnum), detname)
                    dq.append(nextTask)
                    o.write(nextTask)
                    self.maxCores -= self.numCores
                else:
                    # Check for jobs finished
                    for i,val in enumerate(dq):
                        exp, runnum, detname = val.rstrip().split(" ")
                        runstr = "%04d" % int(runnum)
                        if os.path.exists(os.path.join(self.outdir, exp+"_"+runstr+".cxi")):
                            print "found: ", val
                            del dq[i]
                            self.maxCores += self.numCores
                            break
            time.sleep(5)


import json
import h5py
import psana
import sys
import numpy as np
from numba import jit
import subprocess, os
import string, random
import time

ansi_cmap = {"k": '0;30',
        "r": '0;31',
        "g": '0;32',
        "o": '0;33',
        'b': '0;34',
        'p': '0;35',
        'c': '0;36'}

def getMyUnfairShare(numJobs, numWorkers, rank):
    """Returns number of events assigned to the workers calling this function."""
    assert(numJobs >= numWorkers)
    allJobs = np.arange(numJobs)
    jobChunks = np.array_split(allJobs,numWorkers)
    myChunk = jobChunks[rank]
    myJobs = allJobs[myChunk[0]:myChunk[-1]+1]
    return myJobs

def highlight(string, status='k', bold=0):
    attr = []
    if sys.stdout.isatty():
        attr.append(ansi_cmap[status])
        if bold:
            attr.append('1')
        return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)
    else:
        return string

def getNoe(args, facility):
    runStr = "%04d" % args.run
    access = "exp=" + args.exp + ":run=" + runStr + ':idx'
    if 'ffb' in args.access.lower(): access += ':dir=/reg/d/ffb/' + args.exp[:3] + '/' + args.exp + '/xtc'
    ds = psana.DataSource(access)
    run = ds.runs().next()
    times = run.times()
    numJobs = len(times)
    # check if the user requested specific number of events
    if args.noe > -1 and args.noe <= numJobs:
        numJobs = args.noe
    return numJobs

def str2bool(v): return v.lower() in ("yes", "true", "t", "1")

def writeStatus(fname, d):
    json.dump(d, open(fname, 'w'))

# Cheetah-related
def getCheetahDim(detInfo):
    if 'cspad' in detInfo.lower():
        dim0 = 8 * 185
        dim1 = 4 * 388
    elif 'rayonix' in detInfo.lower():
        dim0 = 1920
        dim1 = 1920
    elif 'epix10k' in detInfo.lower() and '2m' in detInfo.lower():
        dim0 = 16 * 352
        dim1 = 1 * 384
    elif 'jungfrau4m' in detInfo.lower():
        dim0 = 8 * 512
        dim1 = 1 * 1024
    else:
        print "detector type not implemented"
        exit()
    return dim0, dim1

def saveCheetahFormatMask(outDir, run=None, detInfo=None, combinedMask=None):
    dim0, dim1 = getCheetahDim(detInfo)
    if run is not None:
        fname = outDir+'/r'+str(run).zfill(4)+'/staticMask.h5'
    else:
        fname = outDir + '/staticMask.h5'
    print "Saving static mask in Cheetah format: ", fname
    myHdf5 = h5py.File(fname, 'w')
    dset = myHdf5.create_dataset('/entry_1/data_1/mask', (dim0, dim1), dtype='int')
    # Convert calib image to cheetah image
    # This ensures mask displayed on GUI gets used in peak finding / indexing
    if combinedMask is None:
        img = np.ones((dim0, dim1))
    else:
        img = pct(detInfo, combinedMask)
    dset[:,:] = img
    myHdf5.close()

def convert_peaks_to_cheetah(detname, s, r, c) :
    """Converts seg, row, col assuming (32,185,388)
       to cheetah 2-d table row and col (8*185, 4*388)
    """
    if isinstance(s, np.ndarray):
        s = s.astype('int')
        r = r.astype('int')
        c = c.astype('int')
    if "cspad" in detname.lower():
        segs, rows, cols = (32, 185, 388)
        row2d = (s % 8) * rows + r  # where s%8 is a segment in quad number [0,7]
        col2d = (s / 8) * cols + c  # where s/8 is a quad number [0,3]
    elif "epix10k" in detname.lower() and "2m" in detname.lower():
        segs, rows, cols = (16, 352, 384)
        row2d = s * rows + r
        col2d = c
    elif "jungfrau4m" in detname.lower():
        segs, rows, cols = (8,512,1024)
        row2d = s * rows + r
        col2d = c
    elif "rayonix" in detname.lower():
        row2d = r
        col2d = c
    else:
        print("Error: This detector is not supported")
        exit()
    return row2d, col2d

def convert_peaks_to_psana(detname, row2d, col2d) :
    """Converts cheetah 2-d table row and col (8*185, 4*388)
       to psana seg, row, col assuming (32,185,388)
    """
    if isinstance(row2d, np.ndarray):
        row2d = row2d.astype('int')
        col2d = col2d.astype('int')
    if "cspad" in detname.lower():
        segs, rows, cols = (32, 185, 388)
        s = (row2d / rows) + (col2d / cols * 8)
        r = row2d % rows
        c = col2d % cols
    elif "epix10k" in detname.lower() and "2m" in detname.lower():
        segs, rows, cols = (16, 352, 384)
        s = (row2d / rows) + (col2d / cols)
        r = row2d % rows
        c = col2d % cols
    elif "jungfrau4m" in detname.lower():
        segs, rows, cols = (8, 512, 1024)
        s = (row2d / rows)
        r = row2d % rows
        c = col2d
    elif "rayonix" in detname.lower():
        s = 0
        r = row2d
        c = col2d
    else:
        print("Error: This detector is not supported")
        exit()
    return s, r, c

def pct(detname, unassembled):
    """
    Transform psana unassembled image to cheetah tile
    :param unassembled: psana unassembled image
    :return: cheetah tile
    """
    #t0 = time.time()
    if "rayonix" in detname.lower():
        img = unassembled[:, :] # TODO: express in terms of x,y,dim0,dim1
        return img
    else:
        if "cspad" in detname.lower():
            row = 8
            col = 4
            x = 185
            y = 388
            dim0 = row * x
            dim1 = col * y
        elif "epix10k" in detname.lower() and "2m" in detname.lower():
            row = 16
            col = 1
            x = 352
            y = 384
            dim0 = row * x
            dim1 = col * y
        elif "jungfrau4m" in detname.lower():
            row = 8
            col = 1
            x = 512
            y = 1024
            dim0 = row * x
            dim1 = col * y
        else:
            print "Error: This detector is not supported"
            exit()

        counter = 0
        #t1 = time.time()
        img = np.zeros((dim0, dim1))
        #t2 = time.time()
        for quad in range(col):
            for seg in range(row):
                img[seg * x:(seg + 1) * x, quad * y:(quad + 1) * y] = unassembled[counter, :, :]
                counter += 1
        #t3 = time.time()
        #print "pct: ", t1-t0, t2-t1, t3-t2
    return img

def ipct(detname, tile):
    """
    Transform cheetah tile to psana unassembled image
    :param tile: cheetah tile
    :return: psana unassembled image
    """
    # Save cheetah format mask
    if "cspad" in detname.lower():
        numQuad = 4
        numAsicsPerQuad = 8
        asicRows = 185
        asicCols = 388
    elif "epix10k" in detname.lower() and "2m" in detname.lower():
        numQuad = 1
        numAsicsPerQuad = 16
        asicRows = 352
        asicCols = 384
    elif "jungfrau4m" in detname.lower():
        numQuad = 1
        numAsicsPerQuad = 8
        asicRows = 512
        asicCols = 1024
    elif "rayonix" in detname.lower():
        _t = tile.shape
        calib = np.zeros((1,_t[0],_t[1]),dtype=tile.dtype)
        calib[0,:,:] = tile
        return calib
    else:
        print "Error: This detector is not supported"
        exit()
    # Convert calib image to cheetah image
    calib = np.zeros((numQuad*numAsicsPerQuad,asicRows,asicCols))
    counter = 0
    for quad in range(numQuad):
        for seg in range(numAsicsPerQuad):
            calib[counter, :, :] = \
                tile[seg * asicRows:(seg + 1) * asicRows, quad * asicCols:(quad + 1) * asicCols]
            counter += 1
    return calib

# HDF5-related

def reshapeHdf5(h5file, dataset, ind, numAppend):
    h5file[dataset].resize((ind + numAppend,))

def cropHdf5(h5file, dataset, ind):
    h5file[dataset].resize((ind,))

def updateHdf5(h5file, dataset, ind, val):
    try:
        h5file[dataset][ind] = val
    except:
        h5file[dataset][ind] = 0

# Upsampling
@jit(nopython=True)
def upsample(warr, dim, binr, binc):
    upCalib = np.zeros(dim)
    for k in range(dim[0]):
        for i, ix in enumerate(xrange(0,dim[1],binr)):
            if ix+binr > dim[1]:
                er = dim[1]+1
            else:
                er = ix+binr
            for j, jy in enumerate(xrange(0,dim[2],binc)):
                if jy+binc > dim[2]:
                    ec = dim[2]+1
                else:
                    ec = jy+binc
                upCalib[k,ix:er,jy:ec] = warr[k,i,j]
    return upCalib

# Compression

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def createFiles(d, p, c, comp, decomp):
    with open(d, 'wb'): pass
    with open(p, 'wb'): pass
    with open(c, 'wb'): pass
    with open(comp, 'wb'): pass
    with open(decomp, 'wb'): pass

def saveCalibPanelSZ(calibFilename, unbonded):
    with open(calibFilename, 'ab') as calibF:
        calibF.write(bytearray(unbonded))
        calibF.flush()

def saveRoiSZ(peaksFilename, s, r, c):
    # Save to binary files for compression
    with open(peaksFilename,"ab") as peaksF:
        nPeaks = len(s)
        peakLen=np.array([nPeaks,nPeaks],np.int64) # duplicate nPeaks twice for python2
        peaksF.write(bytearray(peakLen))
        for j in range(nPeaks):
            pk = np.array([s[j],r[j],c[j]],np.int16)
            peaksF.write(bytearray(pk))
        peaksF.flush()

def saveUnassemSZ(dataFilename, unassem):
    # Save to binary files for compression
    with open(dataFilename,"ab") as dataF:
        dataF.write(bytearray(unassem.astype(np.float32)))
        dataF.flush()

def compDecompSZ(dataFilename, peaksFilename, calibFilename, compFilename, decompFilename):
    # compress and decompress
    cmd = "./exafelSZ_example_sta "+dataFilename+" "+peaksFilename+" "+calibFilename+" "+compFilename+" "+decompFilename
    retcode = subprocess.call(cmd, shell=True)
    try:
        if retcode < 0:
            print >>sys.stderr, "Child was terminated by signal", -retcode
        else:
            pass#print >>sys.stderr, "Child returned", retcode
    except OSError as e:
        print >>sys.stderr, "Execution failed:", e

#    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
#    out, err = process.communicate()

def decomp2unassem(decompFilename):
    dim0, dim1, dim2 = 32, 185, 388 # cspad dimensions
    with open(decompFilename,"rb") as inbf:
        data = inbf.read(dim0*dim1*dim2*4) # float32=4bytes
        return np.fromstring(data,dtype=np.float32).reshape((dim0, dim1, dim2))

def delBinarySZ(dataFilename, peaksFilename, calibFilename, compFilename, decompFilename):
    os.remove(dataFilename)
    os.remove(peaksFilename)  
    os.remove(calibFilename)
    os.remove(compFilename)  
    os.remove(decompFilename)

def unassem2decompRoiSZ(s, r, c, h, w, unbonded, unassem, outdir):
    """
    s: segment positions of the ROIs
    r: row positions of the ROIs
    c: collumn positions of the ROIs
    h: height of ROIs (currently unused)
    w: width of ROIs (currently unused)
    unassem: unassembled image (32,185,388)
    unbonded: unbonded mask image (32,185,388)
    """
    while True:
        dataFilename=os.path.join(outdir, randomString(10)+"_data.bin")
        peaksFilename=os.path.join(outdir, randomString(10)+"_peaks.bin")
        calibFilename=os.path.join(outdir, randomString(10)+"_calibPanel.bin")
        compFilename=os.path.join(outdir, randomString(10)+"_comp.bin")
        decompFilename=os.path.join(outdir, randomString(10)+"_decomp.bin")
        if not os.path.exists(dataFilename) and \
           not os.path.exists(peaksFilename) and \
           not os.path.exists(calibFilename) and \
           not os.path.exists(compFilename) and \
           not os.path.exists(decompFilename):
            createFiles(dataFilename, peaksFilename, calibFilename, compFilename, decompFilename)
            saveCalibPanelSZ(calibFilename, unbonded)
            saveRoiSZ(peaksFilename, s, r, c)
            saveUnassemSZ(dataFilename, unassem)
            compDecompSZ(dataFilename, peaksFilename, calibFilename, compFilename, decompFilename)
            decompRoiSZ = decomp2unassem(decompFilename)
            delBinarySZ(dataFilename, peaksFilename, calibFilename, compFilename, decompFilename)
            return decompRoiSZ







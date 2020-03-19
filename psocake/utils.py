import json
import h5py
import psana
import sys
import numpy as np
from numba import jit
import subprocess, os
import string, random

ansi_cmap = {"k": '0;30',
        "r": '0;31',
        "g": '0;32',
        "o": '0;33',
        'b': '0;34',
        'p': '0;35',
        'c': '0;36'}

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
    if facility == 'LCLS':
        runStr = "%04d" % args.run
        access = "exp=" + args.exp + ":run=" + runStr + ':idx'
        if 'ffb' in args.access.lower(): access += ':dir=/reg/d/ffb/' + args.exp[:3] + '/' + args.exp + '/xtc'
        ds = psana.DataSource(access)
        run = ds.runs().next()
        times = run.times()
        numJobs = len(times)
    elif facility == 'PAL':
        _temp = args.dir + '/' + args.exp[:3] + '/' + args.exp + '/data/run' + str(args.run).zfill(4) + '/*.h5'
        numJobs = len(glob.glob(_temp))
    # check if the user requested specific number of events
    if args.noe > -1 and args.noe <= numJobs:
        numJobs = args.noe
    return numJobs

def str2bool(v): return v.lower() in ("yes", "true", "t", "1")

def writeStatus(fname, d):
    json.dump(d, open(fname, 'w'))

# Cheetah-related

def convert_peaks_to_cheetah(s, r, c) :
    """Converts seg, row, col assuming (32,185,388)
       to cheetah 2-d table row and col (8*185, 4*388)
    """
    segs, rows, cols = (32,185,388)
    row2d = (int(s)%8) * rows + int(r) # where s%8 is a segment in quad number [0,7]
    col2d = (int(s)/8) * cols + int(c) # where s/8 is a quad number [0,3]
    return row2d, col2d

def convert_peaks_to_psana(row2d, col2d) :
    """Converts cheetah 2-d table row and col (8*185, 4*388)
       to psana seg, row, col assuming (32,185,388)
    """
    if isinstance(row2d, np.ndarray):
        row2d = row2d.astype('int')
        col2d = col2d.astype('int')
    segs, rows, cols = (32,185,388)
    s = (row2d / rows) + (col2d / cols * 8)
    r = row2d % rows
    c = col2d % cols
    return s, r, c

def pct(unassembled):
    """
    Transform psana unassembled image to cheetah tile
    :param unassembled: psana unassembled image
    :return: cheetah tile
    """
    counter = 0
    dim0 = 8 * 185
    dim1 = 4 * 388
    img = np.zeros((dim0, dim1))
    for quad in range(4):
        for seg in range(8):
            img[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = unassembled[counter, :, :]
            counter += 1
    return img

def ipct(tile):
    """
    Transform cheetah tile to psana unassembled image
    :param tile: cheetah tile
    :return: psana unassembled image
    """
    # Save cheetah format mask
    numQuad = 4
    numAsicsPerQuad = 8
    asicRows = 185
    asicCols = 388

    # Convert calib image to cheetah image
    calib = np.zeros((32,asicRows,asicCols))
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







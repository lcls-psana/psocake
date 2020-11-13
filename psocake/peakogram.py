#!/usr/bin/env python3

#	
#	peakogram
#
#	Modified code from Tom Grant, HWI, Buffalo NY
#



import sys
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import re
import ntpath

parser = argparse.ArgumentParser()
parser.add_argument("-i", default="peaks.txt", help="peaks.txt file")
parser.add_argument("-n", type=int, help="number of cxi files")
parser.add_argument("-t", default="", help="cxi file tag")
parser.add_argument("-l", action="store_true", help="log scale y-axis")
parser.add_argument("--rmin", type=float, help="minimum pixel resolution cutoff")
parser.add_argument("--rmax", type=float, help="maximum pixel resolution cutoff")
parser.add_argument("--imin", type=float, help="minimum peak intensity cutoff")
parser.add_argument("--imax", type=float, help="maximum peak intensity cutoff")
parser.add_argument("--nmax", default=np.inf, type=int, help="maximum number of peaks to read")
parser.add_argument("-o", default="peakogram", help="output file prefix")
args = parser.parse_args()

data = []
n = 0

if (args.i.endswith(".txt")):
    with open(args.i) as f:
        for line in f:
            if n > args.nmax:
                break
            columns = line.split(',')
            if columns[0].isdigit():
                n += 1
                data.append([columns[8], columns[13]])
                if n % 1000 == 0:
                    sys.stdout.write("\r%i peaks found" % n)
                    sys.stdout.flush()
    data = np.asarray(data, dtype=float)
    sys.stdout.write("\r%i peaks found" % n)
    sys.stdout.flush()
    print("")
    x = data[:, 0]
    y = data[:, 1]

else:
    for i in range(args.n):
        fname = args.i + "_" + str(i)
        if args.t: fname += "_"+args.t
        fname += ".cxi"
        print fname
        f = h5py.File(fname, 'r')
        if i == 0:
            x = f['entry_1/result_1/peakRadius'].value
            y = f['entry_1/result_1/peakMaxIntensity'].value
        else:
            x = np.append(x,f['entry_1/result_1/peakRadius'].value,axis=0)
            y = np.append(y,f['entry_1/result_1/peakMaxIntensity'].value,axis=0)
        f.close()
        print x.shape

xmin = np.min(x[x > 0])
xmax = np.max(x)
ymin = np.min(y[y > 0])
ymax = np.max(y)

if args.rmin is not None:
    xmin = args.rmin
if args.rmax is not None:
    xmax = args.rmax
if args.imin is not None:
    ymin = args.imin
if args.imax is not None:
    ymax = args.imax

keepers = np.where((x > xmin) & (x < xmax) & (y > ymin) & (y < ymax))

x = x[keepers]
y = y[keepers]

if args.l:
    y = np.log10(y)
    ymin = np.log10(ymin)
    ymax = np.log10(ymax)

bins = 300
H, xedges, yedges = np.histogram2d(y, x, bins=bins)

fig = plt.figure()
ax1 = plt.subplot(111)
plot = ax1.pcolormesh(yedges, xedges, H, norm=LogNorm())
cbar = plt.colorbar(plot)
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xlabel("r (npixels, assembled)")
if args.l:
    plt.ylabel("Log(Peak Intensity)")
else:
    plt.ylabel("Peak Intensity")

if (args.i.endswith(".txt")):
    plt.title(args.i)
else:
    if args.t:
        plt.title(args.i+"_"+args.t)
    else:
        plt.title(args.i)
plt.show()
# plt.savefig(args.o, ext="png")

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

elif (args.i.endswith(".cxi")):
    f = h5py.File(args.i, 'r')
    x = f['entry_1/result_1/peakRadiusAll'][()]
    y = f['entry_1/result_1/peakMaxIntensityAll'][()]
    f.close()

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
elif (args.i.endswith(".cxi")):
    if ("/reg/d/psdm/cxi/" in args.i):
        head, tail = ntpath.split(args.i)
        plt.title(tail)
    else:
        plt.title(args.i)
plt.show()
# plt.savefig(args.o, ext="png")

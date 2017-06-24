from psgeom import camera
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e','--exp', help="experiment name", type=str)
parser.add_argument('-r','--run', help="run number", type=int)
parser.add_argument('-d','--det', help="detector name", type=str)
parser.add_argument('--rootDir', help="root directory", type=str)
parser.add_argument('-c','--crystfel', help="crystfel geometry", type=str)
parser.add_argument('-p','--psana', help="psana filename", type=str)
parser.add_argument('-z','--clen', help="home to detector distance (m)", type=float)
args = parser.parse_args()

if 'cspad' in args.det.lower():
    cspad = camera.Cspad.from_crystfel_file(args.crystfel)
    cspad.translate((0,0,args.clen*1e6)) # um
    cspad.to_psana_file(args.psana)
elif 'rayonix' in args.det.lower():
    #print "Not implemented: Converting Rayonix .geom to .data"
    rayonix = camera.CompoundAreaCamera.from_crystfel_file(args.crystfel)
    rayonix.translate((0, 0, args.clen * 1e6))  # um
    rayonix.to_psana_file(args.psana)
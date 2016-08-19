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
args = parser.parse_args()

cspad = camera.Cspad.from_crystfel_file(args.crystfel)
cspad.to_psana_file(args.psana)

from PSCalib.CalibFileFinder import deploy_calib_file
cmts = {'exp': args.exp, 'app': 'psocake', 'comment': 'converted from CrystFEL geometry'}
deploy_calib_file(cdir=args.rootDir+'/calib', src=str(args.det), type='geometry', run_start=args.run, run_end=None, ifname=args.psana, dcmts=cmts, pbits=0)

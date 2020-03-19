from CrystalFindingAnt import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-host", default="", type=str, help="master's host node. Leave blank if master.")
args = parser.parse_args()

cf = CrystalFindingAnt(args.host)

cf.run()

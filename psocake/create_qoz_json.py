import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', "--binSize", type=int, help="bin Size", default = 2)
parser.add_argument('-r', "--roiWindowSize", type=int, help="ROI window Size", default = 9)
parser.add_argument('-e', "--absError", type=float, help="absolute Error bound", default = 10)
parser.add_argument('-f', "--json_file", type=str, help="json file name", default = 'qoz.json')

args = parser.parse_args()

binSize = args.binSize
roiWindowSize = args.roiWindowSize
absError = args.absError


#if type(args.binSize)!=type(None):


fopts = args.json_file

lp_json={
	    "compressor_id": "pressio",
	    "early_config": {
		"pressio": {
		    "pressio:compressor": "roibin",
		    "roibin": {
		        "roibin:metric": "composite",
		        "roibin:background": "mask_binning",
		        "roibin:roi": "fpzip",
		        "background": {
		            "binning:compressor": "pressio",
		            "mask_binning:compressor": "pressio",
		            "pressio": {"pressio:compressor": "qoz"},
		        },
		        "composite": {"composite:plugins": ["size", "time", "input_stats", "error_stat"]},
		    },
		}
	    },
	    "compressor_config": {
		"pressio": {
		    "roibin": {
		        "roibin:roi_size": [roiWindowSize, roiWindowSize, 0],
		        "roibin:centers": None, # "roibin:roi_strategy": "coordinates",
		        "roibin:nthreads": 4,
		        "roi": {"fpzip:prec": 0},
		        "background": {
		            "mask_binning:mask": None,
		            "mask_binning:shape": [binSize, binSize, 1],
		            "mask_binning:nthreads": 4,
		            "pressio": {"pressio:abs": absError, 
		                        "qoz":{'qoz:stride': 8}
		                        #"sz3":{"sz3:stride": 8, 
		                        #       'sz3:pred_dim': 3}
		                       },
		        },
		    }
		}
	    },
	    "name": "pressio",
	}

with open(fopts, "w") as outfile:
    outfile.write(json.dumps(lp_json, indent=4))


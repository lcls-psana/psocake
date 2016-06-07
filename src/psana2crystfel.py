from psgeom import camera
import sys

cspad = camera.Cspad.from_psana_file(sys.argv[1])

cspad.to_crystfel_file(sys.argv[2])

import sys
from psgeom import camera
if 'cspad' in sys.argv[1].lower():
    geom = camera.Cspad.from_psana_file(sys.argv[1])
    geom.to_crystfel_file(sys.argv[2], coffset=float(sys.argv[3]))
else:
    geom = camera.CompoundAreaCamera.from_psana_file(sys.argv[1])
    geom.to_crystfel_file(sys.argv[2], coffset=float(sys.argv[3]))

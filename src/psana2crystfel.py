from psgeom import camera
import sys

if 'cspad' in sys.argv[1].lower():
    geom = camera.Cspad.from_psana_file(sys.argv[1])
    geom.to_crystfel_file(sys.argv[2])
elif 'rayonix' in sys.argv[1].lower():
    geom = camera.CompoundAreaCamera.from_psana_file(sys.argv[1])
    geom.to_crystfel_file(sys.argv[2])

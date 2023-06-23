## Note: this s3df branch was created to run on s3df using python3 only. It was tested with a new branch of psgeom to reflect the latest crystfel geom format:
https://github.com/slaclab/psgeom/tree/new_crystfel_format

# Psocake (py2/py3)

Making Data Analysis for Free-Electron Lasers a piece of cake

## Getting Started at LCLS

For **newer experiments with vertical polarization (starting Run 18)**, use Psocake version v1.0.15 or higher with psana (ana-4.0.14-py3 or higher).
Note that psocake has moved to python3 (and python2 is deprecated).
```
    source /reg/g/psdm/etc/psconda.sh -py3
    source /reg/g/cfel/crystfel/crystfel-dev/setup-sh # CrystFEL compatible version
```

For **older experiments with horizontal polarization (before Run 18)**, use Psocake version v0.6.X. Python2 code is maintained in v00-06-00-branch.  
For SFX, also source CrystFEL v0.8:
```
    source /reg/g/psdm/etc/psconda.sh
    source /reg/g/cfel/crystfel/crystfel-0.8.0/setup-sh # CrystFEL compatible version  
```

## Installation

To make a local install of psocake, please add the following to your ~/.bashrc, where /`<path-to-gitclone>` is the location where you git cloned this repository:  
export PATH=/`<path-to-gitclone>`/app:$PATH   
export PYTHONPATH=/`<path-to-gitclone>`/:$PYTHONPATH    
export PSOCAKE_FACILITY=LCLS  


## Tutorials

Serial Femtosecond Crystallography (SFX):
https://confluence.slac.stanford.edu/display/PSDM/Psocake+SFX+tutorial

Single Particle Imaging (SPI):
https://confluence.slac.stanford.edu/display/PSDM/Psocake+SPI+tutorial

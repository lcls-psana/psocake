# Psocake

Making Data Analysis for Free-Electron Lasers a piece of cake

## Getting Started at LCLS

Psocake (v1.0.15 or higher) with psana (ana-4.0.14-py3 or higher) should be used for experiments with new vertical polarization (starting Run 18).
Note that psocake has moved to python3 (and python2 is deprecated).

For older experiments with horizontal polarization (before 2020), use Psocake version v0.6.X with psana (ana-3.0.2 or lower):
source /cds/sw/ds/ana/conda/manage/bin/psconda.sh
export PATH=/reg/data/ana03/scratch/yoon82/Software/py2/psocake/app:$PATH
export PYTHONPATH=/reg/data/ana03/scratch/yoon82/Software/py2/psocake:$PYTHONPATH
# CrystFEL compatible version
source /reg/g/cfel/crystfel/crystfel-0.8.0/setup-sh

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

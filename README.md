# Psocake

Making Data Analysis for Free-Electron Lasers a piece of cake

## Getting Started at LCLS

Psocake (v1.0.0 or higher) with psana (v1.5.23 or higher) should be used for experiments with new vertical polarization (starting Run 18).

For older experiments with horizontal polarization (before 2020), use Psocake version v0.4.0 with psana (v.1.5.22 or lower)
`conda activate ana-1.5.22`

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

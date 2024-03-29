#!/usr/bin/env python
import h5py
import numpy as np
import psana
import time
import argparse
import os, json, sys

import psanaWhisperer

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument("-e","--exp", help="psana experiment name (e.g. cxic0415)", type=str)
parser.add_argument("-r","--run", help="psana run number (e.g. 15)", type=int)
parser.add_argument("-d","--det",help="psana detector name (e.g. DscCsPad)", type=str)
parser.add_argument("-i","--inDir",help="input directory where files_XXXX.lst exists (e.g. /reg/d/psdm/cxi/cxic0415/scratch)", type=str)
parser.add_argument("-o","--outDir",help="output directory (e.g. /reg/d/psdm/cxi/cxic0415/scratch)", type=str)
parser.add_argument("--sample",help="sample name (e.g. lysozyme)",default='', type=str)
parser.add_argument("--instrument",help="instrument name (e.g. CXI)", type=str)
parser.add_argument("--clen", help="camera length epics name (e.g. CXI:DS1:MMS:06.RBV or CXI:DS2:MMS:06.RBV)", type=str)
parser.add_argument("--coffset", help="camera offset, CXI home position to sample (m)", default=0, type=float)
parser.add_argument("--detectorDistance", help="detector distance from interaction point (m)", default=0, type=float)
parser.add_argument("--cxiVersion", help="cxi version",default=140, type=int)
parser.add_argument("--pixelSize", help="pixel size (m)", type=float)
parser.add_argument("--minPeaks", help="Index only if above minimum number of peaks",default=15, type=int)
parser.add_argument("--maxPeaks", help="Index only if below maximum number of peaks",default=300, type=int)
parser.add_argument("--minRes", help="Index only if above minimum resolution",default=0, type=int)
parser.add_argument("--hitThresh", help="hit only if above minimum number of pixels, or a range specified by a colon (SPI)",default='-1', type=str)
parser.add_argument("--backgroundThresh", help="use as miss if below maximum number of pixels, or a range specified by a colon (SPI)",default='-1', type=str)
parser.add_argument("--aduPerPhoton", help="adu per photon (SPI)",default=1, type=float)
parser.add_argument("--mode",help="type of experiment (e.g. sfx, spi)",default='', type=str)
parser.add_argument("--saveADU", help="Save assembled images in ADU.", action='store_true')
parser.add_argument("--savePhot", help="Save assembled images in photons.", action='store_true')
parser.add_argument("--tag",help="cxi file tag",default="", type=str)
args = parser.parse_args()

def writeStatus(fname,d):
    json.dump(d, open(fname, 'w'))

def getMyUnfairShare(numJobs, numWorkers, rank):
    """Returns number of events assigned to the slave calling this function."""
    assert(numJobs >= numWorkers)
    try:
        allJobs = np.arange(numJobs)
        jobChunks = np.array_split(allJobs, numWorkers)
        myChunk = jobChunks[rank]
        myJobs = allJobs[myChunk[0]:myChunk[-1]+1]
        return myJobs
    except:
        return None

# Set up variable
experimentName = args.exp
runNumber = args.run
detInfo = args.det
sampleName = args.sample
instrumentName = args.instrument.lower()
coffset = args.coffset
(x_pixel_size,y_pixel_size) = (args.pixelSize, args.pixelSize)
mode = args.mode
aduPerPhoton = args.aduPerPhoton
# Munge hitThresh
if ":" in args.hitThresh:
    (hitThreshMin, hitThreshMax) = args.hitThresh.split(':')
    hitThreshMin = float(hitThreshMin)
    hitThreshMax = float(hitThreshMax)
else:
    hitThreshMin = float(args.hitThresh)
    hitThreshMax = float('inf')
# Munge backgroundThresh
if args.backgroundThresh == '-1':
    backgroundThreshMin = -1
    backgroundThreshMax = -1
elif ":" in args.backgroundThresh:
    (backgroundThreshMin, backgroundThreshMax) = args.backgroundThresh.split(':')
    backgroundThreshMin = float(backgroundThreshMin)
    backgroundThreshMax = float(backgroundThreshMax)
else:
    backgroundThreshMin = -1
    backgroundThreshMax = float(args.backgroundThresh)

# Set up psana
ps = psanaWhisperer.psanaWhisperer(experimentName, runNumber, detInfo, args.clen, aduPerPhoton=aduPerPhoton)
ps.setupExperiment()

# Read list of files
runStr = "%04d" % args.run
if args.tag:
    filename = args.inDir + "/" + args.exp + "_" + runStr + "_" + args.tag + ".cxi"
else:
    filename = args.inDir + "/" + args.exp + "_" + runStr + ".cxi"

print("Reading file: %s" % (filename))
if mode == 'sfx':
    statusFname = args.inDir+'/status_index.txt'
elif mode == 'spi':
    statusFname = args.inDir+'/status_hits.txt'

if rank == 0:
    try:
        d = {"message": "#CXIDB"}
        print(statusFname)
        writeStatus(statusFname, d)
    except:
        pass

notDone = 1
while notDone:
    try:
        f = h5py.File(filename, "r")
        if mode == 'sfx':
            nPeaks = f["/entry_1/result_1/nPeaksAll"][()]
            maxRes = f["/entry_1/result_1/maxResAll"][()]
            posX = f["/entry_1/result_1/peakXPosRawAll"][()]
            posY = f["/entry_1/result_1/peakYPosRawAll"][()]
            atot = f["/entry_1/result_1/peakTotalIntensityAll"][()]
            maxRes = f["/entry_1/result_1/maxResAll"][()]
            hitInd = ((nPeaks >= args.minPeaks) & (nPeaks <= args.maxPeaks) & (maxRes >= args.minRes)).nonzero()[0]
            numHits = len(hitInd)
        elif mode == 'spi':
            nHits = f["/entry_1/result_1/nHitsAll"][()]
            hitInd = ((nHits >= hitThreshMin) & (nHits <= hitThreshMax)).nonzero()[0]
            if backgroundThreshMax > -1:
                missInd = ((nHits >= backgroundThreshMin) & (nHits <= backgroundThreshMax)).nonzero()[0]
            numHits = len(hitInd)
        f.close()
        notDone = 0
    except:
        print("Couldn't read h5 file: ", filename)
        print("Number of tries: ", notDone)
        notDone += 1
        if notDone >= 10: sys.exit()
        time.sleep(10)
    if numHits == 0: sys.exit()

if mode == 'sfx' and instrumentName == 'cxi':
    # Get image shape
    firstHit = hitInd[0]
    ps.getEvent(firstHit)
    img = ps.getCheetahImg()
    (dim0, dim1) = img.shape
elif mode == 'sfx' and instrumentName == 'mfx':
    # Get image shape
    firstHit = hitInd[0]
    ps.getEvent(firstHit)
    img = ps.getCalibImg()
    (_, dim0, dim1) = img.shape
elif mode == 'sfx' and instrumentName == 'xpp':
    # Get image shape
    firstHit = hitInd[0]
    ps.getEvent(firstHit)
    img = ps.getCalibImg()
    (_, dim0, dim1) = img.shape
elif mode == 'spi':
    # Get image shape
    firstHit = hitInd[0]
    ps.getEvent(firstHit)
    img = ps.getAssembledImg()
    (dim0, dim1) = img.shape

if rank == 0: tic = time.time()

inDir = args.inDir
assert os.path.isdir(inDir)
if args.outDir is None:
    outDir = inDir
else:
    outDir = args.outDir
    assert os.path.isdir(outDir)

hasCoffset = False
hasDetectorDistance = False
if args.detectorDistance is not 0:
    hasDetectorDistance = True
if args.coffset is not 0:
    hasCoffset = True

ps = psanaWhisperer.psanaWhisperer(experimentName, runNumber, detInfo, args.clen, aduPerPhoton)#, backgroundThreshMin, backgroundThreshMax)
ps.setupExperiment()
startTime = ps.getStartTime()
numEvents = ps.eventTotal
es = ps.ds.env().epicsStore()
try:
    pulseLength = es.value('SIOC:SYS0:ML00:AO820')*1e-15 # s
    numPhotons = es.value('SIOC:SYS0:ML00:AO580')*1e12 # number of photons
except:
    pulseLength = 0
    numPhotons = 0

ebeam = ps.evt.get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
try:
    photonEnergy = ebeam.ebeamPhotonEnergy() * 1.60218e-19 # J
    pulseEnergy = ebeam.ebeamL3Energy() # MeV
except:
    photonEnergy = 0
    pulseEnergy = 0

if hasCoffset:
    detectorDistance = coffset + ps.clen*1e-3 # sample to detector in m
elif hasDetectorDistance:
    detectorDistance = args.detectorDistance

if rank == 0:
    try:
        d = {"message": "#InitCXIDB"}
        writeStatus(statusFname, d)
    except:
        pass

    f = h5py.File(filename, "r+")

    if "/status/xtc2cxidb" in f:
        del f["/status/xtc2cxidb"]
    f["/status/xtc2cxidb"] = 'fail'

    # open the HDF5 CXI file for writing
    if "cxi_version" in f:
        del f["cxi_version"]
    f.create_dataset("cxi_version",data=args.cxiVersion)
    #f.flush()

    ###################
    # LCLS
    ###################
    if "LCLS" in f:
        del f["LCLS"]
    lcls_1 = f.create_group("LCLS")
    lcls_detector_1 = lcls_1.create_group("detector_1")
    ds_lclsDet_1 = lcls_detector_1.create_dataset("EncoderValue",(numHits,), dtype=float)
    ds_lclsDet_1.attrs["axes"] = "experiment_identifier"
    ds_lclsDet_1.attrs["numEvents"] = numHits
    ds_ebeamCharge_1 = lcls_1.create_dataset("electronBeamEnergy",(numHits,), dtype=float)
    ds_ebeamCharge_1.attrs["axes"] = "experiment_identifier"
    ds_ebeamCharge_1.attrs["numEvents"] = numHits
    ds_beamRepRate_1 = lcls_1.create_dataset("beamRepRate",(numHits,), dtype=float)
    ds_beamRepRate_1.attrs["axes"] = "experiment_identifier"
    ds_beamRepRate_1.attrs["numEvents"] = numHits
    ds_particleN_electrons_1 = lcls_1.create_dataset("particleN_electrons",(numHits,), dtype=float)
    ds_particleN_electrons_1.attrs["axes"] = "experiment_identifier"
    ds_particleN_electrons_1.attrs["numEvents"] = numHits
    ds_eVernier_1 = lcls_1.create_dataset("eVernier",(numHits,), dtype=float)
    ds_eVernier_1.attrs["axes"] = "experiment_identifier"
    ds_eVernier_1.attrs["numEvents"] = numHits
    ds_charge_1 = lcls_1.create_dataset("charge",(numHits,), dtype=float)
    ds_charge_1.attrs["axes"] = "experiment_identifier"
    ds_charge_1.attrs["numEvents"] = numHits
    ds_peakCurrentAfterSecondBunchCompressor_1 = lcls_1.create_dataset("peakCurrentAfterSecondBunchCompressor",(numHits,), dtype=float)
    ds_peakCurrentAfterSecondBunchCompressor_1.attrs["axes"] = "experiment_identifier"
    ds_peakCurrentAfterSecondBunchCompressor_1.attrs["numEvents"] = numHits
    ds_pulseLength_1 = lcls_1.create_dataset("pulseLength",(numHits,), dtype=float)
    ds_pulseLength_1.attrs["axes"] = "experiment_identifier"
    ds_pulseLength_1.attrs["numEvents"] = numHits
    ds_ebeamEnergyLossConvertedToPhoton_mJ_1 = lcls_1.create_dataset("ebeamEnergyLossConvertedToPhoton_mJ",(numHits,), dtype=float)
    ds_ebeamEnergyLossConvertedToPhoton_mJ_1.attrs["axes"] = "experiment_identifier"
    ds_ebeamEnergyLossConvertedToPhoton_mJ_1.attrs["numEvents"] = numHits
    ds_calculatedNumberOfPhotons_1 = lcls_1.create_dataset("calculatedNumberOfPhotons",(numHits,), dtype=float)
    ds_calculatedNumberOfPhotons_1.attrs["axes"] = "experiment_identifier"
    ds_calculatedNumberOfPhotons_1.attrs["numEvents"] = numHits
    ds_photonBeamEnergy_1 = lcls_1.create_dataset("photonBeamEnergy",(numHits,), dtype=float)
    ds_photonBeamEnergy_1.attrs["axes"] = "experiment_identifier"
    ds_photonBeamEnergy_1.attrs["numEvents"] = numHits
    ds_wavelength_1 = lcls_1.create_dataset("wavelength",(numHits,), dtype=float)
    ds_wavelength_1.attrs["axes"] = "experiment_identifier"
    ds_wavelength_1.attrs["numEvents"] = numHits
    ds_sec_1 = lcls_1.create_dataset("machineTime",(numHits,),dtype=int)
    ds_sec_1.attrs["axes"] = "experiment_identifier"
    ds_sec_1.attrs["numEvents"] = numHits
    ds_nsec_1 = lcls_1.create_dataset("machineTimeNanoSeconds",(numHits,),dtype=int)
    ds_nsec_1.attrs["axes"] = "experiment_identifier"
    ds_nsec_1.attrs["numEvents"] = numHits
    ds_fid_1 = lcls_1.create_dataset("fiducial",(numHits,),dtype=int)
    ds_fid_1.attrs["axes"] = "experiment_identifier"
    ds_fid_1.attrs["numEvents"] = numHits
    ds_photonEnergy_1 = lcls_1.create_dataset("photon_energy_eV", (numHits,), dtype=float) # photon energy in eV
    ds_photonEnergy_1.attrs["axes"] = "experiment_identifier"
    ds_photonEnergy_1.attrs["numEvents"] = numHits
    ds_wavelengthA_1 = lcls_1.create_dataset("photon_wavelength_A",(numHits,), dtype=float)
    ds_wavelengthA_1.attrs["axes"] = "experiment_identifier"
    ds_wavelengthA_1.attrs["numEvents"] = numHits
    #### Datasets not in Cheetah ###
    ds_evtNum_1 = lcls_1.create_dataset("eventNumber",(numHits,),dtype=int)
    ds_evtNum_1.attrs["axes"] = "experiment_identifier"
    ds_evtNum_1.attrs["numEvents"] = numHits
    #f.flush()
    ###################
    # entry_1
    ###################
    entry_1 = f.require_group("entry_1")

    #dt = h5py.special_dtype(vlen=bytes)
    if "experimental_identifier" in entry_1:
        del entry_1["experimental_identifier"]
    ds_expId = entry_1.create_dataset("experimental_identifier",(numHits,),dtype=int)#dt)
    ds_expId.attrs["axes"] = "experiment_identifier"
    ds_expId.attrs["numEvents"] = numHits
    #f.flush()

    if mode == 'sfx':
        if "entry_1/result_1/nPeaks" in f:
            del f["entry_1/result_1/nPeaks"]
            del f["entry_1/result_1/peakXPosRaw"]
            del f["entry_1/result_1/peakYPosRaw"]
            del f["entry_1/result_1/peakTotalIntensity"]
            del f["entry_1/result_1/maxRes"]
        ds_nPeaks = f.create_dataset("/entry_1/result_1/nPeaks", (numHits,), dtype=int)
        ds_nPeaks.attrs["axes"] = "experiment_identifier"
        ds_nPeaks.attrs["numEvents"] = numHits
        ds_nPeaks.attrs["minPeaks"] = args.minPeaks
        ds_nPeaks.attrs["maxPeaks"] = args.maxPeaks
        ds_nPeaks.attrs["minRes"] = args.minRes
        ds_posX = f.create_dataset("/entry_1/result_1/peakXPosRaw", (numHits,2048), dtype='float32')#, chunks=(1,2048))
        ds_posX.attrs["axes"] = "experiment_identifier:peaks"
        ds_posX.attrs["numEvents"] = numHits
        ds_posY = f.create_dataset("/entry_1/result_1/peakYPosRaw", (numHits,2048), dtype='float32')#, chunks=(1,2048))
        ds_posY.attrs["axes"] = "experiment_identifier:peaks"
        ds_posY.attrs["numEvents"] = numHits
        ds_atot = f.create_dataset("/entry_1/result_1/peakTotalIntensity", (numHits,2048), dtype='float32')#, chunks=(1,2048))
        ds_atot.attrs["axes"] = "experiment_identifier:peaks"
        ds_atot.attrs["numEvents"] = numHits
        ds_maxRes = f.create_dataset("/entry_1/result_1/maxRes", (numHits,), dtype=int)
        ds_maxRes.attrs["axes"] = "experiment_identifier:peaks"
        ds_maxRes.attrs["numEvents"] = numHits
    elif mode == 'spi':
        if "entry_1/result_1/nHits" in f:
            del f["entry_1/result_1/nHits"]
        ds_nHits = f.create_dataset("/entry_1/result_1/nHits", (numHits,), dtype=int)
        ds_nHits.attrs["axes"] = "experiment_identifier"
        ds_nHits.attrs["numEvents"] = numHits
        ds_nHits.attrs["hitThresh"] = args.hitThresh
        ds_nHits.attrs["backgroundThreshMin"] = backgroundThreshMin
        ds_nHits.attrs["backgroundThreshMax"] = backgroundThreshMax
    #f.flush()

    if "start_time" in entry_1:
        del entry_1["start_time"]
    entry_1.create_dataset("start_time",data=startTime)

    if "sample_1" in entry_1:
        del entry_1["sample_1"]
    sample_1 = entry_1.create_group("sample_1")
    sample_1.create_dataset("name",data=sampleName)

    if "instrument_1" in entry_1:
        del entry_1["instrument_1"]
    instrument_1 = entry_1.create_group("instrument_1")
    instrument_1.create_dataset("name",data=instrumentName)

    source_1 = instrument_1.create_group("source_1")
    ds_photonEnergy = source_1.create_dataset("energy", (numHits,), dtype=float) # photon energy in J
    ds_photonEnergy.attrs["axes"] = "experiment_identifier"
    ds_photonEnergy.attrs["numEvents"] = numHits
    ds_pulseEnergy = source_1.create_dataset("pulse_energy", (numHits,), dtype=float) # in J
    ds_pulseEnergy.attrs["axes"] = "experiment_identifier"
    ds_pulseEnergy.attrs["numEvents"] = numHits
    ds_pulseWidth = source_1.create_dataset("pulse_width", (numHits,), dtype=float) # in s
    ds_pulseWidth.attrs["axes"] = "experiment_identifier"
    ds_pulseWidth.attrs["numEvents"] = numHits

    detector_1 = instrument_1.create_group("detector_1")
    ds_dist_1 = detector_1.create_dataset("distance", (numHits,), dtype=float) # in meters
    ds_dist_1.attrs["axes"] = "experiment_identifier"
    ds_dist_1.attrs["numEvents"] = numHits
    ds_x_pixel_size_1 = detector_1.create_dataset("x_pixel_size", (numHits,), dtype=float)
    ds_x_pixel_size_1.attrs["axes"] = "experiment_identifier"
    ds_x_pixel_size_1.attrs["numEvents"] = numHits
    ds_y_pixel_size_1 = detector_1.create_dataset("y_pixel_size", (numHits,), dtype=float)
    ds_y_pixel_size_1.attrs["axes"] = "experiment_identifier"
    ds_y_pixel_size_1.attrs["numEvents"] = numHits
    detector_1.create_dataset("description",data=detInfo)
    #f.flush()

    if mode == 'sfx':
        dset_1 = detector_1.create_dataset("data",(numHits,dim0,dim1),dtype=np.float32)
        dset_1.attrs["axes"] = "experiment_identifier:y:x"
        dset_1.attrs["numEvents"] = numHits
        # Soft links
        if "data_1" in entry_1:
            del entry_1["data_1"]
        data_1 = entry_1.create_group("data_1")
        data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
        source_1["experimental_identifier"] = h5py.SoftLink('/entry_1/experimental_identifier')
    elif mode == 'spi':
        if args.saveADU:
            dset_1 = detector_1.create_dataset("data", (numHits, dim0, dim1), dtype=np.float32)  # ,
            # chunks=(1,dim0,dim1),dtype=float)#,
            # compression='gzip',
            # compression_opts=9)
            dset_1.attrs["axes"] = "experiment_identifier:y:x"
            dset_1.attrs["numEvents"] = numHits
        if args.savePhot:
            dset_2 = detector_1.create_dataset("photons", (numHits, dim0, dim1), dtype=int)
            dset_2.attrs["axes"] = "experiment_identifier:y:x"
            dset_2.attrs["numEvents"] = numHits
        # Soft links
        if "data_1" in entry_1:
            del entry_1["data_1"]
        if args.saveADU:
            data_1 = entry_1.create_group("data_1")
            data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
        source_1["experimental_identifier"] = h5py.SoftLink('/entry_1/experimental_identifier')
    #f.flush()
    f.close()

comm.Barrier()

###################################################
# All workers get the to-do list
###################################################

f = h5py.File(filename, "a") #, driver='mpio', comm=MPI.COMM_WORLD)
myJobs = getMyUnfairShare(numHits,size,rank)

myHitInd = hitInd[myJobs]

ds_expId = f.require_dataset("entry_1/experimental_identifier",(numHits,),dtype=int)
ds_photonEnergy_1 = f.require_dataset("LCLS/photon_energy_eV", (numHits,), dtype=float)
ds_photonEnergy = f.require_dataset("entry_1/instrument_1/source_1/energy", (numHits,), dtype=float)
ds_pulseEnergy = f.require_dataset("entry_1/instrument_1/source_1/pulse_energy", (numHits,), dtype=float)
ds_pulseWidth = f.require_dataset("entry_1/instrument_1/source_1/pulse_width", (numHits,), dtype=float)
ds_dist_1 = f.require_dataset("entry_1/instrument_1/detector_1/distance", (numHits,), dtype=float) # in meters
ds_x_pixel_size_1 = f.require_dataset("entry_1/instrument_1/detector_1/x_pixel_size", (numHits,), dtype=float)
ds_y_pixel_size_1 = f.require_dataset("entry_1/instrument_1/detector_1/y_pixel_size", (numHits,), dtype=float)
ds_lclsDet_1 = f.require_dataset("LCLS/detector_1/EncoderValue",(numHits,), dtype=float)
ds_ebeamCharge_1 = f.require_dataset("LCLS/detector_1/electronBeamEnergy",(numHits,), dtype=float)
ds_beamRepRate_1 = f.require_dataset("LCLS/detector_1/beamRepRate",(numHits,), dtype=float)
ds_particleN_electrons_1 = f.require_dataset("LCLS/detector_1/particleN_electrons",(numHits,), dtype=float)
ds_eVernier_1 = f.require_dataset("LCLS/eVernier",(numHits,), dtype=float)
ds_charge_1 = f.require_dataset("LCLS/charge",(numHits,), dtype=float)
ds_peakCurrentAfterSecondBunchCompressor_1 = f.require_dataset("LCLS/peakCurrentAfterSecondBunchCompressor",(numHits,), dtype=float)
ds_pulseLength_1 = f.require_dataset("LCLS/pulseLength",(numHits,), dtype=float)
ds_ebeamEnergyLossConvertedToPhoton_mJ_1 = f.require_dataset("LCLS/ebeamEnergyLossConvertedToPhoton_mJ",(numHits,), dtype=float)
ds_calculatedNumberOfPhotons_1 = f.require_dataset("LCLS/calculatedNumberOfPhotons",(numHits,), dtype=float)
ds_photonBeamEnergy_1 = f.require_dataset("LCLS/photonBeamEnergy",(numHits,), dtype=float)
ds_wavelength_1 = f.require_dataset("LCLS/wavelength",(numHits,), dtype=float)
ds_wavelengthA_1 = f.require_dataset("LCLS/photon_wavelength_A",(numHits,), dtype=float)
ds_sec_1 = f.require_dataset("LCLS/machineTime",(numHits,),dtype=int)
ds_nsec_1 = f.require_dataset("LCLS/machineTimeNanoSeconds",(numHits,),dtype=int)
ds_fid_1 = f.require_dataset("LCLS/fiducial",(numHits,),dtype=int)
ds_evtNum_1 = f.require_dataset("LCLS/eventNumber",(numHits,),dtype=int)
if mode == 'sfx':
    dset_1 = f.require_dataset("entry_1/instrument_1/detector_1/data", (numHits, dim0, dim1),
                               dtype=np.float32)  # ,chunks=(1,dim0,dim1))
    ds_nPeaks = f.require_dataset("/entry_1/result_1/nPeaks", (numHits,), dtype=int)
    ds_posX = f.require_dataset("/entry_1/result_1/peakXPosRaw", (numHits,2048), dtype='float32')#, chunks=(1,2048))
    ds_posY = f.require_dataset("/entry_1/result_1/peakYPosRaw", (numHits,2048), dtype='float32')#, chunks=(1,2048))
    ds_atot = f.require_dataset("/entry_1/result_1/peakTotalIntensity", (numHits,2048), dtype='float32')#, chunks=(1,2048))
    ds_maxRes = f.require_dataset("/entry_1/result_1/maxRes", (numHits,), dtype=int)
elif mode == 'spi':
    if args.saveADU:
        dset_1 = f.require_dataset("entry_1/instrument_1/detector_1/data", (numHits, dim0, dim1),
                                   dtype=np.float32)  # ,chunks=(1,dim0,dim1))
    if args.savePhot:
        dset_2 = f.require_dataset("entry_1/instrument_1/detector_1/photons", (numHits, dim0, dim1),
                                   dtype=int)
    ds_nHits = f.require_dataset("/entry_1/result_1/nHits", (numHits,), dtype=int)

if rank == 0:
    try:
        d = {"message": "#StartCXIDB"}
        writeStatus(statusFname, d)
    except:
        pass

es = ps.ds.env().epicsStore()

for i,val in enumerate(myHitInd):
    t0 = time.time()
    globalInd = myJobs[0]+i
    ds_expId[globalInd] = val
    ps.getEvent(val)
    t1 = time.time()

    ebeamDet = psana.Detector('EBeam')
    t2 = time.time()
    # Write image in cheetah format
    if mode == 'sfx' and 'cspad' in ps.detInfo.lower():
        img = ps.getCheetahImg()
        assert(img is not None)
        dset_1[globalInd,:,:] = img
    elif mode == 'sfx' and 'rayonix' in ps.detInfo.lower():
        img = ps.getCalibImg()
        assert(img is not None)
        dset_1[globalInd,:,:] = img[0,:,:]
    elif mode == 'spi':
        if backgroundThreshMax > -1:
            ind = abs(missInd - val)
            backgroundEvent = missInd[np.argmin(ind)]
            if args.saveADU: img = ps.getCleanAssembledImg(backgroundEvent)
            if args.savePhot: phot = ps.getCleanAssembledPhotons(backgroundEvent)
        else:
            if args.saveADU:
                dset_1[globalInd, :, :] = ps.getAssembledImg()
            if args.savePhot:
                dset_2[globalInd, :, :] = ps.getAssembledPhotons()
    t3 = time.time()

    try:
        pulseLength = es.value('SIOC:SYS0:ML00:AO820')*1e-15 # s
        numPhotons = es.value('SIOC:SYS0:ML00:AO580')*1e12 # number of photons
    except:
        pulseLength = 0
        numPhotons = 0

    ds_pulseWidth[globalInd] = pulseLength
    ds_dist_1[globalInd] = detectorDistance
    ds_x_pixel_size_1[globalInd] = x_pixel_size
    ds_y_pixel_size_1[globalInd] = y_pixel_size
    #f.flush()
    t4 = time.time()

    # LCLS
    if "cxi" in args.exp:
        ds_lclsDet_1[globalInd] = es.value(args.clen) # mm
    elif "mfx" in args.exp:
        ds_lclsDet_1[globalInd] = 0 # FIXME
    elif "xpp" in args.exp:
        ds_lclsDet_1[globalInd] = es.value(args.clen)  # mm

    try:
        ds_ebeamCharge_1[globalInd] = es.value('BEND:DMP1:400:BDES')
    except:
        ds_ebeamCharge_1[globalInd] = 0

    try:
        ds_beamRepRate_1[globalInd] = es.value('EVNT:SYS0:1:LCLSBEAMRATE')
    except:
        ds_beamRepRate_1[globalInd] = 0

    try:
        ds_particleN_electrons_1[globalInd] = es.value('BPMS:DMP1:199:TMIT1H')
    except:
        ds_particleN_electrons_1[globalInd] = 0

    try:
        ds_eVernier_1[globalInd] = es.value('SIOC:SYS0:ML00:AO289')
    except:
        ds_eVernier_1[globalInd] = 0

    try:
        ds_charge_1[globalInd] = es.value('BEAM:LCLS:ELEC:Q')
    except:
        ds_charge_1[globalInd] = 0

    try:
        ds_peakCurrentAfterSecondBunchCompressor_1[globalInd] = es.value('SIOC:SYS0:ML00:AO195')
    except:
        ds_peakCurrentAfterSecondBunchCompressor_1[globalInd] = 0

    try:
        ds_pulseLength_1[globalInd] = es.value('SIOC:SYS0:ML00:AO820')
    except:
        ds_pulseLength_1[globalInd] = 0

    try:
        ds_ebeamEnergyLossConvertedToPhoton_mJ_1[globalInd] = es.value('SIOC:SYS0:ML00:AO569')
    except:
        ds_ebeamEnergyLossConvertedToPhoton_mJ_1[globalInd] = 0

    try:
        ds_calculatedNumberOfPhotons_1[globalInd] = es.value('SIOC:SYS0:ML00:AO580')
    except:
        ds_calculatedNumberOfPhotons_1[globalInd] = 0

    try:
        ds_photonBeamEnergy_1[globalInd] = es.value('SIOC:SYS0:ML00:AO541')
    except:
        ds_photonBeamEnergy_1[globalInd] = 0

    try:
        ds_wavelength_1[globalInd] = es.value('SIOC:SYS0:ML00:AO192')
    except:
        ds_wavelength_1[globalInd] = 0

    try:
        ds_wavelengthA_1[globalInd] = ds_wavelength_1[globalInd] * 10.
    except:
        ds_wavelengthA_1[globalInd] = 0
    t5 = time.time()

    ebeam = ebeamDet.get(ps.evt)  #ebeam = ps.evt.get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
    try:
        ds_photonEnergy[globalInd] = ebeam.ebeamPhotonEnergy()
    except:
        ds_photonEnergy[globalInd] = 0
    try:
        photonEnergy = ebeam.ebeamPhotonEnergy() * 1.60218e-19 # J
        pulseEnergy = ebeam.ebeamL3Energy() # MeV
    except:
        photonEnergy = 0
        pulseEnergy = 0
        if ds_wavelengthA_1[globalInd] > 0:
            h = 6.626070e-34  # J.m
            c = 2.99792458e8  # m/s
            joulesPerEv = 1.602176621e-19  # J/eV
            photonEnergy = (h / joulesPerEv * c) / (ds_wavelengthA_1[globalInd] * 1e-9)
    ds_photonEnergy_1[globalInd] = photonEnergy
    ds_pulseEnergy[globalInd] = pulseEnergy

    #f.flush()
    t6 = time.time()

    evtId = ps.evt.get(psana.EventId)
    sec = evtId.time()[0]
    nsec = evtId.time()[1]
    fid = evtId.fiducials()

    ds_sec_1[globalInd] = sec
    ds_nsec_1[globalInd] = nsec
    ds_fid_1[globalInd] = fid
    ds_evtNum_1[globalInd] = val
    t7 = time.time()

    if mode == 'sfx':
        ds_nPeaks[globalInd] = nPeaks[val]
        ds_posX[globalInd,:] = posX[val,:]
        ds_posY[globalInd,:] = posY[val,:]
        ds_atot[globalInd,:] = atot[val,:]
        ds_maxRes[globalInd] = maxRes[val]
    elif mode == 'spi':
        ds_nHits[globalInd] = nHits[val]
    t8 = time.time()
    #f.flush()
    t9 = time.time()
    if i%100 == 0: print("Rank: "+str(rank)+", Done "+str(i)+" out of "+str(len(myJobs)))

    if rank == 0 and i%10 == 0:
        try:
            hitRate = numHits*100./numEvents
            fracDone = i*100./len(myJobs)
            d = {"numHits": numHits, "hitRate": hitRate, "fracDone": fracDone}
            writeStatus(statusFname, d)
        except:
            pass
    #t10 = time.time()
    #print "hit convert: ", rank, t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7, t9-t8, t10-t9
f.close()
print("DONE: ", comm.rank)

if rank == 0:
    try:
        hitRate = numHits * 100. / numEvents
        fracDone = 100.
        d = {"numHits": numHits, "hitRate": hitRate, "fracDone": fracDone}
        writeStatus(statusFname, d)
    except:
        pass

    f = h5py.File(filename, "a")
    if "/status/xtc2cxidb" in f:
        del f["/status/xtc2cxidb"]
    f["/status/xtc2cxidb"] = 'success'
    # Add attributes
    if mode == 'spi':
        f.attrs["/entry_1/result_1/nHits"] = numHits
    f.close()
    toc = time.time()
    print("time taken: ", toc-tic)

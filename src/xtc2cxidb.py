#!/usr/bin/env python
from xtc2cxidbMaster import runmaster
from xtc2cxidbClient import runclient
import h5py
import psanaWhisperer
import psana
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'At least 2 MPI ranks required'
numClients = size-1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-e","--exp", help="psana experiment name (e.g. cxic0415)", type=str)
parser.add_argument("-r","--run", help="psana run number (e.g. 15)", type=int)
parser.add_argument("-d","--det",help="psana detector name (e.g. DscCsPad)", type=str)
parser.add_argument("-i","--inDir",help="input directory where files_XXXX.lst exists (e.g. /reg/d/psdm/cxi/cxic0415/scratch)", type=str)
parser.add_argument("-o","--outDir",help="output directory (e.g. /reg/d/psdm/cxi/cxic0415/scratch)", type=str)
parser.add_argument("--sample",help="sample name (e.g. lysozyme)",default='', type=str)
parser.add_argument("--instrument",help="instrument name (e.g. CXI)", type=str)
parser.add_argument("--clen", help="camera length epics name (e.g. CXI:DS1:MMS:06.RBV or CXI:DS2:MMS:06.RBV)", type=str)
parser.add_argument("--coffset", help="camera offset, CXI home position to sample (m)",default=0, type=float)
parser.add_argument("--detectorDistance", help="detector distance from interaction point (m)",default=0, type=float)
parser.add_argument("--cxiVersion", help="cxi version",default=140, type=int)
parser.add_argument("--pixelSize", help="pixel size (m)", type=float)
parser.add_argument("--minPeaks", help="Index only if above minimum number of peaks",default=15, type=int)
parser.add_argument("--maxPeaks", help="Index only if below maximum number of peaks",default=300, type=int)
parser.add_argument("--minRes", help="Index only if above minimum resolution",default=0, type=int)
args = parser.parse_args()

# Set up variable
experimentName = args.exp
runNumber = args.run
detInfo = args.det
sampleName = args.sample
instrumentName = args.instrument
coffset = args.coffset
(x_pixel_size,y_pixel_size) = (args.pixelSize,args.pixelSize)

# Set up psana
ps = psanaWhisperer.psanaWhisperer(experimentName,runNumber,detInfo,args)
ps.setupExperiment()

# Read list of files
runStr = "%04d" % args.run
filename = args.inDir + '/' + args.exp + '_' + runStr + '.cxi'
print "Reading file: %s" % (filename)

f = h5py.File(filename, "r")
nPeaks = f["/entry_1/result_1/nPeaksAll"].value
maxRes = f["/entry_1/result_1/maxResAll"].value
posX = f["/entry_1/result_1/peakXPosRawAll"].value
posY = f["/entry_1/result_1/peakYPosRawAll"].value
atot = f["/entry_1/result_1/peakTotalIntensityAll"].value
maxRes = f["/entry_1/result_1/maxResAll"].value
hitInd = ((nPeaks >= args.minPeaks) & (nPeaks <= args.maxPeaks) & (maxRes >= args.minRes)).nonzero()[0]
numHits = len(hitInd)
print "hitInd, numHits: ", hitInd, numHits
f.close()

# Get image shape
firstHit = hitInd[0]
ps.getEvent(firstHit)
img = ps.getCheetahImg()
(dim0, dim1) = img.shape
print "dim0, dim1: ", dim0, dim1

if rank==0:
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

    startTime = ps.getStartTime()
    numEvents = ps.eventTotal
    es = ps.ds.env().epicsStore()
    pulseLength = es.value('SIOC:SYS0:ML00:AO820')*1e-15 # s
    numPhotons = es.value('SIOC:SYS0:ML00:AO580')*1e12 # number of photons
    ebeam = ps.evt.get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
    photonEnergy = ebeam.ebeamPhotonEnergy() * 1.60218e-19 # J
    pulseEnergy = ebeam.ebeamL3Energy() # MeV
    if hasCoffset:
        detectorDistance = coffset + ps.clen*1e-3 # sample to detector in m
    elif hasDetectorDistance:
        detectorDistance = args.detectorDistance

    f = h5py.File(filename, "r+")

    if "/status/xtc2cxidb" in f:
        del f["/status/xtc2cxidb"]
    f["/status/xtc2cxidb"] = 'fail'

    # open the HDF5 CXI file for writing
    if "cxi_version" in f:
        del f["cxi_version"]
    f.create_dataset("cxi_version", data=args.cxiVersion)

    print "LCLS"
    ###################
    # LCLS
    ###################
    if "LCLS" in f:
        del f["LCLS"]
    lcls_1 = f.create_group("LCLS")
    lcls_detector_1 = lcls_1.create_group("detector_1")
    ds_lclsDet_1 = lcls_detector_1.create_dataset("EncoderValue", (numHits,), dtype=float)
    ds_lclsDet_1.attrs["axes"] = "experiment_identifier"
    ds_lclsDet_1.attrs["numEvents"] = numHits
    ds_ebeamCharge_1 = lcls_1.create_dataset("electronBeamEnergy", (numHits,), dtype=float)
    ds_ebeamCharge_1.attrs["axes"] = "experiment_identifier"
    ds_ebeamCharge_1.attrs["numEvents"] = numHits
    ds_beamRepRate_1 = lcls_1.create_dataset("beamRepRate", (numHits,), dtype=float)
    ds_beamRepRate_1.attrs["axes"] = "experiment_identifier"
    ds_beamRepRate_1.attrs["numEvents"] = numHits
    ds_particleN_electrons_1 = lcls_1.create_dataset("particleN_electrons", (numHits,), dtype=float)
    ds_particleN_electrons_1.attrs["axes"] = "experiment_identifier"
    ds_particleN_electrons_1.attrs["numEvents"] = numHits
    ds_eVernier_1 = lcls_1.create_dataset("eVernier", (numHits,), dtype=float)
    ds_eVernier_1.attrs["axes"] = "experiment_identifier"
    ds_eVernier_1.attrs["numEvents"] = numHits
    ds_charge_1 = lcls_1.create_dataset("charge", (numHits,), dtype=float)
    ds_charge_1.attrs["axes"] = "experiment_identifier"
    ds_charge_1.attrs["numEvents"] = numHits
    ds_peakCurrentAfterSecondBunchCompressor_1 = lcls_1.create_dataset("peakCurrentAfterSecondBunchCompressor", (numHits,),
                                                                       dtype=float)
    ds_peakCurrentAfterSecondBunchCompressor_1.attrs["axes"] = "experiment_identifier"
    ds_peakCurrentAfterSecondBunchCompressor_1.attrs["numEvents"] = numHits
    ds_pulseLength_1 = lcls_1.create_dataset("pulseLength", (numHits,), dtype=float)
    ds_pulseLength_1.attrs["axes"] = "experiment_identifier"
    ds_pulseLength_1.attrs["numEvents"] = numHits
    ds_ebeamEnergyLossConvertedToPhoton_mJ_1 = lcls_1.create_dataset("ebeamEnergyLossConvertedToPhoton_mJ", (numHits,),
                                                                     dtype=float)
    ds_ebeamEnergyLossConvertedToPhoton_mJ_1.attrs["axes"] = "experiment_identifier"
    ds_ebeamEnergyLossConvertedToPhoton_mJ_1.attrs["numEvents"] = numHits
    ds_calculatedNumberOfPhotons_1 = lcls_1.create_dataset("calculatedNumberOfPhotons", (numHits,), dtype=float)
    ds_calculatedNumberOfPhotons_1.attrs["axes"] = "experiment_identifier"
    ds_calculatedNumberOfPhotons_1.attrs["numEvents"] = numHits
    ds_photonBeamEnergy_1 = lcls_1.create_dataset("photonBeamEnergy", (numHits,), dtype=float)
    ds_photonBeamEnergy_1.attrs["axes"] = "experiment_identifier"
    ds_photonBeamEnergy_1.attrs["numEvents"] = numHits
    ds_wavelength_1 = lcls_1.create_dataset("wavelength", (numHits,), dtype=float)
    ds_wavelength_1.attrs["axes"] = "experiment_identifier"
    ds_wavelength_1.attrs["numEvents"] = numHits
    ds_sec_1 = lcls_1.create_dataset("machineTime", (numHits,), dtype=int)
    ds_sec_1.attrs["axes"] = "experiment_identifier"
    ds_sec_1.attrs["numEvents"] = numHits
    ds_nsec_1 = lcls_1.create_dataset("machineTimeNanoSeconds", (numHits,), dtype=int)
    ds_nsec_1.attrs["axes"] = "experiment_identifier"
    ds_nsec_1.attrs["numEvents"] = numHits
    ds_fid_1 = lcls_1.create_dataset("fiducial", (numHits,), dtype=int)
    ds_fid_1.attrs["axes"] = "experiment_identifier"
    ds_fid_1.attrs["numEvents"] = numHits
    ds_photonEnergy_1 = lcls_1.create_dataset("photon_energy_eV", (numHits,), dtype=float)  # photon energy in eV
    ds_photonEnergy_1.attrs["axes"] = "experiment_identifier"
    ds_photonEnergy_1.attrs["numEvents"] = numHits
    ds_wavelengthA_1 = lcls_1.create_dataset("photon_wavelength_A", (numHits,), dtype=float)
    ds_wavelengthA_1.attrs["axes"] = "experiment_identifier"
    ds_wavelengthA_1.attrs["numEvents"] = numHits
    #### Datasets not in Cheetah ###
    ds_evtNum_1 = lcls_1.create_dataset("eventNumber", (numHits,), dtype=int)
    ds_evtNum_1.attrs["axes"] = "experiment_identifier"
    ds_evtNum_1.attrs["numEvents"] = numHits

    print "entry_1"
    ###################
    # entry_1
    ###################
    entry_1 = f.require_group("entry_1")

    # dt = h5py.special_dtype(vlen=bytes)
    if "experimental_identifier" in entry_1:
        del entry_1["experimental_identifier"]
    ds_expId = entry_1.create_dataset("experimental_identifier", (numHits,), dtype=int)  # dt)
    ds_expId.attrs["axes"] = "experiment_identifier"
    ds_expId.attrs["numEvents"] = numHits

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
    ds_posX = f.create_dataset("/entry_1/result_1/peakXPosRaw", (numHits, 2048), dtype='float32')  # , chunks=(1,2048))
    ds_posX.attrs["axes"] = "experiment_identifier:peaks"
    ds_posX.attrs["numEvents"] = numHits
    ds_posY = f.create_dataset("/entry_1/result_1/peakYPosRaw", (numHits, 2048), dtype='float32')  # , chunks=(1,2048))
    ds_posY.attrs["axes"] = "experiment_identifier:peaks"
    ds_posY.attrs["numEvents"] = numHits
    ds_atot = f.create_dataset("/entry_1/result_1/peakTotalIntensity", (numHits, 2048),
                               dtype='float32')  # , chunks=(1,2048))
    ds_atot.attrs["axes"] = "experiment_identifier:peaks"
    ds_atot.attrs["numEvents"] = numHits
    ds_maxRes = f.create_dataset("/entry_1/result_1/maxRes", (numHits,), dtype=int)
    ds_maxRes.attrs["axes"] = "experiment_identifier:peaks"
    ds_maxRes.attrs["numEvents"] = numHits

    if "start_time" in entry_1:
        del entry_1["start_time"]
    entry_1.create_dataset("start_time", data=startTime)

    if "sample_1" in entry_1:
        del entry_1["sample_1"]
    sample_1 = entry_1.create_group("sample_1")
    sample_1.create_dataset("name", data=sampleName)

    if "instrument_1" in entry_1:
        del entry_1["instrument_1"]
    instrument_1 = entry_1.create_group("instrument_1")
    instrument_1.create_dataset("name", data=instrumentName)

    source_1 = instrument_1.create_group("source_1")
    ds_photonEnergy = source_1.create_dataset("energy", (numHits,), dtype=float)  # photon energy in J
    ds_photonEnergy.attrs["axes"] = "experiment_identifier"
    ds_photonEnergy.attrs["numEvents"] = numHits
    ds_pulseEnergy = source_1.create_dataset("pulse_energy", (numHits,), dtype=float)  # in J
    ds_pulseEnergy.attrs["axes"] = "experiment_identifier"
    ds_pulseEnergy.attrs["numEvents"] = numHits
    ds_pulseWidth = source_1.create_dataset("pulse_width", (numHits,), dtype=float)  # in s
    ds_pulseWidth.attrs["axes"] = "experiment_identifier"
    ds_pulseWidth.attrs["numEvents"] = numHits

    detector_1 = instrument_1.create_group("detector_1")
    ds_dist_1 = detector_1.create_dataset("distance", (numHits,), dtype=float)  # in meters
    ds_dist_1.attrs["axes"] = "experiment_identifier"
    ds_dist_1.attrs["numEvents"] = numHits
    ds_x_pixel_size_1 = detector_1.create_dataset("x_pixel_size", (numHits,), dtype=float)
    ds_x_pixel_size_1.attrs["axes"] = "experiment_identifier"
    ds_x_pixel_size_1.attrs["numEvents"] = numHits
    ds_y_pixel_size_1 = detector_1.create_dataset("y_pixel_size", (numHits,), dtype=float)
    ds_y_pixel_size_1.attrs["axes"] = "experiment_identifier"
    ds_y_pixel_size_1.attrs["numEvents"] = numHits
    dset_1 = detector_1.create_dataset("data", (numHits, dim0, dim1), dtype=float)
    # chunks=(1,dim0,dim1), dtype=float)#,
    # compression='gzip',
    # compression_opts=9)
    dset_1.attrs["axes"] = "experiment_identifier:y:x"
    dset_1.attrs["numEvents"] = numHits
    detector_1.create_dataset("description", data=detInfo)

    print "Soft links"
    # Soft links
    if "data_1" in entry_1:
        del entry_1["data_1"]
    data_1 = entry_1.create_group("data_1")
    data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
    source_1["experimental_identifier"] = h5py.SoftLink('/entry_1/experimental_identifier')

    f.close()

comm.Barrier()

print "hitInd, numHits: ", rank, hitInd, nPeaks, posX, atot, maxRes, dim0, numHits

#if rank==0:
#    runmaster(args,numClients)
#else:
#    runclient(args)

MPI.Finalize()


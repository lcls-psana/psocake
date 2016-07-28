#!/usr/bin/env python
import h5py
import numpy as np
import psana
import time
import argparse
import os, json

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
parser.add_argument("--coffset", help="camera offset, CXI home position to sample (m)",default=0, type=float)
parser.add_argument("--detectorDistance", help="detector distance from interaction point (m)",default=0, type=float)
parser.add_argument("--cxiVersion", help="cxi version",default=140, type=int)
parser.add_argument("--pixelSize", help="pixel size (m)", type=float)
#parser.add_argument("--condition",help="comparator operation for choosing input data from an hdf5 dataset."
#                                       "Must be double quotes and comma separated for now."
#                                       "Available comparators are: gt(>), ge(>=), eq(==), le(<=), lt(<)"
#                                       "e.g. /particleSize/corrCoef,ge,0.85 ",default='', type=str)
parser.add_argument("--minPeaks", help="Index only if above minimum number of peaks",default=15, type=int)
parser.add_argument("--maxPeaks", help="Index only if below maximum number of peaks",default=300, type=int)
parser.add_argument("--minRes", help="Index only if above minimum resolution",default=0, type=int)
args = parser.parse_args()

def writeStatus(fname,d):
    json.dump(d, open(fname, 'w'))

class psanaWhisperer():
    def __init__(self, experimentName, runNumber, detInfo):
        self.experimentName = experimentName
        self.runNumber = runNumber
        self.detInfo = detInfo

    def setupExperiment(self):
        self.ds = psana.DataSource('exp=' + str(self.experimentName) + ':run=' + str(self.runNumber) + ':idx')
        self.run = self.ds.runs().next()
        self.times = self.run.times()
        self.eventTotal = len(self.times)
        self.env = self.ds.env()
        self.evt = self.run.event(self.times[0])
        self.det = psana.Detector(str(self.detInfo), self.env)
        # Get epics variable, clen
        if "cxi" in self.experimentName:
            self.epics = self.ds.env().epicsStore()
            self.clen = self.epics.value(args.clen)

    def getEvent(self, number):
        self.evt = self.run.event(self.times[number])

    def getCheetahImg(self):
        """Converts seg, row, col assuming (32,185,388)
           to cheetah 2-d table row and col (8*185, 4*388)
        """
        calib = self.det.calib(self.evt) # (32,185,388)
        img = np.zeros((8 * 185, 4 * 388))
        counter = 0
        for quad in range(4):
            for seg in range(8):
                img[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = calib[counter, :, :]
                counter += 1
        return img

    def getPsanaEvent(self, cheetahFilename):
        # Gets psana event given cheetahFilename, e.g. LCLS_2015_Jul26_r0014_035035_e820.h5
        hrsMinSec = cheetahFilename.split('_')[-2]
        fid = int(cheetahFilename.split('_')[-1].split('.')[0], 16)
        for t in ps.times:
            if t.fiducial() == fid:
                localtime = time.strftime('%H:%M:%S', time.localtime(t.seconds()))
                localtime = localtime.replace(':', '')
                if localtime[0:3] == hrsMinSec[0:3]:
                    self.evt = ps.run.event(t)
                else:
                    self.evt = None

    def getStartTime(self):
        self.evt = self.run.event(self.times[0])
        evtId = self.evt.get(psana.EventId)
        sec = evtId.time()[0]
        nsec = evtId.time()[1]
        fid = evtId.fiducials()
        return time.strftime('%FT%H:%M:%S-0800', time.localtime(sec))  # Hard-coded pacific time

# Set up variable
experimentName = args.exp
runNumber = args.run
detInfo = args.det
sampleName = args.sample
instrumentName = args.instrument
coffset = args.coffset
(x_pixel_size,y_pixel_size) = (args.pixelSize,args.pixelSize)

# Set up psana
ps = psanaWhisperer(experimentName,runNumber,detInfo)
ps.setupExperiment()

# Read list of files
runStr = "%04d" % args.run
filename = args.inDir + '/' + args.exp + '_' + runStr + '.cxi'
print "Reading file: %s" % (filename)
statusFname = args.inDir+'/status_index.txt'

if rank == 0:
    try:
        d = {"message", "#CXIDB"}
        print statusFname
        writeStatus(statusFname, d)
    except:
        pass

f = h5py.File(filename, "r")
nPeaks = f["/entry_1/result_1/nPeaksAll"].value
maxRes = f["/entry_1/result_1/maxResAll"].value
posX = f["/entry_1/result_1/peakXPosRawAll"].value
posY = f["/entry_1/result_1/peakYPosRawAll"].value
atot = f["/entry_1/result_1/peakTotalIntensityAll"].value
maxRes = f["/entry_1/result_1/maxResAll"].value
hitInd = ((nPeaks >= args.minPeaks) & (nPeaks <= args.maxPeaks) & (maxRes >= args.minRes)).nonzero()[0]
numHits = len(hitInd)
f.close()

# Get image shape
firstHit = hitInd[0]
ps.getEvent(firstHit)
img = ps.getCheetahImg()
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
#if hasDetectorDistance is False and hasCoffset is False:
#    print "Need at least hasDetectorDistance or coffset input"
#    exit(0)

def getMyUnfairShare(numJobs,numWorkers,rank):
    """Returns number of events assigned to the slave calling this function."""
    assert(numJobs >= numWorkers)
    try:
        allJobs = np.arange(numJobs)
        jobChunks = np.array_split(allJobs,numWorkers)
        myChunk = jobChunks[rank]
        myJobs = allJobs[myChunk[0]:myChunk[-1]+1]
        return myJobs
    except:
        return None

ps = psanaWhisperer(experimentName,runNumber,detInfo)
ps.setupExperiment()
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

if rank == 0:
    try:
        d = {"message", "#InitCXIDB"}
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
    dset_1 = detector_1.create_dataset("data",(numHits,dim0,dim1),dtype=float)#,
                                       #chunks=(1,dim0,dim1),dtype=float)#,
                                       #compression='gzip',
                                       #compression_opts=9)
    dset_1.attrs["axes"] = "experiment_identifier:y:x"
    dset_1.attrs["numEvents"] = numHits
    detector_1.create_dataset("description",data=detInfo)

    # Soft links
    if "data_1" in entry_1:
        del entry_1["data_1"]
    data_1 = entry_1.create_group("data_1")
    data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
    source_1["experimental_identifier"] = h5py.SoftLink('/entry_1/experimental_identifier')

    f.close()

comm.Barrier()
###################################################
# All workers get the to-do list
###################################################

f = h5py.File(filename, "r+", driver='mpio', comm=MPI.COMM_WORLD)
myJobs = getMyUnfairShare(numHits,size,rank)

myHitInd = hitInd[myJobs]

ds_expId = f.require_dataset("entry_1/experimental_identifier",(numHits,),dtype=int)
dset_1 = f.require_dataset("entry_1/instrument_1/detector_1/data",(numHits,dim0,dim1),dtype=float)#,chunks=(1,dim0,dim1))
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
ds_nPeaks = f.require_dataset("/entry_1/result_1/nPeaks", (numHits,), dtype=int)
ds_posX = f.require_dataset("/entry_1/result_1/peakXPosRaw", (numHits,2048), dtype='float32')#, chunks=(1,2048))
ds_posY = f.require_dataset("/entry_1/result_1/peakYPosRaw", (numHits,2048), dtype='float32')#, chunks=(1,2048))
ds_atot = f.require_dataset("/entry_1/result_1/peakTotalIntensity", (numHits,2048), dtype='float32')#, chunks=(1,2048))
ds_maxRes = f.require_dataset("/entry_1/result_1/maxRes", (numHits,), dtype=int)
ds_evtNum_1 = f.require_dataset("LCLS/eventNumber",(numHits,),dtype=int)

if rank == 0:
    try:
        d = {"message", "#StartCXIDB"}
        writeStatus(statusFname, d)
    except:
        pass

for i,val in enumerate(myHitInd):
    globalInd = myJobs[0]+i
    ds_expId[globalInd] = val #cheetahfilename.split("/")[-1].split(".")[0]

    ps.getEvent(val)
    img = ps.getCheetahImg()
    assert(img is not None)
    dset_1[globalInd,:,:] = img

    es = ps.ds.env().epicsStore()
    pulseLength = es.value('SIOC:SYS0:ML00:AO820')*1e-15 # s
    numPhotons = es.value('SIOC:SYS0:ML00:AO580')*1e12 # number of photons
    ebeam = ps.evt.get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
    photonEnergy = ebeam.ebeamPhotonEnergy() * 1.60218e-19 # J
    pulseEnergy = ebeam.ebeamL3Energy() # MeV


    ds_photonEnergy_1[globalInd] = ebeam.ebeamPhotonEnergy()

    ds_photonEnergy[globalInd] = photonEnergy

    ds_pulseEnergy[globalInd] = pulseEnergy

    ds_pulseWidth[globalInd] = pulseLength

    ds_dist_1[globalInd] = detectorDistance

    ds_x_pixel_size_1[globalInd] = x_pixel_size
    ds_y_pixel_size_1[globalInd] = y_pixel_size

    # LCLS
    ds_lclsDet_1[globalInd] = es.value(args.clen) # mm

    ds_ebeamCharge_1[globalInd] = es.value('BEND:DMP1:400:BDES')

    try:
        ds_beamRepRate_1[globalInd] = es.value('EVNT:SYS0:1:LCLSBEAMRATE')
    except:
        ds_beamRepRate_1[globalInd] = 0

    try:
        ds_particleN_electrons_1[globalInd] = es.value('BPMS:DMP1:199:TMIT1H')
    except:
        ds_particleN_electrons_1[globalInd] = 0


    ds_eVernier_1[globalInd] = es.value('SIOC:SYS0:ML00:AO289')

    ds_charge_1[globalInd] = es.value('BEAM:LCLS:ELEC:Q')

    ds_peakCurrentAfterSecondBunchCompressor_1[globalInd] = es.value('SIOC:SYS0:ML00:AO195')

    ds_pulseLength_1[globalInd] = es.value('SIOC:SYS0:ML00:AO820')

    ds_ebeamEnergyLossConvertedToPhoton_mJ_1[globalInd] = es.value('SIOC:SYS0:ML00:AO569')

    ds_calculatedNumberOfPhotons_1[globalInd] = es.value('SIOC:SYS0:ML00:AO580')

    ds_photonBeamEnergy_1[globalInd] = es.value('SIOC:SYS0:ML00:AO541')

    ds_wavelength_1[globalInd] = es.value('SIOC:SYS0:ML00:AO192')

    ds_wavelengthA_1[globalInd] = ds_wavelength_1[globalInd] * 10.

    evtId = ps.evt.get(psana.EventId)
    sec = evtId.time()[0]
    nsec = evtId.time()[1]
    fid = evtId.fiducials()

    ds_sec_1[globalInd] = sec

    ds_nsec_1[globalInd] = nsec

    ds_fid_1[globalInd] = fid

    ds_nPeaks[globalInd] = nPeaks[val]

    ds_posX[globalInd,:] = posX[val,:]

    ds_posY[globalInd,:] = posY[val,:]

    ds_atot[globalInd,:] = atot[val,:]

    ds_maxRes[globalInd] = maxRes[val]

    ds_evtNum_1[globalInd] = val

    if i%20 == 0: print "Rank: "+str(rank)+", Done "+str(i)+" out of "+str(len(myJobs))

    if rank == 0 and i%20 == 0:
        try:
            d = {"convert": "cxidb", "fracDone": i*100./len(myJobs)}
            writeStatus(statusFname, d)
        except:
            pass

f.close()

if rank == 0:
    try:
        d = {"convert": "cxidb", "fracDone": 100.}
        writeStatus(statusFname, d)
    except:
        pass

    f = h5py.File(filename, "r+")
    if "/status/xtc2cxidb" in f:
        del f["/status/xtc2cxidb"]
    f["/status/xtc2cxidb"] = 'success'
    f.close()
    toc = time.time()
    print "time taken: ", toc-tic
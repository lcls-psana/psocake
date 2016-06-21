import os
import h5py
import psana
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from mpidata import mpidata

def runmaster(args,nClients):
    # Initialize hdf5
    inDir = args.indir
    assert os.path.isdir(inDir)
    if args.outdir is None:
        outDir = inDir
    else:
        outDir = args.outdir
        assert os.path.isdir(outDir)

    hasCoffset = False
    hasDetectorDistance = False
    if args.detectorDistance is not 0:
        hasDetectorDistance = True
    if args.coffset is not 0:
        hasCoffset = True

    experimentName = args.exp
    runNumber = args.run
    detInfo = args.det
    sampleName = args.sample
    instrumentName = args.instrument
    coffset = args.coffset
    (x_pixel_size, y_pixel_size) = (args.pixelSize, args.pixelSize)

    ps = psanaWhisperer(experimentName, runNumber, detInfo)
    ps.setupExperiment()
    startTime = ps.getStartTime()
    numEvents = ps.eventTotal
    es = ps.ds.env().epicsStore()
    pulseLength = es.value('SIOC:SYS0:ML00:AO820') * 1e-15  # s
    numPhotons = es.value('SIOC:SYS0:ML00:AO580') * 1e12  # number of photons
    ebeam = ps.evt.get(psana.Bld.BldDataEBeamV7, psana.Source('BldInfo(EBeam)'))
    photonEnergy = ebeam.ebeamPhotonEnergy() * 1.60218e-19  # J
    pulseEnergy = ebeam.ebeamL3Energy()  # MeV
    if hasCoffset:
        detectorDistance = coffset + ps.clen * 1e-3  # sample to detector in m
    elif hasDetectorDistance:
        detectorDistance = args.detectorDistance

    # Read list of files
    runStr = "%04d" % runNumber
    filename = inDir + '/' + experimentName + '_' + runStr + '.cxi'
    print "Reading file: %s" % (filename)

    f = h5py.File(filename, "r+")

    if "/status/xtc2cxidb" in f:
        del f["/status/xtc2cxidb"]
    f["/status/xtc2cxidb"] = 'fail'

    # Condition:
    if args.condition:
        import operator
        operations = {"lt": operator.lt,
                      "le": operator.le,
                      "eq": operator.eq,
                      "ge": operator.ge,
                      "gt": operator.gt,}

        s = args.condition.split(",")
        ds = s[0]  # hdf5 dataset containing metric
        comparator = s[1]  # operation
        cond = float(s[2])  # conditional value
        print "######### ds,comparator,cond: ", ds, comparator, cond

        metric = f[ds].value
        print "metric: ", metric
        hitInd = np.argwhere(operations[comparator](metric, cond))
        print "hitInd", hitInd
        numHits = len(hitInd)
    nPeaks = f["/entry_1/result_1/nPeaksAll"].value
    posX = f["/entry_1/result_1/peakXPosRawAll"].value
    posY = f["/entry_1/result_1/peakYPosRawAll"].value
    atot = f["/entry_1/result_1/peakTotalIntensityAll"].value

    print "start time: ", startTime
    print "number of hits/events: ", numHits, numEvents
    print "pulseLength (s): ", pulseLength
    print "number of photons : ", numPhotons
    print "photon energy (eV,J): ", ebeam.ebeamPhotonEnergy(), photonEnergy
    print "pulse energy (MeV): ", pulseEnergy
    print "detector distance (m): ", detectorDistance

    # Get image shape
    firstHit = hitInd[0]
    ps.getEvent(firstHit)
    img = ps.getCheetahImg()
    (dim0, dim1) = img.shape

    # open the HDF5 CXI file for writing
    if "cxi_version" in f:
        del f["cxi_version"]
    f.create_dataset("cxi_version", data=args.cxiVersion)

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
    ds_peakCurrentAfterSecondBunchCompressor_1 = lcls_1.create_dataset("peakCurrentAfterSecondBunchCompressor",
                                                                       (numHits,), dtype=float)
    ds_peakCurrentAfterSecondBunchCompressor_1.attrs["axes"] = "experiment_identifier"
    ds_peakCurrentAfterSecondBunchCompressor_1.attrs["numEvents"] = numHits
    ds_pulseLength_1 = lcls_1.create_dataset("pulseLength", (numHits,), dtype=float)
    ds_pulseLength_1.attrs["axes"] = "experiment_identifier"
    ds_pulseLength_1.attrs["numEvents"] = numHits
    ds_ebeamEnergyLossConvertedToPhoton_mJ_1 = lcls_1.create_dataset("ebeamEnergyLossConvertedToPhoton_mJ",
                                                                     (numHits,), dtype=float)
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
    ds_nPeaks = f.create_dataset("/entry_1/result_1/nPeaks", (numHits,), dtype=int)
    ds_nPeaks.attrs["axes"] = "experiment_identifier"
    ds_nPeaks.attrs["numEvents"] = numHits
    ds_posX = f.create_dataset("/entry_1/result_1/peakXPosRaw", (numHits, 2048),
                               dtype='float32')  # , chunks=(1,2048))
    ds_posX.attrs["axes"] = "experiment_identifier:peaks"
    ds_posX.attrs["numEvents"] = numHits
    ds_posY = f.create_dataset("/entry_1/result_1/peakYPosRaw", (numHits, 2048),
                               dtype='float32')  # , chunks=(1,2048))
    ds_posY.attrs["axes"] = "experiment_identifier:peaks"
    ds_posY.attrs["numEvents"] = numHits
    ds_atot = f.create_dataset("/entry_1/result_1/peakTotalIntensity", (numHits, 2048),
                               dtype='float32')  # , chunks=(1,2048))
    ds_atot.attrs["axes"] = "experiment_identifier:peaks"
    ds_atot.attrs["numEvents"] = numHits

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
    dset_1 = detector_1.create_dataset("data", (numHits, dim0, dim1), dtype=float)  # ,
    # chunks=(1,dim0,dim1),dtype=float)#,
    # compression='gzip',
    # compression_opts=9)
    dset_1.attrs["axes"] = "experiment_identifier:y:x"
    dset_1.attrs["numEvents"] = numHits
    detector_1.create_dataset("description", data=detInfo)

    # Soft links
    if "data_1" in entry_1:
        del entry_1["data_1"]
    data_1 = entry_1.create_group("data_1")
    data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
    source_1["experimental_identifier"] = h5py.SoftLink('/entry_1/experimental_identifier')

    f.close()

    while nClients > 0:
        # Remove client if the run ended
        md = mpidata()
        md.recv()
        if md.small.endrun:
            nClients -= 1
        else:
            plot(md)

def plot(md):
    print 'Master received image with shape',md.img.shape,'and intensity',md.small.intensity


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
        self.gain = self.det.gain(self.evt)
        # Get epics variable, clen
        if "cxi" in self.experimentName:
            self.epics = self.ds.env().epicsStore()
            self.clen = self.epics.value(args.clen)

    def getEvent(self, number):
        self.evt = self.run.event(self.times[number])

    def getImg(self, number):
        self.getEvent(number)
        img = self.det.image(self.evt, self.det.calib(self.evt) * self.gain)
        return img

    def getImg(self):
        if self.evt is not None:
            img = self.det.image(self.evt, self.det.calib(self.evt) * self.gain)
            return img
        return None

    def getCheetahImg(self):
        """Converts seg, row, col assuming (32,185,388)
           to cheetah 2-d table row and col (8*185, 4*388)
        """
        calib = self.det.calib(self.evt) * self.gain  # (32,185,388)
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
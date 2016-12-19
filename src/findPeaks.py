# Find Bragg peaks
from peakFinderMaster import runmaster
from peakFinderClient import runclient
import psanaWhisperer
import h5py, psana
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'At least 2 MPI ranks required'
numClients = size-1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-e','--exp', help="experiment name (e.g. cxic0415)", type=str)
parser.add_argument('-r','--run', help="run number (e.g. 24)", type=int)
parser.add_argument('-d','--det', help="detector name (e.g. pnccdFront)", type=str)
parser.add_argument('-o','--outDir', help="output directory where .cxi will be saved (e.g. /reg/d/psdm/cxi/cxic0415/scratch)", type=str)
parser.add_argument("-p","--imageProperty",help="determines what preprocessing is done on the image",default=1, type=int)
parser.add_argument("--algorithm",help="number of events to process",default=1, type=int)
parser.add_argument("--alg_npix_min",help="number of events to process",default=1., type=float)
parser.add_argument("--alg_npix_max",help="number of events to process",default=45., type=float)
parser.add_argument("--alg_amax_thr",help="number of events to process",default=250., type=float)
parser.add_argument("--alg_atot_thr",help="number of events to process",default=330., type=float)
parser.add_argument("--alg_son_min",help="number of events to process",default=10., type=float)
parser.add_argument("--alg1_thr_low",help="number of events to process",default=80., type=float)
parser.add_argument("--alg1_thr_high",help="number of events to process",default=270., type=float)
parser.add_argument("--alg1_rank",help="number of events to process",default=3, type=int)
parser.add_argument("--alg1_radius",help="number of events to process",default=3, type=int)
parser.add_argument("--alg1_dr",help="number of events to process",default=1., type=float)
# parser.add_argument("--alg3_rank",help="number of events to process",default=3, type=int)
# parser.add_argument("--alg3_r0",help="number of events to process",default=5., type=float)
# parser.add_argument("--alg3_dr",help="number of events to process",default=0.05, type=float)
# parser.add_argument("--alg4_thr_low",help="number of events to process",default=10., type=float)
# parser.add_argument("--alg4_thr_high",help="number of events to process",default=150., type=float)
# parser.add_argument("--alg4_rank",help="number of events to process",default=3, type=int)
# parser.add_argument("--alg4_r0",help="number of events to process",default=5, type=int)
# parser.add_argument("--alg4_dr",help="number of events to process",default=0.05, type=float)
parser.add_argument("--streakMask_on",help="streak mask on",default="False", type=str)
parser.add_argument("--streakMask_sigma",help="streak mask sigma above background",default=0., type=float)
parser.add_argument("--streakMask_width",help="streak mask width",default=0, type=float)
parser.add_argument("--userMask_path",help="full path to user mask numpy array",default=None, type=str)
parser.add_argument("--psanaMask_on",help="psana mask on",default="False", type=str)
parser.add_argument("--psanaMask_calib",help="psana calib on",default="False", type=str)
parser.add_argument("--psanaMask_status",help="psana status on",default="False", type=str)
parser.add_argument("--psanaMask_edges",help="psana edges on",default="False", type=str)
parser.add_argument("--psanaMask_central",help="psana central on",default="False", type=str)
parser.add_argument("--psanaMask_unbond",help="psana unbonded pixels on",default="False", type=str)
parser.add_argument("--psanaMask_unbondnrs",help="psana unbonded pixel neighbors on",default="False", type=str)
parser.add_argument("--mask",help="static mask",default='', type=str)
#parser.add_argument("-m","--maxNumPeaks",help="maximum number of peaks to store per event",default=2048, type=int)
parser.add_argument("-n","--noe",help="number of events to process",default=-1, type=int)
parser.add_argument("--medianBackground",help="subtract median background",default=0, type=int)
parser.add_argument("--medianRank",help="median background window size",default=0, type=int)
parser.add_argument("--radialBackground",help="subtract radial background",default=0, type=int)
#parser.add_argument("--distance",help="detector distance used for radial background",default=0, type=float)
parser.add_argument("--sample",help="sample name (e.g. lysozyme)",default='', type=str)
parser.add_argument("--instrument",help="instrument name (e.g. CXI)", type=str)
parser.add_argument("--clen", help="camera length epics name (e.g. CXI:DS1:MMS:06.RBV or CXI:DS2:MMS:06.RBV)", type=str)
parser.add_argument("--coffset", help="camera offset, CXI home position to sample (m)", default=0, type=float)
parser.add_argument("--detectorDistance", help="detector distance from interaction point (m)", default=0, type=float)
parser.add_argument("--pixelSize",help="pixel size",default=0, type=float)
parser.add_argument("--minPeaks", help="Index only if above minimum number of peaks",default=15, type=int)
parser.add_argument("--maxPeaks", help="Index only if below maximum number of peaks",default=2048, type=int)
parser.add_argument("--minRes", help="Index only if above minimum resolution",default=0, type=int)
parser.add_argument("--localCalib", help="Use local calib directory. A calib directory must exist in your current working directory.", action='store_true')
parser.add_argument("--profile", help="Turn on profiling. Saves timing information for calibration, peak finding, and saving to hdf5", action='store_true')
parser.add_argument("--cxiVersion", help="cxi version",default=140, type=int)
args = parser.parse_args()

def getNoe(args):
    runStr = "%04d" % args.run
    ds = psana.DataSource("exp="+args.exp+":run="+runStr+':idx')
    run = ds.runs().next()
    times = run.times()
    # check if the user requested specific number of events
    if args.noe == -1:
        numJobs = len(times)
    else:
        if args.noe <= len(times):
            numJobs = args.noe
        else:
            numJobs = len(times)
    return numJobs

if args.localCalib: psana.setOption('psana.calib-dir','./calib')

if rank == 0:
    # Set up psana
    ps = psanaWhisperer.psanaWhisperer(args.exp, args.run, args.det, args.clen, args.localCalib)
    ps.setupExperiment()
    img = ps.getCheetahImg()
    (dim0, dim1) = img.shape
    numEvents = ps.eventTotal

    runStr = "%04d" % args.run
    fname = args.outDir +"/"+ args.exp +"_"+ runStr + ".cxi"
    # Get number of events to process
    numJobs = getNoe(args)

    # Create hdf5 and save psana input
    myHdf5 = h5py.File(fname, 'w')
    myHdf5['/status/findPeaks'] = 'fail'
    dt = h5py.special_dtype(vlen=bytes)
    myInput = ""
    for key,value in vars(args).iteritems():
        myInput += key
        myInput += " "
        myInput += str(value)
        myInput += "\n"
    dset = myHdf5.create_dataset("/psana/input",(1,), dtype=dt)
    dset[...] = myInput
    myHdf5.flush()

    myHdf5.create_dataset("cxi_version", data=args.cxiVersion)
    myHdf5.flush()

    dt = h5py.special_dtype(vlen=np.float)

    ###################
    # LCLS
    ###################
    lcls_1 = myHdf5.create_group("LCLS")
    lcls_detector_1 = lcls_1.create_group("detector_1")
    ds_lclsDet_1 = lcls_detector_1.create_dataset("EncoderValue",(0,),
                                                  maxshape=(None,),
                                                  dtype=float)
    ds_lclsDet_1.attrs["axes"] = "experiment_identifier"

    ds_ebeamCharge_1 = lcls_detector_1.create_dataset("electronBeamEnergy",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_ebeamCharge_1.attrs["axes"] = "experiment_identifier"

    ds_beamRepRate_1 = lcls_detector_1.create_dataset("beamRepRate",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_beamRepRate_1.attrs["axes"] = "experiment_identifier"

    ds_particleN_electrons_1 = lcls_detector_1.create_dataset("particleN_electrons",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_particleN_electrons_1.attrs["axes"] = "experiment_identifier"

    ds_eVernier_1 = lcls_1.create_dataset("eVernier",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_eVernier_1.attrs["axes"] = "experiment_identifier"

    ds_charge_1 = lcls_1.create_dataset("charge",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_charge_1.attrs["axes"] = "experiment_identifier"

    ds_peakCurrentAfterSecondBunchCompressor_1 = lcls_1.create_dataset("peakCurrentAfterSecondBunchCompressor",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_peakCurrentAfterSecondBunchCompressor_1.attrs["axes"] = "experiment_identifier"

    ds_pulseLength_1 = lcls_1.create_dataset("pulseLength",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_pulseLength_1.attrs["axes"] = "experiment_identifier"

    ds_ebeamEnergyLossConvertedToPhoton_mJ_1 = lcls_1.create_dataset("ebeamEnergyLossConvertedToPhoton_mJ",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_ebeamEnergyLossConvertedToPhoton_mJ_1.attrs["axes"] = "experiment_identifier"

    ds_calculatedNumberOfPhotons_1 = lcls_1.create_dataset("calculatedNumberOfPhotons",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_calculatedNumberOfPhotons_1.attrs["axes"] = "experiment_identifier"

    ds_photonBeamEnergy_1 = lcls_1.create_dataset("photonBeamEnergy",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_photonBeamEnergy_1.attrs["axes"] = "experiment_identifier"

    ds_wavelength_1 = lcls_1.create_dataset("wavelength",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_wavelength_1.attrs["axes"] = "experiment_identifier"

    ds_sec_1 = lcls_1.create_dataset("machineTime",(0,),
                                             maxshape=(None,),
                                             dtype=int)
    ds_sec_1.attrs["axes"] = "experiment_identifier"

    ds_nsec_1 = lcls_1.create_dataset("machineTimeNanoSeconds",(0,),
                                             maxshape=(None,),
                                             dtype=int)
    ds_nsec_1.attrs["axes"] = "experiment_identifier"

    ds_fid_1 = lcls_1.create_dataset("fiducial",(0,),
                                             maxshape=(None,),
                                             dtype=int)
    ds_fid_1.attrs["axes"] = "experiment_identifier"

    ds_photonEnergy_1 = lcls_1.create_dataset("photon_energy_eV",(0,),
                                             maxshape=(None,),
                                             dtype=float) # photon energy in eV
    ds_photonEnergy_1.attrs["axes"] = "experiment_identifier"

    ds_wavelengthA_1 = lcls_1.create_dataset("photon_wavelength_A",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_wavelengthA_1.attrs["axes"] = "experiment_identifier"

    #### Datasets not in Cheetah ###
    ds_evtNum_1 = lcls_1.create_dataset("eventNumber",(0,),
                                             maxshape=(None,),
                                             dtype=int)
    ds_evtNum_1.attrs["axes"] = "experiment_identifier"

    myHdf5.flush()

    ###################
    # entry_1
    ###################
    entry_1 = myHdf5.create_group("entry_1")
    ds_expId = entry_1.create_dataset("experimental_identifier",(0,),
                                             maxshape=(None,),
                                             dtype=int)
    ds_expId.attrs["axes"] = "experiment_identifier"

    myHdf5.create_dataset("/entry_1/result_1/nPeaksAll", data=np.ones(numJobs,)*-1, dtype=int)
    myHdf5.create_dataset("/entry_1/result_1/peakXPosRawAll", (numJobs,2048), dtype=float, chunks=(1,2048))
    myHdf5.create_dataset("/entry_1/result_1/peakYPosRawAll", (numJobs,2048), dtype=float, chunks=(1,2048))
    myHdf5.create_dataset("/entry_1/result_1/peakTotalIntensityAll", (numJobs,2048), dtype=float, chunks=(1,2048))
    myHdf5.create_dataset("/entry_1/result_1/maxResAll", data=np.ones(numJobs,)*-1, dtype=int)
    myHdf5.flush()

    if args.profile:
        myHdf5.create_dataset("/entry_1/result_1/calibTime", data=np.zeros(numJobs, ), dtype=float)
        myHdf5.create_dataset("/entry_1/result_1/peakTime", data=np.zeros(numJobs, ), dtype=float)
        myHdf5.create_dataset("/entry_1/result_1/saveTime", data=np.zeros(numJobs, ), dtype=float)
        myHdf5.create_dataset("/entry_1/result_1/reshapeTime", (0,), maxshape=(None,), dtype=float)
        myHdf5.create_dataset("/entry_1/result_1/totalTime", data=np.zeros(numJobs, ), dtype=float)
        myHdf5.create_dataset("/entry_1/result_1/rankID", data=np.zeros(numJobs, ), dtype=int)
        myHdf5.flush()

    ds_nPeaks = myHdf5.create_dataset("/entry_1/result_1/nPeaks",(0,),
                                             maxshape=(None,),
                                             dtype=int)
    ds_nPeaks.attrs["axes"] = "experiment_identifier"

    ds_nPeaks.attrs["minPeaks"] = args.minPeaks
    ds_nPeaks.attrs["maxPeaks"] = args.maxPeaks
    ds_nPeaks.attrs["minRes"] = args.minRes
    ds_posX = myHdf5.create_dataset("/entry_1/result_1/peakXPosRaw",(0,2048),
                                             maxshape=(None,2048),
                                             chunks = (1, 2048),
                                             compression='gzip',
                                             compression_opts=1,
                                             dtype=float)
    ds_posX.attrs["axes"] = "experiment_identifier:peaks"

    ds_posY = myHdf5.create_dataset("/entry_1/result_1/peakYPosRaw",(0,2048),
                                             maxshape=(None,2048),
                                             chunks=(1, 2048),
                                             compression='gzip',
                                             compression_opts=1,
                                             dtype=float)
    ds_posY.attrs["axes"] = "experiment_identifier:peaks"

    ds_atot = myHdf5.create_dataset("/entry_1/result_1/peakTotalIntensity",(0,2048),
                                             maxshape=(None,2048),
                                             chunks=(1, 2048),
                                             compression='gzip',
                                             compression_opts=1,
                                             dtype=float)
    ds_atot.attrs["axes"] = "experiment_identifier:peaks"

    ds_maxRes = myHdf5.create_dataset("/entry_1/result_1/maxRes",(0,),
                                             maxshape=(None,),
                                             dtype=int)
    ds_maxRes.attrs["axes"] = "experiment_identifier:peaks"

    myHdf5.flush()

    entry_1.create_dataset("start_time",data=ps.getStartTime())
    myHdf5.flush()

    sample_1 = entry_1.create_group("sample_1")
    sample_1.create_dataset("name",data=args.sample)
    myHdf5.flush()

    instrument_1 = entry_1.create_group("instrument_1")
    instrument_1.create_dataset("name", data=args.instrument)
    myHdf5.flush()

    source_1 = instrument_1.create_group("source_1")
    ds_photonEnergy = source_1.create_dataset("energy",(0,),
                                             maxshape=(None,),
                                             dtype=float) # photon energy in J
    ds_photonEnergy.attrs["axes"] = "experiment_identifier"

    ds_pulseEnergy = source_1.create_dataset("pulse_energy",(0,),
                                             maxshape=(None,),
                                             dtype=float) # in J
    ds_pulseEnergy.attrs["axes"] = "experiment_identifier"

    ds_pulseWidth = source_1.create_dataset("pulse_width",(0,),
                                             maxshape=(None,),
                                             dtype=float) # in s
    ds_pulseWidth.attrs["axes"] = "experiment_identifier"

    myHdf5.flush()

    detector_1 = instrument_1.create_group("detector_1")
    ds_data_1 = detector_1.create_dataset("data", (0, dim0, dim1),
                                chunks=(1, dim0, dim1),
                                maxshape=(None, dim0, dim1),
                                compression='gzip',
                                compression_opts=1,
                                dtype=float)
    ds_data_1.attrs["axes"] = "experiment_identifier"

    data_1 = entry_1.create_group("data_1")
    data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')

    # Add x,y,z coordinates
    cx, cy, cz = ps.det.coords_xyz(ps.evt)
    data_1["x"] = ps.getCheetahImg(calib=cx)
    data_1["y"] = ps.getCheetahImg(calib=cy)
    data_1["z"] = ps.getCheetahImg(calib=cz)

    # Add mask in cheetah format
    if args.mask is not None:
        f = h5py.File(args.mask,'r')
        mask = f['/entry_1/data_1/mask'].value
        data_1["mask"] = mask
        f.close()

    ds_dist_1 = detector_1.create_dataset("distance",(0,),
                                             maxshape=(None,),
                                             dtype=float) # in meters
    ds_dist_1.attrs["axes"] = "experiment_identifier"

    ds_x_pixel_size_1 = detector_1.create_dataset("x_pixel_size",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_x_pixel_size_1.attrs["axes"] = "experiment_identifier"

    ds_y_pixel_size_1 = detector_1.create_dataset("y_pixel_size",(0,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_y_pixel_size_1.attrs["axes"] = "experiment_identifier"

    detector_1.create_dataset("description",data=args.det)
    myHdf5.flush()

    myHdf5.close()

comm.Barrier()

if rank==0:
    runmaster(args,numClients)
else:
    runclient(args)

MPI.Finalize()

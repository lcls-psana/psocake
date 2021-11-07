# Find Bragg peaks
import h5py
import numpy as np
from mpi4py import MPI
import os, time
import psana
from psocake.peakFinderClientSlim import runclient
from psocake.utils import *
from psocake import cheetahUtils

facility = 'LCLS'

tic = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'At least 2 MPI ranks required'

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
parser.add_argument("-n","--noe",help="number of events to process",default=-1, type=int)
parser.add_argument("--medianBackground",help="subtract median background",default=0, type=int)
parser.add_argument("--medianRank",help="median background window size",default=0, type=int)
parser.add_argument("--radialBackground",help="subtract radial background",default=0, type=int)
parser.add_argument("--sample",help="sample name (e.g. lysozyme)",default='', type=str)
parser.add_argument("--instrument",help="instrument name (e.g. CXI)", default=None, type=str)
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
parser.add_argument("--auto", help="automatically determine peak finding parameter per event", default="False", type=str)
# LCLS specific
parser.add_argument("-a","--access", help="Set data node access: {ana,ffb}",default="ana", type=str)
parser.add_argument("-t","--tag", help="Set tag for cxi filename",default="", type=str)
parser.add_argument("-i","--inputImages", default="", type=str, help="full path to hdf5 file with calibrated CsPad images saved as /data/data and /eventNumber. It can be in a cheetah format (3D) or psana unassembled format (4D)")
parser.add_argument("--cm0", help="Psana common mode correction parameter 0",default=0, type=int)
parser.add_argument("--cm1", help="Psana common mode correction parameter 1",default=0, type=int)
parser.add_argument("--cm2", help="Psana common mode correction parameter 2",default=0, type=int)
parser.add_argument("--cm3", help="Psana common mode correction parameter 3",default=0, type=int)
args = parser.parse_args()
if args.localCalib: psana.setOption('psana.calib-dir','./calib')

# Get number of events to process all together
runStr = "%04d" % args.run
access = "exp="+args.exp+":run="+runStr+':idx'
if 'ffb' in args.access.lower(): access += ':dir=/cds/data/drpsrcf/' + args.exp[:3] + '/' + args.exp + '/xtc'
ds = psana.DataSource(access)
run = next(ds.runs())
times = run.times()
env = ds.env()
det = psana.Detector(args.det)
det.do_reshape_2d_to_3d(flag=True)
detPsocake = cheetahUtils.SupportedDetectors().parseDetectorName(args.det)
detDesc = getattr(cheetahUtils, detPsocake)() # instantiate detector descriptor class
(dim0, dim1) = detDesc.tileDim

numJobs = 0
# check if the user requested specific number of events
if args.noe == -1:
    numJobs = len(times)
else:
    if args.noe <= len(times):
        numJobs = args.noe
    else:
        numJobs = len(times)

if rank == 0:
    # create a combined mask
    if args.psanaMask_on:
        combinedMask = det.mask(run, calib=True, status=True, edges=True, central=True, unbond=True, unbondnbrs=False)
    else:
        combinedMask = np.ones_like(
            det.mask(run, calib=True, status=True, edges=True, central=True, unbond=True, unbondnbrs=False))
    # pick up manual mask
    manualMaskFname = args.outDir + "/mask.npy"
    if os.path.exists(manualMaskFname):
        combinedMask = combinedMask * np.load(manualMaskFname)
    # outDir already contains runStr, so set run = None
    cheetahUtils.saveCheetahFormatMask(args.outDir, detDesc, None, combinedMask)

def createCxi(fname):
    # Create hdf5 and save psana input
    myHdf5 = h5py.File(fname, 'w')
    myHdf5['/status/findPeaks'] = 'fail'
    # Save user input arguments
    dt = h5py.special_dtype(vlen=bytes)
    myInput = ""
    for key,value in vars(args).items():
        myInput += key
        myInput += " "
        myInput += str(value)
        myInput += "\n"
    dset = myHdf5.create_dataset("/psocake/input",(1,), dtype=dt)
    dset[...] = myInput
    #myHdf5.flush()

    myHdf5.create_dataset("cxi_version", data=args.cxiVersion)
    #myHdf5.flush()

    dt = h5py.special_dtype(vlen=np.float)
    dti = h5py.special_dtype(vlen=np.dtype('int32'))

    ###################
    # LCLS
    ###################
    lcls_1 = myHdf5.create_group("LCLS")
    lcls_detector_1 = lcls_1.create_group("detector_1")
    ds_lclsDet_1 = lcls_detector_1.create_dataset("EncoderValue",(numJobs,),
                                                  maxshape=(None,),
                                                  dtype=float)
    ds_lclsDet_1.attrs["axes"] = "experiment_identifier"

    ds_ebeamCharge_1 = lcls_detector_1.create_dataset("electronBeamEnergy",(numJobs,),
                                                      maxshape=(None,),
                                                      dtype=float)
    ds_ebeamCharge_1.attrs["axes"] = "experiment_identifier"

    ds_beamRepRate_1 = lcls_detector_1.create_dataset("beamRepRate",(numJobs,),
                                                      maxshape=(None,),
                                                      dtype=float)
    ds_beamRepRate_1.attrs["axes"] = "experiment_identifier"

    ds_particleN_electrons_1 = lcls_detector_1.create_dataset("particleN_electrons",(numJobs,),
                                                              maxshape=(None,),
                                                              dtype=float)
    ds_particleN_electrons_1.attrs["axes"] = "experiment_identifier"

    ds_eVernier_1 = lcls_1.create_dataset("eVernier",(numJobs,),
                                          maxshape=(None,),
                                          dtype=float)
    ds_eVernier_1.attrs["axes"] = "experiment_identifier"

    ds_charge_1 = lcls_1.create_dataset("charge",(numJobs,),
                                        maxshape=(None,),
                                        dtype=float)
    ds_charge_1.attrs["axes"] = "experiment_identifier"

    ds_peakCurrentAfterSecondBunchCompressor_1 = lcls_1.create_dataset("peakCurrentAfterSecondBunchCompressor",(numJobs,),
                                                                       maxshape=(None,),
                                                                       dtype=float)
    ds_peakCurrentAfterSecondBunchCompressor_1.attrs["axes"] = "experiment_identifier"

    ds_pulseLength_1 = lcls_1.create_dataset("pulseLength",(numJobs,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_pulseLength_1.attrs["axes"] = "experiment_identifier"

    ds_ebeamEnergyLossConvertedToPhoton_mJ_1 = lcls_1.create_dataset("ebeamEnergyLossConvertedToPhoton_mJ",(numJobs,),
                                                                     maxshape=(None,),
                                                                     dtype=float)
    ds_ebeamEnergyLossConvertedToPhoton_mJ_1.attrs["axes"] = "experiment_identifier"

    ds_calculatedNumberOfPhotons_1 = lcls_1.create_dataset("calculatedNumberOfPhotons",(numJobs,),
                                                           maxshape=(None,),
                                                           dtype=float)
    ds_calculatedNumberOfPhotons_1.attrs["axes"] = "experiment_identifier"

    ds_photonBeamEnergy_1 = lcls_1.create_dataset("photonBeamEnergy",(numJobs,),
                                                  maxshape=(None,),
                                                  dtype=float)
    ds_photonBeamEnergy_1.attrs["axes"] = "experiment_identifier"

    ds_wavelength_1 = lcls_1.create_dataset("wavelength",(numJobs,),
                                            maxshape=(None,),
                                            dtype=float)
    ds_wavelength_1.attrs["axes"] = "experiment_identifier"

    ds_sec_1 = lcls_1.create_dataset("machineTime",(numJobs,),
                                     maxshape=(None,),
                                     dtype=int)
    ds_sec_1.attrs["axes"] = "experiment_identifier"

    ds_nsec_1 = lcls_1.create_dataset("machineTimeNanoSeconds",(numJobs,),
                                      maxshape=(None,),
                                      dtype=int)
    ds_nsec_1.attrs["axes"] = "experiment_identifier"

    ds_fid_1 = lcls_1.create_dataset("fiducial",(numJobs,),
                                     maxshape=(None,),
                                     dtype=int)
    ds_fid_1.attrs["axes"] = "experiment_identifier"

    ds_photonEnergy_1 = lcls_1.create_dataset("photon_energy_eV",(numJobs,),
                                              maxshape=(None,),
                                              dtype=float) # photon energy in eV
    ds_photonEnergy_1.attrs["axes"] = "experiment_identifier"

    ds_wavelengthA_1 = lcls_1.create_dataset("photon_wavelength_A",(numJobs,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_wavelengthA_1.attrs["axes"] = "experiment_identifier"

    #### Datasets not in Cheetah ###
    ds_evtNum_1 = lcls_1.create_dataset("eventNumber",(numJobs,),
                                        maxshape=(None,),
                                        dtype=int)
    ds_evtNum_1.attrs["axes"] = "experiment_identifier"

    ds_evr0_1 = lcls_detector_1.create_dataset("evr0",(numJobs,),
                                               maxshape=(None,),
                                               dtype=dti)
    ds_evr0_1.attrs["axes"] = "experiment_identifier"

    ds_evr1_1 = lcls_detector_1.create_dataset("evr1",(numJobs,),
                                               maxshape=(None,),
                                               dtype=dti)
    ds_evr1_1.attrs["axes"] = "experiment_identifier"

    ds_evr2_1 = lcls_detector_1.create_dataset("evr2",(numJobs,),
                                               maxshape=(None,),
                                               dtype=dti)
    ds_evr2_1.attrs["axes"] = "experiment_identifier"

    ds_ttspecAmpl_1 = lcls_1.create_dataset("ttspecAmpl",(numJobs,),
                                            maxshape=(None,),
                                            dtype=float)
    ds_ttspecAmpl_1.attrs["axes"] = "experiment_identifier"

    ds_ttspecAmplNxt_1 = lcls_1.create_dataset("ttspecAmplNxt",(numJobs,),
                                               maxshape=(None,),
                                               dtype=float)
    ds_ttspecAmplNxt_1.attrs["axes"] = "experiment_identifier"

    ds_ttspecFltpos_1 = lcls_1.create_dataset("ttspecFltPos",(numJobs,),
                                              maxshape=(None,),
                                              dtype=float)
    ds_ttspecFltpos_1.attrs["axes"] = "experiment_identifier"

    ds_ttspecFltposFwhm_1 = lcls_1.create_dataset("ttspecFltPosFwhm",(numJobs,),
                                                  maxshape=(None,),
                                                  dtype=float)
    ds_ttspecFltposFwhm_1.attrs["axes"] = "experiment_identifier"

    ds_ttspecFltposPs_1 = lcls_1.create_dataset("ttspecFltPosPs",(numJobs,),
                                                maxshape=(None,),
                                                dtype=float)
    ds_ttspecFltposPs_1.attrs["axes"] = "experiment_identifier"

    ds_ttspecRefAmpl_1 = lcls_1.create_dataset("ttspecRefAmpl",(numJobs,),
                                               maxshape=(None,),
                                               dtype=float)
    ds_ttspecRefAmpl_1.attrs["axes"] = "experiment_identifier"

    lcls_injector_1 = lcls_1.create_group("injector_1")
    ds_pressure_1 = lcls_injector_1.create_dataset("pressureSDS",(numJobs,),
                                                   maxshape=(None,),
                                                   dtype=float)
    ds_pressure_1.attrs["axes"] = "experiment_identifier"
    ds_pressure_2 = lcls_injector_1.create_dataset("pressureSDSB",(numJobs,),
                                                   maxshape=(None,),
                                                   dtype=float)
    ds_pressure_2.attrs["axes"] = "experiment_identifier"

    #myHdf5.flush()

    ###################
    # entry_1
    ###################
    entry_1 = myHdf5.create_group("entry_1")
    ds_expId = entry_1.create_dataset("experimental_identifier",(numJobs,),
                                      maxshape=(None,),
                                      dtype=int)
    ds_expId.attrs["axes"] = "experiment_identifier"

    myHdf5.create_dataset("/entry_1/result_1/nPeaksAll", data=np.ones(numJobs,)*-1, dtype=int)
    myHdf5.create_dataset("/entry_1/result_1/peakXPosRawAll", (numJobs,args.maxPeaks), dtype=float, chunks=(1,args.maxPeaks))
    myHdf5.create_dataset("/entry_1/result_1/peakYPosRawAll", (numJobs,args.maxPeaks), dtype=float, chunks=(1,args.maxPeaks))
    myHdf5.create_dataset("/entry_1/result_1/peakTotalIntensityAll", (numJobs,args.maxPeaks), dtype=float, chunks=(1,args.maxPeaks))
    myHdf5.create_dataset("/entry_1/result_1/peakMaxIntensityAll", (numJobs,args.maxPeaks), dtype=float, chunks=(1,args.maxPeaks))
    myHdf5.create_dataset("/entry_1/result_1/peakRadiusAll", (numJobs,args.maxPeaks), dtype=float, chunks=(1,args.maxPeaks))
    myHdf5.create_dataset("/entry_1/result_1/maxResAll", data=np.ones(numJobs,)*-1, dtype=int)
    myHdf5.create_dataset("/entry_1/result_1/likelihoodAll", data=np.ones(numJobs, ) * -1, dtype=float)

    myHdf5.create_dataset("/entry_1/result_1/timeToolDelayAll", data=np.ones(numJobs, ) * -1, dtype=float)
    myHdf5.create_dataset("/entry_1/result_1/laserTimeZeroAll", data=np.ones(numJobs, ) * -1, dtype=float)
    myHdf5.create_dataset("/entry_1/result_1/laserTimeDelayAll", data=np.ones(numJobs, ) * -1, dtype=float)
    myHdf5.create_dataset("/entry_1/result_1/laserTimePhaseLockedAll", data=np.ones(numJobs, ) * -1, dtype=float)
    #myHdf5.flush()

    if args.profile:
        myHdf5.create_dataset("/entry_1/result_1/calibTime", data=np.zeros(numJobs, ), dtype=float)
        myHdf5.create_dataset("/entry_1/result_1/peakTime", data=np.zeros(numJobs, ), dtype=float)
        myHdf5.create_dataset("/entry_1/result_1/saveTime", data=np.zeros(numJobs, ), dtype=float)
        myHdf5.create_dataset("/entry_1/result_1/reshapeTime", (0,), maxshape=(None,), dtype=float)
        myHdf5.create_dataset("/entry_1/result_1/totalTime", data=np.zeros(numJobs, ), dtype=float)
        myHdf5.create_dataset("/entry_1/result_1/rankID", data=np.zeros(numJobs, ), dtype=int)
        #myHdf5.flush()

    ds_nPeaks = myHdf5.create_dataset("/entry_1/result_1/nPeaks",(numJobs,),
                                      maxshape=(None,),
                                      dtype=int)
    ds_nPeaks.attrs["axes"] = "experiment_identifier"

    ds_nPeaks.attrs["minPeaks"] = args.minPeaks
    ds_nPeaks.attrs["maxPeaks"] = args.maxPeaks
    ds_nPeaks.attrs["minRes"] = args.minRes
    ds_posX = myHdf5.create_dataset("/entry_1/result_1/peakXPosRaw",(numJobs,args.maxPeaks),
                                    maxshape=(None,args.maxPeaks),
                                    chunks = (1, args.maxPeaks),
                                    dtype=float)
    ds_posX.attrs["axes"] = "experiment_identifier:peaks"

    ds_posY = myHdf5.create_dataset("/entry_1/result_1/peakYPosRaw",(numJobs,args.maxPeaks),
                                    maxshape=(None,args.maxPeaks),
                                    chunks=(1, args.maxPeaks),
                                    dtype=float)
    ds_posY.attrs["axes"] = "experiment_identifier:peaks"

    ds_rcent = myHdf5.create_dataset("/entry_1/result_1/rcent", (numJobs, args.maxPeaks),
                                     maxshape=(None, args.maxPeaks),
                                     chunks=(1, args.maxPeaks),
                                     dtype=float)
    ds_rcent.attrs["axes"] = "experiment_identifier:peaks"

    ds_ccent = myHdf5.create_dataset("/entry_1/result_1/ccent", (numJobs, args.maxPeaks),
                                     maxshape=(None, args.maxPeaks),
                                     chunks=(1, args.maxPeaks),
                                     dtype=float)
    ds_ccent.attrs["axes"] = "experiment_identifier:peaks"

    ds_rmin = myHdf5.create_dataset("/entry_1/result_1/rmin", (numJobs, args.maxPeaks),
                                     maxshape=(None, args.maxPeaks),
                                     chunks=(1, args.maxPeaks),
                                     dtype=float)
    ds_rmin.attrs["axes"] = "experiment_identifier:peaks"

    ds_rmax = myHdf5.create_dataset("/entry_1/result_1/rmax", (numJobs, args.maxPeaks),
                                     maxshape=(None, args.maxPeaks),
                                     chunks=(1, args.maxPeaks),
                                     dtype=float)
    ds_rmax.attrs["axes"] = "experiment_identifier:peaks"

    ds_cmin = myHdf5.create_dataset("/entry_1/result_1/cmin", (numJobs, args.maxPeaks),
                                     maxshape=(None, args.maxPeaks),
                                     chunks=(1, args.maxPeaks),
                                     dtype=float)
    ds_cmin.attrs["axes"] = "experiment_identifier:peaks"

    ds_cmax = myHdf5.create_dataset("/entry_1/result_1/cmax", (numJobs, args.maxPeaks),
                                     maxshape=(None, args.maxPeaks),
                                     chunks=(1, args.maxPeaks),
                                     dtype=float)
    ds_cmax.attrs["axes"] = "experiment_identifier:peaks"

    ds_atot = myHdf5.create_dataset("/entry_1/result_1/peakTotalIntensity",(numJobs,args.maxPeaks),
                                    maxshape=(None,args.maxPeaks),
                                    chunks=(1, args.maxPeaks),
                                    dtype=float)

    ds_atot.attrs["axes"] = "experiment_identifier:peaks"
    ds_amax = myHdf5.create_dataset("/entry_1/result_1/peakMaxIntensity", (numJobs,args.maxPeaks),
                                    maxshape = (None,args.maxPeaks),
                                    chunks = (1,args.maxPeaks),
                                    dtype=float)
    ds_amax.attrs["axes"] = "experiment_identifier:peaks"

    ds_radius = myHdf5.create_dataset("/entry_1/result_1/peakRadius",(numJobs,args.maxPeaks),
                                     maxshape=(None,args.maxPeaks),
                                     chunks=(1, args.maxPeaks),
                                     dtype=float)
    ds_radius.attrs["axes"] = "experiment_identifier:peaks"

    ds_maxRes = myHdf5.create_dataset("/entry_1/result_1/maxRes",(numJobs,),
                                     maxshape=(None,),
                                     dtype=int)
    ds_maxRes.attrs["axes"] = "experiment_identifier:peaks"

    ds_likelihood = myHdf5.create_dataset("/entry_1/result_1/likelihood",(numJobs,),
                                         maxshape=(None,),
                                         dtype=float)
    ds_likelihood.attrs["axes"] = "experiment_identifier"

    ds_timeToolDelay = myHdf5.create_dataset("/entry_1/result_1/timeToolDelay",(numJobs,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_timeToolDelay.attrs["axes"] = "experiment_identifier"

    ds_laserTimeZero = myHdf5.create_dataset("/entry_1/result_1/laserTimeZero",(numJobs,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_laserTimeZero.attrs["axes"] = "experiment_identifier"

    ds_laserTimeDelay = myHdf5.create_dataset("/entry_1/result_1/laserTimeDelay",(numJobs,),
                                             maxshape=(None,),
                                             dtype=float)
    ds_laserTimeDelay.attrs["axes"] = "experiment_identifier"

    ds_laserTimePhaseLocked = myHdf5.create_dataset("/entry_1/result_1/laserTimePhaseLocked",(numJobs,),
                                                   maxshape=(None,),
                                                   dtype=float)
    ds_laserTimePhaseLocked.attrs["axes"] = "experiment_identifier"

    #myHdf5.flush()

    entry_1.create_dataset("start_time",data=0)#ps.getStartTime())
    sample_1 = entry_1.create_group("sample_1")
    sample_1.create_dataset("name",data=args.sample)
    instrument_1 = entry_1.create_group("instrument_1")
    instrument_1.create_dataset("name", data=args.instrument)
    #myHdf5.flush()

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
    #myHdf5.flush()

    detector_1 = instrument_1.create_group("detector_1")
    ds_data_1 = detector_1.create_dataset("data", (numJobs, dim0, dim1),
                                         chunks=(1, dim0, dim1),
                                         maxshape=(None, dim0, dim1),
                                         dtype=np.float32)
    ds_data_1.attrs["axes"] = "experiment_identifier"

    data_1 = entry_1.create_group("data_1")
    data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')

    """
    # Add x,y,z coordinates
    cx, cy, cz = ps.det.coords_xyz(ps.evt)
    ds_x = data_1.create_dataset("x", (dim0, dim1),
                                chunks=(dim0, dim1),
                                maxshape=(dim0, dim1),
                                dtype=float)
    ds_y = data_1.create_dataset("y", (dim0, dim1),
                                chunks=(dim0, dim1),
                                maxshape=(dim0, dim1),
                                dtype=float)
    ds_z = data_1.create_dataset("z", (dim0, dim1),
                                chunks=(dim0, dim1),
                                maxshape=(dim0, dim1),
                                dtype=float)
    ds_x[...] = ps.getCheetahImg(calib=cx)
    ds_y[...] = ps.getCheetahImg(calib=cy)
    ds_z[...] = ps.getCheetahImg(calib=cz)
    """

    # Add mask in cheetah format
    if args.mask is not None:
        data_1.create_dataset("mask", (dim0, dim1),
                              chunks=(dim0, dim1),
                              maxshape=(dim0, dim1),
                              dtype=int)

    data_1.create_dataset("powderHits", (dim0, dim1),
                              chunks=(dim0, dim1),
                              maxshape=(dim0, dim1),
                              dtype=float)
    data_1.create_dataset("powderMisses", (dim0, dim1),
                              chunks=(dim0, dim1),
                              maxshape=(dim0, dim1),
                              dtype=float)

    ds_dist_1 = detector_1.create_dataset("distance",(numJobs,),
                                         maxshape=(None,),
                                         dtype=float) # in meters
    ds_dist_1.attrs["axes"] = "experiment_identifier"

    ds_x_pixel_size_1 = detector_1.create_dataset("x_pixel_size",(numJobs,),
                                                 maxshape=(None,),
                                                 dtype=float)
    ds_x_pixel_size_1.attrs["axes"] = "experiment_identifier"

    ds_y_pixel_size_1 = detector_1.create_dataset("y_pixel_size",(numJobs,),
                                                 maxshape=(None,),
                                                 dtype=float)
    ds_y_pixel_size_1.attrs["axes"] = "experiment_identifier"

    detector_1.create_dataset("description",data=args.det)
    #myHdf5.flush()

    # Close hdf5 file
    myHdf5.close()

# Remove all previous .cxi files (important if rerun with reduced ranks)
def removeOldCxi():
    if args.tag:
        searchWord = args.tag+'.cxi'
    else:
        searchWord = '.cxi'

    for root, dirs, files in os.walk(args.outDir):
        for file in files:
            if str(file).startswith(args.exp) and str(file).endswith(searchWord):
                tok = str(file).split("_")
                if args.tag:
                    if len(tok) == 4:
                        if tok[0] == args.exp and tok[1] == runStr and tok[3] == searchWord:
                            try:
                                is_int = int(tok[2])
                                os.remove(os.path.join(root,str(file)))
                                print("### removing: ", str(file))
                            except:
                                print("### not removing: ", str(file))
                    elif len(tok) == 3:
                        if tok[0] == args.exp and tok[1] == runStr and tok[2] == searchWord:
                            os.remove(os.path.join(root,str(file)))
                            print("### removing: ", str(file))
                else:
                    if len(tok) == 3:
                        if tok[0] == args.exp and tok[1] == runStr:
                            try:
                                is_int = int(tok[2].split(".cxi")[0])
                                os.remove(os.path.join(root,str(file)))
                                print("### removing: ", str(file))
                            except:
                                print("### not removing: ", str(file))
                    elif len(tok) == 2:
                        if tok[0] == args.exp and tok[1] == runStr+searchWord:
                            print("### removing: ", str(file))
                            os.remove(os.path.join(root,str(file)))

if rank == 0:
    removeOldCxi()

comm.Barrier()

# create .cxi per rank
fname = args.outDir + '/' + args.exp +"_"+ runStr +"_"+str(rank)
if args.tag: fname += '_' + args.tag
fname += ".cxi"
createCxi(fname)

# create master .cxi
if rank == 0:
    fnameAll = args.outDir + '/' + args.exp +"_"+ runStr
    if args.tag: fnameAll += '_' + args.tag
    fnameAll += ".cxi"
    createCxi(fnameAll)

toc = time.time()
if rank == 0: print("h5 setup (rank, time): ", rank, toc-tic)

tic = time.time()

runclient(args,ds,run,times,det,numJobs,detDesc)

toc = time.time()
if rank == 0: print("compute time (rank, time): ", rank, toc-tic)

comm.Barrier()

# Write out a status_peaks.txt
if rank == 0:
    numHits = 0
    numProcessed = 0
    masterFname = args.outDir + '/' + args.exp +"_"+ runStr
    if args.tag: masterFname += '_' + args.tag
    masterFname += ".cxi"

    powderHits = None
    powderMisses = None
    with h5py.File(masterFname,"r+") as F:
        for i in range(size):
            fname = args.outDir + '/' + args.exp +"_"+ runStr +"_"+str(i)
            if args.tag: fname += '_' + args.tag
            fname += ".cxi"
            while not os.path.exists(fname):
                time.sleep(1)
            with h5py.File(fname,"r+") as f:
                numHits += len(f['/LCLS/eventNumber'])
                numProcessed += f["/LCLS/eventNumber"].attrs['numEvents']
                # copy over nPeaks from other ranks
                ind = np.where(f['/entry_1/result_1/nPeaksAll'][()]>-1)[0]
                if len(ind) > 0:
                    for i in ind:
                        F['/entry_1/result_1/nPeaksAll'][i] = f['/entry_1/result_1/nPeaksAll'][i]
                if powderHits is not None:
                    powderHits = np.maximum(powderHits, f["/entry_1/data_1/powderHits"][()])
                    powderMisses = np.maximum(powderMisses, f["/entry_1/data_1/powderMisses"][()])
                else:
                    powderHits = f["/entry_1/data_1/powderHits"][()]
                    powderMisses = f["/entry_1/data_1/powderMisses"][()]
        F["/LCLS/eventNumber"].attrs["numCores"] = size # use this to figure out how many files are generated
        F["/entry_1/data_1/powderHits"][...] = powderHits
        F["/entry_1/data_1/powderMisses"][...] = powderMisses

        if args.mask is not None:
            F['/entry_1/data_1/mask'][:, :] = cheetahUtils.readMask(args.mask)
        print("Done writing master .cxi")
    hitRate = numHits * 100. / numProcessed

    statusFname = args.outDir + "/status_peaks"
    if args.tag: statusFname += '_'+args.tag
    statusFname += ".txt"
    d = {"numHits": numHits, "hitRate(%)": hitRate, "fracDone(%)": 100.0}
    writeStatus(statusFname, d)
    print("Done writing status")

MPI.Finalize()

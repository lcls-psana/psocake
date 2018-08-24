Instructions for using the AntFarm (tentative name for this program)

1) Open 3 Terminals. ssh into psanagpuXXX machine.

2) Setup environment with ana-1.3.58 and pymongo/mongodb and other packages:
+ ana-1.3.57 has error handling for faulty idx files.
+ pymongo/mongodb is required to use MongoDB.

At LCLS, use the following conda environment:
$conda activate /reg/neh/home/takeller/.conda/envs/antfarmTest2

Or:

conda create --name antfarm python=2.7 pytorch=0.1.12 torchvision numpy h5py
conda activate antfarm
conda install --channel lcls-rhel7 psana-conda --force
conda install pymongo mongodb
conda install pytorch=0.1.12 torchvision cuda80 -c soumith

3) cd psocake/scripts
If the MongoDB server is not running*, then run startMongoServer.py in one
of the terminals. 

4) In a second terminal,
Export peaknet4antfarm paths:
$export PYTHONPATH=/reg/neh/home/liponan/ai/peaknet4antfarm:$PYTHONPATH
$export PYTHONPATH=/reg/neh/home/liponan/ai/pytorch-yolo2:$PYTHONPATH
$cd psocake/scripts
$python master.py
Keep track of the master's host node name,
it will be used as an argument when running the client. (-host)

5) Finally, in a third terminal,
$export PYTHONPATH=/reg/neh/home/liponan/ai/peaknet4antfarm:$PYTHONPATH
$export PYTHONPATH=/reg/neh/home/liponan/ai/pytorch-yolo2:$PYTHONPATH
$cd psocake/scripts
$python runClients.py -host [master host node]
-host is a required argument
Use -h to read all possible arguments

6) When you are ready, you may open more terminals to run more "Workers"
on separate GPUs. (Follow step 5)

#######################################################

*Our MongoDB server is running on psanagpu114
(This will need to run elsewhere for permanent service)

#######################################################

LCLS Tutorial/Example:

- Open at least 3 terminals, two in psanagpu114, one in psanagpu115
- use the following line in each terminal to activate the LCLS environment, then change directories:
$ conda activate /reg/neh/home/takeller/.conda/envs/antfarmTest2
$ cd psocake/scripts
- Export the right peaknet4antfarm paths:
$export PYTHONPATH=/reg/neh/home/liponan/ai/peaknet4antfarm:$PYTHONPATH
$export PYTHONPATH=/reg/neh/home/liponan/ai/pytorch-yolo2:$PYTHONPATH
- In your first psanagpu114 terminal use the following line to start a server:
./startMongoServer.py
- In your second psanagpu114 terminal use the following line to start your "Queen":
python master.py
- In your psanagpu115 terminal use the following line to start a "Worker":
python runClients.py -host psanagpu114
- Now there is one worker running. You may add more workers on other GPUs by running
the previous line in a new terminal.

#######################################################

To use this client/master setup with an alternative algorithm,
a few function calls within "clientPeakFinder.py" will need to be changed.
algorithm() is an abstract method for the plugin, so it has to be defined.

Within the algorithm() function, PyAlgos and
set_peak_selection_pars() are called. In addtion, within the getPeaks()
function, PyAlgos and alg.peak_finder_v3r3() are
called to find peaks in an nda file. These function calls could be replaced
with alternative functions, given that the argument parameters given to
runClients.py are sufficient. If you would like to save your information to
our MongoDB server, use the -dbname argument when running "runClients.py",
and pass in a name for your database.

#######################################################
OPTIONAL: For peak finding ants, source the following environment.
$export PYTHONPATH=/reg/neh/home/liponan/ai/peaknet4antfarm:$PYTHONPATH
$export PYTHONPATH=/reg/neh/home/liponan/ai/pytorch-yolo2:$PYTHONPATH

In the other terminal, run "python runClients.py" with these arguments:

arguments for runClients.py:
  -host HOST          master's host address
  -type TYPE          type of worker/name of plugin,      ***Optional
                      default = clientPeakFinder
  -npix_min NPIX_MIN  minimum number of pixels for a peak ***Optional
  -npix_max NPIX_MAX  maximum number of pixels for a peak ***Optional
  -amax_thr AMAX_THR  maximum value threshold             ***Optional
  -atot_thr ATOT_THR  integral inside peak                ***Optional
  -son_min SON_MIN    signal over noise ratio             ***Optional
  -name NAME          Name of client for database         ***Optional
  -dbname DBNAME      The database you would like to save ***Optional
                      your information to
  -h, --help          show this help message and exit     ***Optional

runClients.py runs a single client which is able to find experiment/run events with a high likelihood
 of crystals, post these events to a database, and then train PeakNet on these events. A message
 of how many peaks were found is sent to the master as well.

#######################################################

Master/Client Peaknet Steps:

1. Both Queen and Clients make their own Peaknet instances.
2. Queen and Client load DN weights
   Client also calls peaknet.model.cuda()
3. Client tells queen it is ready
4. Queen sends pn.model to client
5. Client updateModel(model from queen)
6. Client trains its Peaknet instance
7. Client sends pn.model to queen
8. Queen updateGradient(model from client)
9. Queen Optimizes
10. Steps 3-10 repeat until Clients stop

To print the database, run Python interpretor:
1. from peakDatabase import PeakDatabase
2. kwargs = {'dbname': 'PeakFindingDatabase', 'host': 'psanagpu114', 'npix_min': 2, 'name': 'Client', 'npix_max': 30, 'server': 'psanagpu114', 'amax_thr': 300, 'son_min': 10, 'atot_thr': 600}
3. myDatabase = PeakDatabase(**kwargs)
4. myDatabase.printDatabase() # prints everything
5. myDatabase.resetDatabase() # deletes everything

#######################################################

To use the Psocake Labeling tool, use:
$conda activate /reg/neh/home/takeller/.conda/envs/Tate2


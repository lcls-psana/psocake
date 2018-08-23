Instructions for using the AntFarm (tentative name for this program)

1) Open 3 Terminals

2) Setup environment with ana-1.3.58 and pymongo/mongodb and other packages:
+ ana-1.3.57 has error handling for faulty idx files.
+ pymongo/mongodb is required to use MongoDB.

At LCLS, use the following conda environment:
$conda activate /reg/neh/home/takeller/.conda/envs/antfarmTest2

Or:

conda create --name ANTFARM python=2.7 pytorch=0.1.12 torchvision numpy h5py
conda activate antfarm
conda install --channel lcls-rhel7 psana-conda --force
conda install pymongo mongodb

3) If the MongoDB server is not running*, then run startMongoServer.sh in one
of the terminals. Keep track of the database's host address,
it will be used as an argument when running the client. (-server)

4) In a second terminal, run "python master.py".
Keep track of the master's host address,
it will be used as an argument when running the client. (-host)

5) Finally, in a third terminal, run "python runClients.py" with 
the arguments -host [master host address] -server [MongoDB server host address]
Use -h to read all possible arguments

6) When you are ready, you may open more terminals to run more "Workers"
on separate GPUs.

#######################################################

*Our MongoDB server is running on psanagpu114
(This will need to run elsewhere for permanent service)

#######################################################

LCLS Tutorial/Example:

- Open at least 3 terminals, two in psanagpu114, one in psanagpu115
- use the following line in each terminal to activate the LCLS environment:
$ conda activate /reg/neh/home/takeller/.conda/envs/antfarmTest2
- In your first psanagpu115 terminal use the following line to start a server:
./startMongoServer.sh
- In your second psanagpu115 terminal use the following line to start your "Queen":
python master.py
- In your psanagpu114 terminal use the following line to start a "Worker":
python runClients.py -host psanagpu115 -server psanagpu115
- Now there is one worker running. You may add more workers on other GPUs by running
the previous line in a new terminal.

#######################################################

To use this client/master setup with an alternative algorithm,
a few function calls within "clientPeakFinder.py" will need to be changed.
Within the algorithm() function, on lines 48 and 49, PyAlgos and
set_peak_selection_pars() are called. In addtion, within the getPeaks()
function (within lines 152-154), PyAlgos and alg.peak_finder_v3r3() are
called to find peaks in an nda file. These function calls could be replaced
with alternative functions, given that the argument parameters given to
runClients.py are sufficient. If you would like to save your information to
our MongoDB server, use the -dbname argument when running "runClients.py",
and pass in a name for your database.

#######################################################
OPTIONAL: For peak finding ants, source the following environment.
$export PYTHONPATH=/reg/neh/home/liponan/ai/psnet:$PYTHONPATH
$export PYTHONPATH=/reg/neh/home/liponan/ai/psnet/examples:$PYTHONPATH
$export PYTHONPATH=/reg/neh/home/liponan/ai/psnet/python:$PYTHONPATH
$export PYTHONPATH=/reg/neh/home/yoon82/Software/peaknet4psocake:$PYTHONPATH

In the other terminal, run "python runClients.py" with these arguments:

arguments for runClients.py:
  -host HOST          master's host address
  -server SERVER      Host address of the MongoDB server
  -npix_min NPIX_MIN  minimum number of pixels for a peak ***Optional, there is a default value
  -npix_max NPIX_MAX  maximum number of pixels for a peak ***Optional, there is a default value
  -amax_thr AMAX_THR  maximum value threshold             ***Optional, there is a default value
  -atot_thr ATOT_THR  integral inside peak                ***Optional, there is a default value
  -son_min SON_MIN    signal over noise ratio             ***Optional, there is a default value
  -name NAME          Name of client for database         ***Optional, there is a default value
  -dbname DBNAME      The database you would like to save ***Optional, there is a default value
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

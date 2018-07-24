To run this program, open two terminals.

Get Tate2 which includes error handling for faulty idx files and pymongo:
conda activate /reg/neh/home/takeller/.conda/envs/Tate2

Source the following environment:
export PYTHONPATH=/reg/neh/home/liponan/ai/psnet:/reg/neh/home/liponan/ai/psnet/examples:/reg/neh/home/liponan/ai/psnet/python:$PYTHONPATH
export PYTHONPATH=/reg/neh/home/yoon82/Software/peaknet4psocake:$PYTHONPATH

If the MongoDB server is not running*, then in either terminal, run startMongoServer.sh. 
Keep track of the database's host address, it will be used as an argument when running the client.

In one terminal, run "python master.py". Keep track of the master's host address, it will be used as an argument when running 
the client.

In the other terminal, run "python runClients.py" with the following arguments:

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


runClients.py runs a single client which is able to experiment/run events with a high likelihood
 of crystals, post these events to a database, and then train PeakNet on these events. A message
 of how many peaks were found is sent to the master as well. 

To use this client/master setup with an alternative algorithm, a few function calls within 
"clientPeakFinder.py" will need to be changed. Within the algorithm() function, on lines 48 and 
49, PyAlgos and set_peak_selection_pars() are called. In addtion, within the getPeaks() function
 (within lines 152-154), PyAlgos and alg.peak_finder_v3r3() are called to find peaks in an nda 
file. These function calls could be replaced with alternative functions, given that the argument
 parameters given to runClients.py are sufficient. If you would like to save your information to
 our MongoDB server, use the -dbname argument when running "runClients.py", and pass in a name 
for your database.


*Our MongoDB server will be running on psanagpu114

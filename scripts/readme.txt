To run this program, open two terminals.

Get Tate2 which includes error handling for faulty idx files and pymongo:
conda activate /reg/neh/home/takeller/.conda/envs/Tate2

Source the following environment:
export PYTHONPATH=/reg/neh/home/liponan/ai/psnet:/reg/neh/home/liponan/ai/psnet/examples:/reg/neh/home/liponan/ai/psnet/python:$PYTHONPATH
export PYTHONPATH=/reg/neh/home/yoon82/Software/peaknet4psocake:$PYTHONPATH

In either terminal, run startMongoServer.sh. Keep track of the database's host address, it will be used as an argument when running the client.

In one terminal, run "python master.py". Keep track of the master's host address, it will be used as an argument when running the client.

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
  -h, --help          show this help message and exit     ***Optional


runClients.py will find events with a high likelihood of crystals, post these events to a database, train PeakNet on these events, and will send a message of how many peaks were found to the master.

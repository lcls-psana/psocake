To run this program, open two terminals.

Get psana version >= 1.3.57 which includes error handling for faulty idx files:
conda activate ana-1.3.57

Source the following environment:
export PYTHONPATH=/reg/neh/home/liponan/ai/psnet:/reg/neh/home/liponan/ai/psnet/examples:/reg/neh/home/liponan/ai/psnet/python:$PYTHONPATH
export PYTHONPATH=/reg/neh/home/yoon82/Software/peaknet4psocake:$PYTHONPATH

In one terminal, run "python master.py".

In the other terminal, run "python allPeakFinding.py". 

allPeakFinding.py will finding events with a high likelihood of crystals, and will send a message of how many peaks were found to the master.

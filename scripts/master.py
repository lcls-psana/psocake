import os
import os.path
import re
import numpy as np
from psana import *
import time
import base64
from pathlib import Path
import json
from masterSocket import masterSocket
import zmq

#Initialize global variables 

#Value that continues the loop of accepting information from clients. Could potentially be used to end the master program if all clients have finished sending information to the master. For now, boolean is always true
boolean = True

#Value that keeps track of the number of clients sending the master information
#numberOfClients = 1

#Value that keeps track of the number of clients who have finished sending the master information. Could be used to end the master program.
#numClientsDone = 0

#The total number of peaks found by the clients
totalPeaks = 0

#The total number of hits found by the clients
totalHits = 0

#If peaks have been reported, then this value will change to true and hits will be reported.
peaksThenHits = False

def munge_json(event):
        """ Convert an array pushed through json back into a numpy array
        
        Arguments:
        event -- array sent through json to be changed into numpy array
        """
        if 'done' in event:
                return None, None
        else:
		val = 0
                for i,evn in enumerate(event):
                        try:
                                val = np.frombuffer(base64.b64decode(event[0]), dtype = np.dtype(event[2])).reshape(event[1])
                        except TypeError:
                                pass
                return val

#Create master socket to recieve information from clients
socket = masterSocket()

#Accepted information from clients.
while (boolean):
        try:
                print("waiting...")
                val = socket.pull()
                #if(val == "Done!"):
                #        numClientsDone += 1
                #boolean = ((val != "Done") and (numClientsDone != numberOfClients))
                integer = (isinstance(val,int))
        except zmq.error.Again:
                break
        if(integer):
                if(peaksThenHits == False):
                        totalPeaks += val
                        print(totalPeaks, "Peaks")
                        peaksThenHits = True
                else:
                        totalHits += val
                        print(totalHits, "Hits")
                        peaksThenHits = False
        #elif(val == "Done!"):
                #fprint("finishing up")
        else:
                b = (munge_json(val))
                print(b)
                print(b[1][0])



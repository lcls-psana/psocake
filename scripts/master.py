import os
import os.path
import re
import numpy as np
from psana import *
import time
import zmq
import base64
from pathlib import Path
import json


#push(val) takes a variable val, and sends it to a receiving client zmq socket
def push(val):
	context = zmq.Context()
	zmq_socket = context.socket(zmq.PUSH)
	zmq_socket.bind("tcp://127.0.0.1:5560")
	print("I am pushing:", val)
	zmq_socket.send_json(val)
	print("Pushed")

#pull() recieves information from a client zmq socket
def pull():
	context = zmq.Context()
	results_receiver = context.socket(zmq.PULL)
        results_receiver.connect("tcp://127.0.0.1:5559")
        result = results_receiver.recv_json()  
	print("I just pulled:", result)  
	return result

#change data back into np array
def munge_json(event):
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

#Read the list
#csPadList = []
#with open('csPadList.txt', 'r') as f:
#        csPadList = json.load(f)

#for val in csPadList:
 #       push(val)


boolean = True
numberOfClients = 1
#numClientsDone = 0
totalPeaks = 0
peaksThenHits = 0

while (boolean):
        try:
                print("waiting...")
                val = pull()
                #if(val == "Done!"):
                #        numClientsDone += 1
                #boolean = ((val != "Done") and (numClientsDone != numberOfClients))
                integer = (isinstance(val,int))
        except zmq.error.Again:
                break
        if(integer):
                if(peaksThenHits == 0):
                        totalPeaks += val
                        print(totalPeaks, "Peaks")
                        peaksThenHits += 1
                else:
                        print(val, "Hits")
        #elif(val == "Done!"):
                #fprint("finishing up")
        else:
                b = (munge_json(val))
                print(b)
                print(b[1][0])



import os
import os.path
import re
import numpy as np
import time
import datetime
import base64
import json
from peaknet import Peaknet
from masterSocket import masterSocket
import zmq

#Initialize variables 

#Value that continues the loop of accepting information from clients.
boolean = True

#Number of models to pass before saving a model to a database
checkpoint = 100

#Create master socket to recieve information from clients
socket = masterSocket()

#Step 1: Both Queen and Clients make their own Peaknet instances.
peaknet = Peaknet()

#Step 2: Queen loads DN weights
peaknet.loadDNWeights()


#Communication with clients begins.
while (boolean):
    try:
        #print(datetime.datetime.now().strftime("--%Y-%m-%d--%H:%M:%S"))
        print("waiting for worker...")
        val = socket.pull()
        #print(datetime.datetime.now().strftime("--%Y-%m-%d--%H:%M:%S"))
    except zmq.error.Again:
        break
    if(val == "Im Ready!"):
        #Step 3: Client tells Queen it is ready
        #Step 4: Queen sends model to Client
        socket.push(peaknet.model)
        #Step 5: Client updateModel(model from queen)
        #Step 6: Client trains its Peaknet instance
    else:
        #Step 7: Queen recieves new model from client
        model = val
        #Step 8: Queen does updateGradient(new model from client)
        peaknet.updateGrad(model)
        #Step 9: Queen Optimizes
        peaknet.optimize()
        #Step 10: Repeat Steps 3-10
    #TODO: Every checkpoint # models, the model will be saved to MongoDB

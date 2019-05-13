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
import torch

#Initialize variables 
runMasterOnGPU = True

#Number of models to pass before saving a model to a database
checkpoint = 100

#Create master socket to recieve information from clients
socket = masterSocket()

#Step 1: Both Queen and Clients make their own Peaknet instances.
peaknet = Peaknet()
lr = 0.001

#Step 2: Queen loads DN weights
peaknet.loadCfg("/reg/neh/home/liponan/ai/pytorch-yolo2/cfg/newpeaksv10-asic.cfg") # FIXME: load latest weights
peaknet.init_model()
#peaknet.model = torch.load("/reg/d/psdm/cxi/cxic0415/res/liponan/antfarm_backup/api_demo_psana_model_000086880")

if runMasterOnGPU: peaknet.model.cuda()
peaknet.set_optimizer(adagrad=True, lr=lr)

#Communication with clients begins.
kk = 0
outdir = "/reg/d/psdm/cxi/cxic0415/res/liponan/antfarm_backup"

while 1:
    try:
        print("waiting for worker...")
        val = socket.pull()
        print("#### master pulled: ", val)
    except zmq.error.Again:
        break

    print("^^^^^^^^^^^^^^^^^^^^: ", val)

    if(val[0] == "Ready"):
        #Step 3: Client tells Queen it is ready
        #Step 4: Queen sends model to Client
        socket.push(['train', peaknet.model])
        #Step 5: Client updateModel(model from queen)
        #Step 6: Client trains its Peaknet instance
        counter = val[1]

        if counter%10000 == 0:
            print("######### validate")
            socket.push(['validate', peaknet.model])
        elif counter%10 == 0:
            print("######### validateSubset")
            socket.push(['validateSubset', peaknet.model])

        fname = os.path.join(outdir, str(kk)+".pkl")
        if kk%3 == 0:
            torch.save(peaknet.model, fname)
        kk += 1

    elif(val[0] == "Gradient"): # val is the gradient
        #Step 7: Queen recieves new model from client
        grads = val[1]
        mini_batch_size = val[2]
        #Step 8: Queen does updateGradient(new model from client)
        peaknet.set_optimizer(adagrad=True, lr=lr)

        peaknet.updateGrad(grads, mini_batch_size, useGPU=runMasterOnGPU)
        #Step 9: Queen Optimizes
        peaknet.optimize()

        #Step 10: Repeat Steps 3-10
        model_dict = dict(peaknet.model.named_parameters())
        #TODO: Every checkpoint # models, the model will be saved to MongoDB



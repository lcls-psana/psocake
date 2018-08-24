import subprocess
import socket
import os

#Start Database
subprocess.call("mongod --dbpath /reg/d/psdm/cxi/cxitut13/res/autosfx/db --bind_ip_all", shell = True)

#Locate GPU Host
host = socket.gethostname()

#Write Location to File
curDir = os.getcwd()
filename = curDir+"/databaseLocation.txt"
file = open(filename,"w") 
file.write("%s"%host) 
file.close()

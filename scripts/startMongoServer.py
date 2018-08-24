import subprocess
import socket
import os

#Locate GPU Host
host = socket.gethostname()

#Write Location to File
curDir = os.getcwd()
filename = "databaseLocation.txt"
file = open(filename,"w")
print filename
file.write("%s"%host)
file.close()

#Start Database
subprocess.call("mongod --dbpath /reg/d/psdm/cxi/cxitut13/res/autosfx/db --bind_ip_all", shell = True)


import subprocess
import socket

#Start Database
subprocess.call("mongod --dbpath /reg/d/psdm/cxi/cxitut13/res/autosfx/db1 --bind_ip_all", shell = True)

#Locate GPU Host
host = socket.gethostname()

#Write Location to File
filename = "databaseLocation.txt"
file = open(filename,”w”) 
file.write("%s"%host) 
file.close()

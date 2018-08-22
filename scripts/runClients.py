import loadClients
import argparse


#runClient takes an argument "-host", enter the master's host name
parser = argparse.ArgumentParser()
parser.add_argument("-host", help="master's host")
parser.add_argument("-npix_min", default = 2, type = int, help = "minimum number of pixels for a peak")
parser.add_argument("-npix_max", default = 30, type = int, help = "maximum number of pixels for a peak")
parser.add_argument("-amax_thr", default = 300, type = int, help = "maximum value threshold")
parser.add_argument("-atot_thr", default = 600, type = int, help = "integral inside peak")
parser.add_argument("-son_min", default = 10, type = int, help = "signal over noise ratio")
parser.add_argument("-name", default = "Client", help = "Name of client for database")
parser.add_argument("-server", default = "psanagpu114", help = "Host address of the MongoDB server")
parser.add_argument("-dbname" , default = "PeakFindingDatabase", help = "The database you would like to save your information to")
args = parser.parse_args()

def invoke_model(model_name, **kwargs):
    """Load a plugin client by giving a client name, and entering arguments for corresponding algorithms

    Arguments:
    model_name -- client plugin name, at the moment, either clientPeakFinder or clientSummer
    **kwargs -- arguments for peakFinding algorithm, master host name, and client name
    """
    model_module_obj = loadClients.load_model(model_name)
    model_module_obj.algorithm(**kwargs)

kwargs = {"npix_min": args.npix_min, "npix_max": args.npix_max, "amax_thr": args.amax_thr, 
          "atot_thr": args.atot_thr, "son_min" : args.son_min, "host" : args.host, "name" : args.name,
          "server" : args.server, "dbname" : args.dbname}

print(kwargs)

#invoke_model("clientPeakFinder",**kwargs)
invoke_model("testClient",**kwargs)

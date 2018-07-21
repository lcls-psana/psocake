import clientAbstract
import zmq
from crawler import Crawler
import psana
from psana import *
import numpy as np

class clientSummer(clientAbstract.clientAbstract):
    def pull(self):
        """ Recieve information from the master zmq socket. 
        When called, the program will wait until information 
        has been pushed by the master zmq socket.
        """
        context = zmq.Context()
        results_receiver = context.socket(zmq.PULL)
        results_receiver.connect("tcp://127.0.0.1:5560")
        result = results_receiver.recv_json()    
        print("I just pulled:", result)  
        return result

    def push(self,val):
        """ Give information to the master zmq socket.
        
        Arguments:
        val -- The information/value that will be pushed to the master zmq socket.
        """
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PUSH)
        zmq_socket.bind("tcp://127.0.0.1:5559")
        print("I am pushing:", val)
        zmq_socket.send_json(val)

    def algorithm(self):
        #self.push(kwargs)
        exp, runnum, det = self.getExpRunDet()
        ds = psana.DataSource('exp=%s:run=%d:idx'%(exp,runnum))
        d = psana.Detector(det)
        d.do_reshape_2d_to_3d(flag=True)
        run = ds.runs().next()
        times = run.times()
        env = ds.env()
        mask = d.mask(runnum,calib=True,status=True,edges=True,central=True,unbond=True,unbondnbrs=True)
        numEvents = len(times)
        for j in range(numEvents):
            evt = run.event(times[j])
            try:
                nda = d.calib(evt) * mask
            except TypeError:
                nda = d.calib(evt)
            if (nda is not None):
                print(exp, runnum, det, j)
                print("Sum is", np.sum(nda))
                break
                        
    def getExpRunDet(self):
        crawler = Crawler()
        exp, runnum, det = crawler.returnOneRandomExpRunDet()
        runnum = int(runnum)
        return [exp,runnum,det]


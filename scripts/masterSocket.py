import zmq
import socket

class masterSocket:
    context = zmq.Context()
    puller = context.socket(zmq.PULL)
    pusher = context.socket(zmq.PUSH)
    #retrieve the host name that the master is running on
    host = socket.gethostname()

    def __init__(self):
        self.puller.bind("tcp://*:5559")
        self.pusher.bind("tcp://*:5560")

    def push(self, val):
        """ Give information to the client zmq socket.
    
        Arguments:
        val -- The information/value that will be pushed to the client zmq socket.
        """
	print("I am pushing:", val)
	self.pusher.send_pyobj(val)
	print("Pushed")

    def pull(self):
        """ Recieve information from the client zmq socket. 
        When called, the program will wait until information 
        has been pushed by the client zmq socket.
        """
        result = self.puller.recv_pyobj()  
	print("I just pulled:", result)  
	return result


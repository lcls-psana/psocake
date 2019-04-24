import zmq

class clientSocket:
    context = zmq.Context()
    puller = context.socket(zmq.PULL)
    pusher = context.socket(zmq.PUSH)

    def __init__(self, **kwargs):
        host = kwargs["host"]
        self.puller.connect("tcp://%s:5560"%host)
        self.pusher.connect("tcp://%s:5559"%host)
        

    def pull(self):
        """ Recieve information from the master zmq socket. 
        When called, the program will wait until information 
        has been pushed by the master zmq socket.
        """
        result = self.puller.recv_pyobj(flags=0)#zmq.NOBLOCK)
        print("### I just pulled:", result)
        return result

    def push(self, val):
        """ Give information to the master zmq socket.
        
        Arguments:
        val -- The information/value that will be pushed to the master zmq socket.
        """
        #print("I am pushing:", val)
        self.pusher.send_pyobj(val)

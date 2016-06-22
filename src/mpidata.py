import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class arrayinfo(object):
    def __init__(self,name,array):
        self.name = name
        self.shape = array.shape
        self.dtype = array.dtype

class small(object):
    def __init__(self):
        self.arrayinfolist = []
        self.endrun = False
    def addarray(self,name,array):
        self.arrayinfolist.append(arrayinfo(name,array))

class mpidata(object):

    def __init__(self):
        self.small=small()
        self.arraylist = []

    def endrun(self):
        self.small.endrun = True
        comm.send(self.small,dest=0,tag=rank)

    def addarray(self,name,array):
        self.arraylist.append(array)
        self.small.addarray(name,array)

    def send(self):
        assert rank!=0
        comm.send(self.small,dest=0,tag=rank)
        for arr in self.arraylist:
            assert arr.flags['C_CONTIGUOUS']
            comm.Send(arr,dest=0,tag=rank)

    def recv(self):
        assert rank==0
        status=MPI.Status()       
        self.small=comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        recvRank = status.Get_source()
        if not self.small.endrun:
            for arrinfo in self.small.arrayinfolist:
                if not hasattr(self,arrinfo.name) or arr.shape!=arrinfo.shape or arr.dtype!=arrinfo.dtype:
                    setattr(self,arrinfo.name,np.empty(arrinfo.shape,dtype=arrinfo.dtype))
                arr = getattr(self,arrinfo.name)
                comm.Recv(arr,source=recvRank,tag=MPI.ANY_TAG)

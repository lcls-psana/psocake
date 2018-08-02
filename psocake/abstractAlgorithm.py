import abc

class abstractAlgorithm(object):
    __metaclass__ = abc.ABCMeta


    def __init__(self): pass

    @abc.abstractmethod
    def setParams(self): pass

    @abc.abstractmethod
    def initParams(self, **kwargs): pass

    @abc.abstractmethod
    def algorithm(self, nda, mask, **kwargs): pass

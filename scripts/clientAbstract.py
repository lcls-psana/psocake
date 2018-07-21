import abc

class clientAbstract(object):
    __metaclass__ = abc.ABCMeta

    def __init__( self ): pass
    
    @abc.abstractmethod
    def algorithm(self, **kwargs): pass

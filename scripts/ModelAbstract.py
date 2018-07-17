import abc

class ModelAbstract(object):
    __metaclass__ = abc.ABCMeta

    def __init__( self ):
        print "Model abstract"

    @abc.abstractproperty
    def get_name(self): pass

    @abc.abstractmethod
    def get_range(self, no_of_elements=1) : pass

    @abc.abstractmethod
    def get_resource(self, no_of_elements=1): pass

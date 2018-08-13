import abc

#See adaptiveAlgorithm for example
class abstractAlgorithm(object):
    __metaclass__ = abc.ABCMeta

    #An Initializer Function, import your algorithm and make an instance
    def __init__(self): pass

    #Set your algorithms parameters based on kwarg inputs
    @abc.abstractmethod
    def setParams(self): pass

    #Save your alorithm parameter values based on kwarg inputs
    @abc.abstractmethod
    def initParams(self, **kwargs): pass

    #Takes parameters as input, if none uses default parameters, then evaluates the imported algorithm
    @abc.abstractmethod
    def algorithm(self, nda, mask, kw): pass

    #Return default parameters in string format for Psocake to display
    @abc.abstractmethod
    def getDefaultParams(self): pass

    #Save a default parameter variable
    def setDefaultParams(self): pass

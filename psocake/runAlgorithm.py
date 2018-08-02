import loadAlgorithm

def invoke_model(model_name, **kwargs):
    """Load a plugin client by giving a client name, and entering arguments for corresponding algorithms

    Arguments:
    model_name -- client plugin name, at the moment, either clientPeakFinder or clientSummer
    **kwargs -- arguments for peakFinding algorithm, master host name, and client name
    """
    model_module_obj = loadAlgorithm.load_model(model_name)
    return model_module_obj.algorithm(**kwargs)

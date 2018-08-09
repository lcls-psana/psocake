import loadAlgorithm

def invoke_model(model_name):
    """Load a plugin algorithm by giving an algorithm name, and optionally entering arguments for parameters

    Arguments:
    model_name -- client plugin name, at the moment, only adaptiveAlgorithm
    """
    model_module_obj = loadAlgorithm.load_model(model_name)
    return model_module_obj

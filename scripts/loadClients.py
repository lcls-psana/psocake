import os, sys

def load_module(modname):
    """Load a plugin by module name at runtime

    Arguments:
    modname -- python module name
    """
    try:
        if modname not in sys.modules: # if not already loaded
            module = __import__(modname) 
        else:
            module = sys.modules[modname]
        class_ = getattr(module, modname)

        #print "Dynamically loaded model :"+modname
        classObj = class_()
        return classObj
    except Exception as e:
        print("No Model : "+modname, "Error : "+e.message)
        raise e

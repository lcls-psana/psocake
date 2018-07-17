import loader

def invoke_model(model_name):
    model_module_obj = loader.load_model(model_name)

    name = model_module_obj.get_name()
    range_val = model_module_obj.get_range(2)
    resource = model_module_obj.get_resource()
    print name, range_val, resource

invoke_model("modelA")
invoke_model("modelB")
invoke_model("modelA")

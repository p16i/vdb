import importlib

def get_network(name):
    module_path = f"nets.{name}"

    print("Taking %s" % module_path)
    model_mod = importlib.import_module(module_path)

    return model_mod.Net

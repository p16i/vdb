import importlib
import yaml
import utils
import datasets

def get_network(name):
    module_path = f"nets.{name}"

    print("Taking %s" % module_path)
    model_mod = importlib.import_module(module_path)

    return model_mod.Net

def load_model(path):

    with open(f"{path}/summary.yml", "r") as fh:
        summary = yaml.safe_load(fh)

        model_name, model_config = summary['model'].split("/")
        model_config = utils.parse_arch(model_config)

        input_shape = datasets.input_dims[summary["dataset"]]
        model_cls = get_network(model_name)
        model = model_cls(
            model_config, input_shape,
            summary['beta'], summary['M']
        )

        model.load_weights(f"{path}/model")

    return model, summary
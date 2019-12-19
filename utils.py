from datetime import datetime
import yaml
import numpy as np
import datasets

def get_experiment_name(prefix=""):
    timestamp = datetime.now().strftime('%Y-%m-%d--%H-%M-%S-%f')
    if prefix:
        return f"{prefix}--{timestamp}"
    else:
        return timestamp

def format_metrics(prefix, metrics, acc):
    return ":: %5s  >> Loss: %.4f | I(Z; Y): %.4f | I(X; Z): %.4f | acc(L1): %.4f | acc(L12): %.4f" \
        % (prefix, *metrics, *acc)

def to_number(a):
    if "." in a:
        return float(a)
    else:
        return int(a)

def parse_arch(arch):
    parts = map(lambda x: x.split(":"), arch.split("|"))
    parts = map(lambda x: (x[0], to_number(x[1])), parts)

    return dict(parts)

def read_summary(fp):
    with open(fp, "r") as fh:
        summary = yaml.safe_load(fh)

    for s, v in summary['metrics'].items():
        for kk, vv in v.items():
            summary[f"metrics:{s}:{kk}"] = vv

    del summary['metrics']
    summary['filename'] = fp.split("/")[-2]

    return summary
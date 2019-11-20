from datetime import datetime

def get_experiment_name(prefix=""):
    timestamp = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    if prefix:
        return f"{prefix}--{timestamp}"
    else:
        return timestamp

def format_metrics(prefix, metrics):
    return ":: %5s  >> Loss: %.4f | I(Z; Y): %.4f | I(X; Z): %.4f | acc: %.4f" \
        % (prefix, *metrics)

def to_number(a):
    if "." in a:
        return float(a)
    else:
        return int(a)

def parse_arch(arch):
    parts = map(lambda x: x.split(":"), arch.split("|"))
    parts = map(lambda x: (x[0], to_number(x[1])), parts)

    return dict(parts)
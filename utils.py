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
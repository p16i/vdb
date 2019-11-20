from datetime import datetime

def get_experiment_name(prefix=""):
    timestamp = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    if prefix:
        return f"{prefix}--{timestamp}"
    else:
        return timestamp


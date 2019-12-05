# Variation Deficiency Bottleneck

## Dependencies
Please run `pip install -r requirements.txt` for installing all depedencies.

## Project Structures
### CLI
`train.py` provides the training CLI. Below is its usage:
```
Usage:
train.py  [--epoch=<epoch> --beta=<beta> -M=<M> --lr=<lr>] --strategy=<strategy> --dataset=<dataset> <model>

Options:
  -h --help                 Show this screen.
  --dataset=<dataset>       One from {mnist, fashion_mnist, cifar10} [default: mnist]
  --beta=<beta>             Value of β [default: 0.001]
  -M=<M>                    Value of M [default: 1]
  --lr=<lr>                 Learning rate [default: 0.001]
  --epoch=<epoch>           Number of epochs [default: 10]
  --strategy=<strategy>     Optimizaton strategy "oneshot" or "seq/d:1|e:10" [default: oneshot]
                            "seq/e:10|d:1" means "decoder get update every epoch while 10 for encoder".
```

### Running Unit tests
```
pytest tests.py
```

**Note:** Currently, we just have a test for verifying results of the class loss from TF1 and TF2 implementations.


### Core Modules
- `vdb.py` contains the details of our model, i.e. how its architecture and computation look like.
- `losses.py` contains the IB loss function and the implementations of optimization strategies:
  - oneshot
  - sequential
- `datasets.py` provides a function to get any dataset from Keras based on `name`.

### Utility Modules
- `plot_helper.py`
- `tfutils.py`
- `utils.py`

## Experiment Artifacts
Each experiment is assigned with a unique name `vdb-<dataset>--<timestamp>`. 
During training, we collect metrics to TensorBoard under `log` directory. These metrics include:
- loss
- accuracy
- I(X; Z) Bound (or something similar)
- I(Y; Z) Bound

When the bottleneck dimensions is `2`, we also plot the latent representation for 1000 test samples to TensorBoard.

We also save the details of each experiment in `summary.yml`.
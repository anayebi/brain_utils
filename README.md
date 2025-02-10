# brainmodel_utils
Basic utilities for comparing models to neural & behavioral data, along with packaging these data in Python (from Matlab).

# Installation
To install run:
```
git clone https://github.com/anayebi/brainmodel_utils
cd brainmodel_utils/
pip install -e .
```

# Usage:
After installing the package, you can use it as follows:

```
from brainmodel_utils.metrics.consistency import get_linregress_consistency
import numpy as np

## Example data (substitute with your actual model/data!)
source = np.random.randn(100, 50)  # 100 stimuli, 50 model features

# Alternatively, if `source` is from another animal (rather than a model), you can include a trials dimension:
target = np.random.randn(20, 100, 50)  # 20 trials, 100 stimuli, 50 units

## Linear regression parameters (we usually recommend Ridge with a user defined alpha that has been cross-validated via median consistency on a val set of your choosing!)
alpha = 1.0
map_kwargs = {
                "map_type": "sklinear",
                "map_kwargs": {
                    "regression_type": "Ridge",
                    "regression_kwargs": {"alpha": alpha},
                },
            }
# see below for other map types

## Compute consistency
consistency_results = get_linregress_consistency(
    source=source,
    target=target,
    map_kwargs=map_kwargs,
    num_bootstrap_iters=1000,  # Number of bootstrap split-half iterations
    num_parallel_jobs=100,     # Parallelization across split-halves for speed
    start_seed=42,             # Reproducibility seed
    metric="pearsonr"          # Use "rsa_pearsonr/spearmanr" for RSA instead, in which case use an Identity Map (see below)
)

print(consistency_results)
```

We support a range of standard map types (listed [here](https://github.com/neuroagents-lab/brainmodel_utils/blob/main/brainmodel_utils/neural_mappers/__init__.py)), from no mapping (`IdentityNeuralMap`) used for RSA (implemented [here](https://github.com/neuroagents-lab/brainmodel_utils/blob/main/brainmodel_utils/metrics/utils.py#L86-L89)), to 1-to-1 simple correlation-based mapping of units from source to target (`PercentileNeuralMap`), Partial Least Squares (`PLSNeuralMap`), and the [`scikit-learn`](https://scikit-learn.org/) mapping functions like Ridge/Lasso/ElasticNet (`SKLinearNeuralMap`).

Example usages are below for each, but feel free to PR your own!

```python
# Identity (only for use with RSA via "rsa_pearsonr/spearmanr")
map_kwargs = {"map_type": "identity"}

# 1-to-1 mapping, finds the best (100th percentile) unit in source to match to each target unit, on train set
map_kwargs = {"map_type": "percentile"}

# 1-to-1 mapping using 95th percentile group rather than absolute best source unit
alpha = 1.0  # we strongly recommend cross-validating this!
map_kwargs = {
    "map_type": "percentile",
    "map_kwargs": {
        "percentile": 95,
        "identity": False,  # must be set if percentile < 100, since you can no longer use the identity transform
        "regression_type": "Ridge",
        "regression_kwargs": {"alpha": alpha},
    },
}

# ElasticNet regression
alpha = 1.0  # we strongly recommend cross-validating this!
l1_ratio = 0.5  # we strongly recommend cross-validating this!
map_kwargs = {
    "map_type": "sklinear",
    "map_kwargs": {
        "regression_type": "ElasticNet",
        "regression_kwargs": {"alpha": alpha, "l1_ratio": l1_ratio},
    },
}

# Partial Least Squares (PLS) regression. Note we do not use `sklinear` for this, since `scikit-learn` sets `scale=True` by default for PLS, which is **not** what we want for neural data (we want `scale=False`).
map_kwargs = {
    "map_type": "pls",
    "map_kwargs": {
        "n_components": 25,  # we recommend cross-validating this, or going as high as feasible; e.g., 100 components is good too!
    },
}
```

Under the hood, [`PipelineNeuralMap`](https://github.com/neuroagents-lab/brainmodel_utils/blob/main/brainmodel_utils/neural_mappers/pipeline_neural_map.py) is called for each of these, which can chain these mappings with additional factorization if you like, by passing in `factor_kwargs`.

# License
MIT

# Contact
If you have any questions or encounter issues, either submit a Github issue here (preferred) or [email me](https://anayebi.github.io/contact/).

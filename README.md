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

# Example data (substitute with your actual model/data!)
source = np.random.randn(100, 50)  # 100 stimuli, 50 model features

# Alternatively, if `source` is from another animal (rather than a model),
# you can include a trials dimension:
target = np.random.randn(20, 100, 50)  # 20 trials, 100 stimuli, 50 units

# Linear regression parameters (in this case, Ridge with a user defined alpha)
map_kwargs = {
                "map_type": "sklinear",
                "map_kwargs": {
                    "regression_type": "Ridge",
                    "regression_kwargs": {"alpha": alpha},
                },
            }

# Compute consistency
consistency_results = get_linregress_consistency(
    source=source,
    target=target,
    map_kwargs=map_kwargs,
    num_bootstrap_iters=1000,  # Number of bootstrap split-half iterations
    num_parallel_jobs=100,     # Parallelization across split-halves
    start_seed=42,             # Reproducibility seed
    metric="pearsonr"          # Use "rsa_pearsonr" for RSA, in which case use an Identity Map
)

print(consistency_results)
```

# License
MIT

# Contact
If you have any questions or encounter issues, either submit a Github issue here (preferred) or [email me](https://anayebi.github.io/contact/).

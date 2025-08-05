# Population Receptive Field Modeling With Multiple Backends

Prototype for fitting population receptive field (pRF) models with multiple backends through keras.

To get started, run the example notebook `test_multi_backend.ipynb`.

## Installation

Install from GitHub via:

```
git clone https://github.com/popylar-org/kerasprf.git
cd kerasprf
pip install .
```

Running the notebook requires the following additional packages:

```
pip install jupyter ipykernel matplotlib
```

## Implementation

Implements a basic 2D Gaussian pRF model (see [Dumoulin & Wandell, 2008](https://doi.org/10.1016/j.neuroimage.2007.09.034)) that can be fit to a single time series. Because the model is implemente in keras 3, the backend used for fitting can be changed using the `KERAS_BACKEND` environment variable.

Under the hood, the package implements a different function to update the parameters of the pRF model using backpropagation for each backend.

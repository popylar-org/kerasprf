
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import pytest

from kerasprf.adapter import Adapter, ParameterTransform
from kerasprf.model.gaussian_2d import Gaussian2DModel
from kerasprf.optmize.backend.base_fitter import ParameterFitter
from kerasprf.stimulus import Stimulus


@pytest.fixture
def num_steps():
    return 100

@pytest.fixture
def grid(num_steps):
    x_range, y_range = np.linspace(-5, 5, num_steps), np.linspace(-5, 5, num_steps)
    xy_grid, yx_grid = np.meshgrid(x_range, y_range)
    grid = np.vstack([xy_grid[None,:,:], yx_grid[None,:,:]]).T
    return grid

@pytest.fixture
def paradigm(num_steps, grid):
    paradigm = (
        np.ones((num_steps, num_steps, num_steps)) * 
        0.5 * np.sin(np.arange(num_steps)) * 
        (np.sin(grid[:,:,0]) + np.cos(grid[:, :, 1]))
    )
    return paradigm

def prf_response_fun(x, centroid, sigma):
    return np.exp(-(np.sum((x - centroid)**2, axis=-1) / (2 * sigma**2)))

@pytest.fixture
def rng():
    return np.random.default_rng(2025)

@pytest.fixture
def simulated_signal(num_steps, grid, paradigm, rng):
    # True pRF parameters
    true_centroid = np.array([-2, 3])
    true_sigma = 2

    # pRF predictions
    true_signal = (prf_response_fun(grid, true_centroid[None, None, :], true_sigma) * paradigm).sum(axis=(0, 1))

    # Add noise
    simulated_signal = true_signal + rng.normal(0, scale=0.2, size=num_steps)

    return simulated_signal

# Sum of squares
def loss_fn(y, y_pred):
    return keras.ops.sum((y - y_pred)**2)

@pytest.fixture
def stimulus(paradigm, grid):
    return Stimulus(
        ["x", "y"],
        paradigm,
        grid
    )


def test_fit(stimulus, simulated_signal):
    # Define some starting values
    start_params = {
        "centroid": np.array([0, 0]),
        "sigma": 1.0
    }

    model = Gaussian2DModel()

    optimizer = keras.optimizers.Adam(learning_rate=0.1)

    adapter = Adapter(transforms=[ParameterTransform("sigma", keras.ops.log, keras.ops.exp)])

    fitter = ParameterFitter(model, stimulus, adapter, optimizer, loss_fn)

    result = fitter.fit(simulated_signal, start_params, num_steps=10)

    assert isinstance(result, dict)

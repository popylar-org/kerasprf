
from abc import ABC, abstractmethod

import keras


class TimeSeriesModel(ABC):
    @property
    @abstractmethod
    def parameter_names(self):
        return []
    
    @abstractmethod
    def predict(self, timeseries, parameters):
        return timeseries


class BaselineAmplitudeModel(TimeSeriesModel):
    @property
    def parameter_names(self):
        return ["baseline", "amplitude"]
    

    def predict(self, timeseries, parameters):
        return parameters["baseline"] + parameters["amplitude"] * timeseries 


class HRFModel(TimeSeriesModel):
    def __init__(self, time_length=32.0, dt=1.0) -> None:
        super().__init__()
        self.time_length = time_length
        self.dt = dt
        self.timestamps = keras.ops.linspace(
            1e-4,
            self.time_length,
            num=int(self.time_length / self.dt)
        )

    @property
    def parameter_names(self):
        return ["tau"]
    

    def calc_hrf(self, parameters):
        return self.timestamps * keras.ops.exp(-self.timestamps / parameters["tau"])


    def predict(self, timeseries, parameters):
        hrf = self.calc_hrf(parameters)
        return keras.ops.correlate(timeseries, hrf)


class NoiseModel(TimeSeriesModel):
    def __init__(self, rng=None) -> None:
        super().__init__()
        self.rng = rng


class GaussianNoiseModel(NoiseModel):
    def __init__(self, rng) -> None:
        super().__init__(rng)

    @property
    def parameter_names(self):
        return ["sigma_noise"]
    

    def predict(self, timeseries, parameters):
        noise = keras.random.normal(
            timeseries.shape,
            mean=0,
            stddev=parameters["sigma_noise"],
            seed=self.rng
        )
        return timeseries + noise
    
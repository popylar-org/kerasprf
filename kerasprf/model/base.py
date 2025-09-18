
from abc import ABC, abstractmethod

import keras


class BaseModel(ABC):
    @property
    @abstractmethod
    def parameter_names(self):
        return []


class ResponseModel(BaseModel):
    @abstractmethod
    def predict(self, stimulus, parameters):
        return stimulus.grid
    

class TimeSeriesModel(BaseModel):
    @abstractmethod
    def predict(self, stimulus, parameters):
        return keras.ops.sum(stimulus.paradigm, axis=(0, 1))


class EncodingModel:
    def __init__(self, response_model):
        self.response_model = response_model

    def predict(self, stimulus, parameters):
        x = keras.ops.matmul(self.response_model.predict(stimulus, parameters), stimulus.paradigm)
        x = keras.ops.sum(x, axis=(0, 1))
        return x


class BaselineAmplitudeModel:
    def __init__(self, model):
        self.model = model

    @property
    def parameter_names(self):
        return ["baseline", "amplitude"]
    

    def predict(self, stimulus, parameters):
        return parameters["baseline"] + parameters["amplitude"] * self.model.predict(stimulus, parameters) 


class HRFModel:
    def __init__(self, model, time_length=32.0, dt=1.0):
        self.model = model
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


    def predict(self, stimulus, parameters):
        hrf = self.calc_hrf(parameters)
        return keras.ops.correlate(self.model.predict(stimulus, parameters), hrf)


class NoiseModel:
    def __init__(self, model, rng=None):
        self.model = model
        self.rng = rng


class GaussianNoiseModel(NoiseModel):
    def __init__(self, model, rng=None):
        super().__init__(model=model, rng=rng)

    @property
    def parameter_names(self):
        return ["sigma_noise"]
    

    def predict(self, stimulus, parameters):
        x = self.model.predict(stimulus, parameters)

        noise = keras.random.normal(
            x.shape,
            mean=0,
            stddev=parameters["sigma_noise"],
            seed=self.rng
        )
        return x + noise
    
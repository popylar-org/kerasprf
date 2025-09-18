
import keras

from kerasprf.model import BaselineAmplitudeModel, EncodingModel, GaussianNoiseModel, HRFModel, ResponseModel


class Gaussian2DResponseModel(ResponseModel):
    @property
    def parameter_names(self):
        return ["x", "y", "sigma"]


    def predict(self, stimulus, parameters):
        coordinates = keras.ops.convert_to_tensor(stimulus.coordinates)
        centroid = keras.ops.stack([parameters["x"], parameters["y"]], axis=-1)
        x = keras.ops.exp(-(keras.ops.sum((coordinates - centroid)**2, axis=-1) / (2 * parameters["sigma"]**2)))
        return x


class Gaussian2DModel:
    def __init__(self, model=None):
        if model is None:
            model = EncodingModel(Gaussian2DResponseModel())

        self.model = model


    def predict(self, stimulus, parameters):
        return self.model.predict(stimulus, parameters)
        

    @classmethod
    def from_default(cls, hrf_model=True, baseline_amplitude_model=True, noise_model=False):
        model = EncodingModel(Gaussian2DResponseModel())

        if hrf_model:
            model = HRFModel(model)
        
        if baseline_amplitude_model:
            model = BaselineAmplitudeModel(model)

        if noise_model:
            model = GaussianNoiseModel(model)

        return cls(model)
    
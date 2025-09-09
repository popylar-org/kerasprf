
import keras

from kerasprf.model.composite_model import CompositeModel
from kerasprf.model.encoding_model import EncodingModel
from kerasprf.model.timeseries_model import BaselineAmplitudeModel, HRFModel, GaussianNoiseModel 


class Gaussian2DModel(EncodingModel):
    @property
    def parameter_names(self):
        return ["centroid", "sigma"]


    def predict(self, stimulus, parameters):
        coordinates = keras.ops.convert_to_tensor(stimulus.coordinates)
        paradigm = keras.ops.convert_to_tensor(stimulus.paradigm)
        x = keras.ops.exp(-(keras.ops.sum((coordinates - parameters["centroid"])**2, axis=-1) / (2 * parameters["sigma"]**2))) * paradigm
        x = keras.ops.sum(x, axis=(0, 1))

        # if not training:
        #     return keras.ops.convert_to_numpy(x)

        return x
    

class Gaussian2DCompositeModel(CompositeModel):
    def __init__(self, encoding_model, *args, **kwargs):
        super().__init__(encoding_model, *args, **kwargs)

    @classmethod
    def from_default(cls, hrf_model=True, baseline_amplitude_model=True, noise_model=False):
        kwargs = {
            "encoding_model": Gaussian2DModel()
        }

        if hrf_model:
            kwargs["hrf_model"] = HRFModel()
        
        if baseline_amplitude_model:
            kwargs["baseline_amplitude_model"] = BaselineAmplitudeModel()

        if noise_model:
            kwargs["noise_model"] = GaussianNoiseModel()

        return cls(**kwargs)
    
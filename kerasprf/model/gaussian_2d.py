
import keras

from kerasprf.model.encoding_model import EncodingModel


class Gaussian2DModel(EncodingModel):
    @property
    def parameter_names(self):
        return ("centroid", "sigma")


    def predict(self, stimulus, parameters):
        coordinates = keras.ops.convert_to_tensor(stimulus.coordinates)
        paradigm = keras.ops.convert_to_tensor(stimulus.paradigm)
        x = keras.ops.exp(-(keras.ops.sum((coordinates - parameters["centroid"])**2, axis=-1) / (2 * parameters["sigma"]**2))) * paradigm
        x = keras.ops.sum(x, axis=(0, 1))

        # if not training:
        #     return keras.ops.convert_to_numpy(x)

        return x
    
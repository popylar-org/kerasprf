
import keras

from .backend.base_model import BackendModel

from kerasprf.adapter import Adapter


class Gaussian2DModel(BackendModel):
    @property
    def required_params(self):
        return set(["centroid", "sigma"])
    
    @staticmethod
    def create_default_adapter():
        return (Adapter()
            .transform(include="sigma", forward_fun=keras.ops.log, inverse_fun=keras.ops.exp)
            .broadcast(include="centroid", shape=(1, 1, 2))
        )
    
    @staticmethod
    def set_default_params():
        return dict(
            centroid=keras.ops.array([0, 0]),
            sigma=1.0
        )


    def call(self, grid, stimulus, training=None):
        params = self.params

        x = keras.ops.exp(-(keras.ops.sum((grid - params["centroid"])**2, axis=-1) / (2 * params["sigma"]**2))) * stimulus
        x = keras.ops.sum(x, axis=(0, 1))

        if not training:
            return keras.ops.convert_to_numpy(x)

        return x
    
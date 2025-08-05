
import keras

from tqdm import tqdm

from .backend.base_model import BackendModel


class Gaussian2DModel(BackendModel):
    def __init__(self, centroid, sigma, scale):
        super().__init__()
        self.centroid = keras.Variable(centroid, dtype="float32", name="centroid")
        self.sigma = keras.Variable(sigma, dtype="float32", name="sigma")
        self.scale = scale


    def call(self, grid, stimulus, training=None):
        x = keras.ops.exp(-(keras.ops.sum((grid - self.centroid[None, None, :])**2, axis=-1) / (2 * self.sigma**2))) * stimulus
        x = keras.ops.sum(x, axis=(0, 1)) / self.scale

        if not training:
            return keras.ops.convert_to_numpy(x)

        return x
    

    def fit(self, x, y, num_steps=1000):
        state = self.get_state()

        with tqdm(range(num_steps)) as pbar:
            for _ in pbar:
                logs, state = self.update_model_weights(x, y, state)
                
                if logs:
                    display_logs = {}
                    for key, value in logs.items():
                        try:
                            if hasattr(value, 'numpy'):
                                display_logs[key] = float(value.numpy())
                            else:
                                display_logs[key] = float(value)
                        except (AttributeError, TypeError):
                            display_logs[key] = str(value)
                    
                    pbar.set_postfix(display_logs)

        if state is not None:
            trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables = state
            for variable, value in zip(self.trainable_variables, trainable_variables):
                variable.assign(value)
            for variable, value in zip(self.non_trainable_variables, non_trainable_variables):
                variable.assign(value)

        return logs
    
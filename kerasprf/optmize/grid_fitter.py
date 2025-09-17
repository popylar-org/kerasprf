

import keras
import numpy as np


class GridFitter:
    def __init__(self, model, stimulus, adapter, optimizer, loss):
        self.model = model
        self.stimulus = stimulus
        self.adapter = adapter
        self.optimizer = optimizer
        self.loss = loss


    def fit(self, data, init_parameters):
        data = keras.ops.convert_to_tensor(data, dtype="float32")
        keys = list(init_parameters.keys())
        arrays = [np.asarray(init_parameters[k]) for k in keys]
        mesh = np.meshgrid(*arrays, indexing='ij')
        combos = keras.ops.convert_to_tensor(np.stack(mesh, axis=-1).reshape(-1, len(keys)), dtype="float32")

        predict_partial = lambda params: self.model.predict(self.stimulus, {k: v for k, v in zip(keys, params)})

        predictions = keras.ops.vectorized_map(predict_partial, combos)

        losses = keras.ops.vectorized_map(lambda y_pred: self.loss(data, y_pred), predictions)

        best_params = {k: v for k, v in zip(keys, combos[keras.ops.argmin(losses)])}

        return {"loss": keras.ops.min(losses)}, best_params

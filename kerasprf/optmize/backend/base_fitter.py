
import warnings

import keras

from tqdm import tqdm

from kerasprf.adapter import Adapter


match keras.backend.backend():
    case "jax":
        from .jax_fitter import JAXModel as BaseBackendModel
    case "tensorflow":
        from .tensorflow_fitter import TensorFlowModel as BaseBackendModel
    case "torch":
        from .torch_fitter import TorchModel as BaseBackendModel
    case other:
        raise ValueError(f"Backend '{other}' is not supported.")


class ParameterFitter(BaseBackendModel):
    def __init__(self, model, stimulus, adapter, optimizer, loss):
        super().__init__()

        self.model = model
        self.stimulus = stimulus
        self.adapter = adapter
        self.optimizer = optimizer
        self.loss = loss

        
    def _create_variables(self, init_parameters):
        init_parameters = self.adapter.forward(init_parameters)
        
        for key, val in init_parameters.items():
            setattr(self, key, keras.Variable(val, dtype="float32", name=key))

    
    def _delete_variables(self):
        pass   


    def fit(self, data, init_parameters, num_steps=1000):
        self._create_variables(init_parameters)

        self.optimizer.build(self.trainable_variables)

        self.compile(optimizer=self.optimizer, loss=self.loss)

        state = self.get_state()

        with tqdm(range(num_steps)) as pbar:
            for _ in pbar:
                logs, state = self.update_model_weights(self.stimulus, data, state)
                
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


import warnings

import keras

from tqdm import tqdm

from kerasprf.adapter import Adapter


match keras.backend.backend():
    case "jax":
        from .jax_model import JAXModel as BaseBackendModel
    case "tensorflow":
        from .tensorflow_model import TensorFlowModel as BaseBackendModel
    case "torch":
        from .torch_model import TorchModel as BaseBackendModel
    case other:
        raise ValueError(f"Backend '{other}' is not supported.")


class BackendModel(BaseBackendModel):
    def __init__(self, adapter=None, params=None):
        super().__init__()

        if adapter is None:
            adapter = self.create_default_adapter()

        if params is None:
            params = self.set_default_params()

        self.check_required_params(params)

        self.adapter = adapter

        params_adapted = self.adapter(params)
        
        for key, val in params_adapted.items():
            setattr(self, key, keras.Variable(val, dtype="float32", name=key))

    @property
    def required_params(self):
        return set()

    @property
    def params(self):
        trainable_varialbes_dict = {param.name: param.value for param in self.trainable_variables}
        return self.adapter(trainable_varialbes_dict, inverse=True)
    
    @staticmethod
    def create_default_adapter():
        return Adapter()
    
    @staticmethod
    def set_default_params():
        return dict()
    

    def check_required_params(self, params):
        required_params = set(self.required_params)
        user_params = set(params.keys())
        diff_required = required_params - user_params
        diff_user = user_params - required_params

        if len(diff_required) > 0:
            raise ValueError(f"Missing required parameters: {diff_required}")
        
        if len(diff_user) > 0:
            warnings.warn(f"Ignoring unused parameters: {diff_user}")


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

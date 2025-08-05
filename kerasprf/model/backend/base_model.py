
import keras


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
    pass

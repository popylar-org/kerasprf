
from abc import ABC, abstractmethod


class EncodingModel(ABC):
    @property
    @abstractmethod
    def parameter_names(self):
        return ()
    
    @abstractmethod
    def predict(stimulus, parameters):
        pass

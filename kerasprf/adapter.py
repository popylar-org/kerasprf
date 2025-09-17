
from abc import ABC, abstractmethod


class Transform(ABC):
    def __init__(self, forward_fun, inverse_fun):
        self.forward_fun = forward_fun
        self.inverse_fun = inverse_fun

    @abstractmethod
    def forward(self, data):
        return dict()

    @abstractmethod
    def inverse(self, data):
        return dict()


class ParameterTransform(Transform):
    def __init__(self, parameter_names, forward_fun, inverse_fun):
        super().__init__(forward_fun, inverse_fun)
        self.parameter_names = parameter_names


    def forward(self, data):
        return {key: (self.forward_fun(val) if key in self.parameter_names else val) for key, val in data.items()}
    

    def inverse(self, data):
        return {key: (self.inverse_fun(val) if key in self.parameter_names else val) for key, val in data.items()}


class Adapter:
    def __init__(self, transforms=None):
        if transforms is None:
            transforms = []
    
        self.transforms = transforms


    def forward(self, data):
        for transform in self.transforms:
            data = transform.forward(data)

        return data
    

    def inverse(self, data):
        for transform in self.transforms:
            data = transform.inverse(data)

        return data
    
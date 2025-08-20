
from abc import ABC, abstractmethod
from collections.abc import MutableSequence

import keras


class BaseTransform(ABC):
    def __init__(self, include):
        super().__init__()
        self.include = include

    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def inverse(self, data):
        pass


class Transform(BaseTransform):
    def __init__(self, forward_fun, inverse_fun, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_fun = forward_fun
        self.inverse_fun = inverse_fun


    def forward(self, data):
        return {key: (self.forward_fun(val) if key in self.include else val) for key, val in data.items()}
    

    def inverse(self, data):
        return {key: (self.inverse_fun(val) if key in self.include else val) for key, val in data.items()}


class Broadcast(BaseTransform):
    def __init__(self, shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = shape


    def forward(self, data):
        return {key: (keras.ops.broadcast_to(val, shape=self.shape) if key in self.include else val) for key, val in data.items()}
    

    def inverse(self, data):
        return data


class Adapter:
    def __init__(self, transforms=None):
        super().__init__()

        if transforms is None:
            transforms = []
    
        self.transforms = list(transforms)


    def forward(self, data):
        for transform in self.transforms:
            data = transform.forward(data)

        return data
    

    def inverse(self, data):
        for transform in self.transforms:
            data = transform.inverse(data)

        return data


    def __call__(self, data, inverse=False):
        if inverse:
            return self.inverse(data)

        return self.forward(data)


    def transform(self, include, forward_fun, inverse_fun):
        self.transforms.append(Transform(include=include, forward_fun=forward_fun, inverse_fun=inverse_fun))

        return self
    

    def broadcast(self, include, shape):
        self.transforms.append(Broadcast(include=include, shape=shape))

        return self
    
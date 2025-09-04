
from abc import ABC, abstractmethod
from collections.abc import MutableSequence

import keras


class BaseTransform(ABC):
    def __init__(self, include):
        self.include = include


    def filter(self, data, fun, *args, **kwargs):
        return {key: (fun(val, *args, **kwargs) if key in self.include else val) for key, val in data.items()}

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
        return self.filter(data=data, fun=self.forward_fun)
    

    def inverse(self, data):
        return self.filter(data=data, fun=self.inverse_fun)


class Broadcast(BaseTransform):
    def __init__(self, shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = shape


    def forward(self, data):
        return self.filter(data=data, fun=keras.ops.broadcast_to, shape=self.shape)
    

    def inverse(self, data):
        return data
    

class ExpandDims(BaseTransform):
    def __init__(self, axis, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis


    def forward(self, data):
        return self.filter(data=data, fun=keras.ops.expand_dims, axis=self.axis)
    

    def inverse(self, data):
        return data


class Repeat(BaseTransform):
    def __init__(self, repeats, axis, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeats = repeats
        self.axis = axis

    def forward(self, data):
        return self.filter(data=data, fun=keras.ops.repeat, repeats=self.repeats, axis=self.axis)
    

    def inverse(self, data):
        return data


class Adapter:
    def __init__(self, transforms=None):
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
    

    def expand_dims(self, include, axis):
        self.transforms.append(ExpandDims(include=include, axis=axis))

        return self
    

    def repeat(self, include, repeats, axis):
        self.transforms.append(Repeat(include=include, repeats=repeats, axis=axis))

        return self
    


class CompositeModel:
    def __init__(self, encoding_model, *args, **kwargs):
        self.encoding_model = encoding_model
        
        models_args = {f"model_{i}": arg for i, arg in enumerate(args)}
        models_kwargs = {key: val for key, val in kwargs.items()}

        self.models = models_args | models_kwargs


    def predict(self, stimulus, parameters):
        timeseries = self.encoding_model.predict(stimulus, parameters)

        for model in self.models.values():
            timeseries = model.predict(timeseries, parameters)

        return timeseries
    
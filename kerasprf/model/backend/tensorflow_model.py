
import keras
import tensorflow as tf


class TensorFlowModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def get_state(self):
        return None
    
    
    def update_model_weights(self, x, y, state=None):
        with tf.GradientTape() as tape:
            y_pred = self(*x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply(gradients, self.trainable_variables)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        logs = {m.name: m.result() for m in self.metrics}

        return logs, state
    
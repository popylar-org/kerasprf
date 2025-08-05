
import keras
import tensorflow as tf


class TensorFlowModel(keras.Model):
    def get_state(self):
        return None
    
    
    def update_model_weights(self, x, y, state=None):
        with tf.GradientTape() as tape:
            y_pred = self(*x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply(gradients, self.trainable_variables)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        logs = {m.name: m.result() for m in self.metrics}

        return logs, state
    
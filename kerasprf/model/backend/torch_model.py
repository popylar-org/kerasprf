
import keras
import torch


class TorchModel(keras.Model):
    def get_state(self):
        return None
    

    def update_model_weights(self, x, y, state=None):
        self.zero_grad()

        y_pred = self(*x, training=True)
        loss = self.compute_loss(y=y, y_pred=y_pred)

        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        logs = {m.name: m.result() for m in self.metrics}

        return logs, state
    
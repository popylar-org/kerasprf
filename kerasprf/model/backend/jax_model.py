
import jax
import jax.numpy as jnp
import keras


class JAXModel(keras.Model):
    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        x,
        y,
    ):
        # Convert JAX arrays to ensure compatibility
        trainable_variables = [jnp.array(v) for v in trainable_variables]
        non_trainable_variables = [jnp.array(v) for v in non_trainable_variables]
        
        state_mapping = []
        state_mapping.extend(zip(self.trainable_variables, trainable_variables))
        state_mapping.extend(zip(self.non_trainable_variables, non_trainable_variables))

        with keras.StatelessScope(state_mapping) as scope:
            y_pred = self(*x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        # update variables
        non_trainable_variables = [scope.get_current_value(v) for v in self.non_trainable_variables]

        return loss, (y_pred, non_trainable_variables)


    def get_state(self):
        return self.trainable_variables, self.non_trainable_variables, self.optimizer.variables, self.metrics_variables


    def update_model_weights(self, x, y, state):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state

        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)

        (loss, (y_pred, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            x,
            y,
        )

        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        new_metrics_vars = []
        logs = {}
        for metric in self.metrics:
            this_metric_vars = metrics_variables[
                len(new_metrics_vars) : len(new_metrics_vars) + len(metric.variables)
            ]
            if metric.name == "loss":
                this_metric_vars = metric.stateless_update_state(this_metric_vars, loss)
            else:
                this_metric_vars = metric.stateless_update_state(
                    this_metric_vars, y, y_pred
                )
            logs[metric.name] = metric.stateless_result(this_metric_vars)
            new_metrics_vars += this_metric_vars

        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )

        return logs, state
    
def AsyncStep(self, closure=None):
    if self.gradient_state.sync_gradients:
        if self.scaler is not None:
            self.scaler.step(self.optimizer, closure)
            self.scaler.update()
        else:
            self.optimizer.step(closure)

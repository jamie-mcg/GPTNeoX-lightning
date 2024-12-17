import lightning.pytorch as pl
import torch

class GradientNormLogger(pl.Callback):
    def __init__(self, norm_type: float = 2.0):
        super().__init__()
        self.norm_type = norm_type

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % trainer.log_every_n_steps == 0:
            grad_norms = self._compute_gradient_norms(pl_module)
            for name, norm in grad_norms.items():
                # WARNING: THIS IS REALLY NOT A GOOD WAY OF ACCESSING THE TENSORBOARD LOGGER
                pl_module.loggers[1].experiment.add_scalar(f'grad_norms/{name}', norm, trainer.global_step)

    def _compute_gradient_norms(self, pl_module):
        grad_norms = {}
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(self.norm_type).item()
                grad_norms[name] = norm
        return grad_norms
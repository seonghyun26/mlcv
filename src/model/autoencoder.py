import torch

from typing import Optional
from torch.optim import Adam
from mlcolvar.cvs import AutoEncoderCV


class AutoEncoderCV(AutoEncoderCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = Adam(self.parameters(), lr=1e-3)
        self.cv_normalize = False

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        optimizer = self.optimizer
        optimizer.step(closure=optimizer_closure)

    def backward(self, loss):
        loss.backward(retain_graph=True)
    
    def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_in is not None:
            x = self.norm_in(x)
        x = self.encoder(x)
        
        if self.cv_normalize:
            x = self._map_range(x)
        return x
    
    def set_cv_range(self, cv_min, cv_max, cv_std):
        self.cv_normalize = True
        self.cv_min = cv_min.detach()
        self.cv_max = cv_max.detach()
        self.cv_std = cv_std.detach()

    def _map_range(self, x):
        out_max = 1
        out_min = -1
        return (x - self.cv_min) * (out_max - out_min) / (self.cv_max - self.cv_min) + out_min
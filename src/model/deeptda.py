import torch

from mlcolvar.cvs import DeepTDA


class DeepTDA(DeepTDA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cv_normalize = False
        
    def set_cv_range(self, cv_min, cv_max, cv_std):
        self.cv_normalize = True
        self.cv_min = cv_min.detach()
        self.cv_max = cv_max.detach()
        self.cv_std = cv_std.detach()

    def _map_range(self, x):
        out_max = 1
        out_min = -1
        return (x - self.cv_min) * (out_max - out_min) / (self.cv_max - self.cv_min) + out_min

    def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.BLOCKS:
            block = getattr(self, b)
            if block is not None:
                x = block(x)

        if self.cv_normalize:
            x = self._map_range(x)
            
        return x

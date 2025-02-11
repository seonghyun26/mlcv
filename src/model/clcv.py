import torch
import lightning
import torch.nn as nn

from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization


class CLCV(BaseCV, lightning.LightningModule):
    BLOCKS = ["norm_in", "encoder",]

    def __init__(
        self,
        encoder_layers: list,
        loss_fn: str = "triplet",
        options: dict = None,
        **kwargs,
    ):
        super().__init__(in_features=encoder_layers[0], out_features=encoder_layers[-1], **kwargs)
        # ======= OPTIONS =======
        options = self.parse_options(options)
        self.cv_normalize = False
        self.cv_min = 0
        self.cv_max = 1

        # =======   LOSS  =======
        if loss_fn == "triplet":
            self.loss_fn = nn.TripletMarginLoss()
        else:
            raise ValueError(f"Loss function {loss_fn} not supported")
        
        # ======= BLOCKS =======
        # initialize norm_in
        o = "norm_in"
        if (options[o] is not False) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features, **options[o])

        # initialize encoder
        o = "encoder"
        self.encoder = FeedForward(encoder_layers, **options[o])

    def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the CV without pre or post/processing modules."""
        if self.norm_in is not None:
            x = self.norm_in(x)
        x = self.encoder(x)
        
        if self.cv_normalize:
            x = self._map_range(x)
        return x

    def set_cv_range(self, cv_min, cv_max, cv_std):
        self.cv_normalize = True
        self.cv_min = cv_min
        self.cv_max = cv_max
        self.cv_std = cv_std

    def _map_range(self, x):
        out_max = 1
        out_min = -1
        return (x - self.cv_min) * (out_max - out_min) / (self.cv_max - self.cv_min) + out_min

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_cv(x)
        # normalized_x = F.normalize(x, p=2, dim=1)
        
        return x

    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        # =================get data===================
        anchor = train_batch["data"]
        positive = train_batch["positive"]
        negative = train_batch["negative"]
        
        # =================forward====================
        anchor_rep = self.encode(anchor)
        positive_rep = self.encode(positive)
        negative_rep = self.encode(negative)
        
        # ===================loss=====================
        loss = self.loss_fn(anchor_rep, positive_rep, negative_rep)
        
        # ====================log=====================
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        return loss


if __name__ == "__main__":
    pass
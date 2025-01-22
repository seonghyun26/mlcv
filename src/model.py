import torch
import lightning

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from mlcolvar.cvs import BaseCV, AutoEncoderCV, VariationalAutoEncoderCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.transform.utils import Inverse


__all__ = ["CLCV"]

class AutoEncoderCV(AutoEncoderCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = Adam(self.parameters(), lr=1e-3)

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

class CLCV(BaseCV, lightning.LightningModule):
    BLOCKS = ["norm_in", "encoder",]

    def __init__(
        self,
        encoder_layers: list,
        options: dict = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        encoder_layers : list
            Number of neurons per layer of the encoder
        options : dict[str,Any], optional
            Options for the building blocks of the model, by default None.
            Available blocks: ['norm_in', 'encoder','decoder'].
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(
            in_features=encoder_layers[0], out_features=encoder_layers[-1], **kwargs
        )

        # =======   LOSS  =======
        # Reconstruction (MSE) loss
        self.loss_fn = nn.TripletMarginLoss()

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

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
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_cv(x)
        # normalized_data = F.normalize(x, p=2, dim=1)
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
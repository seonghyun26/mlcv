import torch
import lightning
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from typing import Any, Optional, Tuple

from mlcolvar.cvs import BaseCV, AutoEncoderCV, VariationalAutoEncoderCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.loss.mse import mse_loss


class VDELoss(torch.nn.Module):
    def forward(
        self,
        target: torch.Tensor,
        output: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        z_t: torch.Tensor,
        z_t_tau: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        elbo_loss = elbo_gaussians_loss(target, output, mean, log_variance, weights)
        auto_correlation_loss = 0
        
        z_t_mean = z_t.mean(dim=0)
        z_t_tau_mean = z_t_tau.mean(dim=0)
        z_t_centered = z_t - z_t_mean.repeat(z_t.shape[0], 1)
        z_t_tau_centered = z_t_tau - z_t_tau_mean.repeat(z_t_tau.shape[0], 1)
        
        auto_correlation_loss = - (z_t_centered @ z_t_tau_centered.T)[torch.eye(z_t.shape[0], dtype=torch.bool, device = z_t.device)].mean()
        auto_correlation_loss = auto_correlation_loss / (z_t.std(dim=0).T @ z_t_tau.std(dim=0))
        
        return elbo_loss + auto_correlation_loss

def elbo_gaussians_loss(
    target: torch.Tensor,
    output: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    kl = -0.5 * (log_variance - log_variance.exp() - mean**2 + 1).sum(dim=1)

    if weights is None:
        kl = kl.mean()
    else:
        weights = weights.squeeze()
        if weights.shape != kl.shape:
            raise ValueError(
                f"weights should be a tensor of shape (n_batches,) or (n_batches,1), not {weights.shape}."
            )
        kl = (kl * weights).sum()

    reconstruction = mse_loss(output, target, weights=weights)

    return reconstruction + kl


class VariationalDynamicsEncoder(VariationalAutoEncoderCV):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # =======   LOSS  =======
        # ELBO loss function when latent space and reconstruction distributions are Gaussians.
        self.loss_fn = VDELoss()
    
    # def encode_decode(
    #     self, x: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     if self.norm_in is not None:
    #         x = self.norm_in(x)

    #     # Encode input into a Gaussian distribution.
    #     z = self.encoder(x)
    #     mean, log_variance = self.mean_nn(z), self.log_var_nn(z)

    #     # Sample from the Gaussian distribution in latent space.
    #     std = torch.exp(log_variance / 2)
    #     z_sampled = torch.distributions.Normal(mean, std).rsample()

    #     # Decode sample.
    #     x_hat = self.decoder(z_sampled)
    #     if self.norm_in is not None:
    #         x_hat = self.norm_in.inverse(x_hat)

    #     return mean, log_variance, x_hat

    
    def training_step(self, train_batch, batch_idx):
        x = train_batch["data"]
        input = x
        loss_kwargs = {}
        if "weights" in train_batch:
            loss_kwargs["weights"] = train_batch["weights"]

        # Encode/decode.
        mean, log_variance, x_hat = self.encode_decode(x)

        # Reference output (compare with a 'target' key if any, otherwise with input 'data')
        if "target" in train_batch:
            x_ref = train_batch["target"]
        else:
            x_ref = x
        
        # Values for autocorrealtion loss
        if self.norm_in is not None:
            input_normalized = self.norm_in(input)
            x_ref_normalized = self.norm_in(x_ref)
        z_t = self.encoder(input_normalized)
        z_t_tau = self.encoder(x_ref_normalized)
        
        # Loss function.
        loss = self.loss_fn(
            x_ref, x_hat, mean, log_variance,
            z_t, z_t_tau,
            **loss_kwargs
        )

        # Log.
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)

        return loss


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
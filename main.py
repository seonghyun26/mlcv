import os
import wandb
import hydra
import torch
import random
import string
import logging
import lightning

from mlcolvar.utils.trainer import MetricsCallback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src import *
from omegaconf import OmegaConf


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="basic"
)
def main(cfg):
    # Load configs and components
    lightning_logger = load_lightning_logger(cfg)
    logger = logging.getLogger("MLCVs")
    logger.info(">> Configs")
    logger.info(OmegaConf.to_yaml(cfg))
    device = "cuda"
    model = load_model(cfg).to(device)
    datamodule = load_data(cfg)
    
    # train
    logger.info(">> Training...")
    metrics = MetricsCallback()
    early_stopping = EarlyStopping(**cfg.model.trainer.early_stopping)
    trainer = lightning.Trainer(
        callbacks=[metrics, early_stopping],
        logger=lightning_logger,
        max_epochs=cfg.model.trainer.max_epochs,
        enable_checkpointing=False
    )
    trainer.fit(model, datamodule)
    logger.info(">> Training complete.!!")
    
    # Plot CVs
    model.eval()
    logger.info("")
    logger.info(">> Saving plots")
    checkpoint_path = f"./model/{cfg.model.name}/{cfg.data.version}"
    if not os.path.exists(f"./model/{cfg.model.name}"):
        os.makedirs(f"./model/{cfg.model.name}")
    if os.path.exists(checkpoint_path + "-jit.pt") or os.path.exists(checkpoint_path + ".pt"):
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        checkpoint_path += f"-{random_suffix}"
    plot_ad_cv(
        cfg = cfg,
        model = model,
        datamodule = datamodule,
        checkpoint_path = checkpoint_path,
    )
    
    # Save model
    logger.info(">> Saving model")
    torch.save(model.state_dict(), checkpoint_path + ".pt")
    example_input = torch.rand(1, cfg.model.input_dim)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(checkpoint_path + "-jit.pt")
    logger.info(f"Model saved at {checkpoint_path}")
    
    # finish
    wandb.finish()
    return


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
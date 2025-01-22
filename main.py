import os
import wandb
import hydra
import torch
import random
import string
import logging
import lightning

import numpy as np

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.io import create_dataset_from_files

from src import *
from tqdm.auto import tqdm, trange
from omegaconf import DictConfig, OmegaConf, open_dict


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="basic"
)
def main(cfg):
    # Load configs)
    lightning_logger = load_lightning_logger(cfg)
    logger = logging.getLogger("CMD")
    logger.info(">> Configs")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Load components
    device = "cuda"
    model = load_model(cfg).to(device)
    datamodule = load_data(cfg)
    
    # train
    logger.info(">> Training")
    metrics = MetricsCallback()
    early_stopping = EarlyStopping(**cfg.trainer.early_stopping)
    trainer = lightning.Trainer(
        callbacks=[metrics, early_stopping],
        logger=lightning_logger,
        max_epochs=None,
        enable_checkpointing=False
    )
    trainer.fit(model, datamodule)
    logger.info("Training complete")
    
    # Save model
    checkpoint_path = f"./model/{cfg.name}/{cfg.data.version}"
    if not os.path.exists(f"./model/{cfg.name}"):
        os.makedirs(f"./model/{cfg.name}")
    if os.path.exists(checkpoint_path + "-jit.pt") or os.path.exists(checkpoint_path + ".pt"):
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        checkpoint_path += f"-{random_suffix}"
    torch.save(model, checkpoint_path + ".pt")
    example_input = torch.randn(1, cfg.input_dim)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(checkpoint_path + "-jit.pt")
    
    logger.info(f"Model saved at {checkpoint_path}")
    
    # Plot CVs
    plot_ad_cv(
        cfg = cfg,
        model = model,
        datamodule = datamodule,
        checkpoint_path = checkpoint_path,
    )
    
    return

if __name__ == "__main__":
    # torch.manual_seed(41)
    main()
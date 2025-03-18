import os
import hydra
import torch
import logging
import lightning

from mlcolvar.utils.trainer import MetricsCallback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src import *
from src import plot_ad_cv
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
    checkpoint_path = f"./model/{cfg.model.name}/{cfg.data.version}"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if cfg.model.checkpoint:
        logger.info(">> Loading model from checkpoint")
        model.load_state_dict(torch.load(f"{checkpoint_path}/{cfg.model.checkpoint_name}.pt"))
        model.eval()
    else:
        logger.info(">> Training...")
        metrics = MetricsCallback()
        early_stopping = EarlyStopping(**cfg.model.trainer.early_stopping)
        trainer = lightning.Trainer(
            callbacks=[metrics, early_stopping],
            logger=lightning_logger,
            max_epochs=cfg.model.trainer.max_epochs,
            enable_checkpointing=False,
            log_every_n_steps=1,
        )
        trainer.fit(model, datamodule)
        model.eval()
        logger.info(">> Training complete.!!")
    
    
        # Save model
        logger.info(">> Saving model")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save(model.state_dict(), checkpoint_path + "/final.pt")
        
        random_input = torch.rand(1, datamodule.dataset["data"].shape[1])
        traced_script_module = torch.jit.trace(model, random_input)
        traced_script_module.save(checkpoint_path + "/final-jit.pt")
        logger.info(f"Model saved at {checkpoint_path}")
    
    
    # Evaluation
    logger.info(">> Evaluating...")
    eval(
        cfg = cfg,
        model = model,
        logger = logger,
        datamodule = datamodule,
        checkpoint_path = checkpoint_path
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
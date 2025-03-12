import os
import torch
import wandb

from torch import optim
from omegaconf import OmegaConf
from mlcolvar.data import DictDataset, DictModule

from .util import *
from .data import *
from .model import *


model_dict = {
    "clcv": CLCV,
    "deeplda": DeepLDA,
    "deeptda": DeepTDA,
    "deeptica": DeepTICA,
    "autoencoder": AutoEncoderCV,
    "timelagged-autoencoder": AutoEncoderCV,
    "vde": VariationalDynamicsEncoder
}


def load_model(cfg):
    if cfg.model.name == "deeptica":
        model = DeepTICA(
            layers = cfg.model.model.layers,
            n_cvs = cfg.model.model.n_cvs,
            options = dict(cfg.model.model.options)
        )
    
    elif cfg.model.name == "gnncv-tica":
        import mlcolvar.graph as mg
        mg.utils.torch_tools.set_default_dtype('float32')
        scheduler_name = cfg.model.trainer.optimizer.lr_scheduler.scheduler
        if scheduler_name == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR
        optimizer_options = {
            'optimizer': cfg.model.trainer.optimizer.optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'gamma': cfg.model.trainer.optimizer.lr_scheduler.gamma
            }
        }
        model = mg.cvs.GraphDeepTICA(
            n_cvs=cfg.n_cvs,
            cutoff=cfg.cutoff,
            atomic_numbers=cfg.atomic_numbers,
            model_options=dict(cfg.model.model),
            optimizer_options=optimizer_options,
        )
    
    elif cfg.model.name in model_dict:
        model = model_dict[cfg.model.name](**cfg.model.model)
    
    else:
        raise ValueError(f"Model {cfg.model.name} not found")
    
    print(">> Model")
    print(model)
    return model


def load_lightning_logger(cfg):
    if cfg.model.logger.name == "wandb":
        from lightning.pytorch.loggers import WandbLogger
        wandb.init(
            project = cfg.model.logger.project,
            entity = "eddy26",
            tags = cfg.model.logger.tags,
            config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        )
        lightning_logger = WandbLogger(
            project = cfg.model.logger.project,
            log_model = cfg.model.logger.log_model
        )
    else:
        lightning_logger = None
        
    return lightning_logger


def load_data(cfg):
    data_dir = os.path.join(
        cfg.data.dir,
        cfg.data.molecule,
        str(cfg.data.temperature),
        cfg.data.version
    )
    
    if cfg.model.name in ["deeplda", "deeptda"]:
        custom_data = torch.load(os.path.join(data_dir, "distance.pt"))
        custom_label = torch.load(os.path.join(data_dir, "label.pt"))
        dataset = DictDataset({
            "data": custom_data,
            "labels": custom_label
        })
        datamodule = DictModule(
            dataset = dataset,
            lengths = [0.8,0.2],
            batch_size=cfg.model.trainer.batch_size
        )
        
    elif cfg.model.name == "deeptica":
        custom_data = torch.load(os.path.join(data_dir, "distance.pt"))
        custom_data_lag = torch.load(os.path.join(data_dir, "distance-timelag.pt"))
        dataset = DictDataset({
            "data": custom_data,
            "data_lag": custom_data_lag,
            "weights": torch.ones(custom_data.shape[0], dtype=torch.float32, device=custom_data.device),
            "weights_lag": torch.ones(custom_data_lag.shape[0], dtype=torch.float32, device=custom_data_lag.device)
        })
        datamodule = DictModule(
            dataset = dataset,
            lengths = [0.8,0.2],
            batch_size=cfg.model.trainer.batch_size
        )
    
    elif cfg.model.name == "autoencoder":
        custom_data = torch.load(os.path.join(data_dir, "xyz-aligned.pt"))
        backbone_atom_data = custom_data[:, ALANINE_BACKBONE_ATOM_IDX]
        dataset = DictDataset({
            "data": backbone_atom_data.reshape(backbone_atom_data.shape[0], -1),
        })
        datamodule = DictModule(
            dataset = dataset,
            lengths = [0.8,0.2],
            batch_size=cfg.model.trainer.batch_size
        )
    
    elif cfg.model.name == "timelagged-autoencoder":
        custom_data = torch.load(os.path.join(data_dir, "xyz-aligned.pt"))
        custom_data_lag = torch.load(os.path.join(data_dir, "xyz-aligned-timelag.pt"))
        backbone_atom_data = custom_data[:, ALANINE_HEAVY_ATOM_IDX]
        backbone_atom_data_lag = custom_data_lag[:, ALANINE_HEAVY_ATOM_IDX]
        backbone_atom_data_lag.requires_grad = True
        dataset = DictDataset({
            "data": backbone_atom_data.reshape(backbone_atom_data.shape[0], -1),
            "target": backbone_atom_data_lag.reshape(backbone_atom_data.shape[0], -1),
        })
        datamodule = DictModule(
            dataset = dataset,
            lengths = [0.8,0.2],
            batch_size=cfg.model.trainer.batch_size
        )
    
    elif cfg.model.name == "gnncv-tica":
        import mlcolvar.graph as mg
        mg.utils.torch_tools.set_default_dtype('float32')
        graph_dataset = torch.load(os.path.join(data_dir, "graph-dataset.pt"))
        datasets = mg.utils.timelagged.create_timelagged_datasets(
            graph_dataset, lag_time=2
        )
        datamodule = mg.data.GraphCombinedDataModule(
            datasets,
            random_split = False,
            batch_size = cfg.model.trainer.batch_size
        )
    
    elif cfg.model.name == "vde":
        custom_data = torch.load(os.path.join(data_dir, "distance-aligned.pt"))
        custom_data_lag = torch.load(os.path.join(data_dir, "distance-timelag.pt"))
        dataset = DictDataset({
            "data": custom_data,
            "target": custom_data_lag,
        })
        datamodule = DictModule(
            dataset = dataset,
            lengths = [0.8,0.2],
            batch_size=cfg.model.trainer.batch_size
        )
    
    else:
        raise ValueError(f"Data not found for model {cfg.model.name}")
    
    print(">> Dataset")
    print(datamodule)
    return datamodule


def load_checkpoint(cfg, model):
    checkpoint_path = f"./model/{cfg.model.name}/{cfg.data.version}"
    
    if not "checkpoint" in cfg.model or cfg.model["checkpoint"]:
        raise ValueError(f"Checkpoint path disabled, check config")
    
    if cfg.model.name == "deeplda":
        model.load_state_dict(torch.load(cfg.model.checkpoint_path))
    
    elif cfg.model.name == "deeptda":
        model.load_state_dict(torch.load(cfg.model.checkpoint_path))
    
    else:
        raise ValueError(f"Checkpoint not found for model {cfg.model.name}")
    
    return model
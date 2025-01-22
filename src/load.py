import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch import optim
from omegaconf import DictConfig, OmegaConf, open_dict

from mlcolvar.cvs import DeepLDA, DeepTDA, DeepTICA
from mlcolvar.data import DictDataset, DictModule

from .util import *
from .dataset import *
from .model import CLCV, AutoEncoderCV, VariationalDynamicsEncoder


model_dict = {
    "clcv": CLCV,
    "deeplda": DeepLDA,
    "deeptda": DeepTDA,
    "deeptica": DeepTICA,
    "autoencoder": AutoEncoderCV,
    "timelagged-autoencoder": AutoEncoderCV,
    "vde": VariationalDynamicsEncoder
}

ALDP_PHI_ANGLE = [4, 6, 8, 14]
ALDP_PSI_ANGLE = [6, 8, 14, 16]
ALANINE_HEAVY_ATOM_IDX = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]
ALANINE_BACKBONE_ATOM_IDX = [1, 4, 6, 8, 10, 14, 16, 18]


def load_model(cfg):
    if cfg.name == "deeptica":
        model = DeepTICA(
            layers = cfg.model.layers,
            n_cvs = cfg.model.n_cvs,
            options = {'nn': {'activation': 'tanh'} }
        )
    
    elif cfg.name in model_dict:
        model = model_dict[cfg.name](**cfg.model)
    
    elif cfg.name == "gnncv-tica":
        import mlcolvar.graph as mg
        mg.utils.torch_tools.set_default_dtype('float32')
        model = mg.cvs.GraphDeepTICA(
            n_cvs=cfg.n_cvs,
            cutoff=cfg.cutoff,
            atomic_numbers=cfg.atomic_numbers,
            model_options=dict(cfg.model),
            optimizer_options=dict(cfg.trainer.optimizer),
        )
        
    elif cfg.name == "gnncv-tda":
        import mlcolvar.graph as mg
        mg.utils.torch_tools.set_default_dtype('float32')
        model = mg.cvs.GraphDeepTDA(
            n_cvs=cfg.n_cvs,
            cutoff=cfg.cutoff,
            atomic_numbers=cfg.atomic_numbers,
            model_options=dict(cfg.model),
            optimizer_options=dict(cfg.trainer.optimizer),
        )
    
    else:
        raise ValueError(f"Model {cfg.name} not found")
    
    print(">> Model")
    print(model)
    return model

def load_lightning_logger(cfg):
    if cfg.trainer.logger.name == "wandb":
        from lightning.pytorch.loggers import WandbLogger
        wandb.init(
            project = cfg.trainer.logger.project,
            entity = "eddy26",
            tags = cfg.trainer.logger.tags,
            config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        )
        lightning_logger = WandbLogger(
            project = cfg.trainer.logger.project,
            log_model = cfg.trainer.logger.log_model
        )
    else:
        lightning_logger = None
        
    return lightning_logger

def load_data(cfg):
    if cfg.name == "clcv":
        custom_dataset = torch.load(f"../../data/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}/cl-distance.pt")
        dataset = DictDataset({
            "data": custom_dataset.x,
            "positive": custom_dataset.x_augmented,
            "negative": custom_dataset.x_augmented_hard,
        })
        datamodule = DictModule(dataset,lengths=[0.8,0.2])

    elif cfg.name in ["deeplda", "deeptda"]:
        custom_data = torch.load(f"../../data/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}/cl-distance.pt")
        custom_label = torch.load(f"../../data/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}/label.pt")
        dataset = DictDataset({
            "data": custom_data,
            "labels": custom_label
        })
        datamodule = DictModule(dataset,lengths=[0.8,0.2])
        
    elif cfg.name == "deeptica":
        custom_data = torch.load(f"../../data/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}/cl-distance.pt")
        custom_data_lag = torch.load(f"../../data/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}/cl-distance-timelag.pt")
        dataset = DictDataset({
            "data": custom_data,
            "data_lag": custom_data_lag,
            "weights": torch.ones(custom_data.shape[0], dtype=torch.float32, device=custom_data.device),
            "weights_lag": torch.ones(custom_data_lag.shape[0], dtype=torch.float32, device=custom_data_lag.device)
        })
        datamodule = DictModule(dataset,lengths=[0.8,0.2])
    
    elif cfg.name == "autoencoder":
        data = torch.load(f"../../data/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}/cl-xyz-aligned.pt")
        backbone_atom_data = data[:, ALANINE_BACKBONE_ATOM_IDX]
        custom_dataset = DictDataset({
            "data": backbone_atom_data.reshape(backbone_atom_data.shape[0], -1),
        })
        datamodule = DictModule(custom_dataset,lengths=[0.8,0.2])
    
    elif cfg.name == "timelagged-autoencoder":
        custom_data = torch.load(f"../../data/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}/cl-xyz-aligned.pt")
        custom_data_lag = torch.load(f"../../data/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}/cl-xyz-aligned-timelag.pt")
        
        backbone_atom_data = custom_data[:, ALANINE_HEAVY_ATOM_IDX]
        backbone_atom_data_lag = custom_data_lag[:, ALANINE_HEAVY_ATOM_IDX]
        backbone_atom_data_lag.requires_grad = True
        custom_dataset = DictDataset({
            "data": backbone_atom_data.reshape(backbone_atom_data.shape[0], -1),
            "target": backbone_atom_data_lag.reshape(backbone_atom_data.shape[0], -1),
        })
        datamodule = DictModule(custom_dataset,lengths=[0.8,0.2])
    
    elif cfg.name in ["gnncv-tica", "gnncv-tda"]:
        import mlcolvar.graph as mg
        mg.utils.torch_tools.set_default_dtype('float32')
        dataset = torch.load(f"../../data/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}/graph-dataset.pt")
        datasets = mg.utils.timelagged.create_timelagged_datasets(
            dataset, lag_time=2
        )
        datamodule = mg.data.GraphCombinedDataModule(
            datasets, random_split=False, batch_size=5000
        )
    
    elif cfg.name == "vde":
        custom_data = torch.load(f"../../data/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}/cl-distance.pt")
        custom_data_lag = torch.load(f"../../data/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}/cl-distance-timelag.pt")
        dataset = DictDataset({
            "data": custom_data,
            "target": custom_data_lag,
        })
        datamodule = DictModule(dataset,lengths=[0.8,0.2])
    
    else:
        raise ValueError(f"Data not found for model {cfg.name}")
    
    print(">> Data")
    print(datamodule)
    return datamodule

import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch import optim
from omegaconf import DictConfig, OmegaConf, open_dict

from mlcolvar.cvs import DeepLDA, DeepTDA, DeepTICA
from mlcolvar.cvs import AutoEncoderCV, VariationalAutoEncoderCV
from mlcolvar.data import DictDataset, DictModule
from mlcolvar.core.transform import Normalization

from .util import *
from .dataset import *
from .model import CLCV

model_dict = {
    "clcv": CLCV,
    "deeplda": DeepLDA,
    "deeptda": DeepTDA,
    "deeptica": DeepTICA,
    "autoencoder": AutoEncoderCV,
    "timelagged-autoencoder": AutoEncoderCV,
    "vae": VariationalAutoEncoderCV
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
        if cfg.name == ["autoencoder", "timelagged-autoencoder"]:
            optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
        custom_dataset = DictDataset({
            "data": data[:, ALANINE_HEAVY_ATOM_IDX].reshape(-1, 22, 3)
        })
        datamodule = DictModule(custom_dataset,lengths=[0.8,0.2])
    
    elif cfg.name == "timelagged-autoencoder":
        custom_dataset = torch.load(f"../../data/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}/cl-xyz-aligned-tae.pt")
        current_state, time_lagged_state = custom_dataset[0, :, ALANINE_HEAVY_ATOM_IDX], custom_dataset[1, :, ALANINE_HEAVY_ATOM_IDX]
        data_num = current_state.shape[0]
        current_state = current_state.reshape(data_num, -1)
        time_lagged_state = time_lagged_state.reshape(data_num, -1)
        
        # Preprocessing 
        # 1. mean-free cooridnates
        current_state_normalized = current_state - torch.mean(current_state, dim=0)
        time_lagged_state_normalized = time_lagged_state - torch.mean(time_lagged_state, dim=0)
        # 2. Whitening with C matrix
        from scipy.linalg import fractional_matrix_power 
        matrix_cov_00 = current_state_normalized.T @ current_state_normalized / data_num
        matrix_cov_tau = time_lagged_state_normalized.T @ time_lagged_state_normalized / data_num
        matrix_cov_00_pow_minus_half = fractional_matrix_power(matrix_cov_00, -0.5)
        matrix_cov_tau_pow_minus_half = fractional_matrix_power(matrix_cov_tau, -0.5)
        current_state_whitened = current_state_normalized @ torch.Tensor(matrix_cov_00_pow_minus_half, device=current_state_normalized.device).T
        time_lagged_state_whitened = time_lagged_state_normalized @ torch.Tensor(matrix_cov_tau_pow_minus_half, device=time_lagged_state_normalized.device).T
        
        custom_dataset = DictDataset({
            "data": current_state_whitened.reshape(data_num, -1),
            "target": time_lagged_state_whitened.reshape(data_num, -1),
        })
        datamodule = DictModule(custom_dataset,lengths=[0.8,0.2])
    
    elif cfg.name == "vae":
        pass
    
    else:
        raise ValueError(f"Data not found for model {cfg.name}")
    
    print(">> Data")
    print(datamodule)
    return datamodule

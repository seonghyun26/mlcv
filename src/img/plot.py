import os
import torch
import wandb

import numpy as np
import pandas as pd
import mdtraj as md

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from ..util import compute_dihedral_torch
from ..util.constant import *


def plot_ad_cv(
    cfg,
    model,
    datamodule,
    checkpoint_path,
):
    if cfg.name in ["deeplda", "deeptda"]:
        cv_dim = 1
    elif cfg.name in ["deeptica", "vde"]:
        cv_dim = cfg.model["n_cvs"]
    elif cfg.name in ["autoencoder", "timelagged-autoencoder", "vde", "clcv"]:
        cv_dim = cfg.model["encoder_layers"][-1]
    else:
        raise ValueError(f"Model {cfg.name} not found")
    
    # Load data
    projection_dataset = datamodule.dataset["data"]
    data_dir = os.path.join(
        cfg.data.dir,
        cfg.data.molecule,
        str(cfg.data.temperature),
        cfg.data.version
    )
    psi_list = np.load(f"{data_dir}/psi.npy")
    phi_list = np.load(f"{data_dir}/phi.npy")
    
    # Compute CV
    cv = model(projection_dataset)
    for i in range(cv_dim):
        wandb.log({
            f"cv/cv{i}/min": cv[:, i].min(),
            f"cv/cv{i}/max": cv[:, i].max(),
            f"cv/cv{i}/std": cv[:, i].std()
        })
    
    # CV Normalization
    print(f"CV range: {cv.min(dim=0)[0].item()} ~ {cv.max(dim=0)[0].item()}")
    model.set_cv_range(cv.min(dim=0)[0], cv.max(dim=0)[0], cv.std(dim=0)[0])
    cv = model(projection_dataset)
    print(f"CV normalized range: {cv.min(dim=0)[0].item()} ~ {cv.max(dim=0)[0].item()}")
    df = pd.DataFrame({
        **{f'CV{i}': cv[:, i].detach().cpu().numpy() for i in range(cv_dim)},
        'psi': psi_list.squeeze(),
        'phi': phi_list.squeeze()
    })

    
    # Plot the projection of CVs
    fig, axs = plt.subplots(2, 2, figsize = ( 15, 12 ) )
    axs = axs.ravel()
    norm = colors.Normalize(
        vmin=min(df[f'CV{i}'].min() for i in range(min(cv_dim, 9))),
        vmax=max(df[f'CV{i}'].max() for i in range(min(cv_dim, 9)))
    )
    
    # Plot CVs
    start_state_xyz = md.load(f"../simulation/data/alanine/c5.pdb").xyz
    goal_state_xyz = md.load(f"../simulation/data/alanine/c7ax.pdb").xyz
    start_state = torch.tensor(start_state_xyz)
    goal_state = torch.tensor(goal_state_xyz)
    phi_start = compute_dihedral_torch(start_state[:, ALDP_PHI_ANGLE])
    psi_start = compute_dihedral_torch(start_state[:, ALDP_PSI_ANGLE])
    phi_goal = compute_dihedral_torch(goal_state[:, ALDP_PHI_ANGLE])
    psi_goal = compute_dihedral_torch(goal_state[:, ALDP_PSI_ANGLE])
    for i in range(min(cv_dim, 4)):
        ax = axs[i]
        df.plot.hexbin(
            'phi','psi', C=f"CV{i}",
            cmap="viridis", ax=ax,
            gridsize=100,
            norm=norm
        )
        ax.scatter(phi_start, psi_start, edgecolors="black", c="w", zorder=101, s=100)
        ax.scatter(phi_goal, psi_goal, edgecolors="black", c="w", zorder=101, s=300, marker="*")
    
    save_dir = checkpoint_path + "-cv-plot.png"
    fig.savefig(save_dir)
    plt.close()
    print(f"CV plot saved at {save_dir}")
    
    wandb.log({"cv-plot": wandb.Image(save_dir)})
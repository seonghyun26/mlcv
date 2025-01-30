import torch
import wandb

import numpy as np
import pandas as pd
import mdtraj as md

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from tqdm import tqdm

from .util import compute_dihedral_torch


ALDP_PHI_ANGLE = [4, 6, 8, 14]
ALDP_PSI_ANGLE = [6, 8, 14, 16]


def map_range(x, in_min, in_max):
    out_max = 1
    out_min = -1
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def plot_ad_cv(
    cfg,
    model,
    datamodule,
    checkpoint_path,
):
    model.eval()
    cv_list = []
    if cfg.name == "deeplda" or cfg.name == "deeptda":
        cv_dim = 1
    elif cfg.name in ["deeptica", "vde"]:
        cv_dim = cfg.model["n_cvs"]
    elif cfg.name in ["autoencoder", "timelagged-autoencoder", "vde", "clcv"]:
        cv_dim = cfg.model["encoder_layers"][-1]
    else:
        raise ValueError(f"Model {cfg.name} not found")
    
    # Load data
    projection_dataset = datamodule.dataset["data"]
    data_dir = f"../../data/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}"
    psi_list = np.load(f"{data_dir}/psi.npy")
    phi_list = np.load(f"{data_dir}/phi.npy")
    
    # Compute CV
    for data in tqdm(
        projection_dataset,
        desc = f"Computing CVs for {cfg.name}"
    ):
        cv = model(data)
        cv_list.append(cv)
    cv_list = torch.stack(cv_list)
    
    # Scaling for some cases
    print(f"CV range: {cv_list.min()} ~ {cv_list.max()}")
    if cfg.name == "deeptda":
        cv_list = cv_list / cfg.output_scale
    elif cfg.name in ["autoencoder", "timelagged-autoencoder", "vde"]:
        cv_list = map_range(cv_list, cv_list.min(), cv_list.max())
    elif cfg.name in ["clcv", "deeptica"]:
        model.set_cv_range(cv_list.min(), cv_list.max())
        cv_list = model(projection_dataset)
    print(f"CV normalized range: {cv_list.min()} ~ {cv_list.max()}")
        
    df = pd.DataFrame({
        **{f'CV{i}': cv_list[:, i].detach().cpu().numpy() for i in range(cv_dim)},
        'psi': psi_list.squeeze(),
        'phi': phi_list.squeeze()
    })
    for i in range(cv_dim):
        wandb.log({
            f"cv/cv{i}/min": df[f'CV{i}'].min(),
            f"cv/cv{i}/max": df[f'CV{i}'].max(),
            f"cv/cv{i}/std": df[f'CV{i}'].std()
        })
    
    # Plot the projection of CVs
    fig, axs = plt.subplots(2, 2, figsize = ( 15, 12 ) )
    axs = axs.ravel()
    norm = colors.Normalize(
        vmin=min(df[f'CV{i}'].min() for i in range(min(cv_dim, 9))),
        vmax=max(df[f'CV{i}'].max() for i in range(min(cv_dim, 9)))
    )
    
    # Plot CVs
    start_state_xyz = md.load(f"../../data/alanine/c5.pdb").xyz
    goal_state_xyz = md.load(f"../../data/alanine/c7ax.pdb").xyz
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
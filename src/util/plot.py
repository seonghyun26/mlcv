import os
import torch
import wandb

import numpy as np
import pandas as pd
import mdtraj as md

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from ..util.constant import *
from ..util.angle import compute_dihedral

def save_plot(dir, fig):
    fig.savefig(f"{dir}")
    print(f"Saved plot at {dir}")
    plt.close()



def plot_angle_distribution(
    cfg,
    trajectory_list,
    checkpoint_path
):
    molecule = cfg.steeredmd.molecule
    
    return


def plot_ad_cv(
    cfg,
    model,
    datamodule,
    checkpoint_path,
):
    if cfg.model.name in ["deeplda", "deeptda"]:
        cv_dim = 1
    elif cfg.model.name in ["deeptica", "vde"]:
        cv_dim = cfg.model["n_cvs"]
    elif cfg.model.name in ["autoencoder", "timelagged-autoencoder", "vde"]:
        cv_dim = cfg.model["encoder_layers"][-1]
    elif cfg.model.name == "tbgcv":
        cv_dim = cfg.model.model["encoder_layers"][-1]
    else:
        raise ValueError(f"Model {cfg.model.name} not found")
    
    # Load data
    projection_dataset = datamodule.dataset["data"].to(model.device)
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
        print(f"CV {i} range: {cv[:, i].min(dim=0)[0].item()} ~ {cv[:, i].max(dim=0)[0].item()}")
    
    # CV Normalization
    model.set_cv_range(cv.min(dim=0)[0], cv.max(dim=0)[0], cv.std(dim=0)[0])
    cv = model(projection_dataset)
    for i in range(cv_dim):
        print(f"CV {i} normalized range: {cv[:, i].min(dim=0)[0].item()} ~ {cv[:, i].max(dim=0)[0].item()}")
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
    c5 = torch.load(f"../simulation/data/alanine/c5.pt")
    c7ax = torch.load(f"../simulation/data/alanine/c7ax.pt")
    phi_start, psi_start = c5["phi"], c5["psi"]
    phi_goal, psi_goal = c7ax["phi"], c7ax["psi"]
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
    
    save_dir = checkpoint_path + "/cv-plot.png"
    fig.savefig(save_dir)
    print(f"CV plot saved at {save_dir}")
    
    wandb.log({"cv-plot": wandb.Image(save_dir)})
    plt.close()

    return save_dir



def plot_paths(cfg, trajectory_list, hit_mask, hit_index, seed, checkpoint_path):
    molecule = cfg.steeredmd.molecule
    
    if molecule == "alanine":
        # Load start, goal state and compute phi, psi
        start_state_xyz = md.load(f"../simulation/data/alanine/{cfg.steeredmd.start_state}.pdb").xyz
        goal_state_xyz = md.load(f"../simulation/data/alanine/{cfg.steeredmd.goal_state}.pdb").xyz
        start_state = torch.tensor(start_state_xyz)
        goal_state = torch.tensor(goal_state_xyz)
        phi_start = compute_dihedral(start_state[:, ALDP_PHI_ANGLE])
        psi_start = compute_dihedral(start_state[:, ALDP_PSI_ANGLE])
        phi_goal = compute_dihedral(goal_state[:, ALDP_PHI_ANGLE])
        psi_goal = compute_dihedral(goal_state[:, ALDP_PSI_ANGLE])
    
        # Compute phi, psi from trajectory_list
        phi_traj_list = [compute_dihedral(trajectory[:, ALDP_PHI_ANGLE]) for trajectory in trajectory_list]
        psi_traj_list = [compute_dihedral(trajectory[:, ALDP_PSI_ANGLE]) for trajectory in trajectory_list]
        
        ram_plot_img = plot_ad_traj(
            checkpoint_path = checkpoint_path,
            traj_dihedral = (phi_traj_list, psi_traj_list),
            start_dihedral = (phi_start, psi_start),
            goal_dihedral = (phi_goal, psi_goal),
            type = "all",
            seed = seed,
        )
        
        hit_phi_traj_list = [phi_traj_list[i][:hit_index[i]] for i in range(len(phi_traj_list)) if hit_mask[i]]
        hit_psi_traj_list = [psi_traj_list[i][:hit_index[i]] for i in range(len(psi_traj_list)) if hit_mask[i]]
        transition_path_plot_img = plot_ad_traj(
            checkpoint_path = checkpoint_path,
            traj_dihedral = (hit_phi_traj_list, hit_psi_traj_list),
            start_dihedral = (phi_start, psi_start),
            goal_dihedral = (phi_goal, psi_goal),
            type = "hits",
            seed = seed,
        )

    elif molecule == "chignolin":
        raise ValueError(f"Projection for molecule {molecule} TBA...")
    
    else:
        raise ValueError(f"Ramachandran plot for molecule {molecule} TBA...")
    
    return wandb.Image(ram_plot_img), wandb.Image(transition_path_plot_img)


def plot_ad_traj(
    checkpoint_path,
    traj_dihedral,
    start_dihedral,
    goal_dihedral,
    type,
    seed,
):
    cv_bound = 0.75
    plt.clf()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    traj_list_phi = traj_dihedral[0]
    traj_list_psi = traj_dihedral[1]
    sample_num = len(traj_dihedral[0])

    # Plot the potential
    xs = np.arange(-np.pi, np.pi + 0.1, 0.1)
    ys = np.arange(-np.pi, np.pi + 0.1, 0.1)
    x, y = np.meshgrid(xs, ys)
    inp = torch.tensor(np.array([x, y])).view(2, -1).T
    potential = AlaninePotential(f"../simulation/data/alanine/final_frame.dat")
    z = potential.potential(inp)
    z = z.view(y.shape[0], y.shape[1])
    plt.contourf(xs, ys, z, levels=100, zorder=0)

    # Plot the trajectory
    cm = plt.get_cmap("gist_rainbow")
    ax.set_prop_cycle(
        color=[cm(1.0 * i / sample_num) for i in range(sample_num)]
    )
    for idx in range(sample_num):
        ax.plot(
            traj_list_phi[idx].cpu(),
            traj_list_psi[idx].cpu(),
            marker="o",
            linestyle="None",
            markersize=3,
            alpha=1.0,
            zorder=100
        )

    # Plot start and goal states
    ax.scatter(
        start_dihedral[0], start_dihedral[1], edgecolors="black", c="w", zorder=101, s=160
    )
    ax.scatter(
        goal_dihedral[0], goal_dihedral[1], edgecolors="black", c="w", zorder=101, s=500, marker="*"
    )
    square = plt.Rectangle(
        (goal_dihedral[0] - cv_bound / 2, goal_dihedral[1] - cv_bound /2),
        cv_bound, cv_bound,
        color='r', fill=False, linewidth=4,
        zorder=101
    )
    plt.gca().add_patch(square)
    
    # Plot the Ramachandran plot
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])
    plt.xlabel("phi")
    plt.ylabel("psi")
    plt.show()
    
    save_plot(
        dir = f"{checkpoint_path}/{seed}-{type}-paths",
        fig = fig
    )
    return fig


class AlaninePotential():
    def __init__(self, landscape_path):
        super().__init__()
        self.open_file(landscape_path)

    def open_file(self, landscape_path):
        with open(landscape_path) as f:
            lines = f.readlines()

        dims = [90, 90]

        self.locations = torch.zeros((int(dims[0]), int(dims[1]), 2))
        self.data = torch.zeros((int(dims[0]), int(dims[1])))

        i = 0
        for line in lines[1:]:
            splits = line[0:-1].split(" ")
            vals = [y for y in splits if y != '']

            x = float(vals[0])
            y = float(vals[1])
            val = float(vals[-1])

            self.locations[i // 90, i % 90, :] = torch.tensor(np.array([x, y]))
            self.data[i // 90, i % 90] = (val)  # / 503.)
            i = i + 1

    def potential(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp, loc.double(), p=2)
        index = distances.argmin(dim=1)

        x = torch.div(index, self.locations.shape[0], rounding_mode='trunc')  # index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        z = self.data[x, y]
        return z

    def drift(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp[:, :2].double(), loc.double(), p=2)
        index = distances.argsort(dim=1)[:, :3]

        x = index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        dims = torch.stack([x, y], 2)

        min = dims.argmin(dim=1)
        max = dims.argmax(dim=1)

        min_x = min[:, 0]
        min_y = min[:, 1]
        max_x = max[:, 0]
        max_y = max[:, 1]

        min_x_dim = dims[range(dims.shape[0]), min_x, :]
        min_y_dim = dims[range(dims.shape[0]), min_y, :]
        max_x_dim = dims[range(dims.shape[0]), max_x, :]
        max_y_dim = dims[range(dims.shape[0]), max_y, :]

        min_x_val = self.data[min_x_dim[:, 0], min_x_dim[:, 1]]
        min_y_val = self.data[min_y_dim[:, 0], min_y_dim[:, 1]]
        max_x_val = self.data[max_x_dim[:, 0], max_x_dim[:, 1]]
        max_y_val = self.data[max_y_dim[:, 0], max_y_dim[:, 1]]

        grad = -1 * torch.stack([max_y_val - min_y_val, max_x_val - min_x_val], dim=1)

        return grad
    
    

import os
import torch
import hydra
import wandb
import jax

import jax.numpy as jnp
import numpy as np
import mdtraj as md

from tqdm import tqdm

from .util.constant import *
from .util.plot import plot_ad_traj, plot_ad_cv
from .util.angle import compute_dihedral
from .util.rotate import kabsch_rmsd

pairwise_distance = torch.cdist


def potential_energy(cfg, trajectory):
    molecule = cfg.data.molecule
    energy_list = []
    
    if molecule == "alanine":
        pbb_file_path = f"../simulation/alanine/c5.pdb"
        simulation = init_simulation(cfg, pbb_file_path)
        
        for frame in trajectory:
            try:
                simulation = set_simulation(simulation, frame)
                energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                energy_list.append(energy._value)
            except Exception as e:
                print(f"Error in computing energy: {e}")
                energy_list.append(10000)
    else: 
        raise ValueError(f"Potential energy for molecule {molecule} TBA")
    
    return energy_list


def compute_thp(
    cfg,
    trajectory_list,
    goal_state
):
    device = trajectory_list.device
    molecule = cfg.steeredmd.molecule
    sample_num = cfg.steeredmd.sample_num
    cv_bound = 0.75
    hit_rate = 0.0
    hit_mask = []
    hit_index = []
    
    if molecule == "alanine":
        psi_goal = compute_dihedral(goal_state[0, ALDP_PSI_ANGLE].reshape(-1, len(ALDP_PSI_ANGLE), 3))
        phi_goal = compute_dihedral(goal_state[0, ALDP_PHI_ANGLE].reshape(-1, len(ALDP_PHI_ANGLE), 3))
        for i in tqdm(
            range(sample_num),
            desc = f"Computing THP for {trajectory_list.shape[0]} trajectories"
        ):
            psi = compute_dihedral(trajectory_list[i, :, ALDP_PSI_ANGLE])
            phi = compute_dihedral(trajectory_list[i, :, ALDP_PHI_ANGLE])
            psi_hit_distance = torch.abs(psi - psi_goal)
            phi_hit_distance = torch.abs(phi - phi_goal)
            cv_distance = torch.sqrt(psi_hit_distance ** 2 + phi_hit_distance ** 2)
            hit_in_path = (psi_hit_distance < cv_bound) & (phi_hit_distance < cv_bound)
            hit_index_in_path = torch.argmin(cv_distance)
            
            if torch.any(hit_in_path):
                hit_rate += 1.0
                hit_mask.append(True)
                hit_index.append(hit_index_in_path)
            else:
                hit_mask.append(False)
                hit_index.append(-1)
                
        hit_rate /= sample_num
        hit_mask = torch.tensor(hit_mask)
        hit_index = torch.tensor(hit_index, dtype=torch.int32)

    elif molecule == "chignolin":
        raise NotImplementedError(f"THP for molecule {molecule} to be implemented")
    
    else:
        raise ValueError(f"THP for molecule {molecule} TBA")
    
    return hit_rate, hit_mask, hit_index


def compute_epd(cfg, trajectory_list, goal_state, hit_mask, hit_index):
    atom_num = cfg.data.atom
    unit_scale_factor = 1000
    hit_trajectory = trajectory_list[hit_mask]
    hit_path_num = hit_mask.sum().item()
    goal_state = goal_state[hit_mask]
    epd = 0.0
    
    hit_state_list = []
    rmsd = []
    for i in tqdm(
        range(hit_path_num),
        desc = f"Computing EPD, RMSD for {hit_path_num} hitting trajectories"
    ):
        hit_state_list.append(hit_trajectory[i, hit_index[i]])
        rmsd.append(kabsch_rmsd(hit_trajectory[i, hit_index[i]], goal_state[i]))
    
    hit_state_list = torch.stack(hit_state_list)
    matrix_f_norm = torch.sqrt(torch.square(
        pairwise_distance(hit_state_list, hit_state_list) - pairwise_distance(goal_state, goal_state)
    ).sum((1, 2)))
    epd = torch.mean(matrix_f_norm / (atom_num ** 2) * unit_scale_factor)
    rmsd = torch.tensor(rmsd)
    rmsd = torch.mean(rmsd)
    
    return epd, rmsd


def compute_energy(cfg, trajectory_list, goal_state, hit_mask, hit_index):
    molecule = cfg.steeredmd.molecule
    sample_num = trajectory_list.shape[0]
    path_length = trajectory_list.shape[1]
    
    try:
        if molecule == "alanine":
            goal_state_file_path = f"data/{cfg.data.molecule}/{cfg.steeredmd.goal_state}.pdb"
            goal_simulation = init_simulation(cfg, goal_state_file_path)
            goal_state_energy = goal_simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
            
            path_energy_list = []
            for trajectory in tqdm(
                trajectory_list[hit_mask],
                desc=f"Computing energy for {trajectory_list[hit_mask].shape[0]} hitting trajectories"
            ):
                energy_trajectory = potential_energy(cfg, trajectory)
                path_energy_list.append(energy_trajectory)
            path_energy_list = np.array(path_energy_list)
            
            path_maximum_energy = np.max(path_energy_list, axis=1)
            path_final_energy_error = np.array(path_energy_list[:, -1]) - goal_state_energy

        else: 
            raise ValueError(f"Energy for molecule {molecule} TBA")
    except Exception as e:
        print(f"Error in computing energy: {e}")
        path_maximum_energy = np.ones(sample_num) * 10000
        path_energy_list = np.ones((sample_num, path_length)) * 10000
        path_final_energy_error = np.ones(sample_num) * 10000
    
    return path_maximum_energy.mean(), path_energy_list[:, -1].mean(), path_final_energy_error.mean()


def compute_projection(cfg, model_wrapper, epoch):
    def map_range(x, in_min, in_max):
        out_max = 1
        out_min = -1
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    molecule = cfg.steeredmd.molecule
    device = model_wrapper.device
    if cfg.model.name in ["clcv", "autoencoder"]:
        cv_dim = cfg.model.params.encoder_layers[-1]
    elif cfg.model.name == "spib":
        cv_dim = cfg.model.params.decoder_output_dim
    elif cfg.model.name == "vde":
        cv_dim = cfg.model.params.n_cvs
    elif cfg.model.name in CLCV_METHODS:
        cv_dim = cfg.model["params"].output_dim
    else:
        cv_dim = 1
    
    if molecule == "alanine":
        data_dir = f"{cfg.data.dir}/projection/{cfg.steeredmd.molecule}/{cfg.steeredmd.metrics.projection.version}"
        phis = np.load(f"{data_dir}/phi.npy")
        psis = np.load(f"{data_dir}/psi.npy")
        temperature = torch.tensor(cfg.steeredmd.simulation.temperature).repeat(phis.shape[0], 1).to(device)
        
        if cfg.model.name in CLCV_METHODS + ["clcv"]:
            if cfg.model.input == "distance":
                projection_file = f"{data_dir}/heavy_atom_distance.pt"
            else:
                raise ValueError(f"Input type {cfg.model.input} not found for CLCV_METHODS")
        
        elif cfg.model.name in ["deeplda", "deeptda", "deeptica", "vde"]:
            projection_file = f"{data_dir}/heavy_atom_distance.pt"
        
        elif cfg.model.name in ["autoencoder", "timelagged-autoencoder", "gnncvtica"]:
            projection_file = f"{data_dir}/xyz-aligned.pt"
        
        elif cfg.model.name == "spib":
            projection_file = f"{data_dir}/four-dihedral.pt"
            
        else:
            raise ValueError(f"Input type {cfg.model.input} not found")
        
        projected_cv = model_wrapper.compute_cv(
            preprocessed_file = projection_file,
            temperature = temperature,
        )
        cv_min = projected_cv.min(dim=0)[0]
        cv_max = projected_cv.max(dim=0)[0]
        print(f"CV min: {cv_min}, CV max: {cv_max}")
        # projected_cv = map_range(projected_cv, cv_min, cv_max)
        # print(f"Normalized CV min: {projected_cv.min(dim=0)[0]}, CV max: {projected_cv.max(dim=0)[0]}")
        if cfg.logging.wandb:
            for i in range(cv_dim):
                wandb.log({
                    f"cv/cv{i}/min": projected_cv[:, i].min(),
                    f"cv/cv{i}/max": projected_cv[:, i].max(),
                    f"cv/cv{i}/std": projected_cv[:, i].std(),
                })
        

        start_state_xyz = md.load(f"./data/{cfg.steeredmd.molecule}/{cfg.steeredmd.start_state}.pdb").xyz
        goal_state_xyz = md.load(f"./data/{cfg.steeredmd.molecule}/{cfg.steeredmd.goal_state}.pdb").xyz
        start_state = torch.tensor(start_state_xyz)
        goal_state = torch.tensor(goal_state_xyz)
        phi_start = compute_dihedral(start_state[:, ALDP_PHI_ANGLE])
        psi_start = compute_dihedral(start_state[:, ALDP_PSI_ANGLE])
        phi_goal = compute_dihedral(goal_state[:, ALDP_PHI_ANGLE])
        psi_goal = compute_dihedral(goal_state[:, ALDP_PSI_ANGLE])
        
        projection_img = plot_ad_cv(
            phi = phis,
            psi = psis,
            cv = projected_cv.cpu().detach().numpy(),
            cv_dim = cv_dim,
            epoch = epoch,
            start_dihedral = (phi_start, psi_start),
            goal_dihedral = (phi_goal, psi_goal),
            cfg_plot = cfg.steeredmd.metrics.projection
        )
    
    elif molecule == "chignolin":
        raise ValueError(f"Projection for molecule {molecule} TBA...")
    
    else:
        raise ValueError(f"Projection for molecule {molecule} not supported")
    
    return wandb.Image(projection_img[0]), wandb.Image(projection_img[1]), wandb.Image(projection_img[2])


def compute_jacobian(cfg, model_wrapper, epoch):
    molecule = cfg.steeredmd.molecule
    device = model_wrapper.device
    
    if molecule == "alanine":
        pass
    
    elif molecule == "chignolin":
        raise ValueError(f"Jacobian for molecule {molecule} TBA...")
    
    else:
        raise ValueError(f"Jacobian for molecule {molecule} not supported")
        
    jacobian_img = None
    return wandb.Image(jacobian_img)


def init_simulation(cfg, pdb_file_path, frame=None):
    pdb = PDBFile(pdb_file_path)
    force_field = load_forcefield(cfg, cfg.steeredmd.molecule)
    system = load_system(cfg, cfg.steeredmd.molecule, pdb, force_field)
    
    cfg_simulation = cfg.steeredmd.simulation
    integrator = LangevinIntegrator(
        cfg_simulation.temperature * kelvin,
        cfg_simulation.friction / femtoseconds,
        cfg_simulation.timestep * femtoseconds
    )
    platform = openmm.Platform.getPlatformByName(cfg_simulation.platform)
    properties = {'DeviceIndex': '0', 'Precision': cfg_simulation.precision}
    

    simulation = Simulation(
        pdb.topology,
        system,
        integrator,
        platform,
        properties
    )        
    
    simulation.context.setPositions(pdb.positions)   
    simulation.minimizeEnergy()
    
    return simulation

def set_simulation(simulation, frame):
    if frame is not None:
        atom_xyz = frame.detach().cpu().numpy()
        atom_list = [Vec3(atom[0], atom[1], atom[2]) for atom in atom_xyz]
        current_state_openmm = Quantity(value=atom_list, unit=nanometer)
        simulation.context.setPositions(current_state_openmm)
    else:
        raise ValueError("Frame is None")
    
    simulation.context.setVelocities(Quantity(value=np.zeros(frame.shape), unit=nanometer/picosecond))
    
    return simulation
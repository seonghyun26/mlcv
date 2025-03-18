import torch
import wandb

import numpy as np

import openmm as mm
import openmm.unit as unit

from openmm import *
from openmm.app import *

from tqdm import tqdm

from .util.constant import *
from .util.angle import compute_dihedral
from .util.rotate import kabsch_rmsd

from .simulation.dynamics import load_forcefield, load_system

pairwise_distance = torch.cdist


def potential_energy(cfg, trajectory):
    molecule = cfg.data.molecule
    energy_list = []
    
    if molecule == "alanine":
        pbb_file_path = f"../simulation/data/alanine/c5.pdb"
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


def compute_epd(
    cfg,
    trajectory_list,
    logger,
    goal_state,
    hit_mask,
    hit_index
):
    atom_num = cfg.data.atom
    unit_scale_factor = 1000
    hit_trajectory = trajectory_list[hit_mask]
    hit_path_num = hit_mask.sum().item()
    goal_state = goal_state[hit_mask]
    epd = 0.0
    
    if hit_path_num != 0:
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
    
    else:
        logger.info("No hitting trajectories found")
        return -1, -1


def compute_energy(
    cfg,
    trajectory_list,
    hit_mask
):
    molecule = cfg.steeredmd.molecule
    
    try:
        if molecule == "alanine":
            if trajectory_list[hit_mask].shape[0] == 0:
                path_max_energy = None
                path_final_energy = None
            else:
                path_energy_list = []
                for trajectory in tqdm(
                    trajectory_list[hit_mask],
                    desc=f"Computing energy for {trajectory_list[hit_mask].shape[0]} hitting trajectories"
                ):
                    energy_trajectory = potential_energy(cfg, trajectory)
                    path_energy_list.append(energy_trajectory)
                
                path_energy_list = np.array(path_energy_list)
                path_maximum_energy = np.max(path_energy_list, axis=1)
                path_max_energy = path_maximum_energy.mean()
                path_final_energy = path_energy_list[:, -1].mean()
        
        elif molecule == "chignolin":
            raise NotImplementedError(f"Energy for molecule {molecule} TBA")
        
        else: 
            raise ValueError(f"Energy for molecule {molecule} not implemented")
    
    except Exception as e:
        print(f"Error in computing energy: {e}")
        path_max_energy = None
        path_final_energy = None
    
    return path_max_energy, path_final_energy


def compute_jacobian(cfg, model):
    molecule = cfg.steeredmd.molecule
    device = model.device
    
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
        cfg_simulation.temperature * unit.kelvin,
        cfg_simulation.friction / unit.femtoseconds,
        cfg_simulation.timestep * unit.femtoseconds
    )
    platform = mm.Platform.getPlatformByName(cfg_simulation.platform)
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
        current_state_openmm = unit.Quantity(value=atom_list, unit=unit.nanometer)
        simulation.context.setPositions(current_state_openmm)
    else:
        raise ValueError("Frame is None")
    
    simulation.context.setVelocities(unit.Quantity(value=np.zeros(frame.shape), unit=unit.nanometer/unit.picosecond))
    
    return simulation


def compute_free_energy_difference(cfg, trajectory_list, logger):
    return
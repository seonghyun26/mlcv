import torch
import numpy as np

from tqdm import tqdm 
from .simulation import *

def simulate_steered_md(cfg, model, logger, repeat_idx, checkpoint_path):
    steered_md_simulation = SteeredMDSimulation(
        cfg = cfg,
        model = model,
    )
    time_horizon = cfg.steeredmd.simulation.time_horizon
    sample_num = cfg.steeredmd.sample_num
    position_list = []
    
    try:
        for step in tqdm(
            range(1, time_horizon + 1),
            desc=f"Genearting {sample_num} trajectories for {time_horizon} steps",
        ):
            position = steered_md_simulation.report()
            position = torch.tensor([list(p) for p in position], dtype=torch.float32, device = model.device)
            # position = np.array([list(p) for p in position], dtype=np.float32)
            position_list.append(position)
            steered_md_simulation.step(step)
        
        if isinstance(position_list[0], torch.Tensor):
            trajectory_list = torch.stack(position_list, dim=1)
        elif isinstance(position_list[0], list):
            trajectory_list = np.stack(position_list, axis=1)
        else:
            raise ValueError(f"Type {type(position_list)} not supported")
        
        save_trajectory(trajectory_list, logger, checkpoint_path, repeat_idx)
    
    except Exception as e:
        logger.error(f"Error in generating trajectory: {e}")
        trajectory_list = None

    return trajectory_list


def simluate_metadynamics(cfg, model, logger, repeat_idx, checkpoint_path):
    metadynamics_simulation = MetadynamicsSimulation(
        cfg = cfg,
        model = model,
        repeat_idx = repeat_idx
    )
    time_horizon = cfg.metadynamics.simulation.time_horizon
    position_list = []
    
    #TODO: metaydnamics simulation by openMM
    trajectory_list = None

    return trajectory_list

def save_trajectory(trajectory_list, logger, checkpoint_path, repeat_idx):
    if isinstance(trajectory_list, torch.Tensor):
        trajectory_list = trajectory_list.detach().cpu().numpy()
    np.save(f"{checkpoint_path}/{repeat_idx}-traj.npy", trajectory_list)
    
    logger.info(f"{trajectory_list.shape[0]} trajectories saved at: {checkpoint_path}")
    return
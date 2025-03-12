import os
import torch
import wandb
import hydra


from .load import *
from .metric import *
from .simulation.run import simulate_steered_md, simluate_metadynamics
from .util.plot import plot_ram


def eval(cfg, model, logger, datamodule, checkpoint_path):
    eval_dict = {}
    eval_dict["eval/ramachandran"] = wandb.Image(plot_ad_cv(cfg, model, datamodule, checkpoint_path))
    eval_dict.update(steered_md(cfg, model, logger, checkpoint_path))
    # eval_dict.update(metadynamics(cfg, model, logger, checkpoint_path))

    wandb.log(eval_dict)
    wandb.finish()

    return


def steered_md(cfg, model, logger, checkpoint_path):
    steered_md_result = {}
    for seed in range(cfg.steeredmd.seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        logger.info(f">> Steered MD evaluation for seed {seed}")
        trajectory_list = simulate_steered_md(cfg, model, logger, seed, checkpoint_path)
        steered_md_metric = evalute_steered_md(cfg, trajectory_list, logger, seed, checkpoint_path)
        
        logger.info(f">> Steered MD result: {steered_md_metric}")
        steered_md_result.update(steered_md_metric)
        
    return steered_md_result


def evalute_steered_md(cfg, trajectory_list, logger, seed, checkpoint_path):
    result_dict = {}
    goal_state = load_state_file(cfg, cfg.steeredmd.goal_state, trajectory_list.device)
    
    result_dict[f"steered_md/{seed}/thp"], hit_mask, hit_index = compute_thp(cfg, trajectory_list, goal_state)
    result_dict[f"steered_md/{seed}/all_paths"], result_dict[f"steered_md/{seed}/hitting_paths"] = plot_ram(cfg, trajectory_list, hit_mask, hit_index, seed, checkpoint_path)
    # epd = compute_epd(cfg, trajectory_list, goal_state, hit_mask, hit_index)
    
    return result_dict


def metadynamics(cfg, model, logger, checkpoint_path):
    metadynamics_result = {}
    for seed in range(5):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        logger.info(f">> Metadynamics evaluation for seed {seed}")
        trajectory_list = simluate_metadynamics(cfg, model, logger, seed, checkpoint_path)
        metadynamics_metric = evaluate_metadynamics(cfg, trajectory_list, logger, seed)
        
        logger.info(f">> Metadynamics result: {metadynamics_metric}")
        metadynamics_result.update(metadynamics_metric)
    return metadynamics_result


def evaluate_metadynamics(cfg, trajectory_list, logger, seed):
    result_dict = {}
    
    result_dict[f"metadynamics/{seed}/free_energy_difference"] = compute_free_energy_difference(cfg, trajectory_list)
    result_dict[f"metadynamics/{seed}/angle_distribution"] = plot_angle_distribution(cfg, trajectory_list)
    
    return result_dict


def load_state_file(cfg, state, device):
    if cfg.steeredmd.molecule == "alanine":
        state_dir = f"../simulation/data/alanine/{state}.pdb"
        state = md.load(state_dir).xyz
        state = torch.tensor(state, dtype=torch.float32, device=device)
        states = state.repeat(cfg.steeredmd.sample_num, 1, 1)
    
    elif cfg.job.molecule == "chignolin":
        raise NotImplementedError("Chignolin state TBA")
    
    else:
        raise ValueError(f"Molecule {cfg.job.molecule} not found")
    
    return states




@hydra.main(
    version_base=None,
    config_path="config",
    config_name="basic"
)
def main(cfg):
    wandb.init(
        project = "mlcv",
        entity = "eddy26",
    )
    logger = logging.getLogger("MLCVs")
    logger.info(">> Configs")
    logger.info(OmegaConf.to_yaml(cfg))
    device = "cuda"
    
    # Load model and data
    logger.info(">> Loading model from checkpoint and datamodule")
    model = load_model(cfg).to(device)
    model = load_checkpoint(cfg, model)
    model.eval()
    datamodule = load_data(cfg)
    
    # Evaluate
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

import os
import torch
import wandb
import hydra


from .load import *
from .metric import *
from .simulation.run import simulate_steered_md


def eval(cfg, model, logger, datamodule, checkpoint_path):
    eval_dict = {}
    eval_dict["eval/ramachandran"] = wandb.Image(plot_ad_cv(cfg, model, datamodule, checkpoint_path))
    eval_dict.update(steered_md(cfg, model, logger, checkpoint_path))
    # eval_dict.update(metadynamics(cfg, model, checkpoint_path))

    wandb.log(eval_dict)
    wandb.finish()

    return


def steered_md(cfg, model, logger, checkpoint_path):
    for seed in range(5):
        logger.info(f"> Steered MD evaluation for seed {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        trajectory_list = simulate_steered_md(cfg, model, logger, seed, checkpoint_path)
        steered_md_metric = evalute_steered_md(cfg, trajectory_list, logger, seed)
        logger.info(f"Steered MD metric: {steered_md_metric}")
        
    return

def metadynamics(cfg, model, logger):
    for seed in range(5):
        torch.manual_seed(seed)
        
    return


def evalute_steered_md(cfg, trajectory_list, logger, seed):
    result_dict = {}
    goal_state = load_state_file(cfg, cfg.steeredmd.goal_state, device)
    
    result_dictf[f"steered_md/{seed}/thp"], hit_mask, hit_index = compute_thp(cfg, trajectory_list, goal_state)
    epd = compute_epd(cfg, trajectory_list, goal_state, hit_mask, hit_index)
    ram = plot_ram(cfg, trajectory_list, hit_mask, hit_index, seed)
    
    return result_dict



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

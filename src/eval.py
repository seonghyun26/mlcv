import os
import torch
import wandb
import hydra
import logging
import matplotlib.pyplot as plt

from .load import *
from .metric import *
from .simulation.run import simulate_steered_md, simluate_metadynamics
from .util.plot import plot_paths
from .util.convert import input2representation

def eval(cfg, model, logger, datamodule, checkpoint_path):
    model.eval()
    eval_dict = {}
    eval_dict["eval/ramachandran"] = wandb.Image(plot_ad_cv(cfg, model, datamodule, checkpoint_path))
    eval_dict["eval/sensitivity"] = wandb.Image(sensitivity(cfg, model, logger, checkpoint_path))
    
    checkpoint_path = f"./model/{cfg.model.name}/{cfg.data.version}/{cfg.steeredmd.simulation.k}"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    eval_dict.update(steered_md(cfg, model, logger, checkpoint_path))
    # eval_dict.update(metadynamics(cfg, model, logger, checkpoint_path))

    keys_with_average = [key for key in eval_dict.keys() if key.endswith("average")]
    eval_dict_avereage = {key: eval_dict[key] for key in keys_with_average}
    wandb.log(eval_dict_avereage)
    wandb.finish()

    return


def steered_md(cfg, model, logger, checkpoint_path):
    steered_md_result = {}
    for repeat_idx in range(0, cfg.steeredmd.repeat):
        np.random.seed(repeat_idx)
        torch.manual_seed(repeat_idx)
        logger.info(f">> Steered MD evaluation #{repeat_idx}")
        trajectory_list = simulate_steered_md(cfg, model, logger, repeat_idx, checkpoint_path)
        steered_md_metric = evalute_steered_md(cfg, trajectory_list, logger, repeat_idx, checkpoint_path)
        
        logger.info(f">> Steered MD result: {steered_md_metric}")
        wandb.log(steered_md_metric)
        steered_md_result.update(steered_md_metric)
    
    if cfg.steeredmd.repeat > 0:
        steered_md_result["steered_md/thp/average"] = np.mean([value for key, value in steered_md_result.items() if key.startswith("steered_md/thp") and value != None])
        epd_list = [value for key, value in steered_md_result.items() if key.startswith("steered_md/epd") and value != None]
        if len(epd_list) > 0:
            steered_md_result["steered_md/epd/average"] = np.mean(epd_list)
        rmsd_list = [value for key, value in steered_md_result.items() if key.startswith("steered_md/rmsd") and value != None]
        if len(rmsd_list) > 0:
            steered_md_result["steered_md/rmsd/average"] = np.mean(rmsd_list)
        max_energy_list = [value for key, value in steered_md_result.items() if key.startswith("steered_md/max_energy") and value is not None]
        if len(max_energy_list) > 0:
            steered_md_result["steered_md/max_energy/average"] = torch.mean(max_energy_list) 
        final_energy_list = [value for key, value in steered_md_result.items() if key.startswith("steered_md/final_energy") and value is not None]
        if len(final_energy_list) > 0:
            steered_md_result["steered_md/final_energy/average"] = torch.mean(final_energy_list)
        
    logger.info(f">> Steered MD average result: {steered_md_result}")
        
    return steered_md_result

def evalute_steered_md(cfg, trajectory_list, logger, repeat_idx, checkpoint_path):
    result_dict = {}
    goal_state = load_state_file(cfg, cfg.steeredmd.goal_state, trajectory_list.device)
    
    result_dict[f"steered_md/thp/{repeat_idx}"], hit_mask, hit_index = compute_thp(cfg, trajectory_list, goal_state)
    result_dict[f"steered_md/all_paths/{repeat_idx}"], result_dict[f"steered_md/hitting_paths/{repeat_idx}"] = plot_paths(cfg, trajectory_list, hit_mask, hit_index, repeat_idx, checkpoint_path)
    result_dict[f"steered_md/epd/{repeat_idx}"], result_dict[f"steered_md/rmsd/{repeat_idx}"] = compute_epd(cfg, trajectory_list, logger, goal_state, hit_mask, hit_index)
    result_dict[f"steered_md/max_energy/{repeat_idx}"], result_dict[f"steered_md/final_energy/{repeat_idx}"] = compute_energy(cfg, trajectory_list, hit_mask)
    
    return result_dict


def metadynamics(cfg, model, logger, checkpoint_path):
    metadynamics_result = {}
    for repeat_idx in range(5):
        np.random.repeat_idx(repeat_idx)
        torch.manual_repeat_idx(repeat_idx)
        
        logger.info(f">> Metadynamics evaluation for repeat_idx {repeat_idx}")
        trajectory_list = simluate_metadynamics(cfg, model, logger, repeat_idx, checkpoint_path)
        metadynamics_metric = evaluate_metadynamics(cfg, trajectory_list, logger, repeat_idx)
        
        logger.info(f">> Metadynamics result: {metadynamics_metric}")
        metadynamics_result.update(metadynamics_metric)
    return metadynamics_result


def evaluate_metadynamics(cfg, trajectory_list, logger, repeat_idx):
    result_dict = {}
    
    result_dict[f"metadynamics/{repeat_idx}/free_energy_difference"] = compute_free_energy_difference(cfg, trajectory_list, logger)
    result_dict[f"metadynamics/{repeat_idx}/angle_distribution"] = plot_angle_distribution(cfg, trajectory_list)
    
    return result_dict


def sensitivity(cfg, model, logger, checkpoint_path):
    # Compute sensitivities
    c5 = torch.load(f"../simulation/data/{cfg.data.molecule}/{cfg.steeredmd.start_state}.pt")['xyz']
    c5_input = input2representation(cfg, c5)
    c5_input.requires_grad = True
    c5_input = c5_input.to(model.device)
    c5_output = model(c5_input)
    
    sensitivities = []
    for i in range(c5_output.shape[1]):
        node_sensitivity = torch.autograd.grad(c5_output[i], c5_input, retain_graph=True)[0]
        node_sensitivity = node_sensitivity / node_sensitivity.sum()
        sensitivities.append(node_sensitivity)
    sensitivities = torch.stack(sensitivities)
    
    # Plot sensitivities
    plt.figure(figsize=(12, 6))
    plt.imshow(sensitivities.reshape(c5_input.shape[1], c5_input.shape[1]).detach().cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Sensitivity')
    plt.xlabel('Input distance')
    plt.ylabel('Output dimension')
    plt.yticks([])
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(checkpoint_path, 'sensitivity.png')
    logger.info(f">> Sensitivity plot saved at {plot_path}, highest for {sensitivities.argmax()}-th intput")
    plt.savefig(plot_path)
    wandb.log({"sensitivity": wandb.Image(plot_path)})
    plt.close()
    
    return plot_path

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
    checkpoint_path = f"./model/{cfg.model.name}/{cfg.data.version}"
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

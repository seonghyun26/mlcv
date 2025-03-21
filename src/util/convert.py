import torch

from .constant import *

def input2representation(cfg, input):
    if cfg.model.representation == "heavy_atom_distance":
        representation = coordinate2distance(input)
    
    elif cfg.model.representation == "backbone_coordinate":
        representation_list = []
        for data in input:
            data = data.reshape(ALANINE_ATOM_NUM, 3)
            representation_list.append(data[ALANINE_HEAVY_ATOM_IDX].reshape(-1))
        representation = torch.stack(representation_list)
    
    elif cfg.model.representation == "backbone":
        raise NotImplementedError(f"Representation {cfg.model.representation} not implemented")
        
    else:
        raise ValueError(f"Representation {cfg.model.representation} not found")    
    
    return representation


def coordinate2distance(position):
    position = position.reshape(-1, 3)
    heavy_atom_position = position[ALANINE_HEAVY_ATOM_IDX]
    num_heavy_atoms = len(heavy_atom_position)
    distance = []
    for i in range(num_heavy_atoms):
        for j in range(i+1, num_heavy_atoms):
            distance.append(torch.norm(heavy_atom_position[i] - heavy_atom_position[j]))
    distance = torch.stack(distance)
    
    return distance
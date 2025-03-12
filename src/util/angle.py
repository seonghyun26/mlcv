import torch 
import numpy as np

def compute_dihedral(positions):
    if isinstance(positions, torch.Tensor):
        angles = compute_dihedral_torch(positions)
    
    elif isinstance(positions, np.ndarray):
        angles = compute_dihedral_numpy(positions)
    
    else:
        raise ValueError("Type not supported for diehdral computation")

    return angles

def compute_dihedral_numpy(positions):
    """http://stackoverflow.com/q/20305272/1128289"""
    def dihedral(p):
        if not isinstance(p, np.ndarray):
            p = p.numpy()
        b = p[:-1] - p[1:]
        b[0] *= -1
        v = np.array([v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
        
        # Normalize vectors
        v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
        b1 = b[1] / np.linalg.norm(b[1])
        x = np.dot(v[0], v[1])
        m = np.cross(v[0], b1)
        y = np.dot(m, v[1])
        
        return np.arctan2(y, x)
    
    return np.array(list(map(dihedral, positions)))
    

def compute_dihedral_torch(
    positions: torch.Tensor,
) -> torch.Tensor:
    """
        Computes the dihedral angle for batches of points P1, P2, P3, P4.
        Args:
            positions: (bacth_size, 4, 3)
        Returns:
            A tensor of shape (batch_size,) containing the dihedral angles in radians.
    """

    P1 = positions[:, 0]
    P2 = positions[:, 1]
    P3 = positions[:, 2]
    P4 = positions[:, 3]
    b1 = P2 - P1
    b2 = P3 - P2
    b3 = P4 - P3
    
    b2_norm = b2 / b2.norm(dim=1, keepdim=True)
    n1 = torch.cross(b1, b2, dim=1)
    n2 = torch.cross(b2, b3, dim=1)
    n1_norm = n1 / n1.norm(dim=1, keepdim=True)
    n2_norm = n2 / n2.norm(dim=1, keepdim=True)
    m1 = torch.cross(n1_norm, b2_norm, dim=1)
    
    # Compute cosine and sine of the angle
    x = (n1_norm * n2_norm).sum(dim=1)
    y = (m1 * n2_norm).sum(dim=1)
    angle = - torch.atan2(y, x)
    
    return angle


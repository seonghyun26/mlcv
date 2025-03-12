import torch

def kabsch(
	reference_position: torch.Tensor,
	position: torch.Tensor,
) -> torch.Tensor:
    '''
        Kabsch algorithm for aligning two sets of points
        Args:
            reference_position (torch.Tensor): Reference positions (N, 3)
            position (torch.Tensor): Positions to align (N, 3)
        Returns:
            torch.Tensor: Aligned positions (N, 3)
    '''
    # Compute centroids
    centroid_ref = torch.mean(reference_position, dim=0, keepdim=True)
    centroid_pos = torch.mean(position, dim=0, keepdim=True)
    ref_centered = reference_position - centroid_ref  
    pos_centered = position - centroid_pos

    # Compute rotation, translation matrix
    covariance = torch.matmul(ref_centered.T, pos_centered)
    U, S, Vt = torch.linalg.svd(covariance)
    d = torch.linalg.det(torch.matmul(Vt.T, U.T))
    if d < 0:
        Vt[-1] *= -1
    rotation = torch.matmul(Vt.T, U.T)

    # Align position to reference_position
    aligned_position = torch.matmul(pos_centered, rotation) + centroid_ref
    return aligned_position


def kabsch_rmsd(
    P: torch.Tensor,
    Q: torch.Tensor
) -> torch.Tensor:
    centroid_P = torch.mean(P, dim=-2, keepdims=True)
    centroid_Q = torch.mean(Q, dim=-2, keepdims=True)
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(-2, -1), q)
    U, S, Vt = torch.linalg.svd(H)
    
    d = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))  # B
    Vt[d < 0.0, -1] *= -1.0

    # Optimal rotation and translation
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    t = centroid_Q - torch.matmul(centroid_P, R.transpose(-2, -1))

    # Calculate RMSD
    P_aligned = torch.matmul(P, R.transpose(-2, -1)) + t
    rmsd = (P_aligned - Q).square().sum(-1).mean(-1).sqrt()
    
    return rmsd
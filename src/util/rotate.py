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

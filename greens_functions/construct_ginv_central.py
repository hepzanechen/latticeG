import torch

def construct_ginv_central(H_BdG: torch.Tensor, E: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    Constructs the inverse Green's function (G_inv) for the central region in RAK space.

    Parameters:
    -----------
    H_BdG : torch.Tensor
        Bogoliubov-de Gennes Hamiltonian matrix for the central region.
    E : float
        Energy value.
    eta : float
        Small imaginary part added to ensure convergence (for retarded and advanced components).

    Returns:
    --------
    torch.Tensor
        The block diagonal inverse Green's function for the central region.
    """
    H_total_size = H_BdG.size(0)  # Get the total size of H_BdG matrix

    # Construct Ginv_central in RAK space
    Ginv_central_R = (E + 1j * eta) * torch.eye(H_total_size, dtype=torch.complex64, device=H_BdG.device) - H_BdG
    Ginv_central_A = (E - 1j * eta) * torch.eye(H_total_size, dtype=torch.complex64, device=H_BdG.device) - H_BdG
    # Ginv_central_K = torch.zeros((H_total_size, H_total_size), dtype=torch.complex64, device=H_BdG.device)  # Assuming G^K = 0 for the central part

    # Assemble Ginv_central in block diagonal form for R and A spaces
    Ginv_central = torch.block_diag(Ginv_central_R, Ginv_central_A)

    return Ginv_central

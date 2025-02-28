import torch
from hamiltonians.Lead import Lead
def construct_ginv_tlc(lead:Lead, Ncentre: int, NLi: int) -> torch.Tensor:
    """
    Construct the tunneling matrix (tLC_combined) for the lead and central region, including counting fields.

    Parameters:
    -----------
    lead : Lead
        Lead object containing parameters for constructing the tunneling matrix.
    Ncentre : int
        Number of lattice sites in the central region.
    NLi : int
        Number of lattice sites in the lead.

    Returns:
    --------
    torch.Tensor
        Combined tunneling matrix for both electron and hole parts.
    """
    lambda_ = lead.lambda_  # Counting field lambda

    # Construct tLC_e (electron part)
    tLC_e = torch.zeros((Ncentre, NLi), dtype=torch.complex64, device=lead.V1alpha.device)

    # Counting fields appear as phase factors
    cos_lambda = torch.cos(lambda_ / 2)
    sin_lambda = -1j * torch.sin(lambda_ / 2)
    phase_factor = torch.stack([torch.stack([cos_lambda, sin_lambda]), torch.stack([sin_lambda, cos_lambda])])
    # Assign values to tLC_e based on lead's position
    for idx in range(len(lead.position)):
        tLC_e[lead.position[idx], :] = -lead.V1alpha[idx, :]

    # Apply phase factor to electron part
    tLC_e = torch.kron(phase_factor, tLC_e)

    # Construct tLC_h (hole part) by taking complex conjugate of tLC_e
    tLC_h = -tLC_e.conj()

    # Combine electron and hole parts
    tLC_combined = torch.kron(tLC_e, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64)) + torch.kron(tLC_h, torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64))

    return tLC_combined

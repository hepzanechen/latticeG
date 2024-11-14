import torch
from .construct_ginv_central import construct_ginv_central
from .add_ginv_leads import add_ginv_leads
from .construct_ginv_tlc import construct_ginv_tlc


def construct_ginv_total(H_BdG: torch.Tensor, E: torch.Tensor, eta: torch.Tensor, leads_info: list) -> torch.Tensor:
    """
    Construct the total G inverse matrix for the system, including the central region and the leads.

    Parameters:
    -----------
    H_BdG : torch.Tensor
        Hamiltonian of the central region in BdG formalism.
    E : torch.Tensor
        Energy value.
    eta : float
        Small imaginary part for regularization.
    leads_info : list
        List of Lead objects containing lead parameters.

    Returns:
    --------
    torch.Tensor
        Combined G inverse matrix for the entire system.
    """
    # Construct Ginv_central
    Ginv_central = construct_ginv_central(H_BdG, E, eta)
    Ncentre = Ginv_central.size(0) / 4

    # Number of leads
    num_leads = len(leads_info)

    # Add leads diagonal part
    Ginv_total_blkdiag = add_ginv_leads(Ginv_central, leads_info, E, num_leads)

    # Initialize tLCBlk matrix with zeros
    tLC_blk = torch.zeros_like(Ginv_total_blkdiag)

    # Position to place tLC
    tLC_position = Ginv_central.size(0)

    # Loop through each lead, constructing tLC and integrating into Ginv_total
    for i in range(num_leads):
        lead = leads_info[i]
        NLi = lead.t.size(0)  # Number of lattice sites in lead 'i'
        tLCi = construct_ginv_tlc(lead, Ncentre, NLi)
        tLC_blk[:4 * Ncentre, tLC_position:tLC_position + 4 * NLi] = tLCi
        tLC_position += 4 * NLi

    # Combine Ginv_total_blkdiag and tLC_blk to form Ginv_total
    Ginv_total = Ginv_total_blkdiag + tLC_blk + tLC_blk.T.conj()

    return Ginv_total

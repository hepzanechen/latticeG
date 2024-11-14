import torch
from utils.lead_decimation import lead_decimation

def construct_ginv_leads(Ginv_central: torch.Tensor, leads_info: list, E: torch.Tensor) -> torch.Tensor:
    """
    Construct the block diagonal Green's function matrix for the leads and central region.

    Parameters:
    -----------
    Ginv_central : torch.Tensor
        The inverse Green's function for the central region.
    leads_info : list
        List of lead objects, containing parameters and coupling information.
    E : torch.Tensor
        Energy value (scalar tensor).

    Returns:
    --------
    torch.Tensor
        The combined inverse Green's function for the central region and all leads in block-diagonal form.
    """
    # Initialize a list to store Ginv blocks for all leads and the central region
    Ginv_blocks = [Ginv_central]  # Start with the central region's Ginv

    # Loop over each lead to calculate and construct their Ginv matrices
    for lead in leads_info:
        # Calculate lead Green's functions for electrons and holes
        gLr_e, gLa_e, gLless_e, gLmore_e = lead_decimation(E, lead.t, lead.epsilon0, lead.mu, lead.temperature, 'e')
        gLr_h, gLa_h, gLless_h, gLmore_h = lead_decimation(E, lead.t, lead.epsilon0, lead.mu, lead.temperature, 'h')
        # Assemble the lead Green's functions in particle-hole space
        gLr = torch.kron(gLr_e, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64)) + \
              torch.kron(gLr_h, torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64))
        
        gLa = torch.kron(gLa_e, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64)) + \
              torch.kron(gLa_h, torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64))
        
        gLk = torch.kron(gLless_e + gLmore_e, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64)) + \
              torch.kron(gLless_h + gLmore_h, torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64))

        # Compute the inverse Green's functions for R and A components
        Ginv_Lead_R = torch.linalg.inv(gLr)  # Inverse of the retarded Green's function
        Ginv_Lead_A = torch.linalg.inv(gLa)  # Inverse of the advanced Green's function
        
        # Combine R and A into Ginv_Lead, and also include the Keldysh component
        # The full Ginv for the lead including Keldysh component, considering g^k = g^< + g^>
        Ginv_Lead_K = -Ginv_Lead_R @ gLk @ Ginv_Lead_A  # Construct the Keldysh part based on g^k
        
        # Assemble the full Ginv for the lead
        Ginv_Lead = torch.cat([
            torch.cat([Ginv_Lead_R, Ginv_Lead_K], dim=1),
            torch.cat([torch.zeros_like(Ginv_Lead_R), Ginv_Lead_A], dim=1)
        ], dim=0)

        # Add Ginv_Lead to Ginv_blocks for block diagonal combination
        Ginv_blocks.append(Ginv_Lead)

    # Combine all Ginv_blocks including central region and all leads
    Ginv_totalBlkdiag = torch.block_diag(*Ginv_blocks)

    return Ginv_totalBlkdiag

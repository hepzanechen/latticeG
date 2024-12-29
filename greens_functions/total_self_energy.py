"""Total self-energy calculations for leads in BdG space."""

import torch
from typing import Dict, List, Tuple
from utils.lead_decimation import lead_decimation

def calculate_total_self_energy(
    E_batch: torch.Tensor,
    leads_info: list,
    system_dim: int,
    Ny: int,
) -> Tuple[torch.Tensor, list]:
    """Calculate total self-energy from all leads in BdG space.
    We first compute full lead surface Green's functions in BdG space.
    and extract from V1alpha the coupling matrix of electrons and holes.
    Then multiply the coupling matrix with the surface Green's function to get the self-energy
    of electrons and holes separately, each is of dim of system_dim_BdG.
    
    Args:
        E_batch: Batch of energy values (batch_size,)
        leads_info: List of lead objects
        system_dim: Dimension of the system (without BdG)
        Ny: Number of sites in y direction
        
    Returns:
        Tuple of (total self-energy matrix, updated leads_info)
    """
    device = E_batch.device
    batch_size = E_batch.size(0)
    system_dim_BdG = system_dim * 2  # BdG space
    
    # Initialize total self-energy matrix
    Sigma_retarded_Total = torch.zeros((batch_size, system_dim_BdG, system_dim_BdG), 
                                     dtype=torch.complex64, device=device)
    
    # Process each lead
    for i, lead in enumerate(leads_info):
        # Calculate surface Green's functions for electrons and holes
        gLr_e, _, _, _ = lead_decimation(
            E_batch, lead.t, lead.epsilon0, lead.mu, lead.temperature, 'e'
        )
        gLr_h, _, _, _ = lead_decimation(
            E_batch, lead.t, lead.epsilon0, lead.mu, lead.temperature, 'h'
        )
        
        # Construct total surface Green's function in BdG space using batch-parallel einsum
        kron_e = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=device)
        kron_h = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=device)
        
        # Perform the Kronecker products using torch.einsum
        gLr_e_BdG = torch.einsum('bij,kl->bikjl', gLr_e, kron_e).reshape(batch_size, lead_iSize*2, lead_iSize*2)
        gLr_h_BdG = torch.einsum('bij,kl->bikjl', gLr_h, kron_h).reshape(batch_size, lead_iSize*2, lead_iSize*2)
        
        # Sum the electron and hole contributions
        lead_iGLeadSurface = gLr_e_BdG + gLr_h_BdG
        
        # Initialize coupling matrices for electron and hole parts
        lead_iSize = len(lead.position)
        lead_iV1alpha_electron = torch.zeros((batch_size, 2*lead_iSize, system_dim_BdG), 
                                           dtype=torch.complex64, device=device)
        lead_iV1alpha_hole = torch.zeros_like(lead_iV1alpha_electron)
        
        # Construct coupling matrices
        for lead_iSite_j in range(lead_iSize):
            lead_iSite_j_electron = 2 * lead_iSite_j
            lead_iSite_j_hole = 2 * lead_iSite_j + 1
            
            # Extract position and coupling strength
            positions = lead.position[lead_iSite_j]
            V1alpha = lead.V1alpha[lead_iSite_j]
            
            # Process each connection point
            for k, (x, y, flavor) in enumerate(positions):
                # Calculate system position in BdG space
                position = (x - 1) * Ny + y
                pos_electron = 2 * position - 1
                pos_hole = 2 * position
                
                # Construct BdG coupling based on flavor
                if flavor == 0:  # Normal electron coupling
                    V1alpha_electron = torch.tensor([[V1alpha[k], 0], [0, 0]], 
                                                  dtype=torch.complex64, device=device)
                    V1alpha_hole = torch.tensor([[0, 0], [0, -V1alpha[k].conj()]], 
                                              dtype=torch.complex64, device=device)
                elif flavor == 1:  # MZM1 coupling
                    V1alpha_electron = (1/torch.sqrt(torch.tensor(2.0))) * torch.tensor(
                        [[V1alpha[k], V1alpha[k]], [0, 0]], 
                        dtype=torch.complex64, device=device
                    )
                    V1alpha_hole = (1/torch.sqrt(torch.tensor(2.0))) * torch.tensor(
                        [[0, 0], [-V1alpha[k].conj(), -V1alpha[k].conj()]], 
                        dtype=torch.complex64, device=device
                    )
                elif flavor == 2:  # MZM2 coupling
                    V1alpha_electron = (1/torch.sqrt(torch.tensor(2.0))) * torch.tensor(
                        [[-1j*V1alpha[k], 1j*V1alpha[k]], [0, 0]], 
                        dtype=torch.complex64, device=device
                    )
                    V1alpha_hole = (1/torch.sqrt(torch.tensor(2.0))) * torch.tensor(
                        [[0, 0], [1j*V1alpha[k].conj(), -1j*V1alpha[k].conj()]], 
                        dtype=torch.complex64, device=device
                    )
                
                # Embed coupling matrices into the total coupling matrices
                lead_iV1alpha_electron[:, lead_iSite_j_electron:lead_iSite_j_hole+1, 
                                     pos_electron:pos_hole+1] = V1alpha_electron.unsqueeze(0)
                lead_iV1alpha_hole[:, lead_iSite_j_electron:lead_iSite_j_hole+1, 
                                 pos_electron:pos_hole+1] = V1alpha_hole.unsqueeze(0)
        
        # Calculate self-energies in BdG space
        lead_iSigma_electron = lead_iV1alpha_electron.transpose(-1, -2) @ \
                              lead_iGLeadSurface @ lead_iV1alpha_electron
        lead_iSigma_hole = lead_iV1alpha_hole.transpose(-1, -2) @ \
                          lead_iGLeadSurface @ lead_iV1alpha_hole
        
        # Update total self-energy
        Sigma_retarded_Total += lead_iSigma_electron + lead_iSigma_hole
        
        # Store gamma matrices for transmission calculations
        leads_info[i].Gamma = {
            'e': 1j * (lead_iSigma_electron - lead_iSigma_electron.transpose(-1, -2).conj()),
            'h': 1j * (lead_iSigma_hole - lead_iSigma_hole.transpose(-1, -2).conj())
        }
    
    return Sigma_retarded_Total, leads_info 
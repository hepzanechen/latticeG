"""Direct Green's function calculations using matrix inversion."""

import torch
from typing import Dict, List, Tuple, Optional
from .total_self_energy import calculate_total_self_energy
from utils.fermi_distribution import fermi_distribution

def calculate_transport_properties(
    E_batch: torch.Tensor,
    H_total: torch.Tensor,
    leads_info: list,
    temperature: torch.Tensor,
    ny: int,
    eta: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Calculate transport properties using direct matrix inversion method.
    
    Args:
        E_batch: Batch of energy values (batch_size,)
        H_total: Total Hamiltonian including leads (N, N)
        leads_info: List of lead objects
        temperature: Temperature for each energy point (batch_size,)
        ny: Number of sites in y direction
        eta: Small imaginary part (batch_size,)
        
    Returns:
        Dictionary containing transport properties
    """
    device = E_batch.device
    batch_size = E_batch.size(0)
    H_total_size = H_total.size(0)
    system_size = H_total_size // 2  # BdG space
    
    # Calculate total self-energy and update leads_info with Gamma matrices
    sigma_total, leads_info = calculate_total_self_energy(
        E_batch, leads_info, system_size, ny, eta
    )
    
    # Calculate retarded Green's function
    eye = torch.eye(H_total_size, dtype=torch.complex64, device=device)
    E_mat = E_batch.view(-1, 1, 1) * eye.unsqueeze(0)
    eta_mat = eta.view(-1, 1, 1) * eye.unsqueeze(0)
    
    G_retarded = torch.linalg.solve(
        E_mat + eta_mat - H_total.unsqueeze(0) - sigma_total,
        eye.unsqueeze(0).expand(batch_size, -1, -1)
    )
    
    # Calculate LDOS
    rho_jj_ee_and_hh = -torch.imag(torch.diagonal(G_retarded, dim1=1, dim2=2)) / torch.pi
    
    # Calculate total DOS
    # Calculate total DOS by summing electron and hole contributions separately
    # rho_e is at odd indices (0,2,4...), rho_h at even indices (1,3,5...)
    rho_e = rho_jj_ee_and_hh[:, ::2]  # Select electron components
    rho_h = rho_jj_ee_and_hh[:, 1::2]  # Select hole components
    
    # Sum over all sites for each component
    total_dos_e = torch.sum(rho_e, dim=1)  # Sum over electron sites
    total_dos_h = torch.sum(rho_h, dim=1)  # Sum over hole sites
    
    # Calculate transmission and noise
    num_leads = len(leads_info)
    transmission = torch.zeros((batch_size, num_leads, num_leads), dtype=torch.float32, device=device)
    andreev = torch.zeros_like(transmission)
    noise = torch.zeros_like(transmission)
    current = torch.zeros((batch_size, num_leads), dtype=torch.float32, device=device)
    
    # Calculate common sum for noise calculations
    common_sum = sum(
        lead.Gamma[ptype] * fermi_distribution(E_batch, lead.mu, temperature, ptype).unsqueeze(-1).unsqueeze(-1)
        for lead in leads_info
        for ptype in ['e', 'h']
    )
    
    # Calculate transmission and noise
    for i in range(num_leads):
        for j in range(num_leads):
            for alpha_idx, alpha in enumerate(['h', 'e']):
                for beta_idx, beta in enumerate(['h', 'e']):
                    # Calculate transmission
                    if i == j and alpha == beta:
                        T_ij = (leads_info[i].t.size(0) + 
                               torch.diagonal(
                                   leads_info[i].Gamma[alpha] @ G_retarded @
                                   leads_info[i].Gamma[alpha] @ G_retarded.conj().transpose(-1, -2) +
                                   1j * leads_info[i].Gamma[alpha] @
                                   (G_retarded.conj().transpose(-1, -2) - G_retarded),
                                   dim1=1, dim2=2
                               ).sum(1))
                    else:
                        T_ij = torch.diagonal(
                            leads_info[i].Gamma[alpha] @ G_retarded @
                            leads_info[j].Gamma[beta] @ G_retarded.conj().transpose(-1, -2),
                            dim1=1, dim2=2
                        ).sum(1)
                    
                    # Store transmission
                    if alpha == 'e' and beta == 'e':
                        transmission[:, i, j] = torch.real(T_ij)
                    elif alpha == 'h' and beta == 'e':
                        andreev[:, i, j] = torch.real(T_ij)
                    
                    # Calculate noise contributions
                    sign_alpha = 1 if alpha == 'e' else -1
                    sign_beta = 1 if beta == 'e' else -1
                    
                    # First term (shot noise)
                    if i == j and alpha == beta:
                        f_alpha = fermi_distribution(E_batch, leads_info[i].mu, temperature, alpha)
                        noise[:, i, j] += leads_info[i].t.size(0) * f_alpha * (1 - f_alpha)
                        current[:, i] += sign_alpha * leads_info[i].t.size(0) * f_alpha
                    
                    # Second term (cross correlations)
                    f_beta = fermi_distribution(E_batch, leads_info[j].mu, temperature, beta)
                    current[:, i] -= sign_alpha * T_ij * f_beta
                    
                    if i == j and alpha == beta:
                        # Third term (thermal noise)
                        for k in range(num_leads):
                            for gamma in ['h', 'e']:
                                noise[:, i, j] += T_ij * fermi_distribution(
                                    E_batch, leads_info[k].mu, temperature, gamma
                                )
                        
                        # Fourth term (quantum noise)
                        delta_term = fermi_distribution(
                            E_batch, leads_info[i].mu, temperature, alpha
                        ).unsqueeze(-1).unsqueeze(-1) * (
                            leads_info[i].Gamma[alpha] / torch.norm(leads_info[i].Gamma[alpha])
                        )
                    else:
                        delta_term = 0
                    
                    # Calculate noise terms
                    ga_term = 1j * leads_info[j].Gamma[beta] @ G_retarded.conj().transpose(-1, -2) * \
                             fermi_distribution(E_batch, leads_info[j].mu, temperature, beta).unsqueeze(-1).unsqueeze(-1)
                    gr_term = -1j * leads_info[j].Gamma[beta] @ G_retarded * \
                             fermi_distribution(E_batch, leads_info[i].mu, temperature, alpha).unsqueeze(-1).unsqueeze(-1)
                    sum_gamma_term = leads_info[j].Gamma[beta] @ G_retarded @ \
                                   common_sum @ G_retarded.conj().transpose(-1, -2)
                    
                    s_s_product = delta_term + ga_term + gr_term + sum_gamma_term
                    noise[:, i, j] -= sign_alpha * sign_beta * torch.diagonal(
                        s_s_product @ s_s_product.conj().transpose(-1, -2),
                        dim1=1, dim2=2
                    ).sum(1)
    
    return {
        'rho_jj': rho_e,
        'rho_electron': total_dos_e,
        'rho_hole': total_dos_h,
        'transmission': transmission,
        'andreev': andreev,
        'current': current,
        'noise': noise
    } 
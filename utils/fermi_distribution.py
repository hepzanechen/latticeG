import torch

def fermi_distribution(E, mu, temperature, particle_type):
    """
    Calculate the Fermi-Dirac distribution for electrons ('e') and holes ('h').
    Assuming k_B = 1 for simplicity and using compatible units.

    Parameters:
    -----------
    E : torch.Tensor
        Energy values. Natural unit is taken, kB=1
    mu : float
        Chemical potential (Fermi level).
    temperature : float
        Temperature.
    particle_type : str
        'e' for electron distribution, 'h' for hole distribution.

    Returns:
    --------
    torch.Tensor
        Fermi-Dirac distribution values for electrons or holes.
    """
    # Assume data is already on GPU
    if particle_type == 'e':
        f = 1 / (torch.exp((E - mu) / temperature) + 1)
    elif particle_type == 'h':
        f = 1 / (torch.exp((E + mu) / temperature) + 1)
    else:
        raise ValueError('Invalid type. Choose either "e" for electron or "h" for hole.')
    
    return f

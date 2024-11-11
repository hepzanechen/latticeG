import torch
from fermi_distribution import fermi_distribution

def lead_decimation(E, t, epsilon0, mu, temperature, particle_type, desired_accuracy=1e-25):
    """
    Calculate the lead's Green's functions using the decimation method.

    Parameters:
    -----------
    E : torch.Tensor
        Energy value (should already be on the appropriate device).
    t : torch.Tensor
        Hopping matrix (should already be on the appropriate device).
    epsilon0 : torch.Tensor
        Onsite energy matrix (should already be on the appropriate device).
    mu : float
        Chemical potential.
    temperature : float
        Temperature.
    particle_type : str
        'e' for electron distribution, 'h' for hole distribution.
    desired_accuracy : float
        Desired accuracy threshold for the iterative decimation process.

    Returns:
    --------
    tuple
        (gLr, gLa, gLless, gLmore) - Retarded, advanced, lesser, and greater Green's functions.
    """
    slice_dim = t.size(0)

    # Fermi distribution for calculating lesser and greater Green's functions
    f = fermi_distribution(E, mu, temperature, particle_type)

    # Add small imaginary part for regularization
    if particle_type == 'h':
        omega = (-E + 1e-2j) * torch.eye(slice_dim,dtype=torch.complex64, device=E.device)
    else:  # Default to 'particle'
        omega = (E + 1e-2j) * torch.eye(slice_dim,dtype=torch.complex64, device=E.device)

    # Initialize variables
    H00 = epsilon0
    H01 = t
    H10 = t.T.conj()
    alpha = (H01 @ torch.linalg.inv(omega - H00)) @ H01
    beta = (H10 @ torch.linalg.inv(omega - H00)) @ H10
    epsilon_s = H00 + (H01 @ torch.linalg.inv(omega - H00)) @ H10
    E_mat = epsilon_s + (H10 @ torch.linalg.inv(omega - H00)) @ H01

    # Iterate until desired accuracy is reached
    while torch.norm(alpha) > desired_accuracy:
        alpha_prev = alpha
        beta_prev = beta
        epsilon_prev = E_mat
        epsilon_s = epsilon_s + (alpha_prev @ torch.linalg.inv(omega - epsilon_prev)) @ beta_prev
        alpha = (alpha_prev @ torch.linalg.inv(omega - epsilon_prev)) @ alpha_prev
        beta = (beta_prev @ torch.linalg.inv(omega - epsilon_prev)) @ beta_prev
        E_mat = epsilon_prev + (alpha_prev @ torch.linalg.inv(omega - epsilon_prev)) @ beta_prev + (beta_prev @ torch.linalg.inv(omega - epsilon_prev)) @ alpha_prev

    # Calculating the retarded Green's function
    gLr = torch.linalg.inv(omega - epsilon_s)

    # The advanced Green's function is the Hermitian conjugate of gLr
    gLa = gLr.T.conj()

    # Lesser and greater Green's functions using the Keldysh formalism
    gLless = (gLa - gLr) * f
    gLmore = (gLr - gLa) * (1 - f)

    return gLr, gLa, gLless, gLmore


from time import time
# Define parameters
t = torch.tensor([[20.0, 0.0], [0.0, 20.0]],dtype=torch.complex64,device='cuda' if torch.cuda.is_available() else 'cpu')
epsilon0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex64, device=t.device)
E = torch.tensor(0.1,dtype=torch.float32, device=t.device)
mu = torch.tensor(0.2,dtype=torch.float32,device=t.device)
temperature = torch.tensor(1e-6,dtype=torch.float32,device=t.device)
particle_type = 'e'

# Time the function execution
start_time = time()

# Call lead_decimation to calculate Green's functions
gLr, gLa, gLless, gLmore = lead_decimation(E, t, epsilon0, mu, temperature, particle_type)
elapsed_time = time() - start_time
# Display results
print('Retarded Greens function:')
print(gLr)

print('Advanced Greens function:')
print(gLa)

print('Lesser Greens function:')
print(gLless)

print('Greater Greens function:')
print(gLmore)

print(f'Elapsed time: {elapsed_time:.4f} seconds')
import torch
from .fermi_distribution import fermi_distribution

def lead_decimation(E: torch.Tensor, t: torch.Tensor, epsilon0: torch.Tensor, mu: torch.Tensor, temperature: torch.Tensor, particle_type: str, desired_accuracy: float = 1e-25) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # Fermi distribution for calculating lesser and greater Green's functions
    f = fermi_distribution(E, mu, temperature, particle_type)

    # Add small imaginary part for regularization
    imaginary_regularization = 1e-2j
    scalar = -E if particle_type == 'h' else E
    omega = torch.diag(torch.full_like(t[:, 0], scalar + imaginary_regularization, dtype=torch.complex64, device=E.device))

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

    # for hole G22(\omega)=-G11(-\omega)^*
    if particle_type == 'h':
        gLr = -gLr.conj()

        
    # The advanced Green's function is the Hermitian conjugate of gLr
    gLa = gLr.T.conj()


    # This for kron operation latter
    gLr = gLr.contiguous()
    gLa = gLa.contiguous()

    # Lesser and greater Green's functions using the Keldysh formalism
    gLless = (gLa - gLr) * f
    gLmore = (gLr - gLa) * (1 - f)

    return gLr, gLa, gLless, gLmore


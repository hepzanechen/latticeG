import torch
from typing import List, Dict, Any
from torch.func import jacrev
from greens_functions.construct_ginv_total import construct_ginv_total

def calculation_cf_autograd(
    H_BdG: torch.Tensor,
    E_batch: torch.Tensor,
    eta: float,
    leads_info: List[Any]
) -> Dict[str, Any]:
    """
    Calculates the generating function and its derivatives up to the 4th order
    for both real and imaginary parts, handling a batch of energies efficiently
    using torch.func.jacrev without explicit loops.

    Parameters:
    -----------
    H_BdG : torch.Tensor
        Hamiltonian of the central region in BdG formalism.
    E_batch : torch.Tensor
        Batch of energy values (batch_size,).
    eta : float
        Small imaginary part for regularization.
    leads_info : List[Any]
        List of Lead objects containing lead parameters.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the generating function and its derivatives up to
        the 4th order for both real and imaginary parts.
    """
    funcDevice = E_batch.device
    num_leads = len(leads_info)
    batch_size = E_batch.size(0)

    # Initialize lambda tensor
    lambda_tensor = torch.zeros(num_leads, dtype=torch.float32, device=funcDevice).requires_grad_(True)

    # Define function to compute gen_func for batched E_batch
    def gen_func(lambda_vals):
        # Update lead lambda values
        for i, lead in enumerate(leads_info):
            lead.lambda_ = lambda_vals[i]

        # Construct Ginv_total (batch_size x ginv_size x ginv_size)
        Ginv_total = construct_ginv_total(
            H_BdG=H_BdG,
            E_batch=E_batch,  # Use batched energies
            eta=eta,
            leads_info=leads_info
        )

        # Compute the generating function: log(det(Ginv_total))
        gen_func_values = torch.logdet(Ginv_total)  # Shape: (batch_size,)
        return gen_func_values.real, gen_func_values.imag  # Each is shape (batch_size,)

    # Compute generating function values
    gen_func_real, gen_func_imag = gen_func(lambda_tensor)  # Each of shape (batch_size,)

    # First-order derivatives
    jacobian_real = jacrev(lambda l: gen_func(l)[0])(lambda_tensor)  # Shape: (num_leads, batch_size)
    jacobian_imag = jacrev(lambda l: gen_func(l)[1])(lambda_tensor)  # Shape: (num_leads, batch_size)

    # Transpose to shape (batch_size, num_leads)
    jacobian_real = jacobian_real.T
    jacobian_imag = jacobian_imag.T

    # Second-order derivatives
    hessian_real = jacrev(jacrev(lambda l: gen_func(l)[0]))(lambda_tensor)  # Shape: (num_leads, num_leads, batch_size)
    hessian_imag = jacrev(jacrev(lambda l: gen_func(l)[1]))(lambda_tensor)

    # Transpose to shape (batch_size, num_leads, num_leads)
    hessian_real = hessian_real.permute(2, 0, 1)
    hessian_imag = hessian_imag.permute(2, 0, 1)

    # Third-order derivatives
    third_order_real = jacrev(jacrev(jacrev(lambda l: gen_func(l)[0])))(lambda_tensor)  # Shape: (num_leads, num_leads, num_leads, batch_size)
    third_order_imag = jacrev(jacrev(jacrev(lambda l: gen_func(l)[1])))(lambda_tensor)

    # Transpose to shape (batch_size, num_leads, num_leads, num_leads)
    third_order_real = third_order_real.permute(3, 0, 1, 2)
    third_order_imag = third_order_imag.permute(3, 0, 1, 2)

    # Fourth-order derivatives
    fourth_order_real = jacrev(jacrev(jacrev(jacrev(lambda l: gen_func(l)[0]))))(lambda_tensor)
    fourth_order_imag = jacrev(jacrev(jacrev(jacrev(lambda l: gen_func(l)[1]))))(lambda_tensor)

    # Transpose to shape (batch_size, num_leads, num_leads, num_leads, num_leads)
    fourth_order_real = fourth_order_real.permute(4, 0, 1, 2, 3)
    fourth_order_imag = fourth_order_imag.permute(4, 0, 1, 2, 3)

    # Store results
    results: Dict[str, Any] = {
        'genFuncZero': {
            'real': gen_func_real,  # Shape: (batch_size,)
            'imag': gen_func_imag   # Shape: (batch_size,)
        },
        'gradientsZero': {
            1: {
                'real': jacobian_real,  # Shape: (batch_size, num_leads)
                'imag': jacobian_imag
            },
            2: {
                'real': hessian_real,  # Shape: (batch_size, num_leads, num_leads)
                'imag': hessian_imag
            },
            3: {
                'real': third_order_real,  # Shape: (batch_size, num_leads, num_leads, num_leads)
                'imag': third_order_imag
            },
            4: {
                'real': fourth_order_real,  # Shape: (batch_size, num_leads, num_leads, num_leads, num_leads)
                'imag': fourth_order_imag
            }
        }
    }

    return results
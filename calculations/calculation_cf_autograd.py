import torch
from typing import List, Dict, Any
from hamiltonians import Lead
from torch.func import jacrev, vmap
from greens_functions.construct_ginv_total import construct_ginv_total

def calculation_cf_autograd(
    H_BdG: torch.Tensor,
    E_batch: torch.Tensor,
    eta: float,
    leads_info: List[Lead]
) -> Dict[str, Any]:
    """
    Calculates the generating function and its derivatives up to the 4th order
    for both real and imaginary parts, handling a batch of energies efficiently
    using torch.func.jacrev and vmap.

    Parameters:
    -----------
    E_batch : torch.Tensor
        Batch of energy values (batch_size,).
    H_BdG : torch.Tensor
        Hamiltonian of the central region in BdG formalism.
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

    # Initialize lambda tensor
    lambda_tensor = torch.zeros(num_leads, dtype=torch.float32, device=funcDevice).requires_grad_(True)

    def gen_func(lambda_vals, E):
        # Update lead lambda values
        for i, lead in enumerate(leads_info):
            lead.lambda_ = lambda_vals[i]

        # Construct Ginv_total for this energy
        Ginv_total = construct_ginv_total(
            H_BdG=H_BdG,
            E_batch=E,
            eta=eta,
            leads_info=leads_info
        )

        # Compute the generating function: log(det(Ginv_total))
        gen_func_value = torch.logdet(Ginv_total)  # Scalar complex tensor
        # Return real and imaginary parts
        return gen_func_value.real, gen_func_value.imag  # Each is scalar

    # Vectorized over batch; outputs are tuples, so we handle them separately
    batched_real_func = lambda l: vmap(lambda E: gen_func(l, E)[0])(E_batch)
    batched_imag_func = lambda l: vmap(lambda E: gen_func(l, E)[1])(E_batch)

    # Compute generating function values
    gen_func_real = batched_real_func(lambda_tensor)  # Shape: (batch_size,)
    gen_func_imag = batched_imag_func(lambda_tensor)  # Shape: (batch_size,)

    # First-order derivatives
    jacobian_real = jacrev(batched_real_func)(lambda_tensor)  # Shape: (num_leads, batch_size)
    jacobian_imag = jacrev(batched_imag_func)(lambda_tensor)  # Shape: (num_leads, batch_size)

    # Transpose to shape (batch_size, num_leads)
    jacobian_real = jacobian_real.T
    jacobian_imag = jacobian_imag.T

    # Second-order derivatives
    hessian_real = jacrev(jacrev(batched_real_func))(lambda_tensor)  # Shape: (num_leads, num_leads, batch_size)
    hessian_imag = jacrev(jacrev(batched_imag_func))(lambda_tensor)

    # Transpose to shape (batch_size, num_leads, num_leads)
    hessian_real = hessian_real.permute(2, 0, 1)
    hessian_imag = hessian_imag.permute(2, 0, 1)

    # Third and Fourth-order derivatives (optional and may be computationally intensive)
    third_order_real = jacrev(jacrev(jacrev(batched_real_func)))(lambda_tensor)  # Shape: (num_leads, num_leads, num_leads, batch_size)
    third_order_imag = jacrev(jacrev(jacrev(batched_imag_func)))(lambda_tensor)

    third_order_real = third_order_real.permute(3, 0, 1, 2)
    third_order_imag = third_order_imag.permute(3, 0, 1, 2)

    fourth_order_real = jacrev(jacrev(jacrev(jacrev(batched_real_func))))(lambda_tensor)
    fourth_order_imag = jacrev(jacrev(jacrev(jacrev(batched_imag_func))))(lambda_tensor)

    fourth_order_real = fourth_order_real.permute(4, 0, 1, 2, 3)
    fourth_order_imag = fourth_order_imag.permute(4, 0, 1, 2, 3)

    # Store results
    results: Dict[str, Any] = {
        'genFuncZero': {
            'real': gen_func_real,
            'imag': gen_func_imag
        },
        'gradientsZero': {
            1: {
                'real': jacobian_real,
                'imag': jacobian_imag
            },
            2: {
                'real': hessian_real,
                'imag': hessian_imag
            },
            3: {
                'real': third_order_real,
                'imag': third_order_imag
            },
            4: {
                'real': fourth_order_real,
                'imag': fourth_order_imag
            }
        }
    }

    return results
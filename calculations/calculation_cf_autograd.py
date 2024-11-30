import torch
from typing import List, Dict, Any
from hamiltonians.Lead import Lead
from greens_functions.construct_ginv_total import construct_ginv_total

def calculation_cf_autograd(
    E_batch: torch.Tensor, 
    eta: float, 
    leads_info: List[Lead], 
    H_BdG: torch.Tensor
) -> Dict[str, Any]:
    """
    Calculates the characteristic function and its derivatives using autograd,
    handling a batch of energies.

    Parameters:
    -----------
    E_batch : torch.Tensor
        Batch of energy values (batch_size,).
    eta : float
        Small imaginary part for regularization.
    leads_info : List[Lead]
        List of Lead objects containing lead parameters.
    H_BdG : torch.Tensor
        The BdG Hamiltonian of the central region (N_c x N_c).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the characteristic function and its derivatives.
    """
    funcDevice = E_batch.device
    batch_size = E_batch.size(0)
    num_leads = len(leads_info)

    # Initialize lambda tensor with requires_grad=True for autograd
    # Shape: (num_leads,)
    lambda_tensor = torch.zeros(num_leads, dtype=torch.float32, device=funcDevice, requires_grad=True)

    # Assign lambda_tensor values to leads_info
    for i, lead in enumerate(leads_info):
        lead.lambda_ = lambda_tensor[i]

    # Construct Ginv_total (batch_size x N_total x N_total)
    Ginv_total = construct_ginv_total(
        H_BdG=H_BdG, 
        E_batch=E_batch, 
        eta=eta, 
        leads_info=leads_info
    )


    # Compute the generating function: log(det(Ginv_total))
    # logdet may have both real and imaginary parts
    gen_func = torch.logdet(Ginv_total)  # Shape: (batch_size,)

    # Extract real and imaginary parts of generating function
    gen_func_real = gen_func.real  # Shape: (batch_size,)
    gen_func_imag = gen_func.imag  # Shape: (batch_size,)

    # Initialize generating function accumulators by summing over the batch
    gen_func_real_sum = gen_func_real.sum()
    gen_func_imag_sum = gen_func_imag.sum()

    # Compute first-order derivatives using autograd
    first_order_grad_real = torch.autograd.grad(
        gen_func_real_sum, lambda_tensor, create_graph=True
    )[0]  # Shape: (num_leads,)

    first_order_grad_imag = torch.autograd.grad(
        gen_func_imag_sum, lambda_tensor, create_graph=True
    )[0]  # Shape: (num_leads,)

    # Store the generating function values and first-order derivatives
    results: Dict[str, Any] = {
        'genFuncZero': {
            'real': gen_func_real,  # Shape: (batch_size,)
            'imag': gen_func_imag   # Shape: (batch_size,)
        },
        'gradientsZero': {
            1: {
                'real': first_order_grad_real,  # Shape: (num_leads,)
                'imag': first_order_grad_imag   # Shape: (num_leads,)
            }
        }
    }

    # Initialize lists to store second-order derivatives
    second_order_grads_real: List[torch.Tensor] = []
    second_order_grads_imag: List[torch.Tensor] = []

    # Compute second-order derivatives
    for i in range(num_leads):
        # Compute the second-order gradient for the real part
        second_order_grad_real = torch.autograd.grad(
            first_order_grad_real[i], lambda_tensor, create_graph=True, retain_graph=True
        )[0]
        second_order_grads_real.append(second_order_grad_real)

        # Compute the second-order gradient for the imaginary part
        second_order_grad_imag = torch.autograd.grad(
            first_order_grad_imag[i], lambda_tensor, create_graph=True, retain_graph=True
        )[0]
        second_order_grads_imag.append(second_order_grad_imag)

    # Stack the second-order gradients 
    # Shape: (num_leads, num_leads)
    second_order_grads_real = torch.stack(second_order_grads_real)
    second_order_grads_imag = torch.stack(second_order_grads_imag)

    # Store second-order derivatives
    results['gradientsZero'][2] = {
        'real': second_order_grads_real,  # Shape: (num_leads, num_leads)
        'imag': second_order_grads_imag   # Shape: (num_leads, num_leads)
    }

    return results
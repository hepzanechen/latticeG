import torch
from typing import List, Dict, Any
from torch.func import vmap, jacrev
from greens_functions.construct_ginv_total import construct_ginv_total

def calculation_cf_autograd_vmap(
    H_BdG: torch.Tensor,
    E_batch: torch.Tensor,
    eta: float,
    leads_info: List[Any],
    max_derivative_order: int = 4  # Maximum derivative order to compute
) -> Dict[str, Any]:
    """
    Calculates the generating function and its derivatives with respect to lambda
    up to the specified order at different energy values E. Uses torch.vmap to 
    vectorize computations over E for efficiency.

    Parameters:
    -----------
    H_BdG : torch.Tensor
        Hamiltonian of the central region in BdG formalism.
    E_batch : torch.Tensor
        Tensor of energy values (batch_size,).
    eta : float
        Small imaginary part for regularization.
    leads_info : List[Any]
        List of Lead objects containing lead parameters.
    max_derivative_order : int
        Maximum order of derivatives to compute (default is 4).

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the generating function and its derivatives up to
        the specified order for each energy value in E_batch.
    """
    funcDevice = H_BdG.device
    num_leads = len(leads_info)
    batch_size = E_batch.size(0)

    # Initialize lambda tensor
    lambda_tensor = torch.zeros(num_leads, dtype=torch.float32, device=funcDevice).requires_grad_(True)

    # Define function to compute the generating function for a single E
    def gen_func_single_E(lambda_vals, E_val):
        # Update lead lambda values
        for i, lead in enumerate(leads_info):
            lead.lambda_ = lambda_vals[i]

        # Construct Ginv_total for a single E
        Ginv_total = construct_ginv_total(
            H_BdG=H_BdG,
            E_batch=E_val.unsqueeze(0),  # Make it batch_size=1
            eta=eta,
            leads_info=leads_info
        )

        # Compute the generating function: log(det(Ginv_total))
        gen_func_value = torch.logdet(Ginv_total)[0]  # Extract the scalar value
        return gen_func_value

    # Vectorize gen_func_single_E over E_batch using torch.vmap
    gen_func = vmap(lambda E_val: gen_func_single_E(lambda_tensor, E_val))(E_batch)

    # Detach and store generating function values
    gen_func_values_real = gen_func.real.detach().cpu()
    gen_func_values_imag = gen_func.imag.detach().cpu()

    # Initialize container for derivatives
    derivatives = {}

    # Define function to compute the generating function for vmap over lambda
    def gen_func_vmap_lambda(lambda_vals):
        # Update lead lambda values
        for i, lead in enumerate(leads_info):
            lead.lambda_ = lambda_vals[i]

        # Vectorized over E_batch
        def gen_func_E(E_val):
            Ginv_total = construct_ginv_total(
                H_BdG=H_BdG,
                E_batch=E_val.unsqueeze(0),  # Make it batch_size=1
                eta=eta,
                leads_info=leads_info
            )
            gen_func_value = torch.logdet(Ginv_total)[0]
            return gen_func_value

        # Apply vmap over E_batch
        gen_func_values = vmap(gen_func_E)(E_batch)
        return gen_func_values

    # Compute derivatives up to max_derivative_order
    for order in range(1, max_derivative_order + 1):
        # Use higher-order jacrev
        derivative_func_real = lambda l: gen_func_vmap_lambda(l).real
        derivative_func_imag = lambda l: gen_func_vmap_lambda(l).imag

        if order == 1:
            # First-order derivatives
            derivative_real = jacrev(derivative_func_real)(lambda_tensor)
            derivative_imag = jacrev(derivative_func_imag)(lambda_tensor)
        else:
            # Higher-order derivatives
            for _ in range(order - 1):
                derivative_func_real = jacrev(derivative_func_real)
                derivative_func_imag = jacrev(derivative_func_imag)
            derivative_real = derivative_func_real(lambda_tensor)
            derivative_imag = derivative_func_imag(lambda_tensor)

        # Select the appropriate derivative based on the order
        if order % 2 == 0:
            derivative = derivative_real.detach().cpu()
        else:
            derivative = derivative_imag.detach().cpu()

        # Store the derivative
        derivatives[f'order_{order}'] = derivative

    # Organize results
    results: Dict[str, Any] = {
        'gen_func_values_real': gen_func_values_real,  # Shape: (batch_size,)
        'gen_func_values_imag': gen_func_values_imag,  # Shape: (batch_size,)
        'derivatives': derivatives  # Contains derivatives up to max_derivative_order
    }

    return results
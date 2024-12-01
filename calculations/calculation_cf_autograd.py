import torch
from greens_functions.construct_ginv_total import construct_ginv_total

def calculation_cf_autograd(E: torch.Tensor, H_BdG: torch.Tensor, eta: torch.Tensor, leads_info: list) -> dict:
    """
    Calculates the generating function and its derivatives up to the 4th order for both real and imaginary parts.

    Parameters:
    -----------
    E : torch.Tensor
        Energy value.
    H_BdG : torch.Tensor
        Hamiltonian of the central region in BdG formalism.
    eta : torch.Tensor
        Small imaginary part for regularization.
    leads_info : list
        List of Lead objects containing lead parameters.

    Returns:
    --------
    dict
        A dictionary containing the generating function matrix and its derivatives up to the 4th order for both real and imaginary parts.
    """
    # Initialize results dictionary
    results = {}
    num_leads = len(leads_info)

    # Set lambda tensor with requires_grad=True for autograd
    lambda_tensor = torch.zeros(num_leads, dtype=torch.float32, requires_grad=True)

    # Update lead lambda values
    for i, lead in enumerate(leads_info):
        lead.lambda_ = lambda_tensor[i]

    # Construct Ginv_total
    Ginv_total = construct_ginv_total(H_BdG=H_BdG, E=E, eta=eta, leads_info=leads_info)

    # Compute the generating function
    gen_func = torch.logdet(Ginv_total)

    # Extract real and imaginary parts of generating function
    gen_func_real = gen_func.real
    gen_func_imag = gen_func.imag

    # Store the generating function values
    results['genFuncValueReal'] = gen_func_real.item()
    results['genFuncValueImag'] = gen_func_imag.item()

    # First-order derivative: Real part
    first_order_grad_real = torch.autograd.grad(gen_func_real, lambda_tensor, create_graph=True)[0]
    # First-order derivative: Imaginary part
    first_order_grad_imag = torch.autograd.grad(gen_func_imag, lambda_tensor, create_graph=True)[0]

    # Store first-order derivatives
    results['gradientsZero'] = {1: {'real': first_order_grad_real, 'imag': first_order_grad_imag}}

    # Second-order derivatives
    second_order_grads_real = []
    second_order_grads_imag = []

    for i in range(num_leads):
        # Compute the second-order gradient for real part
        second_order_grad_real = torch.autograd.grad(first_order_grad_real[i], lambda_tensor, create_graph=True, retain_graph=True)[0]
        second_order_grads_real.append(second_order_grad_real)

        # Compute the second-order gradient for imaginary part
        second_order_grad_imag = torch.autograd.grad(first_order_grad_imag[i], lambda_tensor, create_graph=True, retain_graph=True)[0]
        second_order_grads_imag.append(second_order_grad_imag)

    # Stack second-order gradients into tensors
    second_order_grads_real = torch.stack(second_order_grads_real)
    second_order_grads_imag = torch.stack(second_order_grads_imag)
    results['gradientsZero'][2] = {'real': second_order_grads_real, 'imag': second_order_grads_imag}

    # Third-order derivatives
    third_order_grads_real = []
    third_order_grads_imag = []

    for i in range(second_order_grads_real.numel()):
        # Compute the third-order gradient for real part
        third_order_grad_real = torch.autograd.grad(second_order_grads_real.view(-1)[i], lambda_tensor, create_graph=True, retain_graph=True)[0]
        third_order_grads_real.append(third_order_grad_real)

        # Compute the third-order gradient for imaginary part
        third_order_grad_imag = torch.autograd.grad(second_order_grads_imag.view(-1)[i], lambda_tensor, create_graph=True, retain_graph=True)[0]
        third_order_grads_imag.append(third_order_grad_imag)

    # Stack third-order gradients into tensors
    third_order_grads_real = torch.stack(third_order_grads_real)
    third_order_grads_imag = torch.stack(third_order_grads_imag)
    results['gradientsZero'][3] = {'real': third_order_grads_real, 'imag': third_order_grads_imag}

    # Fourth-order derivatives
    fourth_order_grads_real = []
    fourth_order_grads_imag = []

    for i in range(third_order_grads_real.numel()):
        # Compute the fourth-order gradient for real part
        fourth_order_grad_real = torch.autograd.grad(third_order_grads_real.view(-1)[i], lambda_tensor, create_graph=True, retain_graph=True)[0]
        fourth_order_grads_real.append(fourth_order_grad_real)

        # Compute the fourth-order gradient for imaginary part
        fourth_order_grad_imag = torch.autograd.grad(third_order_grads_imag.view(-1)[i], lambda_tensor, create_graph=True, retain_graph=True)[0]
        fourth_order_grads_imag.append(fourth_order_grad_imag)

    # Stack fourth-order gradients into tensors
    fourth_order_grads_real = torch.stack(fourth_order_grads_real)
    fourth_order_grads_imag = torch.stack(fourth_order_grads_imag)
    results['gradientsZero'][4] = {'real': fourth_order_grads_real, 'imag': fourth_order_grads_imag}

    return results

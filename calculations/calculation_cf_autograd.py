import torch
from greens_functions.construct_ginv_total import construct_ginv_total

def calculation_cf_autograd(E: torch.Tensor, H_BdG: torch.Tensor, eta: torch.Tensor, leads_info: list) -> dict:
    """
    Calculates the generating function and its derivatives up to the 4th order.

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
        A dictionary containing the generating function matrix and its derivatives up to the 4th order.
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

    # Store the generating function value
    results['genFuncValue'] = gen_func.item()

    # First-order derivative
    first_order_grad = torch.autograd.grad(gen_func, lambda_tensor, create_graph=True)[0]
    results['gradientsZero'] = {1: first_order_grad}

    # Second-order derivative (Hessian)
    second_order_grads = []
    for grad in first_order_grad:
        second_order_grad = torch.autograd.grad(grad, lambda_tensor, create_graph=True)[0]
        second_order_grads.append(second_order_grad)
    second_order_grads = torch.stack(second_order_grads)
    results['gradientsZero'][2] = second_order_grads

    # Third-order derivative
    third_order_grads = []
    for grad in second_order_grads.view(-1):
        third_order_grad = torch.autograd.grad(grad, lambda_tensor, create_graph=True)[0]
        third_order_grads.append(third_order_grad)
    third_order_grads = torch.stack(third_order_grads)
    results['gradientsZero'][3] = third_order_grads

    # Fourth-order derivative
    fourth_order_grads = []
    for grad in third_order_grads.view(-1):
        fourth_order_grad = torch.autograd.grad(grad, lambda_tensor, create_graph=True)[0]
        fourth_order_grads.append(fourth_order_grad)
    fourth_order_grads = torch.stack(fourth_order_grads)
    results['gradientsZero'][4] = fourth_order_grads

    return results


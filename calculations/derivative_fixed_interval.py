import torch
from greens_functions.construct_ginv_total import construct_ginv_total
from calculations.calculation_cf_autograd import calculation_cf_autograd

def adaptive_derivative_calculation(
    E_values: torch.Tensor, 
    H_BdG: torch.Tensor, 
    eta: torch.Tensor, 
    leads_info: list
) -> dict:
    """
    Calculates the generating function and its derivatives up to the 4th order for a fixed set of energy values.

    Parameters:
    -----------
    E_values : torch.Tensor
        Tensor of energy values over which to calculate the generating function and derivatives.
    H_BdG : torch.Tensor
        Hamiltonian of the central region in BdG formalism.
    eta : torch.Tensor
        Small imaginary part for regularization.
    leads_info : list
        List of Lead objects containing lead parameters.

    Returns:
    --------
    dict
        A dictionary containing the generating function matrix and its derivatives up to the 4th order for all energy values.
    """
    # Initialize results dictionary
    results = {'E': E_values.cpu().numpy(), 'gradientsZero': [], 'genFuncMatrix': []}

    # Calculate generating function and derivatives for all energy values
    for E in E_values:
        # Perform calculation using autograd for generating function and derivatives
        calc_results = calculation_cf_autograd(E, H_BdG, eta, leads_info)
        
        # Store the results for each energy
        results['genFuncMatrix'].append(calc_results['genFuncValue'])
        results['gradientsZero'].append(calc_results['gradientsZero'])

    # Convert lists to tensors and move to CPU for easier manipulation
    results['genFuncMatrix'] = torch.tensor(results['genFuncMatrix']).cpu()
    results['gradientsZero'] = [{order: grad.cpu() for order, grad in res.items()} for res in results['gradientsZero']]

    # Integration across energies using trapezoidal rule (inspired by MATLAB's `trapz`)
    results['IntgradientsZero'] = integrate_gradients_across_energies(results['gradientsZero'], E_values)

    return results

def integrate_gradients_across_energies(
    gradients_zero: list, 
    E_values: torch.Tensor
) -> list:
    """
    Integrates gradients across energy values using the trapezoidal rule.

    Parameters:
    -----------
    gradients_zero : list of dict
        List containing the gradients for each order at each energy value.
    E_values : torch.Tensor
        Tensor of energy values over which to integrate.

    Returns:
    --------
    list of torch.Tensor
        List of integrated gradients for each order.
    """
    # Assuming all orders have the same number of leads
    num_orders = len(gradients_zero[0])

    integrated_gradients = []
    for order_idx in range(1, num_orders + 1):
        # Stack gradients along energy axis to create a tensor
        order_gradients = torch.stack([grad[order_idx] for grad in gradients_zero], dim=-1)
        # Integrate along energy axis using trapezoidal rule
        int_gradient = 0.5 * torch.trapz(order_gradients, E_values, dim=-1)
        integrated_gradients.append(int_gradient)

    return integrated_gradients


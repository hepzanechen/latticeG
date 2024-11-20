import torch
from torch.func import vmap
from greens_functions.construct_ginv_total import construct_ginv_total
from calculations.calculation_cf_autograd import calculation_cf_autograd

def calculation_cf_autograd_vmap(E_batch: torch.Tensor, H_BdG: torch.Tensor, eta: torch.Tensor, leads_info: list) -> dict:
    """
    Vectorized calculation of the generating function and its derivatives up to the 4th order for both real and imaginary parts,
    using a batch of energy values.

    Parameters:
    -----------
    E_batch : torch.Tensor
        Batch of energy values (e.g., [E1, E2, ...]).
    H_BdG : torch.Tensor
        Hamiltonian of the central region in BdG formalism.
    eta : torch.Tensor
        Small imaginary part for regularization.
    leads_info : list
        List of Lead objects containing lead parameters.

    Returns:
    --------
    dict
        A dictionary containing the generating function matrix and its derivatives up to the 4th order for both real and imaginary parts
        for each energy value in the batch.
    """
    # Wrap the original calculation function for a single energy value
    def single_energy_calc(E):
        # Use the existing autograd calculation function
        return calculation_cf_autograd(E, H_BdG, eta, leads_info)
    
    # Use vmap to vectorize over batch of energy values
    vectorized_results = vmap(single_energy_calc)(E_batch)

    # Organize the results for easier access and analysis
    # Since vmap results are nested dicts, we extract them appropriately
    final_results = {
        'genFuncValuesReal': torch.tensor([res['genFuncValueReal'] for res in vectorized_results]),
        'genFuncValuesImag': torch.tensor([res['genFuncValueImag'] for res in vectorized_results]),
        'gradientsZero': {
            1: {
                'real': torch.stack([res['gradientsZero'][1]['real'] for res in vectorized_results]),
                'imag': torch.stack([res['gradientsZero'][1]['imag'] for res in vectorized_results]),
            },
            2: {
                'real': torch.stack([res['gradientsZero'][2]['real'] for res in vectorized_results]),
                'imag': torch.stack([res['gradientsZero'][2]['imag'] for res in vectorized_results]),
            },
            3: {
                'real': torch.stack([res['gradientsZero'][3]['real'] for res in vectorized_results]),
                'imag': torch.stack([res['gradientsZero'][3]['imag'] for res in vectorized_results]),
            },
            4: {
                'real': torch.stack([res['gradientsZero'][4]['real'] for res in vectorized_results]),
                'imag': torch.stack([res['gradientsZero'][4]['imag'] for res in vectorized_results]),
            }
        }
    }

    return final_results
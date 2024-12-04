import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import json
from typing import Dict, Any

def parse_results(results: Dict[str, Any], E: torch.Tensor):
    """
    Parses the results from calculation_cf_autograd.

    Parameters:
    -----------
    results : Dict[str, Any]
        The results dictionary returned by calculation_cf_autograd.
    E : torch.Tensor
        Tensor of energy values (batch_size,).

    Returns:
    --------
    parsed_data : Dict[str, Any]
        Dictionary containing numpy arrays of generating functions and derivatives.
    """
    parsed_data = {}
    E_np = E.detach().cpu().numpy()
    parsed_data['E'] = E_np  # Shape: (batch_size,)

    # Parse Generating Function
    parsed_data['genFuncZero_real'] = results['genFuncZero_real'].detach().cpu().numpy()
    parsed_data['genFuncZero_imag'] = results['genFuncZero_imag'].detach().cpu().numpy()

    # Parse Gradients
    parsed_data['grad_order_real'] = results['grad_order_real'].numpy()  # Shape: (max_order, batch_size, num_leads)
    parsed_data['grad_order_imag'] = results['grad_order_imag'].numpy()  # Shape: (max_order, batch_size, num_leads)

    # Parse Hessians
    parsed_data['hessian_order_real'] = results['hessian_order_real'].numpy()  # Shape: (1, batch_size, num_leads, num_leads)
    parsed_data['hessian_order_imag'] = results['hessian_order_imag'].numpy()  # Shape: (1, batch_size, num_leads, num_leads)

    # Parse Third-Order Derivatives
    parsed_data['third_order_real'] = results['third_order_real'].numpy()    # Shape: (1, batch_size, num_leads, num_leads, num_leads)
    parsed_data['third_order_imag'] = results['third_order_imag'].numpy()    # Shape: (1, batch_size, num_leads, num_leads, num_leads)

    # Parse Fourth-Order Derivatives
    parsed_data['fourth_order_real'] = results['fourth_order_real'].numpy()  # Shape: (1, batch_size, num_leads, num_leads, num_leads, num_leads)
    parsed_data['fourth_order_imag'] = results['fourth_order_imag'].numpy()  # Shape: (1, batch_size, num_leads, num_leads, num_leads, num_leads)

    return parsed_data

def save_data(parsed_data: Dict[str, Any], mat_filename: str, json_filename: str):
    """
    Saves the parsed data to .mat and .json files.

    Parameters:
    -----------
    parsed_data : Dict[str, Any]
        The parsed data dictionary.
    mat_filename : str
        Filename for the .mat file.
    json_filename : str
        Filename for the .json file.
    """
    # Save as .mat
    sio.savemat(mat_filename, parsed_data)
    print(f"Data saved to {mat_filename} successfully.")

    # Save as .json
    # Convert numpy arrays to lists for JSON serialization
    json_ready_data = {key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in parsed_data.items()}
    
    with open(json_filename, 'w') as json_file:
        json.dump(json_ready_data, json_file, indent=4)
    
    print(f"Data saved to {json_filename} successfully.")

def plot_generating_function(parsed_data: Dict[str, Any]):
    """
    Plots the real and imaginary parts of the generating function versus Energy.

    Parameters:
    -----------
    parsed_data : Dict[str, Any]
        The parsed data dictionary.
    """
    E = parsed_data['E']
    gen_real = parsed_data['genFuncZero_real']
    gen_imag = parsed_data['genFuncZero_imag']

    plt.figure(figsize=(10, 6))
    plt.plot(E, gen_real, label='GenFunc Real')
    plt.plot(E, gen_imag, label='GenFunc Imag', linestyle='--')
    plt.xlabel('Energy (E)')
    plt.ylabel('Generating Function')
    plt.title('Generating Function vs Energy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_gradients(parsed_data: Dict[str, Any], num_leads: int):
    """
    Plots the real and imaginary parts of the gradients versus Energy for each lead and derivative order.

    Parameters:
    -----------
    parsed_data : Dict[str, Any]
        The parsed data dictionary.
    num_leads : int
        Number of leads.
    """
    E = parsed_data['E']
    grad_real = parsed_data['grad_order_real']  # Shape: (max_order, batch_size, num_leads)
    grad_imag = parsed_data['grad_order_imag']  # Shape: (max_order, batch_size, num_leads)
    max_order = grad_real.shape[0]

    for order in range(max_order):
        plt.figure(figsize=(12, 6))
        for lead in range(num_leads):
            plt.subplot(2, num_leads, lead+1)
            plt.plot(E, grad_real[order, :, lead], label=f'Order {order+1} Lead {lead+1} Real')
            plt.xlabel('Energy (E)')
            plt.ylabel('Gradient Real')
            plt.title(f'Order {order+1} - Lead {lead+1} Real')
            plt.legend()
            plt.grid(True)

            plt.subplot(2, num_leads, num_leads + lead+1)
            plt.plot(E, grad_imag[order, :, lead], label=f'Order {order+1} Lead {lead+1} Imag')
            plt.xlabel('Energy (E)')
            plt.ylabel('Gradient Imag')
            plt.title(f'Order {order+1} - Lead {lead+1} Imag')
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_hessian(parsed_data: Dict[str, Any], num_leads: int):
    """
    Plots specific components of the Hessian versus Energy.

    Parameters:
    -----------
    parsed_data : Dict[str, Any]
        The parsed data dictionary.
    num_leads : int
        Number of leads.
    """
    E = parsed_data['E']
    hessian_real = parsed_data['hessian_order_real'][0]  # Shape: (batch_size, num_leads, num_leads)
    hessian_imag = parsed_data['hessian_order_imag'][0]  # Shape: (batch_size, num_leads, num_leads)

    for lead1 in range(num_leads):
        for lead2 in range(num_leads):
            plt.figure(figsize=(10, 4))
            
            # Real part
            plt.subplot(1, 2, 1)
            plt.plot(E, hessian_real[:, lead1, lead2], label=f'Real [{lead1+1},{lead2+1}]')
            plt.xlabel('Energy (E)')
            plt.ylabel('Hessian Real')
            plt.title(f'Hessian Real [{lead1+1},{lead2+1}] vs E')
            plt.legend()
            plt.grid(True)
            
            # Imaginary part
            plt.subplot(1, 2, 2)
            plt.plot(E, hessian_imag[:, lead1, lead2], label=f'Imag [{lead1+1},{lead2+1}]')
            plt.xlabel('Energy (E)')
            plt.ylabel('Hessian Imag')
            plt.title(f'Hessian Imag [{lead1+1},{lead2+1}] vs E')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()

def plot_third_order(parsed_data: Dict[str, Any], num_leads: int):
    """
    Plots specific components of the third-order derivatives versus Energy.

    Parameters:
    -----------
    parsed_data : Dict[str, Any]
        The parsed data dictionary.
    num_leads : int
        Number of leads.
    """
    E = parsed_data['E']
    third_order_real = parsed_data['third_order_real'][0]  # Shape: (batch_size, num_leads, num_leads, num_leads)
    third_order_imag = parsed_data['third_order_imag'][0]  # Shape: (batch_size, num_leads, num_leads, num_leads)

    # Example: Select specific components to plot
    lead_indices = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    for idx, (lead1, lead2, lead3) in enumerate(lead_indices):
        plt.figure(figsize=(10, 4))
        
        # Real part
        plt.subplot(1, 2, 1)
        plt.plot(E, third_order_real[:, lead1, lead2, lead3], label=f'Real [{lead1+1},{lead2+1},{lead3+1}]')
        plt.xlabel('Energy (E)')
        plt.ylabel('3rd Order Deriv Real')
        plt.title(f'3rd Order Deriv Real [{lead1+1},{lead2+1},{lead3+1}] vs E')
        plt.legend()
        plt.grid(True)
        
        # Imaginary part
        plt.subplot(1, 2, 2)
        plt.plot(E, third_order_imag[:, lead1, lead2, lead3], label=f'Imag [{lead1+1},{lead2+1},{lead3+1}]')
        plt.xlabel('Energy (E)')
        plt.ylabel('3rd Order Deriv Imag')
        plt.title(f'3rd Order Deriv Imag [{lead1+1},{lead2+1},{lead3+1}] vs E')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def plot_fourth_order(parsed_data: Dict[str, Any], num_leads: int):
    """
    Plots specific components of the fourth-order derivatives versus Energy.

    Parameters:
    -----------
    parsed_data : Dict[str, Any]
        The parsed data dictionary.
    num_leads : int
        Number of leads.
    """
    E = parsed_data['E']
    fourth_order_real = parsed_data['fourth_order_real'][0]  # Shape: (batch_size, num_leads, num_leads, num_leads, num_leads)
    fourth_order_imag = parsed_data['fourth_order_imag'][0]  # Shape: (batch_size, num_leads, num_leads, num_leads, num_leads)

    # Example: Select specific components to plot
    lead_indices = [(0, 0, 0, 0), (0, 1, 1, 1), (1, 0, 1, 0), (1, 1, 0, 1)]
    for idx, (lead1, lead2, lead3, lead4) in enumerate(lead_indices):
        plt.figure(figsize=(10, 4))
        
        # Real part
        plt.subplot(1, 2, 1)
        plt.plot(E, fourth_order_real[:, lead1, lead2, lead3, lead4], label=f'Real [{lead1+1},{lead2+1},{lead3+1},{lead4+1}]')
        plt.xlabel('Energy (E)')
        plt.ylabel('4th Order Deriv Real')
        plt.title(f'4th Order Deriv Real [{lead1+1},{lead2+1},{lead3+1},{lead4+1}] vs E')
        plt.legend()
        plt.grid(True)
        
        # Imaginary part
        plt.subplot(1, 2, 2)
        plt.plot(E, fourth_order_imag[:, lead1, lead2, lead3, lead4], label=f'Imag [{lead1+1},{lead2+1},{lead3+1},{lead4+1}]')
        plt.xlabel('Energy (E)')
        plt.ylabel('4th Order Deriv Imag')
        plt.title(f'4th Order Deriv Imag [{lead1+1},{lead2+1},{lead3+1},{lead4+1}] vs E')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def plot_all(parsed_data: Dict[str, Any], num_leads: int):
    """
    Calls all plotting functions to visualize the generating function and its derivatives.

    Parameters:
    -----------
    parsed_data : Dict[str, Any]
        The parsed data dictionary.
    num_leads : int
        Number of leads.
    """
    # Plot Generating Function
    plot_generating_function(parsed_data)
    
    # Plot Gradients
    plot_gradients(parsed_data, num_leads)
    
    # Plot Hessian (2nd Order Derivative)
    plot_hessian(parsed_data, num_leads)
    
    # Plot Third-Order Derivative
    plot_third_order(parsed_data, num_leads)
    
    # Plot Fourth-Order Derivative
    plot_fourth_order(parsed_data, num_leads)

# Example Usage
if __name__ == "__main__":
    # Example 'results' structure as provided
    results = {
        'genFuncZero_real': torch.tensor([116.9482, 114.2998], requires_grad=True),
        'genFuncZero_imag': torch.tensor([-1.1921e-07,  1.7884e-07], requires_grad=True),
        'grad_order_real': torch.tensor([
            [[1.0431e-07, 2.0862e-07],
             [-1.9372e-07, 0.0000e+00]],
            # Add more orders if max_order > 1
        ]),
        'grad_order_imag': torch.tensor([
            [[-0.2522, -1.1974],
             [0.2522, 1.1974]],
            # Add more orders if max_order > 1
        ]),
        'hessian_order_real': torch.tensor([
            [[-0.2278, 0.1642],
             [-0.5086, 0.1499]],
            [[0.1642, -0.2278],
             [0.1499, -0.5086]]
        ]).unsqueeze(0),  # Shape: (1, batch_size, num_leads, num_leads)
        'hessian_order_imag': torch.tensor([
            [[-8.1956e-08, -4.1910e-08],
             [-1.8626e-08, -1.9372e-07]],
            [[-5.4017e-08, 5.9605e-08],
             [-3.8743e-07, 1.6019e-07]]
        ]).unsqueeze(0),  # Shape: (1, batch_size, num_leads, num_leads)
        'third_order_real': torch.tensor([
            [[[-4.3772e-08, -4.5635e-08],
              [-4.4703e-08, 2.7008e-08]],
             [[0.0000e+00, 1.2107e-07],
              [-3.7253e-09, -7.8231e-08]]],
            [[[-3.4459e-08, 4.2841e-08],
              [3.3528e-08, 1.8626e-09]],
             [[1.8999e-07, -1.8626e-09],
              [-7.8231e-08, 1.6019e-07]]]
        ]).unsqueeze(0),  # Shape: (1, batch_size, num_leads, num_leads, num_leads)
        'third_order_imag': torch.tensor([
            [[[0.1842, -0.1332],
              [-0.1332, 0.1332]],
             [[-0.0610, 0.0040],
              [0.0040, -0.0040]]],
            [[[-0.1332, 0.1332],
              [0.1332, -0.1842]],
             [[0.0040, -0.0040],
              [-0.0040, 0.0610]]]
        ]).unsqueeze(0),  # Shape: (1, batch_size, num_leads, num_leads, num_leads)
        'fourth_order_real': torch.tensor([
            [[[[0.1110, -0.0811],
               [-0.0811, 0.0891]],
              [[-0.0811, 0.0891],
               [0.0891, -0.0811]]],
             [[[-0.1986, 0.0416],
               [0.0416, 0.0047]],
              [[0.0416, 0.0047],
               [0.0047, 0.0416]]]],
            [[[[ -0.0811, 0.0891],
               [0.0891, -0.0811]],
              [[0.0891, -0.0811],
               [-0.0811, 0.1110]]],
             [[[0.0416, 0.0047],
               [0.0047, 0.0416]],
              [[0.0047, 0.0416],
               [0.0416, -0.1986]]]]
        ]).unsqueeze(0),  # Shape: (1, batch_size, num_leads, num_leads, num_leads, num_leads)
        'fourth_order_imag': torch.tensor([
            [[[[-7.2177e-09, 5.3085e-08],
               [5.1223e-08, -3.6322e-08]],
              [[4.7730e-08, -1.3970e-08],
               [-1.3970e-08, 1.3970e-09]]],
             [[[ -1.2014e-07, -5.1223e-08],
               [-5.1223e-08, 4.2841e-08]],
              [[-1.3597e-07, 3.5390e-08],
               [3.5390e-08, 2.2352e-08]]]],
            [[[[2.6543e-08, -1.8626e-08],
               [-1.9558e-08, 3.2131e-08]],
              [[-2.2352e-08, 3.1432e-08],
               [3.1432e-08, -5.3202e-08]]],
             [[[7.4506e-09, 5.2154e-08],
               [5.2154e-08, -1.5553e-07]],
              [[6.5193e-08, -3.7719e-08],
               [-3.7719e-08, -1.4808e-07]]]]
        ]).unsqueeze(0)  # Shape: (1, batch_size, num_leads, num_leads, num_leads, num_leads)
    }
    
    # Example energy tensor
    E = torch.tensor([0.1, 0.2], dtype=torch.float32)
    
    # Parse the results
    parsed_data = parse_results(results, E)
    
    # Save the data
    save_data(parsed_data, 'calculation_results.mat', 'calculation_results.json')
    
    # Plot the data
    num_leads = 2  # Update based on your specific case
    plot_all(parsed_data, num_leads)
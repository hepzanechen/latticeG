import torch
import numpy as np
from calculations.calculation_cf_autograd import calculation_cf_autograd
from scipy.signal import find_peaks
from scipy.integrate import trapz

def adaptive_derivative_calculation(E_values, threshold, H_BdG, eta, leads_info):
    # Initial calculation for all E values
    results = [calculation_cf_autograd(E, H_BdG, eta, leads_info) for E in E_values]
    first_derivative = [res['gradientsZero'][1] for res in results]

    iteration_counter = 0
    max_iterations = 2

    # Find peaks and their locations
    derivatives_values = np.array([x[0].item() for x in first_derivative])
    peaks, _ = find_peaks(np.abs(derivatives_values), prominence=threshold)

    # Estimate the bandwidth (Gamma) for each rho peak and add points around it
    for peak in peaks:
        # Add points around the peak based on the estimated Gamma
        new_points = np.linspace(E_values[peak - 1], E_values[peak + 1], 100)
        E_values = np.concatenate((E_values, new_points))

    # Sort E values and recalculate results
    E_values = np.unique(E_values)
    results = [calculation_cf_autograd(E, H_BdG, eta, leads_info) for E in E_values]
    first_derivative = [res['gradientsZero'][1] for res in results]

    # Loop for refining the energy range based on the results
    while True:
        iteration_counter += 1

        # Find rapid changes in derivative values
        derivatives_values = np.array([x[0].item() for x in first_derivative])
        small_constant = 1e-6  # to prevent division by zero
        diff_first_derivative = np.diff(derivatives_values)
        change_ratio = np.abs(diff_first_derivative) / (np.abs(derivatives_values[:-1]) + small_constant)
        rough_indices = np.where(change_ratio > threshold)[0]

        # If no rapid changes are found or max iterations reached, exit the loop
        if len(rough_indices) == 0 or iteration_counter > max_iterations:
            break

        # Add finer energy points around high variation areas
        new_results = []
        for idx in rough_indices:
            num_new_points = min(10, round(change_ratio[idx] / threshold))
            new_points = np.linspace(E_values[idx], E_values[idx + 1], num_new_points)[1:-1]  # Avoid duplicating existing points
            E_values = np.concatenate((E_values, new_points))
            new_results.extend([calculation_cf_autograd(E, H_BdG, eta, leads_info) for E in new_points])

        # Combine new results with existing ones
        results.extend(new_results)

        # Sort arrays based on E for plotting and further calculations
        sorted_indices = np.argsort(E_values)
        E_values = E_values[sorted_indices]
        results = [results[i] for i in sorted_indices]

    # Construct the final results structure
    results_struct = {
        'E': E_values,
        'gradientsZero': [res['gradientsZero'] for res in results],
        'genFuncMatrix': [res['genFuncMatrix'] for res in results]
    }

    # Integrate gradients across energies
    results_struct['IntgradientsZero'] = integrate_gradients_across_energies(results_struct['gradientsZero'], results_struct['E'])

    return results_struct

def integrate_gradients_across_energies(gradients_zero, energies):
    # Assuming the first element is representative, take out a gradient at a given E value
    num_orders = len(gradients_zero[0])

    int_gradients = [None] * num_orders

    for order_idx in range(num_orders):
        # Extract the gradients of the current order for all energy values
        order_gradients = [grad[order_idx] for grad in gradients_zero]

        # Concatenate along a new axis (order_idx + 1)
        concatenated = np.stack(order_gradients, axis=order_idx + 1)

        # Integrate along the energy axis
        int_gradients[order_idx] = 0.5 * trapz(concatenated, energies, axis=order_idx + 1)

    return int_gradients

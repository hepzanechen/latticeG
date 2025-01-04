"""LDOS plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union
import torch

def plot_ldos_surface(
    E_values: Union[np.ndarray, torch.Tensor],
    rho_jj_values: Union[np.ndarray, torch.Tensor],
    E_lower: float,
    E_upper: float,
    Nx: int,
    Ny: int,
    is_spin: bool = True,
    save_path: Optional[str] = None
) -> None:
    """Plot the Local Density of States (LDOS) surface.
    
    Args:
        E_values: Array of energy values (batch_size,)
        rho_jj_values: Array of rho_jj values (batch_size, num_sites)
        E_lower: Lower bound of energy range to integrate
        E_upper: Upper bound of energy range to integrate
        Nx: Number of sites in x direction
        Ny: Number of sites in y direction
        is_spin: Whether the system includes spin
        save_path: Optional path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(E_values, torch.Tensor):
        E_values = E_values.cpu().numpy()
    if isinstance(rho_jj_values, torch.Tensor):
        rho_jj_values = rho_jj_values.cpu().numpy()
    
    # Filter energy values and corresponding rho_jj values
    mask = (E_values >= E_lower) & (E_values <= E_upper)
    E_filtered = E_values[mask]
    rho_jj_filtered = rho_jj_values[mask]
    
    # Calculate LDOS by integrating rho_jj over the filtered energy values
    ldos = np.trapz(rho_jj_filtered, E_filtered, axis=0)
    
    # Generate meshgrid for plotting
    X, Y = np.meshgrid(np.arange(1, Ny + 1), np.arange(1, Nx + 1))
    
    if is_spin:
        # Split and reshape LDOS for spin-up and spin-down
        ldos_reshaped = ldos.reshape(2, -1)
        ldos_spinup = ldos_reshaped[0].reshape(Nx, Ny)
        ldos_spindown = ldos_reshaped[1].reshape(Nx, Ny)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
        
        # Plot spin-up LDOS
        im1 = ax1.pcolormesh(X, Y, ldos_spinup, shading='auto')
        ax1.set_title('Spin-up LDOS')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label='LDOS')
        
        # Plot spin-down LDOS
        im2 = ax2.pcolormesh(X, Y, ldos_spindown, shading='auto')
        ax2.set_title('Spin-down LDOS')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='LDOS')
        
    else:
        # Reshape and plot total LDOS
        ldos_matrix = ldos.reshape(Nx, Ny)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(X, Y, ldos_matrix, shading='auto')
        ax.set_title(f'Local Density of States (LDOS) across energy interval [{E_lower:.2f}, {E_upper:.2f}]')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='LDOS')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_ldos_energy_slice(
    E_values: Union[np.ndarray, torch.Tensor],
    rho_jj_values: Union[np.ndarray, torch.Tensor],
    energy: float,
    Nx: int,
    Ny: int,
    is_spin: bool = True,
    energy_tolerance: float = 1e-6,
    save_path: Optional[str] = None
) -> None:
    """Plot the LDOS at a specific energy.
    
    Args:
        E_values: Array of energy values (batch_size,)
        rho_jj_values: Array of rho_jj values (batch_size, num_sites)
        energy: Energy value to plot
        Nx: Number of sites in x direction
        Ny: Number of sites in y direction
        is_spin: Whether the system includes spin
        energy_tolerance: Tolerance for finding the energy value
        save_path: Optional path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(E_values, torch.Tensor):
        E_values = E_values.cpu().numpy()
    if isinstance(rho_jj_values, torch.Tensor):
        rho_jj_values = rho_jj_values.cpu().numpy()
    
    # Find closest energy value
    idx = np.abs(E_values - energy).argmin()
    if np.abs(E_values[idx] - energy) > energy_tolerance:
        print(f"Warning: No exact energy match found. Using closest value: {E_values[idx]:.6f}")
    
    # Get LDOS at this energy
    ldos = rho_jj_values[idx]
    
    # Generate meshgrid for plotting
    X, Y = np.meshgrid(np.arange(1, Ny + 1), np.arange(1, Nx + 1))
    
    if is_spin:
        # Split and reshape LDOS for spin-up and spin-down
        ldos_reshaped = ldos.reshape(2, -1)
        ldos_spinup = ldos_reshaped[0].reshape(Nx, Ny)
        ldos_spindown = ldos_reshaped[1].reshape(Nx, Ny)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
        
        # Plot spin-up LDOS
        im1 = ax1.pcolormesh(X, Y, ldos_spinup, shading='auto')
        ax1.set_title(f'Spin-up LDOS at E = {E_values[idx]:.6f}')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label='LDOS')
        
        # Plot spin-down LDOS
        im2 = ax2.pcolormesh(X, Y, ldos_spindown, shading='auto')
        ax2.set_title(f'Spin-down LDOS at E = {E_values[idx]:.6f}')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='LDOS')
        
    else:
        # Reshape and plot total LDOS
        ldos_matrix = ldos.reshape(Nx, Ny)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(X, Y, ldos_matrix, shading='auto')
        ax.set_title(f'Local Density of States (LDOS) at E = {E_values[idx]:.6f}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='LDOS')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close() 
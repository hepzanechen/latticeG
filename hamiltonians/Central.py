import torch
from typing import List, Tuple
import random

class Central:
    def __init__(self, Ny: int, Nx: int, t_y: torch.Tensor, t_x: torch.Tensor):
        """
        Initializes the base central Hamiltonian with roles of x and y switched.

        Parameters:
        -----------
        Ny : int
            Number of lattice sites in the y-direction.
        Nx : int
            Number of lattice sites in the x-direction.
        t_y : torch.Tensor
            Hopping parameter in the y-direction.
        t_x : torch.Tensor
            Hopping parameter in the x-direction.
        """
        self.Ny = Ny
        self.Nx = Nx
        self.t_y = t_y
        self.t_x = t_x
        self.funcDevice=t_x.device

        # Construct the Hamiltonian matrices
        self.H_chain_y = self._construct_chain_y()
        #  TODO: make * to kron when expand t_x, now it seems only support scalr t_x
        self.H_along_x = self.t_x * torch.eye(Ny, dtype=torch.complex64,device=self.funcDevice)

        # Assemble the full Hamiltonian
        self.H_full = self._assemble_full_hamiltonian()

    def _construct_chain_y(self) -> torch.Tensor:
        """Constructs the Hamiltonian matrix for a single chain along the y-direction."""
        #  TODO: make * to kron when expand t_y
        H_inter_y = self.t_y * torch.diag(torch.ones(self.Ny - 1, dtype=torch.complex64,device=self.funcDevice), 1)
        H_chain_y = H_inter_y + H_inter_y.T.conj()
        return H_chain_y

    def _assemble_full_hamiltonian(self) -> torch.Tensor:
        """Assembles the full Hamiltonian matrix without disorder effects."""
        H_full_diag = torch.kron(torch.eye(self.Nx, dtype=torch.complex64,device=self.funcDevice), self.H_chain_y)
        H_full_diag1 = torch.kron(torch.diag(torch.ones(self.Nx - 1, dtype=torch.complex64,device=self.funcDevice), 1), self.H_along_x)
        return H_full_diag + H_full_diag1 + H_full_diag1.T.conj()

    def __repr__(self):
        return f"Central(Ny={self.Ny}, Nx={self.Nx}, t_y={self.t_y}, t_x={self.t_x})"



class DisorderedCentral(Central):
    def __init__(self, Nx: int, Ny: int, t_x: torch.Tensor, t_y: torch.Tensor, deltaV: torch.Tensor, N_imp: int, xi: torch.Tensor):
        """
        Initializes a disordered central Hamiltonian.

        Parameters:
        -----------
        Nx, Ny, t_x, t_y : same as Central.
        deltaV : torch.Tensor
            Amplitude range for the disorder potential.
        N_imp : int
            Number of impurities in the lattice.
        xi : torch.Tensor
            Correlation length of the disorder potential.
        """
        super().__init__(Nx, Ny, t_x, t_y)
        self.deltaV = deltaV
        self.N_imp = N_imp
        self.xi = xi

        # Add disorder potential
        self.H_full += self._construct_disorder_potential()

    def _construct_disorder_potential(self) -> torch.Tensor:
        """Generates the disorder potential for the lattice."""
        U = torch.zeros((self.Nx, self.Ny), dtype=torch.float32,device=self.funcDevice)

        # Generate random positions for scatterers
        R_k_x = torch.randint(0, self.Nx, (self.N_imp,), dtype=torch.int32,device=self.funcDevice)
        R_k_y = torch.randint(0, self.Ny, (self.N_imp,), dtype=torch.int32,device=self.funcDevice)

        # Generate random amplitudes for scatterers
        U_k = 2 * self.deltaV * torch.rand(self.N_imp) - self.deltaV

        # Compute the disorder potential for each lattice site
        for n in range(self.Nx):
            for j in range(self.Ny):
                r_nj = torch.tensor([n, j], dtype=torch.float32,device=self.funcDevice)
                for k in range(self.N_imp):
                    R_k = torch.tensor([R_k_x[k], R_k_y[k]], dtype=torch.float32,device=self.funcDevice)
                    U[n, j] += U_k[k] * torch.exp(-torch.norm(r_nj - R_k) ** 2 / (2 * self.xi ** 2))

        return torch.diag(U.flatten())


class CentralBdG(Central):
    def __init__(self, Ny: int, Nx: int, t_y: torch.Tensor, t_x: torch.Tensor, Delta: torch.Tensor):
        """
        Initializes a BdG central Hamiltonian with superconducting pairing.

        Parameters:
        ----------- 
        Ny : int
            Number of lattice sites in the y-direction.
        Nx : int
            Number of lattice sites in the x-direction.
        t_y : torch.Tensor
            Hopping parameter in the y-direction.
        t_x : torch.Tensor
            Hopping parameter in the x-direction.
        Delta : torch.Tensor
            Pairing potential for superconductivity.
        """
        super().__init__(Ny, Nx, t_y, t_x)
        self.Delta = Delta

        # Construct BdG Hamiltonian
        self.H_full_BdG = self._construct_bdg_with_pairing()

    def _construct_bdg_with_pairing(self) -> torch.Tensor:
        """Constructs the BdG Hamiltonian with superconducting pairing."""
        pairing_matrix = torch.eye(self.Ny * self.Nx, dtype=torch.complex64,device=self.funcDevice) * self.Delta
        H_full_BdG = torch.kron(self.H_full, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64,device=self.funcDevice)) + \
                     torch.kron(-self.H_full.conj(), torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64,device=self.funcDevice)) + \
                     torch.kron(pairing_matrix, torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64,device=self.funcDevice))+ \
                     torch.kron(pairing_matrix.conj(), torch.tensor([[0, 0], [1, 0]], dtype=torch.complex64,device=self.funcDevice))
        return H_full_BdG

    def __repr__(self):
        return f"CentralBdG(Ny={self.Ny}, Nx={self.Nx}, t_y={self.t_y}, t_x={self.t_x}, Delta={self.Delta})"

class DisorderedCentralBdG(DisorderedCentral):
    def __init__(self, Ny, Nx, t_y, t_x, Delta, disorder_strength):
        super().__init__(Ny, Nx, t_y, t_x, disorder_strength)
        self.Delta = Delta
        self.H_full_BdG = self._construct_bdg_with_pairing()

    def _construct_bdg_with_pairing(self) -> torch.Tensor:
        """Constructs the BdG Hamiltonian with superconducting pairing."""
        pairing_matrix = torch.eye(self.Ny * self.Nx, dtype=torch.complex64, device=self.H_full.device) * self.Delta
        H_full_BdG = torch.kron(self.H_full, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=self.H_full.device)) + \
                     torch.kron(-self.H_full.conj(), torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=self.H_full.device)) + \
                     torch.kron(pairing_matrix, torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64, device=self.H_full.device)) + \
                     torch.kron(pairing_matrix.conj(), torch.tensor([[0, 0], [1, 0]], dtype=torch.complex64, device=self.H_full.device))
        return H_full_BdG


class CentralUniformDisorder:
    def __init__(self, Ny, Nx, t_y, t_x, U0, salt):
        self.Ny = Ny
        self.Nx = Nx
        self.t_y = t_y
        self.t_x = t_x
        self.U0 = U0
        self.salt = salt
        self.funcDevice = t_x.device
        # Assemble the full Hamiltonian
        self.H_full = self._assemble_full_hamiltonian()

    def _construct_chain_y(self) -> torch.Tensor:
        """Constructs the Hamiltonian matrix for a single chain along the y-direction with disorder."""
        H_inter_y = self.t_y * torch.diag(torch.ones(self.Ny - 1, dtype=torch.complex64, device=self.funcDevice), 1)
        H_chain_y = H_inter_y + H_inter_y.T.conj()

        # Add disorder to the on-site terms using a vectorized approach
        disorder = self.U0 * (torch.rand(self.Ny, dtype=torch.complex64, device=self.funcDevice) - 0.5)
        H_chain_y += torch.diag(disorder)

        return H_chain_y

    def _assemble_full_hamiltonian(self) -> torch.Tensor:
        # Implementation of the full Hamiltonian assembly
        pass

class CentralBdGDisorder(CentralUniformDisorder):
    """# Example usage
    if __name__ == "__main__":
        Ny = 2
        Nx = 2
        t_y = torch.tensor(2.0, dtype=torch.complex64)
        t_x = torch.tensor(1.0, dtype=torch.complex64)
        Delta = torch.tensor(0, dtype=torch.complex64)
        U0 = 1.0
        salt = 13

        central_bdg_disorder = CentralBdGDisorder(Ny, Nx, t_y, t_x, Delta, U0, salt)
        H_full = central_bdg_disorder.H_full
        print(H_full)
    """
    def __init__(self, Ny, Nx, t_y, t_x, Delta, U0, salt):
        self.Delta = Delta
        super().__init__(Ny, Nx, t_y, t_x, U0, salt)

    def _assemble_full_hamiltonian(self) -> torch.Tensor:
        # Implementation of the full Hamiltonian assembly for BdG
        pass

    def H_full_BdG(self) -> torch.Tensor:
        # Implementation of the BdG Hamiltonian
        pass




class TopologicalSurface2D:
    """Class for constructing 2D topological surface state Hamiltonian."""
    
    def __init__(self, Ny: int, Nx: int, t_y: torch.Tensor, t_x: torch.Tensor,
                 mu: torch.Tensor, B: torch.Tensor, M: torch.Tensor):
        """Initialize 2D topological surface state Hamiltonian.
        
        Args:
            Ny, Nx: System dimensions
            t_y, t_x: Spin-orbit coupling parameters
            mu: Chemical potential
            M: Mass parameter
        """
        self.Ny = Ny
        self.Nx = Nx
        self.t_y = t_y
        self.t_x = t_x
        self.mu = mu
        self.B = B
        self.M = M
        self.funcDevice = t_x.device
        
        # Pauli matrices in spin space
        self.sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=self.funcDevice)
        self.sigma_0 = torch.eye(2, dtype=torch.complex64, device=self.funcDevice)
        
        # Construct the full Hamiltonian
        self.H_full = self._construct_topological_hamiltonian()
        
    def _construct_chain_y(self) -> torch.Tensor:
        """Constructs the Hamiltonian matrix for a single chain along y-direction."""
        # Chemical potential and onsite terms
        H_onsite = self.B *(-self.M - 2 )* self.sigma_z - self.mu * self.sigma_0
        H_chain = torch.kron(torch.eye(self.Ny, dtype=torch.complex64, device=self.funcDevice), 
                           H_onsite)
        
        # Nearest neighbor hopping along y
        t_nn_y = (4/3)*(self.B * self.sigma_z + 1j * self.t_y * self.sigma_y)/2
        H_nn_y = torch.kron(
            torch.diag(torch.ones(self.Ny - 1, dtype=torch.complex64, device=self.funcDevice), 1),
            t_nn_y
        )
        
        # Next-nearest neighbor hopping along y
        t_nnn_y = (-1/6) * (2 * self.B * self.sigma_z + 1j * self.t_y * self.sigma_y)/2
        H_nnn_y = torch.kron(
            torch.diag(torch.ones(self.Ny - 2, dtype=torch.complex64, device=self.funcDevice), 2),
            t_nnn_y
        )
        
        # Add all terms and their Hermitian conjugates
        H_chain = H_chain + H_nn_y + H_nn_y.conj().T + H_nnn_y + H_nnn_y.conj().T
        
        return H_chain
        
    def _construct_topological_hamiltonian(self) -> torch.Tensor:
        """Constructs the full 2D topological surface state Hamiltonian."""
        # Construct hopping along x-direction (nearest neighbor)
        t_nn_x = (4/3)*(self.B * self.sigma_z + 1j * self.t_x * self.sigma_x)/2
        H_nn_x = torch.kron(
            torch.diag(torch.ones(self.Nx - 1, dtype=torch.complex64, device=self.funcDevice), 1),
            torch.kron(torch.eye(self.Ny, dtype=torch.complex64, device=self.funcDevice), t_nn_x)
        )
        
        # Next-nearest neighbor hopping along x
        t_nnn_x = (-1/6) * (2 * self.B * self.sigma_z + 1j * self.t_x * self.sigma_x)/2
        H_nnn_x = torch.kron(
            torch.diag(torch.ones(self.Nx - 2, dtype=torch.complex64, device=self.funcDevice), 2),
            torch.kron(torch.eye(self.Ny, dtype=torch.complex64, device=self.funcDevice), t_nnn_x)
        )
        
        # Full Hamiltonian with all terms
        H_full = torch.kron(torch.eye(self.Nx, dtype=torch.complex64, device=self.funcDevice), 
                           self._construct_chain_y()) + \
                 H_nn_x + H_nn_x.conj().T + H_nnn_x + H_nnn_x.conj().T
        
        return H_full


class MZMVortexHamiltonian(TopologicalSurface2D):
    """Class for constructing BdG Hamiltonian with Majorana zero mode vortices."""
    
    def __init__(self, Ny: int, Nx: int, t_y: torch.Tensor, t_x: torch.Tensor, 
                 Delta_0: torch.Tensor, xi_0: torch.Tensor, lambda_L: torch.Tensor, vortex_positions: List[Tuple[torch.Tensor, torch.Tensor]],
                 mu: torch.Tensor, B: torch.Tensor, M: torch.Tensor):
        """Initialize vortex Hamiltonian for 2D topological surface state."""
        super().__init__(Ny, Nx, t_y, t_x, mu,B, M)
        self.Delta = Delta_0
        self.xi_0 = xi_0
        self.lambda_L = lambda_L.item()
        self.vortex_positions = vortex_positions
        self.hbar = torch.tensor(1, dtype=torch.float32, device=self.funcDevice)
        self.e = torch.tensor(1, dtype=torch.float32, device=self.funcDevice)
        self.Phi_0 = torch.pi * self.hbar / self.e # Half Magnetic flux quantum in natural units
        
        # Apply Peierls substitution to normal state Hamiltonian
        self.H_full = self._apply_peierls_substitution(self.H_full)
        
        # Construct full BdG Hamiltonian with vortices
        self.H_full_BdG = self._construct_vortex_hamiltonian()
        
    def _apply_peierls_substitution(self, h_normal: torch.Tensor) -> torch.Tensor:
        """Apply Peierls substitution to modify hopping terms with vector potential."""
        h_modified = h_normal.clone()
        
        # Loop over all sites
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                site1 = ix * self.Ny + iy
                x1, y1 = torch.tensor(ix, dtype=torch.float32, device=self.funcDevice), torch.tensor(iy, dtype=torch.float32, device=self.funcDevice)  # Actual lattice positions
                
                # Only process right and up hoppings
                neighbors = [
                    (ix+1, iy),   # Right only
                    (ix, iy+1),   # Up only
                ]
                
                # Only process right2 and up2 for next nearest neighbors
                nnn_neighbors = [
                    (ix+2, iy),   # Right 2 only
                    (ix, iy+2),   # Up 2 only
                ]
                
                # Process neighbors
                for nx, ny in neighbors + nnn_neighbors:
                    if 0 <= nx < self.Nx and 0 <= ny < self.Ny:
                        site2 = nx * self.Ny + ny
                        x2, y2 = torch.tensor(nx, dtype=torch.float32, device=self.funcDevice), torch.tensor(ny, dtype=torch.float32, device=self.funcDevice)
                        
                        # Calculate Peierls phase
                        phase = self._calculate_peierls_phase(x1, y1, x2, y2)
                        
                        # Modify hopping terms for all spin combinations
                        for spin1 in [0, 1]:
                            for spin2 in [0, 1]:
                                idx1 = site1 * 2 + spin1
                                idx2 = site2 * 2 + spin2
                                if h_modified[idx1, idx2] != 0:
                                    h_modified[idx1, idx2] *= phase
                                    h_modified[idx2, idx1] = h_modified[idx1, idx2].conj()
        
        return h_modified
    
    def _calculate_delta(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate superconducting gap at position (x,y)."""
        delta = torch.ones(1, dtype=torch.complex64, device=self.funcDevice) * self.Delta
        
        for x_j, y_j in self.vortex_positions:
            r = torch.sqrt((x - x_j)**2 + (y - y_j)**2)
            if r == 0:
                return torch.zeros(1, dtype=torch.complex64, device=self.funcDevice)[0]
            
            # Amplitude factor with tanh profile
            tanh_factor = torch.tanh(r / self.xi_0)
            
            # Phase factor (x+iy)/r for p-wave like vortex
            phase = ((x - x_j) + 1j*(y - y_j))/r
            
            delta *= tanh_factor * phase
            # Convert delta to scalar tensor by taking first element
        return delta[0]
        
    def _calculate_vector_potential(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate vector potential at position (x,y)."""
        Ax = torch.zeros(1, dtype=torch.float32, device=self.funcDevice)
        Ay = torch.zeros(1, dtype=torch.float32, device=self.funcDevice)
        
        for x_j, y_j in self.vortex_positions:
            r = torch.sqrt((x - x_j)**2 + (y - y_j)**2)
            if r == 0:
                continue
                
            # Calculate K1 Bessel function
            from scipy.special import k1
            K1_factor = torch.tensor(k1(r.cpu().numpy() / self.lambda_L), 
                                   device=self.funcDevice)
            
            # Vector potential from London equation solution
            common = self.Phi_0/(2*torch.pi*r) * (1 - r/self.lambda_L * K1_factor)
            
            # Add contribution in polar coordinates
            Ax += -common * (y - y_j)/r  # -sin(θ) component
            Ay += common * (x - x_j)/r   # cos(θ) component
            
        return Ax, Ay
        
    def _calculate_peierls_phase(self, x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Calculate Peierls phase between two points."""
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        Ax, Ay = self._calculate_vector_potential(x_mid, y_mid)
        dx = x2 - x1
        dy = y2 - y1
        phase = self.e * (Ax * dx + Ay * dy) / self.hbar
        # Convert phase to scalar tensor by taking first element
        return torch.exp(1j * phase)[0]
    
    def _construct_vortex_hamiltonian(self) -> torch.Tensor:
        """Construct full BdG Hamiltonian with vortices."""
        h_bdg = torch.kron(self.H_full, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64,device=self.funcDevice)) + \
                torch.kron(-self.H_full.conj(), torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64,device=self.funcDevice)) 
        
        # Add pairing terms
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                pos = (ix * self.Ny + iy) * 2 * 2  # *2 for spin and 2 for BdG
                x, y = torch.tensor(ix, dtype=torch.float32, device=self.funcDevice), torch.tensor(iy, dtype=torch.float32, device=self.funcDevice)  # Use actual lattice positions
                delta = self._calculate_delta(x, y)
                # Pairing in the basis |c_↑†,c_↑,c_↓†,c_↓⟩
                h_bdg[pos, pos+3] = delta  # c_↑† to c_↓
                h_bdg[pos+1, pos+2] = -delta.conj()  # c_↑ to c_↓†
                h_bdg[pos+2, pos+1] = -delta  # c_↓† to c_↑
                h_bdg[pos+3, pos] = delta.conj()  # c_↓ to c_↑†
        
        return h_bdg

    def visualize_vector_potential(self):
        """Visualize vector potential field using quiver plot."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create grid points
        x = np.linspace(0, self.Nx-1, self.Nx)
        y = np.linspace(0, self.Ny-1, self.Ny)
        X, Y = np.meshgrid(x, y)
        
        # Calculate vector potential at each point
        Ax = np.zeros((self.Ny, self.Nx))
        Ay = np.zeros((self.Ny, self.Nx))
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                x_tensor = torch.tensor(float(i), dtype=torch.float32, device=self.funcDevice)
                y_tensor = torch.tensor(float(j), dtype=torch.float32, device=self.funcDevice)
                Ax_t, Ay_t = self._calculate_vector_potential(x_tensor, y_tensor)
                Ax[j,i] = Ax_t.cpu().numpy()
                Ay[j,i] = Ay_t.cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.quiver(X, Y, Ax, Ay)
        
        # Plot vortex positions
        vortex_x = [pos[0] for pos in self.vortex_positions]
        vortex_y = [pos[1] for pos in self.vortex_positions]
        plt.plot(vortex_x, vortex_y, 'ro', label='Vortices')
        
        plt.title('Vector Potential Field')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.colorbar()
        plt.savefig('vector_potential.png')
        plt.close()

    def visualize_delta_field(self):
        """Visualize superconducting order parameter."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Calculate Delta at each point
        Delta_mag = np.zeros((self.Ny, self.Nx))
        Delta_phase = np.zeros((self.Ny, self.Nx))
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                x_tensor = torch.tensor(float(i), dtype=torch.float32, device=self.funcDevice)
                y_tensor = torch.tensor(float(j), dtype=torch.float32, device=self.funcDevice)
                delta = self._calculate_delta(x_tensor, y_tensor)
                Delta_mag[j,i] = abs(delta.cpu().numpy())
                Delta_phase[j,i] = np.angle(delta.cpu().numpy())
        
        # Plot magnitude
        plt.figure(figsize=(15, 5))
        
        plt.subplot(121)
        im1 = plt.imshow(Delta_mag, origin='lower', extent=[0, self.Nx-1, 0, self.Ny-1])
        plt.colorbar(im1, label='|Δ|')
        plt.title('Superconducting Gap Magnitude')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Plot vortex positions
        vortex_x = [pos[0].cpu().numpy() for pos in self.vortex_positions]
        vortex_y = [pos[1].cpu().numpy() for pos in self.vortex_positions]
        plt.plot(vortex_x, vortex_y, 'wo', markeredgecolor='black', label='Vortices')
        plt.legend()
        
        # Plot phase
        plt.subplot(122)
        im2 = plt.imshow(Delta_phase, origin='lower', extent=[0, self.Nx-1, 0, self.Ny-1], 
                         cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im2, label='arg(Δ)')
        plt.title('Superconducting Phase')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(vortex_x, vortex_y, 'wo', markeredgecolor='black', label='Vortices')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('delta_field.png')
        plt.close()

    def visualize_peierls_phase(self):
        """Visualize Peierls phase for nearest-neighbor hoppings."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Calculate Peierls phase for horizontal and vertical bonds
        phase_x = np.zeros((self.Ny, self.Nx-1))  # horizontal bonds
        phase_y = np.zeros((self.Ny-1, self.Nx))  # vertical bonds
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                x1 = torch.tensor(float(i), dtype=torch.float32, device=self.funcDevice)
                y1 = torch.tensor(float(j), dtype=torch.float32, device=self.funcDevice)
                
                # Horizontal bonds
                if i < self.Nx-1:
                    x2 = torch.tensor(float(i+1), dtype=torch.float32, device=self.funcDevice)
                    phase = self._calculate_peierls_phase(x1, y1, x2, y1)
                    phase_x[j,i] = np.angle(phase.cpu().numpy())
                
                # Vertical bonds
                if j < self.Ny-1:
                    y2 = torch.tensor(float(j+1), dtype=torch.float32, device=self.funcDevice)
                    phase = self._calculate_peierls_phase(x1, y1, x1, y2)
                    phase_y[j,i] = np.angle(phase.cpu().numpy())
        
        # Plot phases
        plt.figure(figsize=(15, 5))
        
        plt.subplot(121)
        im1 = plt.imshow(phase_x, origin='lower', extent=[0, self.Nx-2, 0, self.Ny-1], 
                         cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im1, label='Phase (rad)')
        plt.title('Peierls Phase - Horizontal Bonds')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.subplot(122)
        im2 = plt.imshow(phase_y, origin='lower', extent=[0, self.Nx-1, 0, self.Ny-2], 
                         cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im2, label='Phase (rad)')
        plt.title('Peierls Phase - Vertical Bonds')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.tight_layout()
        plt.savefig('peierls_phase.png')
        plt.close()
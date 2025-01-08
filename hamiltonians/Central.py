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
    def __init__(self, Nx: int, Ny: int, t_x: torch.Tensor, t_y: torch.Tensor, deltaV: torch.Tensor, N_imp: int, xi: float):
        """
        Initializes a disordered central Hamiltonian.

        Parameters:
        -----------
        Nx, Ny, t_x, t_y : same as Central.
        deltaV : torch.Tensor
            Amplitude range for the disorder potential.
        N_imp : int
            Number of impurities in the lattice.
        xi : float
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


class VortexHamiltonian(CentralBdG):
    """Class for constructing BdG Hamiltonian with vortices."""
    
    def __init__(self, Ny: int, Nx: int, t_y: torch.Tensor, t_x: torch.Tensor, 
                 Delta_0: float, xi_0: float, lambda_L: float, vortex_positions: List[Tuple[float, float]],
                 mu: float = 0.0, m: float = 2.0, v_F: float = 1.0):
        """Initialize vortex Hamiltonian.
        
        Args:
            Ny, Nx: System dimensions
            t_y, t_x: Hopping parameters
            Delta_0: Base superconducting gap
            xi_0: Superconducting coherence length
            lambda_L: London penetration depth
            vortex_positions: List of (x,y) positions of vortices
            mu: Chemical potential
            m: Mass parameter
            v_F: Fermi velocity
        """
        super().__init__(Ny, Nx, t_y, t_x, Delta_0)
        self.xi_0 = xi_0
        self.lambda_L = lambda_L
        self.vortex_positions = vortex_positions
        self.mu = mu
        self.m = m
        self.v_F = v_F
        #self.Phi_0 = 2.067833848 * 1e-15  # Magnetic flux quantum in Wb
        self.Phi_0 = 1  # Magnetic flux quantum in custom unit
        
        # Pauli matrices
        self.sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=self.funcDevice)
        self.tau_0 = torch.eye(2, dtype=torch.complex64, device=self.funcDevice)
        
        # Construct full Hamiltonian with vortices
        self.H_full_BdG = self._construct_vortex_hamiltonian()
        
    def _calculate_delta(self, x: float, y: float) -> torch.Tensor:
        """Calculate superconducting gap at position (x,y)."""
        delta = torch.ones(1, dtype=torch.complex64, device=self.funcDevice) * self.Delta
        
        for x_j, y_j in self.vortex_positions:
            r = torch.sqrt((x - x_j)**2 + (y - y_j)**2)
            if r == 0:
                # In the vortex core, the gap is zero
                return torch.zeros(1, dtype=torch.complex64, device=self.funcDevice)
            
            # Amplitude factor
            tanh_factor = torch.tanh(r / self.xi_0)
            
            # Phase factor
            phase = torch.complex((x - x_j)/r, (y - y_j)/r)
            
            delta *= tanh_factor * phase
            
        return delta
        
    def _calculate_vector_potential(self, x: float, y: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate vector potential at position (x,y)."""
        Ax = torch.zeros(1, dtype=torch.float32, device=self.funcDevice)
        Ay = torch.zeros(1, dtype=torch.float32, device=self.funcDevice)
        
        for x_j, y_j in self.vortex_positions:
            r = torch.sqrt((x - x_j)**2 + (y - y_j)**2)
            if r == 0:
                # In the vortex core, the vector potential is screened zero
                return torch.zeros(1, dtype=torch.float32, device=self.funcDevice), torch.zeros(1, dtype=torch.float32, device=self.funcDevice)
                
            # Calculate K1 Bessel function (using scipy and converting to torch)
            from scipy.special import k1
            K1_factor = torch.tensor(k1(r.cpu().numpy() / self.lambda_L), 
                                   device=self.funcDevice)
            
            # Common factor
            common = self.Phi_0 * (1/r - K1_factor/self.lambda_L)
            
            # Add contribution to Ax and Ay
            Ax += (y - y_j) * common / r
            Ay += (x - x_j) * common / r
            
        return Ax, Ay
        
    def _calculate_peierls_phase(self, x1: float, y1: float, x2: float, y2: float) -> torch.Tensor:
        """Calculate Peierls phase between two points."""
        # Midpoint for vector potential
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        
        # Get vector potential at midpoint
        Ax, Ay = self._calculate_vector_potential(x_mid, y_mid)
        
        # Calculate line integral A·dl
        dx = x2 - x1
        dy = y2 - y1
        
        phase = -(Ax * dx + Ay * dy) * self.e / self.hbar
        return torch.exp(1j * phase)
        
    def _construct_onsite_term(self) -> torch.Tensor:
        """Construct onsite term H_0(m)."""
        return (self.v_F * self.m * self.sigma_z - self.mu * self.tau_0)
        
    def _construct_hopping_terms(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct nearest-neighbor hopping terms H_nn."""
        # x-direction hopping
        t_x = -self.v_F * self.sigma_z/2 + 1j * self.v_F * self.sigma_x/2
        
        # y-direction hopping
        t_y = -self.v_F * self.sigma_z/2 + 1j * self.v_F * self.sigma_y/2
        
        return t_y, t_x
        
    def _construct_vortex_hamiltonian(self) -> torch.Tensor:
        """Construct full BdG Hamiltonian with vortices."""
        h_size = self.Nx * self.Ny
        h_bdg = torch.zeros((2*h_size, 2*h_size), dtype=torch.complex64, device=self.funcDevice)
        
        # Onsite terms
        h_onsite = self._construct_onsite_term()
        t_y, t_x = self._construct_hopping_terms()
        
        # Fill Hamiltonian
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                pos = ix * self.Ny + iy
                x, y = ix + 0.5, iy + 0.5  # Center coordinates
                
                # Onsite energy and pairing
                delta = self._calculate_delta(x, y)
                h_bdg[2*pos:2*pos+2, 2*pos:2*pos+2] = h_onsite
                h_bdg[2*pos:2*pos+2, 2*h_size+2*pos:2*h_size+2*pos+2] = -1j * delta * self.sigma_y
                h_bdg[2*h_size+2*pos:2*h_size+2*pos+2, 2*pos:2*pos+2] = 1j * delta.conj() * self.sigma_y
                h_bdg[2*h_size+2*pos:2*h_size+2*pos+2, 2*h_size+2*pos:2*h_size+2*pos+2] = -h_onsite.conj()
                
                # Hopping terms with Peierls phase
                # x-direction
                if ix < self.Nx - 1:
                    phase = self._calculate_peierls_phase(x, y, x+1, y)
                    h_nn = t_x * phase
                    next_pos = (ix+1) * self.Ny + iy
                    
                    # Electron part
                    h_bdg[2*pos:2*pos+2, 2*next_pos:2*next_pos+2] = h_nn
                    h_bdg[2*next_pos:2*next_pos+2, 2*pos:2*pos+2] = h_nn.conj().T
                    
                    # Hole part
                    h_bdg[2*h_size+2*pos:2*h_size+2*pos+2, 2*h_size+2*next_pos:2*h_size+2*next_pos+2] = \
                        -h_nn.conj() * phase.conj()
                    h_bdg[2*h_size+2*next_pos:2*h_size+2*next_pos+2, 2*h_size+2*pos:2*h_size+2*pos+2] = \
                        -(h_nn.conj() * phase.conj()).conj().T
                
                # y-direction
                if iy < self.Ny - 1:
                    phase = self._calculate_peierls_phase(x, y, x, y+1)
                    h_nn = t_y * phase
                    next_pos = ix * self.Ny + (iy+1)
                    
                    # Electron part
                    h_bdg[2*pos:2*pos+2, 2*next_pos:2*next_pos+2] = h_nn
                    h_bdg[2*next_pos:2*next_pos+2, 2*pos:2*pos+2] = h_nn.conj().T
                    
                    # Hole part
                    h_bdg[2*h_size+2*pos:2*h_size+2*pos+2, 2*h_size+2*next_pos:2*h_size+2*next_pos+2] = \
                        -h_nn.conj() * phase.conj()
                    h_bdg[2*h_size+2*next_pos:2*h_size+2*next_pos+2, 2*h_size+2*pos:2*h_size+2*pos+2] = \
                        -(h_nn.conj() * phase.conj()).conj().T
        
        return h_bdg
    
class VortexHamiltonian2D(CentralBdG):
    """Class for constructing 2D topological surface state BdG Hamiltonian with vortices."""
    
    def __init__(self, Ny: int, Nx: int, t_y: torch.Tensor, t_x: torch.Tensor, 
                 Delta_0: float, xi_0: float, lambda_L: float, vortex_positions: List[Tuple[float, float]],
                 mu: float = 0.0, v: float = 1.0):
        """Initialize vortex Hamiltonian for 2D topological surface state.
        
        Args:
            Ny, Nx: System dimensions
            t_y, t_x: Hopping parameters
            Delta_0: Base superconducting gap
            xi_0: Superconducting coherence length
            lambda_L: London penetration depth
            vortex_positions: List of (x,y) positions of vortices
            mu: Chemical potential
            v: Velocity of surface Dirac fermions
        """
        super().__init__(Ny, Nx, t_y, t_x, Delta_0)
        self.xi_0 = xi_0
        self.lambda_L = lambda_L
        self.vortex_positions = vortex_positions
        self.mu = mu
        self.v = v
        self.Phi_0 = 1.0  # Magnetic flux quantum in natural units
        
        # Pauli matrices in spin space
        self.sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=self.funcDevice)
        
        # Pauli matrices in particle-hole space
        self.tau_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.tau_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.tau_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=self.funcDevice)
        
        # Identity matrices
        self.sigma_0 = torch.eye(2, dtype=torch.complex64, device=self.funcDevice)
        self.tau_0 = torch.eye(2, dtype=torch.complex64, device=self.funcDevice)
        
        # Construct full Hamiltonian with vortices
        self.H_full_BdG = self._construct_vortex_hamiltonian()
        
    def _calculate_delta(self, x: float, y: float) -> torch.Tensor:
        """Calculate superconducting gap at position (x,y)."""
        delta = torch.ones(1, dtype=torch.complex64, device=self.funcDevice) * self.Delta
        
        for x_j, y_j in self.vortex_positions:
            r = torch.sqrt((x - x_j)**2 + (y - y_j)**2)
            if r == 0:
                return torch.zeros(1, dtype=torch.complex64, device=self.funcDevice)
            
            # Amplitude factor with tanh profile
            tanh_factor = torch.tanh(r / self.xi_0)
            
            # Phase factor (x+iy)/r for p-wave like vortex
            phase = ((x - x_j) + 1j*(y - y_j))/r
            
            delta *= tanh_factor * phase
            
        return delta
        
    def _calculate_vector_potential(self, x: float, y: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate vector potential at position (x,y)."""
        Ax = torch.zeros(1, dtype=torch.float32, device=self.funcDevice)
        Ay = torch.zeros(1, dtype=torch.float32, device=self.funcDevice)
        
        for x_j, y_j in self.vortex_positions:
            r = torch.sqrt((x - x_j)**2 + (y - y_j)**2)
            if r == 0:
                return torch.zeros(1, dtype=torch.float32, device=self.funcDevice), torch.zeros(1, dtype=torch.float32, device=self.funcDevice)
                
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
        
    def _calculate_peierls_phase(self, x1: float, y1: float, x2: float, y2: float) -> torch.Tensor:
        """Calculate Peierls phase between two points."""
        # Midpoint for vector potential
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        
        # Get vector potential at midpoint
        Ax, Ay = self._calculate_vector_potential(x_mid, y_mid)
        
        # Calculate line integral A·dl
        dx = x2 - x1
        dy = y2 - y1
        
        phase = -(Ax * dx + Ay * dy)
        return torch.exp(1j * phase)
        
    def _construct_kinetic_term(self) -> torch.Tensor:
        """Construct kinetic term v[σx*px + σy*py] - μ."""
        return -self.mu * self.sigma_0
        
    def _construct_vortex_hamiltonian(self) -> torch.Tensor:
        """Construct full BdG Hamiltonian with vortices."""
        h_size = self.Nx * self.Ny
        h_bdg = torch.zeros((4*h_size, 4*h_size), dtype=torch.complex64, device=self.funcDevice)
        
        # Kinetic term
        h_0 = self._construct_kinetic_term()
        
        # Fill Hamiltonian
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                pos = ix * self.Ny + iy
                x, y = ix + 0.5, iy + 0.5  # Center coordinates
                
                # Calculate pairing at this site
                delta = self._calculate_delta(x, y)
                
                # Block indices for this site
                e_idx = slice(4*pos, 4*pos+2)  # Electron block
                h_idx = slice(4*pos+2, 4*pos+4)  # Hole block
                
                # Diagonal blocks (kinetic terms)
                h_bdg[e_idx, e_idx] = h_0
                h_bdg[h_idx, h_idx] = -h_0.conj()
                
                # Off-diagonal blocks (pairing terms)
                h_bdg[e_idx, h_idx] = delta * self.sigma_y
                h_bdg[h_idx, e_idx] = delta.conj() * self.sigma_y
                
                # Hopping terms
                # x-direction
                if ix < self.Nx - 1:
                    phase = self._calculate_peierls_phase(x, y, x+1, y)
                    next_pos = (ix+1) * self.Ny + iy
                    next_e_idx = slice(4*next_pos, 4*next_pos+2)
                    next_h_idx = slice(4*next_pos+2, 4*next_pos+4)
                    
                    # Electron hopping
                    h_nn = self.v * phase * self.sigma_x / 2
                    h_bdg[e_idx, next_e_idx] = h_nn
                    h_bdg[next_e_idx, e_idx] = h_nn.conj().T
                    
                    # Hole hopping
                    h_bdg[h_idx, next_h_idx] = -h_nn.conj()
                    h_bdg[next_h_idx, h_idx] = -(h_nn.conj()).conj().T
                
                # y-direction
                if iy < self.Ny - 1:
                    phase = self._calculate_peierls_phase(x, y, x, y+1)
                    next_pos = ix * self.Ny + (iy+1)
                    next_e_idx = slice(4*next_pos, 4*next_pos+2)
                    next_h_idx = slice(4*next_pos+2, 4*next_pos+4)
                    
                    # Electron hopping
                    h_nn = self.v * phase * self.sigma_y / 2
                    h_bdg[e_idx, next_e_idx] = h_nn
                    h_bdg[next_e_idx, e_idx] = h_nn.conj().T
                    
                    # Hole hopping
                    h_bdg[h_idx, next_h_idx] = -h_nn.conj()
                    h_bdg[next_h_idx, h_idx] = -(h_nn.conj()).conj().T
        
        return h_bdg

class TopologicalSurface2D(Central):
    """Class for constructing 2D topological surface state Hamiltonian."""
    
    def __init__(self, Ny: int, Nx: int, t_y: torch.Tensor, t_x: torch.Tensor,
                 mu: torch.Tensor , v: torch.Tensor ):
        """Initialize 2D topological surface state Hamiltonian.
        
        Args:
            Ny, Nx: System dimensions
            t_y, t_x: Hopping parameters
            mu: Chemical potential
            v: Velocity of surface Dirac fermions
        """
        super().__init__(Ny, Nx, t_y, t_x)
        self.mu = mu
        self.v = v
        
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
        H_onsite = -2 * self.sigma_z - self.mu * self.sigma_0
        H_chain = torch.kron(torch.eye(self.Ny, dtype=torch.complex64, device=self.funcDevice), 
                           H_onsite)
        
        # Nearest neighbor hopping along y (4/3 factor from improved regularization)
        t_nn_y = (4/3) * (self.sigma_z + 1j * self.sigma_y) / 2
        H_nn_y = torch.kron(
            torch.diag(torch.ones(self.Ny - 1, dtype=torch.complex64, device=self.funcDevice), 1),
            t_nn_y
        )
        
        # Next-nearest neighbor hopping along y (-1/6 factor from improved regularization)
        t_nnn_y = (-1/6) * (2 * self.sigma_z + 1j * self.sigma_y) / 2
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
        t_nn_x = (4/3) * (self.sigma_z + 1j * self.sigma_x) / 2
        H_nn_x = torch.kron(
            torch.diag(torch.ones(self.Nx - 1, dtype=torch.complex64, device=self.funcDevice), 1),
            torch.kron(torch.eye(self.Ny, dtype=torch.complex64, device=self.funcDevice), t_nn_x)
        )
        
        # Next-nearest neighbor hopping along x
        t_nnn_x = (-1/6) * (2 * self.sigma_z + 1j * self.sigma_x) / 2
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
                 Delta_0: float, xi_0: float, lambda_L: float, vortex_positions: List[Tuple[float, float]],
                 mu: float = 0.0, v: float = 1.0):
        """Initialize vortex Hamiltonian for 2D topological surface state.
        
        Args:
            Ny, Nx: System dimensions
            t_y, t_x: Hopping parameters
            Delta_0: Base superconducting gap
            xi_0: Superconducting coherence length
            lambda_L: London penetration depth
            vortex_positions: List of (x,y) positions of vortices
            mu: Chemical potential
            v: Velocity of surface Dirac fermions
        """
        super().__init__(Ny, Nx, t_y, t_x, mu, v)
        self.Delta = Delta_0
        self.xi_0 = xi_0
        self.lambda_L = lambda_L
        self.vortex_positions = vortex_positions
        self.Phi_0 = 1.0  # Magnetic flux quantum in natural units
        
        # Pauli matrices in particle-hole space
        self.tau_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.tau_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=self.funcDevice)
        self.tau_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=self.funcDevice)
        self.tau_0 = torch.eye(2, dtype=torch.complex64, device=self.funcDevice)
        
        # Construct full BdG Hamiltonian with vortices
        self.H_full_BdG = self._construct_vortex_hamiltonian()
        
    def _calculate_delta(self, x: float, y: float) -> torch.Tensor:
        """Calculate superconducting gap at position (x,y)."""
        delta = torch.ones(1, dtype=torch.complex64, device=self.funcDevice) * self.Delta
        
        for x_j, y_j in self.vortex_positions:
            r = torch.sqrt((x - x_j)**2 + (y - y_j)**2)
            if r == 0:
                return torch.zeros(1, dtype=torch.complex64, device=self.funcDevice)
            
            # Amplitude factor with tanh profile
            tanh_factor = torch.tanh(r / self.xi_0)
            
            # Phase factor (x+iy)/r for p-wave like vortex
            phase = ((x - x_j) + 1j*(y - y_j))/r
            
            delta *= tanh_factor * phase
            
        return delta
        
    def _calculate_vector_potential(self, x: float, y: float) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
    def _calculate_peierls_phase(self, x1: float, y1: float, x2: float, y2: float) -> torch.Tensor:
        """Calculate Peierls phase between two points."""
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        Ax, Ay = self._calculate_vector_potential(x_mid, y_mid)
        dx = x2 - x1
        dy = y2 - y1
        phase = -(Ax * dx + Ay * dy)
        return torch.exp(1j * phase)
        
    def _construct_vortex_hamiltonian(self) -> torch.Tensor:
        """Construct full BdG Hamiltonian with vortices."""
        h_size = self.Nx * self.Ny * 2  # Factor of 2 for spin
        h_bdg = torch.zeros((2*h_size, 2*h_size), dtype=torch.complex64, device=self.funcDevice)
        
        # Embed normal state Hamiltonian in electron and hole sectors
        h_bdg[:h_size, :h_size] = self.H_full
        h_bdg[h_size:, h_size:] = -self.H_full.conj()
        
        # Add pairing terms
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                pos = (ix * self.Ny + iy) * 2  # *2 for spin
                x, y = ix + 0.5, iy + 0.5
                
                delta = self._calculate_delta(x, y)
                
                # Pairing in the basis |c_↑†,c_↑,c_↓†,c_↓⟩
                h_bdg[pos, pos+3] = delta  # c_↑† to c_↓
                h_bdg[pos+1, pos+2] = -delta.conj()  # c_↑ to c_↓†
                h_bdg[pos+2, pos+1] = -delta  # c_↓† to c_↑
                h_bdg[pos+3, pos] = delta.conj()  # c_↓ to c_↑†
        
        return h_bdg 
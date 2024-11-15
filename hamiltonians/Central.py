import torch

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

        # Construct the Hamiltonian matrices
        self.H_chain_y = self._construct_chain_y()
        #  TODO: make * to kron when expand t_x, now it seems only support scalr t_x
        self.H_along_x = self.t_x * torch.eye(Ny, dtype=torch.complex64)

        # Assemble the full Hamiltonian
        self.H_full = self._assemble_full_hamiltonian()

    def _construct_chain_y(self) -> torch.Tensor:
        """Constructs the Hamiltonian matrix for a single chain along the y-direction."""
        #  TODO: make * to kron when expand t_y
        H_inter_y = self.t_y * torch.diag(torch.ones(self.Ny - 1, dtype=torch.complex64), 1)
        H_chain_y = H_inter_y + H_inter_y.T.conj()
        return H_chain_y

    def _assemble_full_hamiltonian(self) -> torch.Tensor:
        """Assembles the full Hamiltonian matrix without disorder effects."""
        H_full_diag = torch.kron(torch.eye(self.Nx, dtype=torch.complex64), self.H_chain_y)
        H_full_diag1 = torch.kron(torch.diag(torch.ones(self.Nx - 1, dtype=torch.complex64), 1), self.H_along_x)
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
        U = torch.zeros((self.Nx, self.Ny), dtype=torch.float32)

        # Generate random positions for scatterers
        R_k_x = torch.randint(0, self.Nx, (self.N_imp,), dtype=torch.int32)
        R_k_y = torch.randint(0, self.Ny, (self.N_imp,), dtype=torch.int32)

        # Generate random amplitudes for scatterers
        U_k = 2 * self.deltaV * torch.rand(self.N_imp) - self.deltaV

        # Compute the disorder potential for each lattice site
        for n in range(self.Nx):
            for j in range(self.Ny):
                r_nj = torch.tensor([n, j], dtype=torch.float32)
                for k in range(self.N_imp):
                    R_k = torch.tensor([R_k_x[k], R_k_y[k]], dtype=torch.float32)
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
        pairing_matrix = torch.eye(self.Ny * self.Nx, dtype=torch.complex64) * self.Delta
        H_full_BdG = torch.kron(self.H_full, torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64)) + \
                     torch.kron(-self.H_full.conj(), torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64)) + \
                     torch.kron(pairing_matrix, torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64))
        return H_full_BdG

    def __repr__(self):
        return f"CentralBdG(Ny={self.Ny}, Nx={self.Nx}, t_y={self.t_y}, t_x={self.t_x}, Delta={self.Delta})"

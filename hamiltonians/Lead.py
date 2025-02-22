import torch
from typing import Optional, Tuple

class Lead:
    def __init__(self, mu:torch.Tensor, t_lead_central:torch.Tensor, temperature:torch.Tensor, 
                 Ny:int, t_lead:torch.Tensor, lead_pos:Tuple[int, int]):
        """
        Base Lead class inherited for different types of leads.
        
        Parameters:
        -----------
        mu : float
            Chemical potential for the lead.
        t_lead_central : float 
            Coupling strength between lead and central region.
        temperature : float
            Temperature.
        Ny : int
            Number of sites in y-direction.
        t_lead : float
            Hopping parameter within the lead.
        lead_pos : tuple(int, int), optional
            Position of lead in (x,y) coordinates. If None, position needs to be set later.
        """
        self.funcDevice = mu.device
        self.mu = mu
        self.temperature = temperature
        self.lambda_ = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=self.funcDevice)
        self.Ny = Ny
        self.lead_pos = [lead_pos] if isinstance(lead_pos[0], int) else lead_pos      
        # Initialize intraCell and position based on the lead type
        self.initialize_intracell()
        if lead_pos is not None:
            self.generate_positions()

        # Construct matrices
        self.t = t_lead * torch.kron(
            torch.eye(Ny, dtype=torch.complex64, device=self.funcDevice),
            self.intralCell
        )
        
        lead_inter_chains = t_lead * torch.kron(
            torch.diag(torch.ones(Ny - 1, device=self.funcDevice), 1),
            self.intralCell
        )
        self.epsilon0 = lead_inter_chains + lead_inter_chains.T.conj()
        
        self.V1alpha = t_lead_central * torch.kron(
            torch.eye(Ny, dtype=torch.complex64, device=self.funcDevice),
            self.intralCell
        )

    def initialize_intracell(self):
        """
        Initialize the intralCell structure. To be implemented by child classes.
        """
        raise NotImplementedError("Child classes must implement initialize_intracell")

    def get_orbital_multiplier(self) -> int:
        """
        Get the number of states per site (accounting for spin and orbitals).
        To be implemented by child classes.
        """
        raise NotImplementedError("Child classes must implement get_orbital_multiplier")

    def generate_positions(self) -> None:
        """Generate position indices for all lead sites."""
        if self.lead_pos is None:
            raise ValueError("Lead positions must be set before generating indices")
            
        orbital_multiplier = self.get_orbital_multiplier()
        expanded_indices = []
        
        # Process all positions uniformly
        for x, y in self.lead_pos:
            base_idx = (x - 1) * self.Ny + y
            expanded_indices.extend([base_idx * orbital_multiplier + orb 
                                  for orb in range(orbital_multiplier)])
                
        self.position = torch.tensor(expanded_indices, device=self.funcDevice)
    def __repr__(self):
        pos_str = f", pos={self.lead_pos}" if self.lead_pos is not None else ""
        return f"{self.__class__.__name__}(mu={self.mu}, temperature={self.temperature}, lambda_={self.lambda_}{pos_str})"


class SpinlessLead(Lead):
    """Lead class for spinless systems (single orbital per site)."""
    
    def initialize_intracell(self):
        """Initialize single-orbital structure."""
        self.intralCell = torch.tensor(1, dtype=torch.complex64, device=self.funcDevice)
        
    def get_orbital_multiplier(self) -> int:
        """Return number of states per site for spinless case."""
        return 1

    def generate_positions(self):
        """Generate single orbital positions."""
        x, y = self.lead_pos
        base_indices = torch.tensor([(x - 1) * self.Ny + y], device=self.funcDevice)
        self.position = base_indices


class SpinfulLead(Lead):
    """Lead class for systems with spin (two spin states per site)."""
    
    def initialize_intracell(self):
        """Initialize spin-1/2 structure."""
        self.intralCell = torch.eye(2, dtype=torch.complex64, device=self.funcDevice)
        
    def get_orbital_multiplier(self) -> int:
        """Return number of states per site for spinful case."""
        return 2

    def generate_positions(self):
        """Generate positions for both spin states."""
        x, y = self.lead_pos
        base_indices = torch.tensor([(x - 1) * self.Ny + y], device=self.funcDevice)
        expanded_indices = []
        for idx in base_indices:
            expanded_indices.extend([idx * 2, idx * 2 + 1])
        self.position = torch.tensor(expanded_indices, device=self.funcDevice)


class MultiOrbitalLead(Lead):
    """Lead class for systems with multiple orbitals per site."""
    
    def __init__(self, mu:torch.Tensor, t_lead_central:torch.Tensor, temperature:torch.Tensor, 
                 Ny:int, t_lead:torch.Tensor, num_orbitals:int, lead_pos:Optional[Tuple[int, int]]=None):
        """
        Initialize a lead with multiple orbitals per site.
        
        Parameters:
        -----------
        num_orbitals : int
            Number of orbitals per site.
        lead_pos : tuple(int, int), optional
            Position of the lead in (x, y) coordinates.
        """
        self.num_orbitals = num_orbitals
        super().__init__(mu, t_lead_central, temperature, Ny, t_lead, lead_pos)

    def initialize_intracell(self):
        """Initialize multi-orbital structure."""
        self.intralCell = torch.eye(self.num_orbitals, dtype=torch.complex64, device=self.funcDevice)
        
    def get_orbital_multiplier(self) -> int:
        """Return number of states per site for multi-orbital case."""
        return self.num_orbitals

    def generate_positions(self):
        """Generate positions for multiple orbitals."""
        x, y = self.lead_pos  
        base_indices = torch.tensor([(x - 1) * self.Ny + y], device=self.funcDevice)
        expanded_indices = []
        for idx in base_indices:
            expanded_indices.extend([idx * self.num_orbitals + i for i in range(self.num_orbitals)])
        self.position = torch.tensor(expanded_indices, device=self.funcDevice)

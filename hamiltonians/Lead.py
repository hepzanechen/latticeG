import torch

class Lead:
    def __init__(self, mu:torch.Tensor, t_lead_central:torch.Tensor, temperature:torch.Tensor, Ny:int,t_lead:torch.Tensor):
        """
        Initializes a lead with the specified properties.

        Parameters:
        -----------
        mu : float
            Chemical potential for the lead.
        t_lead_central : float
            Coupling strength between lead and central region.
        temperature : float
            Temperature of the lead.
        Ny : int
            Number of sites in the x-direction for the lead.
        t_lead : float, optional
            Hopping parameter within the lead (default is 20).
        """
        self.funcDevice = mu.device
        self.mu = mu
        self.temperature = temperature
        # Note: lambda_ will be initialized and overwritten later in calcation_cf_autograd.py
        self.lambda_ = torch.tensor(0,dtype=torch.float32,requires_grad=True,device=self.funcDevice)  # Assuming lambda is initialized as zero
        self.position = None  # Position will be set later

        # Construct matrices
        self.t = t_lead * torch.eye(Ny, dtype=torch.complex64,device=self.funcDevice)  # Hopping within the lead
        lead_inter_chains = t_lead * torch.diag(torch.ones(Ny - 1,device=self.funcDevice), 1)
        lead_inter_chains = lead_inter_chains + lead_inter_chains.T.conj()
        self.epsilon0 = lead_inter_chains  # Onsite energy matrix within the lead
        self.V1alpha = t_lead_central * torch.eye(Ny, dtype=torch.complex64,device=self.funcDevice)  # Coupling to central region

    def __repr__(self):
        return f"Lead(mu={self.mu}, temperature={self.temperature}, lambda_={self.lambda_})"



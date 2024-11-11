import torch

class Lead:
    def __init__(self, mu, t_lead_central, temperature, Nx, Ny,t_lead=20.0):
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
        Nx : int
            Number of sites in the x-direction for the lead.
        t_lead : float, optional
            Hopping parameter within the lead (default is 20).
        """
        self.mu = mu
        self.temperature = temperature
        self.lambda_ = 0  # Assuming lambda is initialized as zero
        self.position = None  # Position will be set later

        # Construct matrices
        self.t = t_lead * torch.eye(Nx, dtype=torch.complex64)  # Hopping within the lead
        lead_inter_chains = t_lead * torch.diag(torch.ones(Nx - 1), 1)
        lead_inter_chains = lead_inter_chains + lead_inter_chains.T
        self.epsilon0 = lead_inter_chains  # Onsite energy matrix within the lead
        self.V1alpha = t_lead_central * torch.eye(Nx, dtype=torch.complex64)  # Coupling to central region

    def __repr__(self):
        return f"Lead(mu={self.mu}, temperature={self.temperature}, lambda_={self.lambda_})"

# Example usage
mu_values = [20, -20, 1]
temperature = 300
Nx = 10
Ny=1
t_lead_central = 15

# Create lead objects
leads = [Lead(mu, t_lead_central, temperature, Nx,Ny) for mu in mu_values]

# Set positions for leads
leads[0].position = torch.arange(1, Nx + 1)
leads[1].position = leads[0].position + Nx * (Ny - 1)  # Assuming Ny is defined elsewhere
leads[2].position = torch.tensor([3 * Nx + 1])

# Print lead information for debugging
for lead in leads:
    print(lead)

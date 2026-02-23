import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, num_layers, obs_size, act_fn : nn.functional, continuous_dec : bool, device):
        """ Decoder is a simple MLP that does the decoding from observation and hidden toward an estimated observation

        Args:
            obs_size (_type_): size of the observation
            hidden_size (_type_): hidden size (ie number of hidden neurons)
            num_layers (_type_): layer numbers
            latent_size (_type_): Latent space size
        """
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.obs_dim = obs_size
        self.act_fn = act_fn

        layers = []

        layers.append(nn.Linear(latent_size, hidden_size))
        layers.append(self.act_fn)

        # Hidden layers
        for _ in range(num_layers - 2):  # Subtract 2 for input and output layers
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self.act_fn)

        # Output layer
        if continuous_dec:
            layers.append(nn.Linear(hidden_size, obs_size*2))
        else:
            layers.append(nn.Linear(hidden_size, obs_size))

        self.mlp = nn.Sequential(*layers).to(device)

        self.continuous_dec = continuous_dec
        self.min_std = 0.01

    def forward(self, latent):

        
        if self.continuous_dec:
            x = self.mlp(latent)
            mu, raw_std = torch.split(x, self.obs_dim, dim=-1)
            std = nn.functional.softplus(raw_std) + self.min_std
            dist = torch.distributions.independent.Independent(torch.distributions.Normal(loc=mu, scale=std), 1)
            return dist
        else:
            obs_hat = self.mlp(latent)
            return obs_hat
    

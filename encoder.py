import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, obs_size, hidden_size, num_layers, latent_size, act_fn : nn.functional, device):
        """ Encoder is a simple MLP that does the encoding of observation toward a latent space

        Args:
            obs_size (_type_): size of the observation
            hidden_size (_type_): hidden size (ie number of hidden neurons)
            num_layers (_type_): layer numbers
            latent_size (_type_): Latent space size
        """
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.act_fn = act_fn

        layers = []

        layers.append(nn.Linear(obs_size, hidden_size))
        layers.append(self.act_fn)

        # Hidden layers
        for _ in range(num_layers - 2):  # Subtract 2 for input and output layers
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self.act_fn)

        # Output layer
        layers.append(nn.Linear(hidden_size, latent_size))

        self.mlp = nn.Sequential(*layers).to(device)



    def forward(self, input):
        
        latent = self.mlp(input)
        return latent
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from RSSM import RSSM
from encoder import Encoder
from decoder import Decoder
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import os
import json  
from tools import *
from torchviz import make_dot

class WorldModel(nn.Module):

    def __init__(self, obs_size, latent_size, deter_size, stoch_size, num_classes, 
                 hidden_encoder, hidden_decoder, hidden_prior, hidden_posterior, 
                 num_layers_encoder, num_layers_decoder, num_layers_prior, num_layers_posterior, 
                 act_fn, free_bit: int, horizon: int, min_std: float, kl_weight: float, continuous_dist : bool, continuous_dec : bool,
                 latent_weight: float, temperature: float = 1.0, beta_dynamics: float = 0.5, beta_representation: float = 0.1, device='cpu'):
        super(WorldModel, self).__init__()

        self.obs_size = obs_size
        self.latent_size = latent_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.num_classes = num_classes
        self.act_fn = act_fn
        self.train_step = 0
        self.continuous_dist = continuous_dist
        self.continuous_dec = continuous_dec

        self.dynamics_model = RSSM(latent_size, stoch_dim=stoch_size, num_classes=num_classes, 
                                   deter_dim=deter_size, hidden_prior=hidden_prior, 
                                   hidden_post=hidden_posterior, 
                                   num_layer_prior=num_layers_prior, 
                                   num_layer_posterior=num_layers_posterior, 
                                   temperature=temperature, 
                                   continuous_dist=continuous_dist,
                                   min_std=min_std,
                                   device=device)
        
        self.encoder = Encoder(obs_size, hidden_encoder, 
                               num_layers_encoder, latent_size, act_fn, device)
        
        dec_in = deter_size + stoch_size if continuous_dist else deter_size + stoch_size*num_classes

        self.decoder = Decoder(dec_in, hidden_decoder, 
                                   num_layers_decoder, obs_size, act_fn, continuous_dec, device)
        
        self.prev_state = None
        self.device = device
        self.free_bit = free_bit
        self.horizon = horizon
        self.kl_weight = kl_weight
        self.latent_weight = latent_weight
        self.beta_dyn = beta_dynamics
        self.beta_rep = beta_representation

    @classmethod
    def from_config(cls, config, device='cpu'):
        """
        Create a WorldModel instance from a config dictionary.
        
        Args:
            config: Dictionary containing model hyperparameters
            device: Device to place model on (default: 'cpu')
        
        Returns:
            WorldModel instance
        """
        # Handle activation function
        act_fn_name = config.get('act_fn', 'elu').lower()
        act_fn_map = {
            'elu': nn.ELU(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leakyrelu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        act_fn = act_fn_map.get(act_fn_name, nn.SiLU())
        
        return cls(
            obs_size=config['obs_size'],
            latent_size=config['latent_size'],
            deter_size=config['deter_size'],
            stoch_size=config['stoch_size'],
            num_classes=config['num_classes'],
            hidden_encoder=config['hidden_encoder'],
            hidden_decoder=config['hidden_decoder'],
            hidden_prior=config['hidden_prior'],
            hidden_posterior=config['hidden_posterior'],
            num_layers_encoder=config['num_layers_encoder'],
            num_layers_decoder=config['num_layers_decoder'],
            num_layers_prior=config['num_layer_prior'],
            num_layers_posterior=config['num_layer_posterior'],
            act_fn=act_fn,
            free_bit=config['free_bits'],
            horizon=config['horizon'],
            min_std=config['min_std'],
            kl_weight=config['klweight'],
            continuous_dist=config['continuous_dist'],
            continuous_dec=config.get('continuous_dec', True),
            latent_weight=config['latentweight'],
            temperature=config.get('temperature', 1.0),
            beta_dynamics=config['betadyn'],
            beta_representation=config['betarep'],
            device=device
        )


    def forward(self, obs): 
        pass

    def get_features(self, states):
        """Extract features from posterior states and deterministic states"""
        stochs = torch.stack([p['stoch'] for p in states], dim=1)
        deters = torch.stack([p['deter'] for p in states], dim=1)  # (B, S, Z*C)
        return torch.cat([stochs, deters], dim=-1)  # (B, S, Z*C + H)


    
    def _train_step(self, data):
        """
        Training step for the world model
        
        Args:
            data: Observation tensor of shape (batch, seq_len, obs_size)
        
        Returns:
            Dictionary containing loss components
        """
        batch, seq_len, obs_shape = data.shape

        # Encode observations to latent space (B, S, latent_size)
        embed = self.encoder(data)
        
        # Run RSSM observation and imagination
        out = self.dynamics_model.observe(embed, self.horizon, state=None)

        # Get features for reconstruction
        features = self.get_features(out['posteriors'])
        
        dec_out = self.decoder(features)

        # === RECONSTRUCTION LOSS ===
        if self.horizon is not None:
            # Only reconstruct observations that were observed (not imagined)
            data_rec = data[:, :-self.horizon, :]
        else: 
            data_rec = data
        
        if self.continuous_dec:
            # recon_loss = -dec_out.log_prob(data_rec).sum(dim=-1).mean()
            recon_loss = F.mse_loss(dec_out.mean, data_rec)


        else:
            recon_loss = F.mse_loss(dec_out, data_rec)

        # === KL LOSS (DreamerV3 style with free bits) ===
        post_dists = out['posteriors']
        prior_dists = out['priors']
        
        kl_loss = 0
        kl_value = 0
        total_dyn_loss = 0
        total_rep_loss = 0
        
        for post_dist, prior_dist in zip(post_dists, prior_dists):
            loss, value, dyn_loss, rep_loss = self.dynamics_model.compute_kl_loss(
                post_dist,
                prior_dist,
                self.free_bit,  # Free bits threshold (e.g., 1.0)
                dyn_scale=self.beta_dyn,  # DreamerV3 default
                rep_scale=self.beta_rep   # DreamerV3 default
            )
            kl_loss += loss.mean()
            kl_value += value.mean()
            total_dyn_loss += dyn_loss.mean()
            total_rep_loss += rep_loss.mean()

        kl_loss = kl_loss / len(post_dists)
        kl_value = kl_value / len(post_dists)
        total_dyn_loss = total_dyn_loss / len(post_dists)
        total_rep_loss = total_rep_loss / len(post_dists)

        # === IMAGINATION/LATENT OVERSHOOT LOSS ===
        if self.horizon is not None and 'imagined_state' in out:
            imagined_states = out['imagined_state']
            
            # Extract features from imagined states
            deters = torch.stack([s["deter"] for s in imagined_states], dim=1)  # (B, H, deter_dim)
            imagined_stoch = torch.stack([s["stoch"] for s in imagined_states], dim=1)  # (B, H, stoch*classes)
            
            imagined_features = torch.cat([imagined_stoch, deters], dim=-1)
            
            # Decode imagined features to observations
            imagined_obs = self.decoder(imagined_features)
            
            # Compare with true future observations
            true_future_obs = data[:, -self.horizon:, :]
            
            # MSE loss on imagined observations
            overshoot_loss = F.mse_loss(imagined_obs.mean, true_future_obs)
            
        else: 
            overshoot_loss = torch.tensor(0.0, device=self.device)

        # === TOTAL LOSS WITH WARMUP ===
        # kl_warmup = warmup(self.train_step, start=250, warmup=500)
        kl_warmup = 1
        # latent_warmup = warmup(self.train_step, start=0, warmup=0)
        
        total_loss = (recon_loss + 
                     self.kl_weight * kl_warmup * kl_loss + 
                     self.latent_weight  * overshoot_loss)
        
        # total_loss = recon_loss + self.kl_weight * kl_warmup * kl_loss

        return {
            'loss': total_loss, 
            'recon_loss': recon_loss, 
            'kl_loss': kl_loss,
            'kl_value': kl_value,
            'dyn_loss': total_dyn_loss,
            'rep_loss': total_rep_loss,
            'latent_loss': overshoot_loss.item() if torch.is_tensor(overshoot_loss) else overshoot_loss
        }

    @torch.no_grad()
    def imagine(self, initial_obs, horizon):
        """
        Imagine future states from initial observation
        
        Args:
            initial_obs: Initial observation (batch, obs_size)
            horizon: Number of steps to imagine
        
        Returns:
            Imagined observations and states
        """
        self.eval()
        
        if initial_obs.dim() == 2:
            initial_obs = initial_obs.unsqueeze(1)  # (B, 1, obs_size)
        
        # Encode initial observation
        embed = self.encoder(initial_obs)
        
        # Get initial state from observation
        out = self.dynamics_model.observe(embed, horizon=None, state=None)
        state = out['last_state']

        
        # Rollout imagination
        imagined_states = self.dynamics_model._rollout_imagine(state, horizon, use_hard=False)
        
        # Decode imagined states
        deters = torch.stack([s["deter"] for s in imagined_states], dim=1)
        imagined_stoch = torch.stack([s["stoch"] for s in imagined_states], dim=1)
        imagined_features = torch.cat([imagined_stoch, deters], dim=-1)
        imagined_obs = self.decoder(imagined_features)
        
        self.train()
        
        return {
            'observations': imagined_obs,
            'states': imagined_states,
            'features': imagined_features
        }


    def _get_categorical_dist(self, logits):
        """
        Create categorical distribution from logits
        
        Args:
            logits: Tensor of shape (B, stoch_dim, num_classes)
        
        Returns:
            Independent categorical distribution
        """
        # Reshape if needed
        if logits.dim() == 2:
            # (B, stoch*classes) -> (B, stoch, classes)
            logits = logits.view(-1, self.stoch_size, self.num_classes)
        
        # Create independent categorical distribution over stoch_dim
        dist = torch.distributions.Independent(
            torch.distributions.Categorical(logits=logits),
            1  # Make independent over stoch_dim
        )
        
        return dist
    
def load_wm(log_path):
    """Load a trained world model from checkpoint"""
    model_path = os.path.join(log_path, 'worldmodel.pt')
    config_path = os.path.join(log_path, 'config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract config
    obs_size = config["obs_size"]
    latent_size = config["latent_size"]
    stoch_size = config["stoch_size"]
    deter_size = config["deter_size"]
    num_classes = config.get("num_classes", 32)
    hidden_encoder = config["hidden_encoder"]
    hidden_decoder = config["hidden_decoder"]
    hidden_prior = config.get("hidden_prior", 256)
    hidden_posterior = config.get("hidden_posterior", 256)
    num_layers_encoder = config["num_layers_encoder"]
    num_layers_decoder = config["num_layers_decoder"]
    num_layers_prior = config.get("num_layers_prior", 3)
    num_layers_posterior = config.get("num_layers_posterior", 3)
    free_bits = config["free_bits"]
    min_std = config["min_std"]
    kl_weight = config["klweight"]
    latent_weight = config["latentweight"]
    horizon = config["horizon"]
    temperature = config.get("temperature", 1.0)
    
    worldmodel = WorldModel(
        obs_size, latent_size, deter_size, stoch_size, num_classes,
        hidden_encoder, hidden_decoder, hidden_prior, hidden_posterior,
        num_layers_encoder, num_layers_decoder, num_layers_prior, num_layers_posterior,
        nn.SiLU(), free_bit=free_bits, horizon=horizon, min_std=min_std, 
        kl_weight=kl_weight, latent_weight=latent_weight, temperature=temperature,
    )
    
    state_dict = torch.load(model_path, weights_only=False)
    worldmodel.load_state_dict(state_dict)
    
    return worldmodel, config

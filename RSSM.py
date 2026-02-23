import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM) - Observation-only version

    The RSSM maintains two types of state:
    - Deterministic state (h): Updated by RNN based on previous stochastic state
    - Stochastic state (z): Sampled from a distribution conditioned on deterministic state

    This version works with observations only (no actions).
    """

    def __init__(
        self,
        latent_dim,  # Dimension of the latent observation
        stoch_dim=32,  # Dimension of stochastic state
        num_classes=32,
        deter_dim=256,  # Dimension of deterministic state
        hidden_prior=256,  # Hidden dimension for MLPs
        hidden_post=256,
        num_layer_prior=3,
        num_layer_posterior=3,  # Minimum standard deviation
        temperature=1.0,  # Temperature for Gumbel-Softmax
        continuous_dist=False,
        min_std=1e-6,
        device="cpu",
    ):
        super().__init__()

        self.stoch_dim = stoch_dim
        self.num_classes = num_classes
        self.deter_dim = deter_dim
        self.temperature = temperature
        self.continuous_dist = continuous_dist
        self.min_std = min_std

        # RNN cell for deterministic state (only takes previous stochastic state)
        if self.continuous_dist:
            self.rnn = nn.GRUCell(stoch_dim, deter_dim, device=device)
        else:
            self.rnn = nn.GRUCell(stoch_dim * num_classes, deter_dim, device=device)
        # Prior network: p(z_t | h_t)
        layer_prior = []
        layer_prior.append(nn.Linear(deter_dim, hidden_prior))
        layer_prior.append(nn.SiLU())

        for n in range(num_layer_prior):
            layer_prior.append(nn.Linear(hidden_prior, hidden_prior))
            layer_prior.append(nn.SiLU())

        if self.continuous_dist:
            # output is mean and standard deviation
            layer_prior.append(nn.Linear(hidden_prior, 2 * stoch_dim))
        else:
            layer_prior.append(nn.Linear(hidden_prior, num_classes * stoch_dim))

        self.prior_net = nn.Sequential(*layer_prior).to(device)

        # Posterior network: q(z_t | h_t, o_t) where o_t is encoded observation
        layer_posterior = []
        layer_posterior.append(nn.Linear(deter_dim + latent_dim, hidden_post))
        layer_posterior.append(nn.SiLU())

        for n in range(num_layer_posterior):
            layer_posterior.append(nn.Linear(hidden_post, hidden_post))
            layer_posterior.append(nn.SiLU())

        if self.continuous_dist:
            # output is mean and standard deviation
            layer_posterior.append(nn.Linear(hidden_post, 2 * stoch_dim))
        else:
            layer_posterior.append(nn.Linear(hidden_post, num_classes * stoch_dim))

        self.posterior_net = nn.Sequential(*layer_posterior).to(device)

    def initial_state(self, batch_size, device):
        """Initialize the RSSM state"""

        if self.continuous_dist:
            state = {
                # mean of continual latent
                "mean": torch.zeros(batch_size, self.stoch_dim, device=device),
                "std": self.min_std
                * torch.ones(batch_size, self.stoch_dim, device=device),
                # deterministic state
                "deter": torch.zeros(batch_size, self.deter_dim, device=device),
                "stoch": torch.zeros(batch_size, self.stoch_dim, device=device),
            }
            return state
        else:
            state = {
                # logits for categorical latent
                "logits": torch.zeros(
                    batch_size, self.stoch_dim, self.num_classes, device=device
                ),
                # uniform categorical distribution
                "probs": torch.full(
                    (batch_size, self.stoch_dim, self.num_classes),
                    1.0 / self.num_classes,
                    device=device,
                ),
                # flattened relaxed categorical sample
                "stoch": torch.full(
                    (batch_size, self.stoch_dim * self.num_classes),
                    1.0 / self.num_classes,
                    device=device,
                ),
                # deterministic state
                "deter": torch.zeros(batch_size, self.deter_dim, device=device),
            }
            return state

    def get_distribution(self, stats, use_hard=True):
        """Create a categorical distribution from logits"""

        if self.continuous_dist:
            # already in good forms
            dist_dict = self._continuous_dist(stats)
        else:
            # from the stoch_dim * num_classes grid
            stats = stats.view(-1, self.stoch_dim, self.num_classes)

            dist_dict = self._categorical_dist(
                stats, temperature=self.temperature, hard=use_hard
            )

        return dist_dict

    def _compute_stoch_state(self, stats, use_hard=True):
        dist_dict = self.get_distribution(stats, use_hard=use_hard)

        return dist_dict

    def prior(self, state, use_hard=True):
        """
        Compute prior p(z_t | h_t) where h_t = f(h_{t-1}, z_{t-1})

        Args:
            state: Dictionary with 'stoch' and 'deter' keys
            use_hard: Whether to use hard categorical samples

        Returns:
            Dictionary with 'stoch', 'deter', and distribution keys
        """
        prior_logits = self.prior_net(state["deter"])
        stoch_state = self._compute_stoch_state(prior_logits, use_hard=use_hard)

        return {
            "stoch": stoch_state["stoch"],
            "deter": state["deter"],
            "dist": stoch_state["dist"],
            **stoch_state,
        }

    def posterior(self, state, obs_embedding, use_hard=True):
        """
        Compute posterior q(z_t | h_t, o_t)

        Args:
            state: Dictionary with 'stoch' and 'deter' keys
            obs_embedding: Encoded observation of shape (batch, hidden_dim)
            use_hard: Whether to use hard categorical samples

        Returns:
            Dictionary with 'stoch', 'deter', and distribution keys
        """
        x = torch.cat([state["deter"], obs_embedding], dim=-1)
        post_logits = self.posterior_net(x)
        stoch_state = self._compute_stoch_state(post_logits, use_hard=use_hard)
        return {
            "stoch": stoch_state["stoch"],
            "deter": state["deter"],
            "dist": stoch_state["dist"],
            **stoch_state,
        }

    def observe(self, obs_embeddings, horizon=None, state=None):
        """
        Update the RSSM state based on observations

        Args:
            obs_embeddings: Encoded observations of shape (batch, seq_len, hidden_dim)
            horizon: Number of steps to imagine ahead
            state: Initial state (if None, creates initial state)
        """
        batch_size, sequence_length, _ = obs_embeddings.shape
        device = obs_embeddings.device

        if state is None:
            state = self.initial_state(batch_size, device)

        priors = []
        posteriors = []

        # Determine how many steps to observe (leave room for imagination)
        if horizon is not None:
            observe_length = sequence_length - horizon
        else:
            observe_length = sequence_length

        # Observation phase
        for t in range(observe_length):
            obs_embed_t = obs_embeddings[:, t]

            # Update deterministic state
            state = self.update(state)

            # Compute posterior distribution q(z_t | h_t, o_t)
            post_state = self.posterior(state, obs_embed_t, use_hard=True)

            # Compute prior distribution p(z_t | h_t)
            prior_state = self.prior(state, use_hard=True)

            # Use posterior for next state
            state = {"deter": post_state["deter"], "stoch": post_state["stoch"]}

            if self.continuous_dist:
                state["mean"] = post_state["mean"]
                state["std"] = post_state["std"]
            else:
                state["logits"] = post_state["logits"]
                state["probs"] = post_state["probs"]

            priors.append(prior_state)
            posteriors.append(post_state)

        # Imagination phase
        results = {"priors": priors, "posteriors": posteriors, "last_state": state}

        if horizon is not None:
            # Detach state before imagination to prevent gradient flow
            if self.continuous_dist:
                detached_state = {
                    "deter": state["deter"],
                    "stoch": state["stoch"],
                    "mean": state["mean"],
                    "std": state["std"],
                }
            else:
                detached_state = {
                    "deter": state["deter"].detach(),
                    "stoch": state["stoch"].detach(),
                    "logits": state["logits"].detach(),
                    "probs": state["probs"].detach(),
                }

            imagined_state = self._rollout_imagine(
                detached_state, horizon, use_hard=False
            )
            results["imagined_state"] = imagined_state

        return results

    def _rollout_imagine(self, current_state, horizon, use_hard=False):
        """
        Rollout imagination for categorical RSSM.
        """

        rollouts = []

        if self.continuous_dist:
            # Start from the detached state
            state_rollout = {
                "deter": current_state["deter"],
                "stoch": current_state["stoch"],
                "mean": current_state["mean"],
                "std": current_state["std"],
            }
        else:
            state_rollout = {
                "deter": current_state["deter"],
                "stoch": current_state["stoch"],
                "logits": current_state["logits"],
                "probs": current_state["probs"],
            }

        for _ in range(horizon):
            # Deterministic transition
            state_rollout = self.update(state_rollout)

            # Prior prediction (no observation)
            state_rollout = self.prior(state_rollout, use_hard=use_hard)

            if self.continuous_dist:
                imagined_state = {
                    "deter": state_rollout["deter"],
                    "stoch": state_rollout["stoch"],
                    "mean": state_rollout["mean"],
                    "std": state_rollout["std"],
                }
            else:
                imagined_state = {
                    "deter": state_rollout["deter"],
                    "stoch": state_rollout["stoch"],
                    "logits": state_rollout["logits"],
                    "probs": state_rollout["probs"],
                }

            rollouts.append(imagined_state)

        return rollouts

    def update(self, state):
        """
        Update the deterministic state using RNN

        Args:
            state: Current state dictionary
        """
        state["deter"] = self.rnn(state["stoch"], state["deter"])
        return state

    def _categorical_dist(self, logits, temperature=1.0, hard=True):
        """
        Sample from categorical distribution using Gumbel-Softmax

        Args:
            logits: Logits of shape (batch, stoch_dim, num_classes)
            temperature: Temperature for Gumbel-Softmax
            hard: Whether to use hard (straight-through) or soft samples
        """
        # print(logits.shape)
        # result in 10/02 obtain with this
        probs = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        dist = torch.distributions.Independent(
            torch.distributions.Categorical(logits=logits),
            1,  # Make independent over stoch_dim
        )

        stoch = probs.reshape(probs.shape[0], -1)
        return {"logits": logits, "probs": probs, "stoch": stoch, "dist": dist}

        # dist = torch.distributions.Categorical(probs)
        # sample = dist.sample()
        return probs

        # B = logits.shape[0]
        # logits = logits.view(B, self.stoch_dim, self.num_classes)

        # # Softmax probabilities
        # probs = F.softmax(logits / temperature, dim=-1)

        # # 1% uniform mixing (unimix)
        # uniform = torch.ones_like(probs) / self.num_classes
        # probs = 0.99 * probs + 0.01 * uniform

        # # Sample categorical
        # dist = torch.distributions.Categorical(probs)
        # sample = dist.sample()

        # # Convert to one-hot
        # one_hot = F.one_hot(sample, self.num_classes).float()

        # # straight-through
        # stoch = one_hot + probs - probs.detach()

        # # Flatten stochastic state
        # stoch = stoch.reshape(B, -1)

        return {"logits": logits, "probs": probs, "stoch": stoch, "dist": dist}

    def _continuous_dist(self, stats):

        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + self.min_std
        dist = torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)
        stoch = dist.rsample()
        return {"mean": mean, "std": std, "stoch": stoch, "dist": dist}

    def compute_kl_loss(self, posteriors, priors, free, dyn_scale=0.5, rep_scale=0.1):
        """
        Calculate KL divergence loss for categorical distributions (DreamerV3 style)

        Args:
            post: Posterior state dict with 'logits' key
            prior: Prior state dict with 'logits' key
            free: Free bits threshold (float)
            dyn_scale: Scale for dynamics loss (default 0.5)
            rep_scale: Scale for representation loss (default 0.1)

        Returns:
            loss: Scaled total KL loss
            value: Unscaled KL divergence value
            dyn_loss: Dynamics loss component
            rep_loss: Representation loss component
        """
        if self.continuous_dist:
            # Get stats
            post_stats = torch.cat((posteriors["mean"], posteriors["std"]), dim=1)
            prior_stats = torch.cat((priors["mean"], priors["std"]), dim=1)

            post_dist = self._continuous_dist(post_stats)["dist"]
            prior_dist = self._continuous_dist(prior_stats)["dist"]

            sg = lambda x: self._continuous_dist(x.detach())["dist"]

            # Representation loss: KL(post || sg(prior))
            rep_loss = torch.distributions.kl.kl_divergence(post_dist, sg(prior_stats))

            # Dynamics loss: KL(sg(post) || prior)
            dyn_loss = torch.distributions.kl.kl_divergence(sg(post_stats), prior_dist)

        else:
            # Get logits
            post_logits = posteriors["logits"]  # (B, stoch_dim, num_classes)
            prior_logits = priors["logits"]  # (B, stoch_dim, num_classes)

            post_dist = self._categorical_dist(post_logits)["dist"]
            prior_dist = self._categorical_dist(prior_logits)["dist"]

            sg = lambda x: self._categorical_dist(x.detach())["dist"]

            # Representation loss: KL(post || sg(prior))
            rep_loss = torch.distributions.kl.kl_divergence(post_dist, sg(prior_logits))

            # Dynamics loss: KL(sg(post) || prior)
            dyn_loss = torch.distributions.kl.kl_divergence(sg(post_logits), prior_dist)

        rep_loss = torch.clamp(rep_loss, min=free)
        dyn_loss = torch.clamp(dyn_loss, min=free)

        value = rep_loss

        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss

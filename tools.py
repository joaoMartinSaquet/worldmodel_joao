from torch.distributions import Normal
import torch
import yaml
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
# from WorldModel.worldModel import WorldModel
from pathlib import Path
import pandas as pd
import json
import os

def detach_dist(dist):
    # Helper to stop gradients on distribution parameters
    return Normal(dist.loc.detach(), dist.scale.detach())

def warmup(steps, start = 0, warmup = 2000):
    if steps < start:
        return 0
    else:
        return min(1.0, abs(steps - start) / warmup)

def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))

def load_best_model(wm, checkpoint_path):
    """
    Load a saved checkpoint into the world model.

    Args:
        wm: WorldModel instance
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=wm.device)
    wm.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"Validation error: {checkpoint['validation_error']:.4f}")
    logger.info(f"Train steps: {checkpoint['train_step']}")

    return checkpoint

def validate_multiple_trajectories(wm, dataset, trajectory_indices=None,
                                   context_ratio=0.75, horizon=10, normalize=True, transform="simexp"):
    """
    Validate the world model on multiple trajectories using batch processing.
    """
    device = wm.device

    # Determine which trajectories to validate
    if trajectory_indices is None:
        trajectory_indices = list(range(len(dataset.trajectories)))

    # Validate trajectory indices
    max_traj_idx = len(dataset.trajectories) - 1
    trajectory_indices = [idx for idx in trajectory_indices if 0 <= idx <= max_traj_idx]

    if not trajectory_indices:
        logger.error("No valid trajectory indices to validate!")
        return None, None

    results = {}
    horizon_errors = []
    full_errors = []

    if horizon is None:
        horizon = 15

    wm.eval()

    for traj_idx in trajectory_indices:
        try:
            val_data = dataset.get_full_trajectory(traj_idx=traj_idx, normalize=normalize)

            if val_data is None or len(val_data) == 0:
                continue

            ld = len(val_data)
            context_len = int(context_ratio * ld)

            if context_len < 1 or ld - context_len < 1:
                continue

            actual_horizon = ld - context_len

            # Prepare data
            val_data_np = np.array(val_data, dtype=np.float64)
            val_data_tensor = torch.tensor(val_data_np, dtype=torch.float32).to(device)
            val_data_sym = symlog(val_data_tensor)

            with torch.no_grad():

                context_obs = val_data_sym[:context_len].unsqueeze(0)  # (1, context_len, obs_size)
                context_embed = wm.encoder(context_obs)  # (1, context_len, latent_size)

                # Run RSSM on context
                out = wm.dynamics_model.observe(context_embed, horizon=None, state=None)
                # Get reconstructions
                features = wm.get_features(out['posteriors'])
                decoder_out = wm.decoder(features)  # (1, context_len, obs_size) or Normal dist

                # Handle decoder output based on type
                if wm.continuous_dec:
                    # Decoder outputs a Normal distribution
                    recons = decoder_out.mean  # (1, context_len, obs_size)
                    recons_std = decoder_out.stddev  # Optional: store std for uncertainty
                else:
                    # Decoder outputs observations directly
                    recons = decoder_out  # (1, context_len, obs_size)

                # === IMAGINATION PHASE: Predict future ===
                last_state = out['last_state']

                # Imagine future states
                imagined_states = wm.dynamics_model._rollout_imagine(
                    last_state,
                    horizon=actual_horizon,
                    use_hard=True
                )


                # Decode imagined states
                deters = torch.stack([s["deter"] for s in imagined_states], dim=1)
                imagined_stoch = torch.stack([s["stoch"] for s in imagined_states], dim=1)
                imagined_features = torch.cat([imagined_stoch, deters], dim=-1)
                decoder_preds = wm.decoder(imagined_features)  # (1, horizon, obs_size) or Normal dist

                # Handle decoder output based on type
                if wm.continuous_dec:
                    # Decoder outputs a Normal distribution
                    preds = decoder_preds.mean  # (1, horizon, obs_size)
                    preds_std = decoder_preds.stddev  # Optional: store std for uncertainty
                else:
                    # Decoder outputs observations directly
                    preds = decoder_preds  # (1, horizon, obs_size)

            # Convert to numpy
            recons_np = recons[0].cpu().numpy()  # (context_len, obs_size)
            preds_np = preds[0].cpu().numpy()  # (horizon, obs_size)

            # Calculate errors
            horizon_err = np.linalg.norm(
                val_data_sym[context_len:context_len + actual_horizon, :2].cpu().numpy() -
                preds_np[:actual_horizon, :2]
            )

                        

            if transform == None:
                gt = val_data_sym.cpu().numpy()
            else:
                recons_np = symexp(torch.tensor(recons_np)).cpu().numpy()
                preds_np = symexp(torch.tensor(preds_np)).cpu().numpy()
                gt = symexp(val_data_sym).cpu().numpy()

            full_hat = np.vstack((recons_np, preds_np))
            full_err = np.linalg.norm(gt - full_hat)
            # Store results
            result_dict = {
                'ground_truth': gt,
                'reconstructions': recons_np,
                'predictions': preds_np,
                'context_length': context_len,
                'horizon_error': horizon_err,
                'full_error': full_err,
                'actual_horizon': actual_horizon,
            }

            # Optionally store std if continuous decoder
            if wm.continuous_dec:
                result_dict['reconstructions_std'] = recons_std[0].cpu().numpy()
                result_dict['predictions_std'] = preds_std[0].cpu().numpy()

            results[traj_idx] = result_dict

            horizon_errors.append(horizon_err)
            full_errors.append(full_err)

        except Exception as e:
            logger.error(f"Error validating trajectory {traj_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        logger.error("No trajectories were successfully validated!")
        return None, None

    # Summary statistics
    summary = {
        'num_validated': len(results),
        'mean_horizon_error': np.mean(horizon_errors),
        'std_horizon_error': np.std(horizon_errors),
        'mean_full_error': np.mean(full_errors),
        'std_full_error': np.std(full_errors),
        'min_horizon_error': np.min(horizon_errors),
        'max_horizon_error': np.max(horizon_errors),
    }

    wm.train()  # Switch back to training mode

    return results, summary

def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def dump_config(config_path, config_data):
    """Dump the config to path

    Args:
        config_path (_type_): _description_
        config_data (_type_): _description_
    """

    with open(config_path, 'w+') as f:
        yaml.dump(config_data, f)

def log_config(config):
    logger.info("--- World Model Configuration ---")
    # Using a table-like format for readability
    for key, value in config.items():
        logger.info(f"{key:>20} : {value}")
    logger.info("----------------------------------")

def save_experiment_meta(config, log_dir):
    """Saves the hyperparameter config to a JSON file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {log_dir}/config.json")

def save_training_metrics(history_dict, log_dir):
    """Saves loss history to a CSV for easy plotting later."""
    df = pd.DataFrame(history_dict)
    df.to_csv(os.path.join(log_dir, "metrics.csv"), index=False)
    logger.info(f"Metrics saved to {log_dir}/metrics.csv")

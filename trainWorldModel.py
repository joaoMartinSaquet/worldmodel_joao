from WorldModel.worldModel import WorldModel
import torch
import pandas as pd
import numpy as np
from loguru import logger
import worldModelDataset as wmd2
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import os
from tools import *
from visu import *
from Datasets.split_train_test import user_split_train_test_sapimouse

def train_wmodel(n_epochs, wm : WorldModel, train_data, val_data, learning_rate, batch_size, normalize, cx_ratio,
                 save_dir = "checkpoints", model_name = "worldmodel", max_norm = 1000., nval=6, ):
    # for now the state is carried over steps
    best_model_path =  Path(os.path.join(save_dir, f'{model_name}_best.pt'))
    optimizer = torch.optim.Adam(wm.parameters(), lr=learning_rate)
    device = wm.device


    elbo_history = []
    recon_history = []
    kl_history = []
    latent_history = []

    logger.info(f"Start of training of world model")
    logger.info(f"Learning rate {learning_rate}, batch size : {batch_size}")
    logger.info(f"Devices {device}")

    wm.train()
    wm.train_step = 0
    validation_error = []

    valdation_idx = np.random.choice(len(val_data.trajectories), size=nval, replace=False)
    best_val_error = np.inf

    for e in range(n_epochs):
        epoch_losses = []
        epoch_recon = []
        epoch_kl = []
        epoch_latent = []

        indices = np.random.permutation(len(train_data))
        n_batches = (len(train_data) + batch_size - 1)// batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx*batch_size
            end_idx = min(start_idx + batch_size, len(train_data))
            batch_indices = indices[start_idx:end_idx]

            # prepare batches
            data = np.array([train_data[i] for i in batch_indices], dtype=np.float64)
            batch_data = torch.tensor(data, dtype=torch.float).to(device)
            batch_data_sym = symlog(batch_data)

            # training step
            optimizer.zero_grad()
            loss_dict = wm._train_step(batch_data_sym)
            loss_dict["loss"].backward()

            # clip gradient to stabilize learning
            torch.nn.utils.clip_grad_norm_(wm.parameters(), max_norm=max_norm)

            optimizer.step()

            wm.train_step += 1

            # record losses
            epoch_losses.append(loss_dict["loss"].item())
            epoch_recon.append(loss_dict["recon_loss"].item())
            epoch_kl.append(loss_dict["kl_loss"].item())
            epoch_latent.append(loss_dict["latent_loss"])



        _, summary = validate_multiple_trajectories(wm, val_data, valdation_idx, context_ratio=cx_ratio, horizon=wm.horizon, normalize=normalize)

        validation_error.append(summary["mean_full_error"])
        current_val_error = summary["mean_full_error"]
        if current_val_error < best_val_error:
                best_val_error = current_val_error
                best_epoch = e + 1

                # Helper function to convert numpy to native Python types
                def to_native(obj):
                    """Convert numpy types to native Python types."""
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {key: to_native(val) for key, val in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(to_native(item) for item in obj)
                    else:
                        return obj

                # Save the best model - convert numpy values to Python native types
                checkpoint = {
                    'epoch': int(e + 1),
                    'model_state_dict': wm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_step': int(wm.train_step),
                    'validation_error': float(current_val_error),
                    'train_loss': float(np.mean(epoch_losses)),
                    'config': {
                        'learning_rate': float(learning_rate),
                        'batch_size': int(batch_size),
                        'normalize': bool(normalize),
                        'context_ratio': float(cx_ratio),
                        'max_norm': float(max_norm),
                    }
                }

                # Ensure parent directory exists and save as string path
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, str(best_model_path))

                logger.info(f" New best model saved to {best_model_path} with error {current_val_error}")

        avg_loss = np.mean(epoch_losses)
        avg_recon = np.mean(epoch_recon)
        avg_kl = np.mean(epoch_kl)
        avg_latent = np.mean(epoch_latent)

        elbo_history.extend(epoch_losses)
        recon_history.extend(epoch_recon)
        kl_history.extend(epoch_kl)
        latent_history.extend(epoch_latent)


        if (e + 1) % 1 == 0 or e == 0:
                logger.info(f"Epoch {e+1:3d}/{n_epochs} | Avg "
                      f"Loss: {avg_loss:.2f} | "
                      f"Recon: {avg_recon:.6f} | "
                      f"KL: {avg_kl:.6f} | "
                      f"latent loss: {avg_latent:.6f} | "
                      f"Validation error: {validation_error[-1]:.2f} | "
                      f"Steps: {wm.train_step}")

    return elbo_history, recon_history, kl_history, validation_error, best_model_path


if __name__ == "__main__":

    load = False
    dump = False
    if not load:
        # Hyperparameters
        config_path = "WorldModel/config_sapimouse_continuous.yaml"
        config = load_config(config_path)

        log_config(config)

        if dump:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            script_path = os.path.dirname(os.path.realpath(__file__))
            log_dir = os.path.join(script_path, f"experiments/worldmodel_{config['  task']}_{ts}")

            save_experiment_meta(config, log_dir)

        task = config["task"]
        lr = config["lr"]
        nepochs = config["nepochs"]
        batch_size = config["batch_size"]
        free_bits = config["free_bits"]
        min_std = config["min_std"]
        betadyn = config["betadyn"]
        betarep = config["betarep"]
        obs_size = config["obs_size"]
        latent_size = config["latent_size"]
        stoch_size = config["stoch_size"]
        num_classes = config["num_classes"]
        deter_size = config["deter_size"]
        hidden_encoder = config["hidden_encoder"]
        hidden_decoder = config["hidden_decoder"]
        hidden_prior = config["hidden_prior"]
        hidden_posterior = config["hidden_posterior"]
        num_layers_encoder = config["num_layers_encoder"]
        num_layers_decoder = config["num_layers_decoder"]
        num_layers_prior = config["num_layer_prior"]
        num_layers_posterior = config["num_layer_posterior"]
        min_std = config["min_std"]
        kl_weight = config["klweight"]
        latent_weight = config["latentweight"]
        horizon = config["horizon"]
        normalize = config["normalize"]
        cx_ratio = config["cx_ratio"]
        continuous = config["continuous_dist"]

        checkpoint_path = "checkpoints/experiment_1/wm_best.pt"

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if task == "sapimouse":
            # train_ds, val_ds = get_dataset("pendulum", 500, seq_len)

            data = pd.read_json("/home/jmartinsaquet/Documents/code/endpoint_prediction/Datasets/sapiuser62treated.json")
            N = 1000  # your threshold
            # 1. Keep rows where the trajectory is LONG ENOUGH (max timestamp > N)
            data = data[data["timestamp"].apply(lambda ts: max(ts) < N)]
            
            train_ds, val_ds = user_split_train_test_sapimouse(data, users=[62], train=0.8, test=0.2)
            train_ds = wmd2.df_to_trajectory_dataset(train_ds, window_size=64, stride=12, normalize=False, num_endpoint_padding=10, include_timestamp=False)
            val_ds = wmd2.df_to_trajectory_dataset(val_ds, window_size=64, stride=12, normalize=False, num_endpoint_padding=10, include_timestamp=False)

            # train_ds = wmd2.generate_sapimouse_dataset("Datasets/sapimouse/segmented_ds_click.json", window_size=64, stride=10, normalize=normalize, include_velocity=False, filter_users=None, min_trajectory_length=32, max_trajectory_length = 300, num_padding_zeros=20)
        else:
            train_ds, val_ds = wmd2.create_train_val_datasets(task, num_train_trajectories=500, num_val_trajectories=5, trajectory_length=100, window_size=64, stride=None, normalize=normalize)

        # worldmodel = WorldModel(obs_size, latent_size, deter_size, stoch_size, num_classes, hidden_encoder, hidden_decoder,hidden_prior, hidden_posterior, num_layers_encoder, num_layers_decoder, num_layers_prior, num_layers_posterior, nn.SiLU(),
        #                         free_bit=free_bits, horizon=horizon, min_std=min_std, kl_weight=kl_weight, latent_weight=latent_weight, continuous_dist=continuous, continuous_dec=True,
        #                         device=device)
        worldmodel = WorldModel.from_config(config, device=device)
        elbo_history, recon_history, kl_history, val_error, best_model_path  = train_wmodel(nepochs, worldmodel, train_ds, val_ds, lr, batch_size, normalize, cx_ratio, save_dir = "checkpoints_new/", model_name = "worldmodel", max_norm = 1000., nval=6)


        # plot losses
        plot_fig = plot_losses(elbo_history, recon_history, kl_history, val_error)

        # checkpoint = load_best_model(
        # wm=worldmodel,
        # checkpoint_path=best_model_path)
        # dump_config(best_model_path + "/" + config_path.split('/')[-1], config)
        # validation
        # Validate on multiple trajectories
        trajectory_indices = [0, 1, 2, 3, 4, 5]  # or None for all trajectories
        results, summary = validate_multiple_trajectories(
            worldmodel,
            train_ds,
            trajectory_indices=trajectory_indices,
            context_ratio=cx_ratio,
            horizon=16,
            normalize=normalize
        )
        logger.info(f"Validation Summary: {summary}")
        oned_fig, twod_fig, thirdd_fig = plot_multiple_trajectories(results, summary, save_dir=None)
        if dump:
            model_path = os.path.join(log_dir, "worldmodel.pt")
            torch.save(worldmodel.state_dict(), model_path)
            plot_fig.savefig(os.path.join(log_dir, "losses.png"))
            twod_fig.savefig(os.path.join(log_dir, "2dreconstruct.png"))
            oned_fig.savefig(os.path.join(log_dir, "1dreconstruct.png"))

        # logger.info(f"total erro on validation :  {err}: ")
        plt.show()

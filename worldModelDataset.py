"""
Windowed Trajectory Dataset for World Models

This module generates full trajectories and slices them into overlapping or non-overlapping
windows for training. This is more efficient and realistic than generating many short trajectories.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dataset_generation import *
from loguru import logger
from torch.utils.data import DataLoader, Dataset

import Datasets.split_train_test as stt


class TrajectoryWindowDataset(Dataset):
    """
    PyTorch Dataset that slices full trajectories into windows for training.

    This approach is better than generating many short trajectories because:
    1. More efficient - generates fewer, longer trajectories
    2. More realistic - captures long-term dynamics
    3. Better data augmentation - overlapping windows provide more training samples
    4. Temporal consistency - windows from same trajectory share context
    """

    def __init__(
        self, trajectories, window_size=50, stride=None, normalize=True, num_padding_zeros =0, device="cpu"
    ):
        """
        Args:
            trajectories: List of full trajectory arrays, each shape (full_len, obs_dim)
            window_size: Size of each training window
            stride: Step size between windows (default: window_size for non-overlapping)
            normalize: Whether to normalize data to zero mean, unit variance
            device: Device for tensors
        """
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.device = device
        self.num_padding_zeros = num_padding_zeros 
        self.length = len(trajectories)
        # Store trajectories and compute windows
        if type(trajectories) is pd.Series:
            self.trajectories = trajectories[trajectories.apply(len) >= window_size * 6]
        else:
            self.trajectories = trajectories

        self.windows = []
        self.trajectory_indices = []  # Track which trajectory each window came from
            
        if num_padding_zeros > 0:
            padded = []
            for traj in trajectories:
                # ✅ Repeat first point at start, last point at end
                start_padding = np.tile(traj[0], (num_padding_zeros, 1))   # shape (N, obs_dim)
                end_padding   = np.tile(traj[-1], (num_padding_zeros, 1))  # shape (N, obs_dim)
                padded.append(np.vstack([start_padding, traj, end_padding]))
            trajectories = padded
        self.trajectories = trajectories
        # Extract windows from all trajectories
        for traj_idx, traj in enumerate(trajectories):
            lt = len(traj)
            num_windows = (lt - window_size) // self.stride + 1
            for i in range(num_windows):
                start_idx = i * self.stride
                end_idx = start_idx + window_size

                if end_idx <= lt:
                    window = traj[start_idx:end_idx]
                    self.windows.append(window)
                    self.trajectory_indices.append(traj_idx)

        self.windows = np.array(self.windows, dtype=np.float32)

        # Compute normalization statistics
        if normalize:
            # # Compute mean and std across all windows
            # all_data = self.windows.reshape(-1, self.windows.shape[-1])
            # self.mean = np.mean(all_data, axis=0)
            # self.std = np.std(all_data, axis=0) + 1e-8
            all_data = self.windows.reshape(-1, self.windows.shape[-1])
            self.max = np.max(all_data, axis=0)
            self.min = np.min(all_data, axis=0)
            self.mean = np.zeros(self.windows.shape[-1])
            self.std = np.ones(self.windows.shape[-1])
            # Normalize
            self.windows = (self.windows - self.min) / (self.max - self.min)
        else:
            self.max = 0
            self.min = 0
            self.mean = 0
            self.std = 1

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """Return a single window as a tensor"""
        window = torch.tensor(self.windows[idx], dtype=torch.float32)
        return window

    def normalize(self, data, compute_stats=True):

        self.max = 1920
        self.min = 0
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8

        # normalized = (data - self.min)/(self.max - self.min)

        normalized = (data - self.mean) / self.std
        return normalized
        # Normalize

    def denormalize(self, data):
        """Denormalize data back to original scale"""
        if isinstance(data, torch.Tensor):
            mean = torch.tensor(self.mean, device=data.device, dtype=data.dtype)
            std = torch.tensor(self.std, device=data.device, dtype=data.dtype)
            return data * std + mean
        else:
            return data * self.std + self.mean

    def get_full_trajectory(self, traj_idx, normalize=True):
        """Get a full trajectory (denormalized)"""
        if type(self.trajectories) is pd.Series:
            traj = self.trajectories.iloc[traj_idx]
        else:
            traj = self.trajectories[traj_idx]

        if normalize:
            traj = self.normalize(traj, compute_stats=False)
            return traj
        else:
            return traj

    def get_statistics(self):
        """Get dataset statistics"""
        return {
            "num_trajectories": len(self.trajectories),
            "num_windows": len(self.windows),
            "window_size": self.window_size,
            "stride": self.stride,
            "obs_dim": self.windows.shape[-1],
            "mean": self.mean,
            "std": self.std,
            "paddin_zeros": self.num_padding_zeros,
            "windows_per_trajectory": len(self.windows) / len(self.trajectories),
        }


def generate_full_trajectories(
    dataset_type="double_pendulum",
    num_trajectories=10,
    trajectory_length=500,
    dt=0.05,
    **kwargs,
):
    """
    Generate full, long trajectories for windowed training.

    Args:
        dataset_type: Type of dataset to generate
            - 'double_pendulum': Chaotic double pendulum
            - 'simple_pendulum': Simple pendulum
            - 'coupled_oscillators': Coupled oscillators
            - 'circular': Circular motion
            - 'linear': Linear motion
            - 'sine': Sine wave
            - 'figure8': Figure-eight pattern
        num_trajectories: Number of full trajectories to generate
        trajectory_length: Length of each full trajectory
        dt: Time step
        **kwargs: Additional arguments passed to generator

    Returns:
        List of trajectory arrays, each shape (trajectory_length, obs_dim)
    """

    if dataset_type == "double_pendulum":
        trajectories = generate_double_pendulum_dataset(
            num_sequences=num_trajectories,
            seq_len=trajectory_length,
            dt=dt,
            output_type=kwargs.get("output_type", "tip_position"),
            chaos_level=kwargs.get("chaos_level", "High"),
            noise=kwargs.get("noise", 0.00),
        )

    elif dataset_type == "simple_pendulum":
        trajectories = generate_simple_pendulum_dataset(
            num_sequences=num_trajectories,
            seq_len=trajectory_length,
            dt=dt,
            damping=kwargs.get("damping", 1.0),
            noise=kwargs.get("noise", 0.0),
        )

    elif dataset_type == "coupled_oscillators":
        trajectories = generate_coupled_oscillators_dataset(
            num_sequences=num_trajectories,
            seq_len=trajectory_length,
            dt=dt,
            coupling_strength=kwargs.get("coupling_strength", 0.5),
            noise=kwargs.get("noise", 0.00),
        )

    elif dataset_type == "circular":
        trajectories = generate_circular_trajectory(
            num_sequences=num_trajectories,
            seq_len=trajectory_length,
            radius=kwargs.get("radius", 1.0),
            noise=kwargs.get("noise", 0.01),
        )

    elif dataset_type == "linear":
        trajectories = generate_constant_velocity_dataset(
            num_sequences=num_trajectories,
            seq_len=trajectory_length,
            noise=kwargs.get("noise", 0.01),
        )

    elif dataset_type == "sine":
        trajectories = generate_sine_wave_dataset(
            num_sequences=num_trajectories,
            seq_len=trajectory_length,
            noise=kwargs.get("noise", 0.00),
        )

    elif dataset_type == "figure8":
        trajectories = generate_figure_eight_dataset(
            num_sequences=num_trajectories,
            seq_len=trajectory_length,
            noise=kwargs.get("noise", 0.01),
        )

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return trajectories


def create_train_val_datasets(
    dataset_type="double_pendulum",
    num_train_trajectories=20,
    num_val_trajectories=5,
    trajectory_length=500,
    window_size=50,
    stride=None,
    normalize=True,
    dt=0.05,
    **kwargs,
):
    """
    Create training and validation windowed datasets.

    Args:
        dataset_type: Type of dataset
        num_train_trajectories: Number of training trajectories
        num_val_trajectories: Number of validation trajectories
        trajectory_length: Length of each full trajectory
        window_size: Size of training windows
        stride: Stride between windows (None = non-overlapping)
        normalize: Whether to normalize data
        dt: Time step
        **kwargs: Additional arguments for dataset generation

    Returns:
        train_dataset, val_dataset: TrajectoryWindowDataset objects
    """

    print(f"Generating {dataset_type} dataset...")
    print(f"  Training trajectories: {num_train_trajectories}")
    print(f"  Validation trajectories: {num_val_trajectories}")
    print(f"  Trajectory length: {trajectory_length}")
    print(f"  Window size: {window_size}")
    print(f"  Stride: {stride if stride else window_size}")

    # Generate full trajectories
    train_trajectories = generate_full_trajectories(
        dataset_type=dataset_type,
        num_trajectories=num_train_trajectories,
        trajectory_length=trajectory_length,
        dt=dt,
        **kwargs,
    )

    val_trajectories = generate_full_trajectories(
        dataset_type=dataset_type,
        num_trajectories=num_val_trajectories,
        trajectory_length=trajectory_length,
        dt=dt,
        **kwargs,
    )

    # Create windowed datasets
    train_dataset = TrajectoryWindowDataset(
        trajectories=train_trajectories,
        window_size=window_size,
        stride=stride,
        normalize=normalize,
    )

    # Use same normalization for validation
    val_dataset = TrajectoryWindowDataset(
        trajectories=val_trajectories,
        window_size=window_size,
        stride=stride,
        normalize=False,  # Don't renormalize
    )

    if normalize:
        # Apply training normalization to validation
        val_dataset.windows = (
            val_dataset.windows - train_dataset.mean
        ) / train_dataset.std
        val_dataset.mean = train_dataset.mean
        val_dataset.std = train_dataset.std

    # Print statistics
    train_stats = train_dataset.get_statistics()
    val_stats = val_dataset.get_statistics()

    print(f"\nTraining set:")
    print(f"  Total windows: {train_stats['num_windows']}")
    print(f"  Windows per trajectory: {train_stats['windows_per_trajectory']:.1f}")
    print(f"  Observation dim: {train_stats['obs_dim']}")

    print(f"\nValidation set:")
    print(f"  Total windows: {val_stats['num_windows']}")
    print(f"  Windows per trajectory: {val_stats['windows_per_trajectory']:.1f}")

    return train_dataset, val_dataset


class SapiMouseTrajectoryLoader:
    """Load and prepare SapiMouse trajectories for training"""

    def __init__(self, json_path: str):
        """
        Args:
            json_path: Path to segmented_trajectories.json
        """
        self.json_path = Path(json_path)
        self.trajectories = None
        self.metadata = None

        self._load_data()

    def _load_data(self):
        """Load trajectories from JSON file"""
        with open(self.json_path, "r") as f:
            data = json.load(f)

        self.trajectories = data
        print(f"Loaded trajectories for {len(data)} users")

    def extract_features(
        self,
        include_time: bool = False,
        include_velocity: bool = False,
        include_position: bool = True,
        relative_coords: bool = False,
        time_delta: bool = False,
    ) -> List[np.ndarray]:
        """
        Extract feature arrays from trajectories

        Args:
            include_time: Include timestamp features
            include_velocity: Include velocity values
            include_position: Include x, y coordinates
            relative_coords: Use relative coordinates (delta x, delta y)
            time_delta: Use time deltas instead of absolute timestamps

        Returns:
            List of numpy arrays, each shape (num_points, num_features)
        """
        all_trajectories = []

        for user_id, user_trajs in self.trajectories.items():
            for traj in user_trajs:
                features = []

                # ts, x, y
                points = np.array(traj["points"])  # Shape: (n, 3) - [time, x, y]
                velocities = np.array(traj["velocities"])  # Shape: (n-1,)

                n_points = len(points)

                # Time features
                if include_time:
                    if time_delta:
                        # Time deltas between consecutive points
                        time_deltas = np.diff(points[:, 0])
                        # Pad first point with 0 or mean
                        time_deltas = np.concatenate(
                            [[np.mean(time_deltas)], time_deltas]
                        )
                        features.append(time_deltas.reshape(-1, 1))
                    else:
                        # Absolute timestamps (normalized by first timestamp)
                        timestamps = points[:, 0] - points[0, 0]
                        features.append(timestamps.reshape(-1, 1))

                # Position features
                if include_position:
                    if relative_coords or False:
                        # Delta x, delta y between consecutive points
                        delta_x = np.diff(points[:, 1])
                        delta_y = np.diff(points[:, 2])
                        # Pad first point
                        delta_x = np.concatenate([[0], delta_x])
                        delta_y = np.concatenate([[0], delta_y])
                        features.append(delta_x.reshape(-1, 1))
                        features.append(delta_y.reshape(-1, 1))
                    else:
                        # Absolute x, y coordinates
                        features.append(points[:, 1].reshape(-1, 1))
                        features.append(points[:, 2].reshape(-1, 1))

                # Velocity features
                if include_velocity:
                    # Pad velocities to match number of points
                    # Last point gets same velocity as second-to-last
                    velocities_padded = np.concatenate([velocities, [velocities[-1]]])
                    features.append(velocities_padded.reshape(-1, 1))

                # Combine all features
                if features:
                    traj_array = np.hstack(features)
                    all_trajectories.append(traj_array)

        return all_trajectories

    def create_dataset(
        self,
        window_size: int = 50,
        stride: Optional[int] = None,
        normalize: bool = True,
        min_trajectory_length: Optional[int] = 50,
        max_trajectory_length: Optional[int] = 300,
        include_time: bool = False,
        include_velocity: bool = True,
        include_position: bool = True,
        relative_coords: bool = False,
        time_delta: bool = False,
        num_padding_zeros : int = 0,
        filter_users: Optional[List[str]] = None,
        filter_events: Optional[List[str]] = None,
    ) -> TrajectoryWindowDataset:
        """
        Create a TrajectoryWindowDataset from loaded trajectories

        Args:
            window_size: Size of each training window
            stride: Step between windows (default: window_size)
            normalize: Whether to normalize features
            min_trajectory_length: Minimum trajectory length to include
            include_time: Include time features
            include_velocity: Include velocity features
            include_position: Include position features
            relative_coords: Use relative coordinates
            time_delta: Use time deltas
            filter_users: List of user IDs to include (None = all)
            filter_events: List of event types to include (None = all)

        Returns:
            TrajectoryWindowDataset
        """
        # Filter trajectories if needed
        if filter_users or filter_events or min_trajectory_length or max_trajectory_length:
            filtered_data = {}
            for user_id, user_trajs in self.trajectories.items():
                if filter_users and user_id not in filter_users:
                    continue

                filtered_trajs = []
                for traj in user_trajs:
                    # Filter by event type
                    if filter_events:
                        if (
                            traj["start_event"] not in filter_events
                            and traj["end_event"] not in filter_events
                        ):
                            continue

                    # Filter by length
                    if min_trajectory_length:
                        if len(traj["points"]) < min_trajectory_length:
                            continue
                    if max_trajectory_length:
                        if len(traj["points"]) > max_trajectory_length:
                            continue

                    filtered_trajs.append(traj)

                if filtered_trajs:
                    filtered_data[user_id] = filtered_trajs

            # Temporarily replace trajectories
            original_trajs = self.trajectories
            self.trajectories = filtered_data

        # Extract features
        trajectories = self.extract_features(
            include_time=include_time,
            include_velocity=include_velocity,
            include_position=include_position,
            relative_coords=relative_coords,
            time_delta=time_delta,
        )

        # Restore original trajectories if we filtered
        if filter_users or filter_events or min_trajectory_length:
            self.trajectories = original_trajs

        print(f"Extracted {len(trajectories)} trajectories")
        print(f"Feature dimension: {trajectories[0].shape[1] if trajectories else 0}")

        # Create dataset
        dataset = TrajectoryWindowDataset(
            trajectories=trajectories,
            window_size=window_size,
            stride=stride,
            normalize=normalize,
            num_padding_zeros = num_padding_zeros
        )

        return dataset

    def create_user_specific_datasets(
        self,
        window_size: int = 50,
        stride: Optional[int] = None,
        normalize: bool = True,
        **feature_kwargs,
    ) -> Dict[str, TrajectoryWindowDataset]:
        """
        Create separate datasets for each user

        Returns:
            Dictionary mapping user_id to TrajectoryWindowDataset
        """
        datasets = {}

        for user_id in self.trajectories.keys():
            dataset = self.create_dataset(
                window_size=window_size,
                stride=stride,
                normalize=normalize,
                filter_users=[user_id],
                **feature_kwargs,
            )
            datasets[user_id] = dataset

            stats = dataset.get_statistics()
            print(
                f"{user_id}: {stats['num_trajectories']} trajectories, {stats['num_windows']} windows"
            )

        return datasets

    def get_trajectory_metadata(self) -> pd.DataFrame:
        """
        Get metadata about all trajectories

        Returns:
            DataFrame with trajectory information
        """
        rows = []

        for user_id, user_trajs in self.trajectories.items():
            for idx, traj in enumerate(user_trajs):
                row = {
                    "user_id": user_id,
                    "trajectory_idx": idx,
                    "session_id": traj["session_id"],
                    "num_points": len(traj["points"]),
                    "duration": traj["duration"],
                    "start_event": traj["start_event"],
                    "end_event": traj["end_event"],
                    **traj["features"],
                }
                rows.append(row)

        return pd.DataFrame(rows)


def visualize_trajectory_windows(dataset, traj_idx=0, num_windows=5):
    """
    Visualize how a full trajectory is sliced into windows.

    Args:
        dataset: TrajectoryWindowDataset
        traj_idx: Index of trajectory to visualize
        num_windows: Number of windows to highlight
    """

    # Get full trajectory
    full_traj = dataset.get_full_trajectory(traj_idx)

    # Find windows from this trajectory
    window_indices = [
        i for i, ti in enumerate(dataset.trajectory_indices) if ti == traj_idx
    ]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    obs_dim = full_traj.shape[1]

    # Plot 1: Full trajectory with window highlights
    if obs_dim == 2:
        axes[0].plot(
            full_traj[:, 0],
            full_traj[:, 1],
            "k-",
            alpha=0.3,
            linewidth=1,
            label="Full Trajectory",
        )
        axes[0].plot(
            full_traj[0, 0], full_traj[0, 1], "go", markersize=10, label="Start"
        )

        # Highlight windows
        colors = plt.cm.rainbow(
            np.linspace(0, 1, min(num_windows, len(window_indices)))
        )
        for i, (win_idx, color) in enumerate(zip(window_indices[:num_windows], colors)):
            window = dataset.denormalize(dataset.windows[win_idx])
            axes[0].plot(
                window[:, 0],
                window[:, 1],
                "-",
                color=color,
                linewidth=2,
                label=f"Window {i + 1}",
                alpha=0.8,
            )
            axes[0].plot(window[0, 0], window[0, 1], "o", color=color, markersize=8)

        axes[0].set_xlabel("X Position", fontsize=12)
        axes[0].set_ylabel("Y Position", fontsize=12)
        axes[0].set_title(
            f"Trajectory {traj_idx}: Full Path with Window Samples",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect("equal")

    # Plot 2: Timeline showing window coverage
    axes[1].plot(
        full_traj[:, 0], "k-", alpha=0.3, linewidth=1, label="Full Trajectory (X)"
    )

    # Show window coverage
    for i, win_idx in enumerate(window_indices[:num_windows]):
        start = i * dataset.stride
        end = start + dataset.window_size

        color = colors[i] if i < num_windows else "gray"
        axes[1].axvspan(
            start,
            end,
            alpha=0.3,
            color=color,
            label=f"Window {i + 1}" if i < num_windows else None,
        )

    axes[1].set_xlabel("Time Step", fontsize=12)
    axes[1].set_ylabel("X Position", fontsize=12)
    axes[1].set_title(
        f"Window Coverage Over Time (Stride={dataset.stride})",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_batch_samples(dataset, num_samples=8):
    """
    Visualize random window samples from the dataset.
    """

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    rows = 2
    cols = num_samples // rows
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, ax in zip(indices, axes):
        window = dataset.denormalize(dataset.windows[idx])
        traj_idx = dataset.trajectory_indices[idx]

        if window.shape[1] == 2:
            ax.plot(window[:, 0], window[:, 1], "b-", linewidth=2)
            ax.plot(window[0, 0], window[0, 1], "go", markersize=8, label="Start")
            ax.plot(window[-1, 0], window[-1, 1], "ro", markersize=8, label="End")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal")
        else:
            for dim in range(window.shape[1]):
                ax.plot(window[:, dim], label=f"Dim {dim}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.legend()

        ax.set_title(f"Window {idx} (Traj {traj_idx})", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Random Window Samples from Dataset", fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


def generate_sapimouse_dataset(
    json_path: str,
    window_size: int = 50,
    stride: int = None,
    normalize: bool = True,
    min_trajectory_length: int = 50,
    max_trajectory_length: int = 300,
    include_time: bool = False,
    include_velocity: bool = False,
    include_position: bool = True,
    relative_coords: bool = False,
    time_delta: bool = False,
    num_padding_zeros: int = 0,
    filter_users: str = None,
    filter_events: str = None,
):
    loader = SapiMouseTrajectoryLoader(json_path)
    dataset = loader.create_dataset(
        window_size,
        stride,
        normalize,
        min_trajectory_length,
        max_trajectory_length,
        include_time,
        include_velocity,
        include_position,
        relative_coords,
        time_delta,
        num_padding_zeros,
        filter_users,
        filter_events,
    )

    return dataset


def df_to_trajectory_dataset(
    data,
    window_size=50,
    stride=None,
    normalize=True,
    num_endpoint_padding=0,
    include_timestamp=False,
):
    """
    Convert SapiMouse DataFrame to TrajectoryWindowDataset.

    Args:
        data: DataFrame with list-valued x, y, timestamp columns
        window_size: Size of each training window
        stride: Step between windows (None = non-overlapping)
        normalize: Whether to normalize features
        num_endpoint_padding: Number of endpoint repetitions at each end
        include_timestamp: Whether to include timestamp as a feature
    """
    trajectories = []

    for idx, row in data.iterrows():
        x  = np.array(row["x"],         dtype=np.float32)
        y  = np.array(row["y"],         dtype=np.float32)
        ts = np.array(row["timestamp"], dtype=np.float32)

        if include_timestamp:
            # shape: (num_points, 3) → [timestamp, x, y]
            traj = np.stack([ts, x, y], axis=1)
        else:
            # shape: (num_points, 2) → [x, y]
            traj = np.stack([x, y], axis=1)

        # Skip trajectories that are too short for even one window
        if len(traj) >= window_size:
            trajectories.append(traj)

    print(f"Converted {len(trajectories)} valid trajectories from DataFrame")

    dataset = TrajectoryWindowDataset(
        trajectories=trajectories,
        window_size=window_size,
        stride=stride,
        normalize=normalize,
        num_padding_zeros=num_endpoint_padding,
    )

    return dataset


# " ============================================================================
# EXAMPLE USAGE SAPIMOUSE
# ============================================================================


def example_basic_usage():
    """Basic example: Load and create dataset"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)

    # Load trajectories
    loader = SapiMouseTrajectoryLoader("./sapimouse/segmented_trajectories.json")

    # Create dataset with default settings
    dataset = loader.create_dataset(
        window_size=50,
        stride=25,  # 50% overlap
        normalize=True,
    )

    # Print statistics
    stats = dataset.get_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Get a sample
    sample = dataset[0]
    print(f"\nSample window shape: {sample.shape}")
    print(f"Sample window:\n{sample[:5]}")  # First 5 timesteps

    return dataset


def example_custom_features():
    """Example with custom feature configuration"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Features")
    print("=" * 70)

    loader = SapiMouseTrajectoryLoader("./sapimouse/segmented_trajectories.json")

    # Create dataset with relative coordinates and velocity
    dataset = loader.create_dataset(
        window_size=50,
        stride=50,  # Non-overlapping windows
        normalize=True,
        include_time=True,
        include_velocity=True,
        include_position=True,
        relative_coords=True,  # Use delta x, delta y instead of absolute
        time_delta=True,  # Use time deltas instead of absolute timestamps
    )

    stats = dataset.get_statistics()
    print(f"\nFeature dimension: {stats['obs_dim']}")
    print(f"Features: [time_delta, delta_x, delta_y, velocity]")
    print(f"Number of windows: {stats['num_windows']}")

    return dataset


def example_filtered_dataset():
    """Example with filtering"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Filtered Dataset")
    print("=" * 70)

    loader = SapiMouseTrajectoryLoader("./sapimouse/segmented_trajectories.json")

    # Get metadata to see what's available
    metadata = loader.get_trajectory_metadata()
    print(f"\nTotal trajectories: {len(metadata)}")
    print(f"\nEvent types:")
    print(metadata["start_event"].value_counts())

    # Create dataset with only click-based trajectories
    dataset = loader.create_dataset(
        window_size=50,
        stride=25,
        normalize=True,
        min_trajectory_length=100,  # Only trajectories with 100+ points
        filter_events=["click", "press", "release"],  # Only click events
    )

    stats = dataset.get_statistics()
    print(f"\nFiltered dataset windows: {stats['num_windows']}")

    return dataset


def example_user_specific_datasets():
    """Example: Create separate dataset for each user"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: User-Specific Datasets")
    print("=" * 70)

    loader = SapiMouseTrajectoryLoader("./sapimouse/segmented_trajectories.json")

    # Create dataset per user
    user_datasets = loader.create_user_specific_datasets(
        window_size=50, stride=25, normalize=True
    )

    print(f"\nCreated datasets for {len(user_datasets)} users")

    # Access specific user dataset
    user_id = list(user_datasets.keys())[0]
    dataset = user_datasets[user_id]
    sample = dataset[0]
    print(f"\n{user_id} sample shape: {sample.shape}")

    return user_datasets


def example_with_dataloader():
    """Example: Use with PyTorch DataLoader"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: PyTorch DataLoader")
    print("=" * 70)

    loader = SapiMouseTrajectoryLoader("./sapimouse/segmented_trajectories.json")

    dataset = loader.create_dataset(window_size=50, stride=25, normalize=True)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Set to >0 for parallel loading
    )

    # Iterate through batches
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}: {batch.shape}")

        if batch_idx >= 2:  # Show first 3 batches
            break

    return dataloader


def example_feature_combinations():
    """Example: Different feature combinations"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Feature Combinations")
    print("=" * 70)

    loader = SapiMouseTrajectoryLoader("./sapimouse/segmented_trajectories.json")

    # Position + Velocity only
    dataset1 = loader.create_dataset(
        window_size=50,
        include_time=False,
        include_velocity=True,
        include_position=True,
        relative_coords=False,
    )
    print(f"Position + Velocity: {dataset1.get_statistics()['obs_dim']} features")
    # Expected: 3 features [x, y, velocity]

    # Time + Relative coords + Velocity
    dataset2 = loader.create_dataset(
        window_size=50,
        include_time=True,
        include_velocity=True,
        include_position=True,
        relative_coords=True,
        time_delta=True,
    )
    print(
        f"Time + RelCoords + Velocity: {dataset2.get_statistics()['obs_dim']} features"
    )
    # Expected: 4 features [time_delta, delta_x, delta_y, velocity]

    # Position only (no velocity, no time)
    dataset3 = loader.create_dataset(
        window_size=50,
        include_time=False,
        include_velocity=False,
        include_position=True,
        relative_coords=False,
    )
    print(f"Position only: {dataset3.get_statistics()['obs_dim']} features")
    # Expected: 2 features [x, y]

    return dataset1, dataset2, dataset3

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
from pathlib import Path
from loguru import logger


def visualize_multi_trajectory_predictions(trajectories, warmup=5,  
                                           figsize=(16, 10)):
    """
    Visualize predictions across multiple trajectories and context lengths.
    
    Args:
        trajectories: List of trajectory dictionaries with keys:
                     'ground_truth', 'reconstructions', 'predictions', 'context_length'
        warmup: Warmup steps
        trajectory_indices: Indices of trajectories to visualize (None = use first few)
        feature_idx: Which feature dimension to plot (default: 0)
        figsize: Figure size
    """

    trajectory_indices = np.arange(len(trajectories)).tolist()
    feature_num = trajectories[0]['ground_truth'].shape[1] if len(trajectories[0]['ground_truth'].shape) > 1 else 1
    n_trajs = len(trajectory_indices)
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_trajs, feature_num, figure=fig, hspace=0.3, wspace=0.3)
    

    for i, traj_idx in enumerate(trajectory_indices):
        context_len = trajectories[i]['context_length']
        for feature_idx in range(feature_num):
            j = feature_idx
            ax = fig.add_subplot(gs[i, j])
            # tr = trajectories[i]
            # Get the trajectory for this context length and index
            traj_dict = trajectories[traj_idx]
                
            gt = traj_dict['ground_truth']
            recons = traj_dict['reconstructions']
            preds = traj_dict['predictions']
                
            # Time indices
            context_end = warmup + context_len
            t_recon = np.arange(warmup, warmup + len(recons))
            t_pred = np.arange(context_end, context_end + len(preds))
            t_gt = np.arange(warmup, warmup + len(gt))
                
            # Plot ground truth
            ax.plot(t_gt, gt[:, feature_idx], 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
            
            # Plot reconstructions
            if len(recons) > 0:
                ax.plot(t_recon, recons[:, feature_idx], 'b--', linewidth=1.5, label='Reconstruction')
            
            # Plot predictions
            if len(preds) > 0:
                ax.plot(t_pred, preds[:, feature_idx], 'r--', linewidth=1.5, label='Prediction')

                # Add vertical line at context boundary
                ax.axvline(x=context_end, color='green', linestyle=':', linewidth=2, alpha=0.5)
            
            # Labels and title
            if i == 0:
                ax.set_title(f'Context: {context_len} steps', fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'Traj {traj_idx}\nFeature {feature_idx}', fontsize=9)
            if i == n_trajs - 1:
                ax.set_xlabel('Time Step', fontsize=9)
            
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='best')
    
    plt.suptitle(f'World Model Predictions: Multiple Trajectories & Context Lengths (Warmup={warmup})', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def compute_prediction_errors(trajectories):
    """
    Compute prediction errors (MSE) for different context lengths.
    
    Args:
        trajectories: List of trajectory dictionaries
    
    Returns:
        avg_errors: Average MSE for each context length
        std_errors: Standard deviation of MSE for each context length
    """
    # Group by context length
    context_errors = {}
    
    for traj_dict in trajectories:
        context_len = traj_dict['context_length']
        
        if context_len not in context_errors:
            context_errors[context_len] = []
        
        # Get prediction error (already computed in the dict)
        if 'horizon_error' in traj_dict:
            context_errors[context_len].append(traj_dict['horizon_error'])
        else:
            # Compute if not present
            gt = traj_dict['ground_truth']
            preds = traj_dict['predictions']
            
            if len(preds) > 0:
                pred_gt = gt[context_len:][:len(preds)]
                mse = np.mean((preds - pred_gt) ** 2)
                context_errors[context_len].append(mse)
    
    # Compute statistics
    avg_errors = {cl: np.mean(errors) if errors else np.nan 
                  for cl, errors in context_errors.items()}
    std_errors = {cl: np.std(errors) if errors else np.nan 
                  for cl, errors in context_errors.items()}
    
    return avg_errors, std_errors


def plot_error_vs_context(avg_errors, std_errors, figsize=(10, 6)):
    """
    Plot prediction error as a function of context length.
    """
    context_lengths = sorted(avg_errors.keys())
    means = [avg_errors[cl] for cl in context_lengths]
    stds = [std_errors[cl] for cl in context_lengths]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.errorbar(context_lengths, means, yerr=stds, marker='o', 
                linewidth=2, markersize=8, capsize=5, capthick=2)
    
    ax.set_xlabel('Context Length (steps)', fontsize=12)
    ax.set_ylabel('Prediction MSE', fontsize=12)
    ax.set_title('Prediction Error vs Context Length', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig


def visualize_single_trajectory_multicontext(trajectories, traj_idx=0, warmup=5,
                                             feature_indices=None, figsize=(14, 8)):
    """
    Visualize a single trajectory with multiple context lengths side by side.
    
    Args:
        trajectories: List of trajectory dictionaries
        traj_idx: Which trajectory index to visualize for each context length
        warmup: Warmup steps
        feature_indices: List of feature indices to plot
        figsize: Figure size
    """
    # Group trajectories by context length
    context_groups = {}
    for traj in trajectories:
        context_len = traj['context_length']
        if context_len not in context_groups:
            context_groups[context_len] = []
        context_groups[context_len].append(traj)
    
    context_lengths = sorted(context_groups.keys())
    
    # Determine feature indices
    if feature_indices is None:
        first_traj = trajectories[0]
        n_features = first_traj['ground_truth'].shape[1] if len(first_traj['ground_truth'].shape) > 1 else 1
        feature_indices = list(range(min(3, n_features)))
    
    n_features = len(feature_indices)
    n_contexts = len(context_lengths)
    
    fig, axes = plt.subplots(n_features, n_contexts, figsize=figsize, squeeze=False)
    
    for j, context_len in enumerate(context_lengths):
        # Get the trajectory for this context length
        if traj_idx < len(context_groups[context_len]):
            traj_dict = context_groups[context_len][traj_idx]
            
            gt = traj_dict['ground_truth']
            recons = traj_dict['reconstructions']
            preds = traj_dict['predictions']
            
            context_end = warmup + context_len
            t_recon = np.arange(warmup, warmup + len(recons))
            t_pred = np.arange(context_end, context_end + len(preds))
            t_gt = np.arange(warmup, warmup + len(gt))
            
            for i, feat_idx in enumerate(feature_indices):
                ax = axes[i, j]
                
                # Plot
                ax.plot(t_gt, gt[:, feat_idx], 'k-', linewidth=2, label='GT', alpha=0.7)
                if len(recons) > 0:
                    ax.plot(t_recon, recons[:, feat_idx], 'b--', linewidth=1.5, label='Recon')
                if len(preds) > 0:
                    ax.plot(t_pred, preds[:, feat_idx], 'r--', linewidth=1.5, label='Pred')
                
                ax.axvline(x=context_end, color='green', linestyle=':', linewidth=2, alpha=0.5)
                
                if i == 0:
                    ax.set_title(f'Context: {context_len}', fontsize=11, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(f'Feature {feat_idx}', fontsize=10)
                if i == n_features - 1:
                    ax.set_xlabel('Time Step', fontsize=9)
                
                ax.grid(True, alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)
    
    plt.suptitle('Single Trajectory: Different Context Lengths', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_multiple_trajectories(results, summary, save_dir=None):
    """
    Create beautiful visualization plots for multiple trajectory validation results.
    Shows individual uncertainty ellipses for each observation in 2D space.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.collections import LineCollection
    import matplotlib.patheffects as path_effects
    
    n_traj = len(results)
    
    # Check if we have uncertainty information (continuous decoder)
    has_uncertainty = 'reconstructions_std' in list(results.values())[0]
    
    # Set beautiful style
    colors = {
        'gt': '#2E3440',
        'recon': '#5E81AC',
        'pred': '#BF616A',
        'context_line': '#A3BE8C',
        'recon_fill': '#88C0D0',
        'pred_fill': '#D08770'
    }
    
    # Plot 1: Error distribution with modern styling
    fig1, ax1 = plt.subplots(figsize=(12, 6), facecolor='white')
    traj_ids = list(results.keys())
    horizon_errors = [results[tid]['horizon_error'] for tid in traj_ids]
    full_errors = [results[tid]['full_error'] for tid in traj_ids]
    
    x = np.arange(len(traj_ids))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, horizon_errors, width, label='Horizon Error', 
                    alpha=0.85, color=colors['pred'], edgecolor='white', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, full_errors, width, label='Full Error', 
                    alpha=0.85, color=colors['recon'], edgecolor='white', linewidth=1.5)
    
    ax1.set_xlabel('Trajectory Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Error (L2 Norm)', fontsize=12, fontweight='bold')
    ax1.set_title('Validation Errors Across Trajectories', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(traj_ids, fontsize=10)
    ax1.legend(fontsize=11, framealpha=0.95, edgecolor='gray')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Plot 2: Individual trajectory plots with uncertainty ellipses (up to 6)
    plot_limit = min(6, n_traj)
    fig2, axs = plt.subplots(2, 3, figsize=(20, 13), facecolor='white')
    axs = axs.flatten()
    
    for i, (traj_idx, data) in enumerate(list(results.items())[:plot_limit]):
        ax = axs[i]
        gt = data['ground_truth']
        recons = data['reconstructions']
        preds = data['predictions']
        context_len = data['context_length']
        
        # Create gradient colors for temporal progression
        recon_colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(recons)))
        pred_colors = plt.cm.Reds(np.linspace(0.5, 0.9, len(preds)))
        
        # Plot ground truth with gradient
        gt_colors = plt.cm.Greys(np.linspace(0.3, 0.7, len(gt)))
        for j in range(len(gt) - 1):
            ax.plot(gt[j:j+2, 0], gt[j:j+2, 1], '-', 
                   color=gt_colors[j], alpha=0.6, linewidth=2, zorder=1)
        ax.scatter(gt[:, 0], gt[:, 1], c='black', s=40, alpha=0.5, 
                  zorder=2, edgecolors='white', linewidth=1, label='Ground Truth')
        
        # Plot reconstructions with uncertainty ellipses
        for j in range(len(recons)):
            if j < len(recons) - 1:
                ax.plot(recons[j:j+2, 0], recons[j:j+2, 1], '-', 
                       color=recon_colors[j], alpha=0.8, linewidth=2.5, zorder=3)
            
            if has_uncertainty:
                recons_std = data['reconstructions_std']
                # Create uncertainty ellipse (2σ = ~95% confidence)
                ellipse = Ellipse(
                    xy=(recons[j, 0], recons[j, 1]),
                    width=4 * recons_std[j, 0],  # 2σ on each side
                    height=4 * recons_std[j, 1],
                    alpha=0.15,
                    facecolor=colors['recon_fill'],
                    edgecolor=colors['recon'],
                    linewidth=1,
                    zorder=2
                )
                ax.add_patch(ellipse)
        
        ax.scatter(recons[:, 0], recons[:, 1], c=colors['recon'], s=60, 
                  marker='s', alpha=0.9, zorder=4, edgecolors='white', 
                  linewidth=1.5, label='Reconstruction')
        
        # Plot predictions with uncertainty ellipses
        for j in range(len(preds)):
            if j < len(preds) - 1:
                ax.plot(preds[j:j+2, 0], preds[j:j+2, 1], '-', 
                       color=pred_colors[j], alpha=0.8, linewidth=2.5, zorder=3)
            
            if has_uncertainty:
                preds_std = data['predictions_std']
                # Create uncertainty ellipse (2σ = ~95% confidence)
                ellipse = Ellipse(
                    xy=(preds[j, 0], preds[j, 1]),
                    width=4 * preds_std[j, 0],  # 2σ on each side
                    height=4 * preds_std[j, 1],
                    alpha=0.15,
                    facecolor=colors['pred_fill'],
                    edgecolor=colors['pred'],
                    linewidth=1,
                    zorder=2
                )
                ax.add_patch(ellipse)
        
        ax.scatter(preds[:, 0], preds[:, 1], c=colors['pred'], s=60, 
                  marker='^', alpha=0.9, zorder=4, edgecolors='white', 
                  linewidth=1.5, label='Prediction')
        
        # Add start and end markers
        ax.scatter(gt[0, 0], gt[0, 1], c='green', s=150, marker='*', 
                  zorder=5, edgecolors='white', linewidth=2, label='Start')
        ax.scatter(gt[-1, 0], gt[-1, 1], c='red', s=150, marker='X', 
                  zorder=5, edgecolors='white', linewidth=2, label='End')
        
        # Add context boundary line
        if context_len > 0 and context_len < len(gt):
            # Find the approximate boundary in space
            if context_len < len(recons):
                boundary_point = recons[context_len - 1]
            else:
                boundary_point = recons[-1]
            
            # Add annotation
            ax.annotate('Context → Prediction', 
                       xy=(boundary_point[0], boundary_point[1]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc=colors['context_line'], alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                     color=colors['context_line'], lw=2),
                       fontsize=9, fontweight='bold')
        
        # Styling
        uncertainty_text = " (95% CI)" if has_uncertainty else ""
        title = f'Trajectory {traj_idx}{uncertainty_text}'
        subtitle = f'Horizon Error: {data["horizon_error"]:.3f} | Full Error: {data["full_error"]:.3f}'
        
        ax.set_title(f'{title}\n{subtitle}', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('x', fontsize=10, fontweight='bold')
        ax.set_ylabel('y', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='best', framealpha=0.95, edgecolor='gray')
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#FAFAFA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Equal aspect ratio for proper circles
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Plot 3: Uncertainty evolution (only if continuous decoder)
    fig3 = None
    if has_uncertainty:
        fig3, axs3 = plt.subplots(2, 3, figsize=(20, 13), facecolor='white')
        axs3 = axs3.flatten()
        
        for i, (traj_idx, data) in enumerate(list(results.items())[:plot_limit]):
            ax = axs3[i]
            recons_std = data['reconstructions_std']
            preds_std = data['predictions_std']
            context_len = data['context_length']
            
            # Calculate std for each dimension
            recons_std_x = recons_std[:, 0]
            recons_std_y = recons_std[:, 1]
            preds_std_x = preds_std[:, 0]
            preds_std_y = preds_std[:, 1]
            
            # Time indices
            recons_t = np.arange(len(recons_std))
            preds_t = np.arange(context_len, context_len + len(preds_std))
            
            # Plot uncertainty over time for both dimensions
            ax.plot(recons_t, recons_std_x, 'o-', label='Recon σ_x', 
                   alpha=0.8, markersize=4, color='#5E81AC', linewidth=2)
            ax.plot(recons_t, recons_std_y, 's-', label='Recon σ_y', 
                   alpha=0.8, markersize=4, color='#81A1C1', linewidth=2)
            ax.plot(preds_t, preds_std_x, '^-', label='Pred σ_x', 
                   alpha=0.8, markersize=4, color='#BF616A', linewidth=2)
            ax.plot(preds_t, preds_std_y, 'v-', label='Pred σ_y', 
                   alpha=0.8, markersize=4, color='#D08770', linewidth=2)
            
            # Fill between for context and prediction phases
            ax.axvspan(0, context_len, alpha=0.1, color=colors['recon'], label='Context Phase')
            ax.axvspan(context_len, preds_t[-1], alpha=0.1, color=colors['pred'], label='Prediction Phase')
            
            # Mark context boundary
            ax.axvline(x=context_len, color=colors['context_line'], linestyle='--', 
                      linewidth=2.5, alpha=0.7, label='Context End')
            
            ax.set_title(f'Uncertainty Evolution - Trajectory {traj_idx}', 
                        fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Time Step', fontsize=10, fontweight='bold')
            ax.set_ylabel('Standard Deviation (σ)', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, loc='best', framealpha=0.95, edgecolor='gray', ncol=2)
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
            ax.set_facecolor('#FAFAFA')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
    
    # Save plots
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig1.savefig(os.path.join(save_dir, "multi_traj_errors.png"), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, "multi_traj_plots.png"), dpi=300, bbox_inches='tight')
        if fig3 is not None:
            fig3.savefig(os.path.join(save_dir, "multi_traj_uncertainty.png"), dpi=300, bbox_inches='tight')
        logger.info(f"Beautiful plots saved to {save_dir}")
    
    return fig1, fig2, fig3 if has_uncertainty else (fig1, fig2)

def plot_losses(elbo_history : list, recon_history, kl_history, validation_error):

    fig, axs = plt.subplots(1, 4, figsize = (15, 10))
    axs[0].plot(elbo_history)
    axs[0].set_title("elbo loss")

    axs[1].plot(recon_history)
    axs[1].set_title("recon loss")

    axs[2].plot(kl_history)
    axs[2].set_title("kl loss")
    
    axs[3].plot(validation_error)
    axs[3].set_title("validation error")
    # plt.show()
    return fig


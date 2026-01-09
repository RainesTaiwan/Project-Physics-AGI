"""
Visualization Tools
可視化工具 - 用於分析和理解 World Model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import yaml

from src.trainer import WorldModel
from src.utils import make_env


def visualize_reconstruction(
    world_model: WorldModel,
    observation: np.ndarray,
    device: torch.device
):
    """
    可視化觀測重建
    
    展示編碼器-解碼器是否保留了必要信息
    """
    # Encode
    obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(device)
    obs_latent = world_model.encode_observation(obs_tensor)
    
    # Create state
    state = {
        'h': torch.zeros(1, world_model.rssm.deterministic_size).to(device),
        'z': obs_latent
    }
    
    # Decode
    with torch.no_grad():
        reconstruction = world_model.decode_state(state)
    
    # Convert to numpy
    obs_np = observation.transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
    recon_np = reconstruction[0].cpu().numpy().transpose(1, 2, 0)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(obs_np / 255.0)
    axes[0].set_title('Original Observation')
    axes[0].axis('off')
    
    axes[1].imshow(recon_np / 255.0)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_imagination_rollout(
    world_model: WorldModel,
    initial_obs: np.ndarray,
    actions: np.ndarray,
    device: torch.device
):
    """
    可視化想像軌跡
    
    展示 RSSM 如何「在腦海中」模擬未來
    
    Args:
        world_model: trained model
        initial_obs: [C, H, W] initial observation
        actions: [T, A] sequence of actions
        device: torch device
    """
    # Initialize state
    obs_tensor = torch.from_numpy(initial_obs).unsqueeze(0).to(device)
    obs_latent = world_model.encode_observation(obs_tensor)
    
    state = {
        'h': torch.zeros(1, world_model.rssm.deterministic_size).to(device),
        'z': obs_latent
    }
    
    # Imagine rollout
    imagined_frames = []
    
    with torch.no_grad():
        # First frame (real)
        recon = world_model.decode_state(state)
        imagined_frames.append(recon[0].cpu().numpy())
        
        # Imagine future
        action_tensor = torch.from_numpy(actions).unsqueeze(0).to(device)
        trajectory = world_model.rssm.imagine_rollout(
            state,
            action_tensor,
            horizon=actions.shape[0]
        )
        
        # Decode imagined states
        for t in range(actions.shape[0]):
            state_t = {
                'h': trajectory['h'][:, t],
                'z': trajectory['z'][:, t]
            }
            recon_t = world_model.decode_state(state_t)
            imagined_frames.append(recon_t[0].cpu().numpy())
    
    # Plot imagination sequence
    num_frames = len(imagined_frames)
    cols = min(8, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if num_frames > 1 else [axes]
    
    for i, frame in enumerate(imagined_frames):
        frame_np = frame.transpose(1, 2, 0) / 255.0
        axes[i].imshow(frame_np)
        axes[i].set_title(f't={i}' if i > 0 else 't=0 (real)')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_latent_space(
    world_model: WorldModel,
    observations: List[np.ndarray],
    device: torch.device
):
    """
    可視化潛在空間
    
    使用 PCA 或 t-SNE 降維到 2D
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("需要安裝 scikit-learn: pip install scikit-learn")
        return None
    
    latents = []
    
    with torch.no_grad():
        for obs in observations:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            z = world_model.encode_observation(obs_tensor)
            latents.append(z.cpu().numpy()[0])
    
    latents = np.array(latents)  # [N, latent_dim]
    
    # PCA to 2D
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=np.arange(len(latents)),
        cmap='viridis',
        alpha=0.6
    )
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Latent Space Visualization')
    plt.colorbar(scatter, label='Time Step')
    
    explained_var = pca.explained_variance_ratio_
    ax.text(
        0.02, 0.98,
        f'Explained Variance:\nPC1: {explained_var[0]:.2%}\nPC2: {explained_var[1]:.2%}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    return fig


def plot_training_curves(log_dir: str):
    """
    繪製訓練曲線
    
    從 TensorBoard 日誌中提取數據
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("需要 TensorBoard")
        return None
    
    # Load events
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Get scalars
    tags = ea.Tags()['scalars']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot reconstruction loss
    if 'train/dynamics/reconstruction_loss' in tags:
        data = ea.Scalars('train/dynamics/reconstruction_loss')
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[0, 0].plot(steps, values)
        axes[0, 0].set_title('Reconstruction Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
    
    # Plot KL loss
    if 'train/dynamics/kl_loss' in tags:
        data = ea.Scalars('train/dynamics/kl_loss')
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[0, 1].plot(steps, values)
        axes[0, 1].set_title('KL Divergence (想像 vs 現實)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('KL')
        axes[0, 1].axhline(y=3.0, color='r', linestyle='--', label='Free Nats')
        axes[0, 1].legend()
    
    # Plot episode reward
    if 'collect/mean_episode_reward' in tags:
        data = ea.Scalars('collect/mean_episode_reward')
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[1, 0].plot(steps, values)
        axes[1, 0].set_title('Episode Reward')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Reward')
    
    # Plot value loss
    if 'train/behavior/value_loss' in tags:
        data = ea.Scalars('train/behavior/value_loss')
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[1, 1].plot(steps, values)
        axes[1, 1].set_title('Value Loss')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize World Model')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--mode', type=str, default='reconstruction',
                       choices=['reconstruction', 'imagination', 'latent'])
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    world_model = WorldModel(config).to(device)
    env = make_env(config['environment'])
    world_model.initialize_dynamics(env.action_dim)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    world_model.load_state_dict(checkpoint['world_model'])
    world_model.eval()
    
    # Get test observation
    obs = env.reset()
    
    if args.mode == 'reconstruction':
        fig = visualize_reconstruction(world_model, obs, device)
        plt.savefig('reconstruction.png')
        print("Saved: reconstruction.png")
    
    elif args.mode == 'imagination':
        # Random actions
        actions = np.random.uniform(
            env.action_min,
            env.action_max,
            size=(15, env.action_dim)
        )
        fig = visualize_imagination_rollout(world_model, obs, actions, device)
        plt.savefig('imagination.png')
        print("Saved: imagination.png")
    
    elif args.mode == 'latent':
        # Collect observations
        observations = []
        for _ in range(100):
            action = np.random.uniform(env.action_min, env.action_max, env.action_dim)
            obs, _, done, _ = env.step(action)
            observations.append(obs)
            if done:
                obs = env.reset()
        
        fig = visualize_latent_space(world_model, observations, device)
        plt.savefig('latent_space.png')
        print("Saved: latent_space.png")
    
    plt.show()
    env.close()

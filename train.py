"""
Main Training Script
主訓練腳本 - 完整的訓練流程
"""

import os
# Set up headless rendering for MuJoCo before importing dm_control
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
from typing import Dict
from torch.utils.tensorboard import SummaryWriter

from src.trainer import WorldModel, WorldModelTrainer
from src.utils import ReplayBuffer, make_env


class Agent:
    """
    完整的 Agent 系統
    整合實時交互循環和夢境訓練循環
    """
    
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set seed
        self.seed = self.config['seed']
        self._set_seed(self.seed)
        
        # Device
        self.device = torch.device(
            self.config['infrastructure']['device']
            if torch.cuda.is_available()
            else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Create environment
        print("Creating environment...")
        self.env = make_env(self.config['environment'])
        
        # Update action dim in config
        self.config['rssm']['action_dim'] = self.env.action_dim
        
        # Create world model
        print("Building World Model...")
        self.world_model = WorldModel(self.config).to(self.device)
        self.world_model.initialize_dynamics(self.env.action_dim)
        
        # Create trainer
        self.trainer = WorldModelTrainer(
            self.world_model,
            self.config,
            self.device
        )
        
        # Create replay buffer
        print("Initializing Replay Buffer...")
        self.replay_buffer = ReplayBuffer(
            capacity=self.config['training']['buffer_size'],
            observation_shape=self.env.observation_shape,
            action_dim=self.env.action_dim,
            sequence_length=self.config['training']['sequence_length'],
            device=self.device
        )
        
        # Logging
        log_dir = Path(self.config['infrastructure']['log_dir']) / \
                  self.config['infrastructure']['experiment_name']
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        # Training state
        self.total_steps = 0
        self.total_episodes = 0
        
        print(f"\n{'='*60}")
        print(f"Environment: {self.config['environment']['name']}")
        print(f"Action dimension: {self.env.action_dim}")
        print(f"Observation shape: {self.env.observation_shape}")
        print(f"{'='*60}\n")
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def collect_experience(self, num_steps: int, random: bool = False) -> Dict:
        """
        實時交互循環 (Inference Loop)
        
        收集環境交互數據
        
        Args:
            num_steps: number of steps to collect
            random: if True, use random actions (for prefill)
            
        Returns:
            stats: collection statistics
        """
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        obs = self.env.reset()
        state = self.world_model.rssm.initial_state(1, self.device)
        
        for step in range(num_steps):
            # Select action
            if random:
                # Random exploration
                action = np.random.uniform(
                    self.env.action_min,
                    self.env.action_max,
                    size=self.env.action_dim
                )
            else:
                # Policy action
                with torch.no_grad():
                    # Encode observation
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    obs_latent = self.world_model.encode_observation(obs_tensor)
                    
                    # Update state
                    if step == 0:
                        state['z'] = obs_latent
                    else:
                        action_tensor = torch.from_numpy(prev_action).float().unsqueeze(0).to(self.device)
                        state, _, _ = self.world_model.rssm.observe_step(
                            state, action_tensor, obs_latent
                        )
                    
                    # Sample action
                    action_tensor = self.world_model.actor_critic.act(state, deterministic=False)
                    action = action_tensor.cpu().numpy()[0]
                
                # Add exploration noise
                noise_scale = self.config['inference'].get('exploration_noise', 0.3)
                action = action + np.random.normal(0, noise_scale, size=action.shape)
                action = np.clip(action, self.env.action_min, self.env.action_max)
            
            # Execute action
            next_obs, reward, done, info = self.env.step(action)
            
            # Store in buffer
            self.replay_buffer.add(obs, action, reward, done)
            
            # Update stats
            current_episode_reward += reward
            current_episode_length += 1
            self.total_steps += 1
            
            # Handle episode end
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                self.total_episodes += 1
                
                # Reset
                obs = self.env.reset()
                state = self.world_model.rssm.initial_state(1, self.device)
                current_episode_reward = 0
                current_episode_length = 0
            else:
                obs = next_obs
                prev_action = action
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0
        }
    
    def train(self, total_steps: int = 1000000):
        """
        完整訓練流程
        
        整合：
        1. 實時交互循環 (收集數據)
        2. 夢境訓練循環 (學習模型)
        """
        train_cfg = self.config['training']
        
        # Phase 1: Prefill buffer with random data
        print("Phase 1: Prefilling replay buffer with random exploration...")
        prefill_steps = train_cfg['prefill_steps']
        self.collect_experience(prefill_steps, random=True)
        print(f"Buffer size: {len(self.replay_buffer)}/{self.replay_buffer.capacity}")
        
        # Phase 2: Training loop
        print("\nPhase 2: Training World Model...")
        print(f"Target: {total_steps} total environment steps")
        
        steps_per_iter = train_cfg['train_every']
        train_steps_per_iter = train_cfg['train_steps']
        
        pbar = tqdm(total=total_steps - prefill_steps, desc="Training")
        
        while self.total_steps < total_steps:
            # Collect experience
            collect_stats = self.collect_experience(steps_per_iter, random=False)
            
            # Log collection stats
            if collect_stats['episode_rewards']:
                self.writer.add_scalar(
                    'collect/mean_episode_reward',
                    collect_stats['mean_reward'],
                    self.total_steps
                )
                self.writer.add_scalar(
                    'collect/mean_episode_length',
                    collect_stats['mean_length'],
                    self.total_steps
                )
            
            # Train on collected data
            for _ in range(train_steps_per_iter):
                # Sample batch
                batch = self.replay_buffer.sample_sequence(
                    batch_size=train_cfg['batch_size']
                )
                
                # Training step
                losses = self.trainer.train_step(batch)
                
                # Log training losses
                for key, value in losses.items():
                    if key != 'train_step':
                        self.writer.add_scalar(
                            f'train/{key}',
                            value,
                            self.trainer.train_step
                        )
            
            # Update progress bar
            pbar.update(steps_per_iter)
            pbar.set_postfix({
                'episodes': self.total_episodes,
                'buffer': len(self.replay_buffer),
                'reward': f"{collect_stats['mean_reward']:.2f}" if collect_stats['episode_rewards'] else 'N/A'
            })
            
            # Save checkpoint
            if self.total_steps % train_cfg['save_every'] == 0:
                self.save_checkpoint()
        
        pbar.close()
        print("\n✓ Training completed!")
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['infrastructure']['log_dir']) / \
                        self.config['infrastructure']['experiment_name'] / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_{self.total_steps}.pt'
        
        torch.save({
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'world_model': self.world_model.state_dict(),
            'model_optimizer': self.trainer.model_optimizer.state_dict(),
            'actor_optimizer': self.trainer.actor_optimizer.state_dict(),
            'value_optimizer': self.trainer.value_optimizer.state_dict(),
            'config': self.config
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.trainer.model_optimizer.load_state_dict(checkpoint['model_optimizer'])
        self.trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.trainer.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        
        self.total_steps = checkpoint['total_steps']
        self.total_episodes = checkpoint['total_episodes']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resumed at step {self.total_steps}, episode {self.total_episodes}")


def main():
    parser = argparse.ArgumentParser(description='Train World Model for Physics-AGI')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=1000000,
        help='Total training steps'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Create agent
    agent = Agent(args.config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        agent.load_checkpoint(args.checkpoint)
    
    # Train
    agent.train(total_steps=args.steps)
    
    # Cleanup
    agent.env.close()
    agent.writer.close()


if __name__ == '__main__':
    main()

"""
Evaluation Script
評估訓練好的模型
"""

import yaml
import torch
import numpy as np
import argparse
from pathlib import Path

from src.trainer import WorldModel
from src.utils import make_env


def evaluate(
    world_model: WorldModel,
    env,
    device: torch.device,
    num_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False
):
    """
    Evaluate trained world model
    
    Args:
        world_model: trained model
        env: environment
        device: torch device
        num_episodes: number of episodes to evaluate
        deterministic: use deterministic actions
        render: render episodes
        
    Returns:
        stats: evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    
    world_model.eval()
    
    for episode in range(num_episodes):
        obs = env.reset()
        state = world_model.rssm.initial_state(1, device)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            with torch.no_grad():
                # Encode observation
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                obs_latent = world_model.encode_observation(obs_tensor)
                
                # Update state
                if episode_length == 0:
                    state['z'] = obs_latent
                else:
                    action_tensor = torch.from_numpy(prev_action).unsqueeze(0).to(device)
                    state, _, _ = world_model.rssm.observe_step(
                        state, action_tensor, obs_latent
                    )
                
                # Get action
                action_tensor = world_model.actor_critic.act(state, deterministic=deterministic)
                action = action_tensor.cpu().numpy()[0]
            
            # Execute
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            prev_action = action
            
            if render:
                # Optionally render
                pass
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards)
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate World Model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of episodes to evaluate'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Use deterministic actions'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    print("Creating environment...")
    env = make_env(config['environment'])
    
    # Create model
    print("Loading model...")
    world_model = WorldModel(config).to(device)
    world_model.initialize_dynamics(env.action_dim)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    world_model.load_state_dict(checkpoint['world_model'])
    print(f"Loaded checkpoint from step {checkpoint['total_steps']}")
    
    # Evaluate
    print(f"\nEvaluating for {args.episodes} episodes...")
    stats = evaluate(
        world_model,
        env,
        device,
        num_episodes=args.episodes,
        deterministic=args.deterministic
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results:")
    print(f"{'='*60}")
    print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Mean Length: {stats['mean_length']:.2f} ± {stats['std_length']:.2f}")
    print(f"Min/Max Reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
    print(f"{'='*60}\n")
    
    env.close()


if __name__ == '__main__':
    main()

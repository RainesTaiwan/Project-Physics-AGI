"""
World Model Trainer
完整的 World Model 訓練系統
整合 Dynamics Learning (學物理) 和 Behavior Learning (學策略)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

from src.models import (
    VariationalEncoder,
    CNNDecoder,
    RSSM,
    ActorCritic
)
from src.models.encoder import reconstruction_loss, kl_divergence_gaussian


class WorldModel(nn.Module):
    """
    完整的 World Model 系統
    
    組件：
    1. Encoder: 觀測 -> 潛在特徵
    2. RSSM: 動力學模型 (物理引擎)
    3. Decoder: 潛在特徵 -> 重建觀測
    4. Reward/Value/Actor: 決策系統
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract config
        self.config = config
        encoder_cfg = config['encoder']
        rssm_cfg = config['rssm']
        decoder_cfg = config['decoder']
        actor_cfg = config['actor']
        reward_cfg = config['reward_value']
        
        # Action dimension (to be set dynamically)
        self.action_dim = None
        
        # Build modules
        self.encoder = VariationalEncoder(
            input_shape=tuple(encoder_cfg['input_shape']),
            latent_dim=encoder_cfg['latent_dim'],
            hidden_dims=encoder_cfg['hidden_dims'],
            activation=encoder_cfg['activation']
        )
        
        self.decoder = CNNDecoder(
            latent_dim=rssm_cfg['stochastic_size'] + rssm_cfg['deterministic_size'],
            output_shape=tuple(decoder_cfg['output_shape']),
            hidden_dims=decoder_cfg['hidden_dims']
        )
        
        # RSSM and Actor-Critic will be initialized after knowing action_dim
        self.rssm = None
        self.actor_critic = None
        
    def initialize_dynamics(self, action_dim: int):
        """Initialize RSSM and Actor-Critic with action dimension"""
        self.action_dim = action_dim
        
        rssm_cfg = self.config['rssm']
        actor_cfg = self.config['actor']
        reward_cfg = self.config['reward_value']
        
        self.rssm = RSSM(
            action_dim=action_dim,
            stochastic_size=rssm_cfg['stochastic_size'],
            deterministic_size=rssm_cfg['deterministic_size'],
            hidden_size=rssm_cfg['hidden_size'],
            activation=rssm_cfg['activation'],
            free_nats=rssm_cfg['free_nats'],
            kl_balance_scale=rssm_cfg['kl_balance_scale']
        )
        
        self.actor_critic = ActorCritic(
            action_dim=action_dim,
            stochastic_size=rssm_cfg['stochastic_size'],
            deterministic_size=rssm_cfg['deterministic_size'],
            actor_hidden_dims=actor_cfg['hidden_dims'],
            value_hidden_dims=reward_cfg['value_hidden_dims'],
            reward_hidden_dims=reward_cfg['reward_hidden_dims']
        )
    
    def encode_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent feature"""
        z, mean, std = self.encoder(observation)
        return z
    
    def decode_state(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode RSSM state to reconstructed observation"""
        # Concatenate h and z
        h = state['h']
        z = state['z']
        
        if h.dim() == 3:  # Sequence [B, T, D]
            B, T = h.shape[:2]
            latent = torch.cat([h, z], dim=-1)
            latent_flat = latent.view(B * T, -1)
            recon_flat = self.decoder(latent_flat)
            reconstruction = recon_flat.view(B, T, *recon_flat.shape[1:])
        else:  # Single step [B, D]
            latent = torch.cat([h, z], dim=-1)
            reconstruction = self.decoder(latent)
        
        return reconstruction
    
    def get_state_features(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get concatenated state features for downstream tasks"""
        return torch.cat([state['h'], state['z']], dim=-1)


class WorldModelTrainer:
    """
    Trainer for World Model
    
    實現兩個訓練循環：
    1. Dynamics Learning: 學習物理動力學
    2. Behavior Learning: 學習策略
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        config: Dict,
        device: torch.device
    ):
        self.world_model = world_model
        self.config = config
        self.device = device
        
        train_cfg = config['training']
        
        # Optimizers
        # Model optimizer: Encoder + RSSM + Decoder + Reward
        self.model_params = list(world_model.encoder.parameters()) + \
                           list(world_model.decoder.parameters()) + \
                           list(world_model.rssm.parameters()) + \
                           list(world_model.actor_critic.reward.parameters())
        
        self.model_optimizer = torch.optim.Adam(
            self.model_params,
            lr=train_cfg['model_lr'],
            eps=train_cfg['adam_eps']
        )
        
        # Actor optimizer
        self.actor_optimizer = torch.optim.Adam(
            world_model.actor_critic.actor.parameters(),
            lr=train_cfg['actor_lr'],
            eps=train_cfg['adam_eps']
        )
        
        # Value optimizer
        self.value_optimizer = torch.optim.Adam(
            world_model.actor_critic.value.parameters(),
            lr=train_cfg['value_lr'],
            eps=train_cfg['adam_eps']
        )
        
        # Loss weights
        self.reconstruction_weight = train_cfg['reconstruction_weight']
        self.kl_weight = train_cfg['kl_weight']
        self.reward_weight = train_cfg['reward_weight']
        
        # Gradient clipping
        self.grad_clip_norm = train_cfg['grad_clip_norm']
        
        # Training state
        self._step_count = 0
    
    def train_dynamics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Dynamics Learning (學習物理動力學)
        
        目標：
        1. 重建觀測 (Reconstruction)
        2. 最小化 Prior-Posterior KL (學習物理法則)
        3. 預測獎勵 (Reward Prediction)
        
        Args:
            batch: {
                'observations': [B, L, C, H, W]
                'actions': [B, L, A]
                'rewards': [B, L]
                'dones': [B, L]
            }
            
        Returns:
            losses: dict of loss values
        """
        B, L = batch['observations'].shape[:2]
        
        # Move to device
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        
        # Initialize RSSM state
        state = self.world_model.rssm.initial_state(B, self.device)
        
        # Storage for losses
        reconstruction_losses = []
        kl_losses = []
        reward_losses = []
        
        # Roll out through sequence
        for t in range(L):
            # Current observation and action
            obs_t = observations[:, t]
            action_t = actions[:, t] if t > 0 else torch.zeros(B, self.world_model.action_dim).to(self.device)
            reward_t = rewards[:, t]
            
            # Encode observation
            obs_latent = self.world_model.encode_observation(obs_t)
            
            # Update RSSM state (posterior path)
            if t == 0:
                # First step: just encode
                state['z'] = obs_latent
            else:
                # Subsequent steps: observe_step
                state, prior_dist, posterior_dist = self.world_model.rssm.observe_step(
                    state, action_t, obs_latent
                )
                
                # KL divergence loss
                kl_loss = self.world_model.rssm.kl_loss(prior_dist, posterior_dist)
                kl_losses.append(kl_loss)
            
            # Reconstruction loss
            recon_obs = self.world_model.decode_state(state)
            recon_loss = reconstruction_loss(recon_obs, obs_t)
            reconstruction_losses.append(recon_loss)
            
            # Reward prediction loss
            pred_reward = self.world_model.actor_critic.reward(state)
            reward_loss = F.mse_loss(pred_reward, reward_t)
            reward_losses.append(reward_loss)
        
        # Average losses
        total_recon_loss = torch.mean(torch.stack(reconstruction_losses))
        total_kl_loss = torch.mean(torch.stack(kl_losses)) if kl_losses else torch.tensor(0.0).to(self.device)
        total_reward_loss = torch.mean(torch.stack(reward_losses))
        
        # Combined loss
        total_loss = (
            self.reconstruction_weight * total_recon_loss +
            self.kl_weight * total_kl_loss +
            self.reward_weight * total_reward_loss
        )
        
        # Optimize
        self.model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_params, self.grad_clip_norm)
        self.model_optimizer.step()
        
        return {
            'dynamics/total_loss': total_loss.item(),
            'dynamics/reconstruction_loss': total_recon_loss.item(),
            'dynamics/kl_loss': total_kl_loss.item(),
            'dynamics/reward_loss': total_reward_loss.item()
        }
    
    def train_behavior(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Behavior Learning (學習策略)
        
        在想像空間中展開軌跡，優化 Actor 和 Critic
        
        核心：凍結動力學模型，在 Latent Space 中做夢
        """
        B, L = batch['observations'].shape[:2]
        
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        
        # Build initial states from real data
        with torch.no_grad():
            state = self.world_model.rssm.initial_state(B, self.device)
            
            # Warm-up: run through first few real steps
            warmup_steps = min(5, L)
            for t in range(warmup_steps):
                obs_t = observations[:, t]
                action_t = actions[:, t] if t > 0 else torch.zeros(B, self.world_model.action_dim).to(self.device)
                
                obs_latent = self.world_model.encode_observation(obs_t)
                
                if t == 0:
                    state['z'] = obs_latent
                else:
                    state, _, _ = self.world_model.rssm.observe_step(state, action_t, obs_latent)
        
        # Imagine rollout
        imagination_horizon = self.config['rssm']['imagination_horizon']
        
        # Storage for imagination
        states_h = [state['h']]
        states_z = [state['z']]
        actions_imagined = []
        log_probs = []
        
        # Roll out in imagination
        for t in range(imagination_horizon):
            # Sample action from policy
            action, dist = self.world_model.actor_critic.actor(state, deterministic=False)
            log_prob = self.world_model.actor_critic.actor.log_prob(dist, action)
            
            # Imagine next state (detach to avoid retaining graph)
            next_state, _, _ = self.world_model.rssm.imagine_step(state, action.detach())
            state = {
                'h': next_state['h'].detach(),
                'z': next_state['z'].detach()
            }
            
            states_h.append(state['h'])
            states_z.append(state['z'])
            actions_imagined.append(action)
            log_probs.append(log_prob)
        
        # Stack imagined trajectory
        states_h = torch.stack(states_h[:-1], dim=1)  # [B, H, D]
        states_z = torch.stack(states_z[:-1], dim=1)  # [B, H, S]
        imagined_states = {'h': states_h, 'z': states_z}
        
        # Predict rewards and values
        pred_rewards = self.world_model.actor_critic.reward(imagined_states)  # [B, H]
        pred_values = self.world_model.actor_critic.value(imagined_states)    # [B, H]
        
        # Compute returns (lambda-return)
        gamma = self.config['reward_value']['gamma']
        lambda_ = self.config['reward_value']['lambda_']
        
        returns = self._compute_lambda_returns(pred_rewards, pred_values, gamma, lambda_)
        
        # Actor loss (policy gradient)
        advantages = (returns - pred_values).detach()
        log_probs_stacked = torch.stack(log_probs, dim=1)  # [B, H]
        actor_loss = -(log_probs_stacked * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(pred_values, returns.detach())
        
        # Combined loss for simplicity
        total_behavior_loss = actor_loss + value_loss
        
        # Optimize both
        self.actor_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_behavior_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.world_model.actor_critic.actor.parameters(),
            self.grad_clip_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.world_model.actor_critic.value.parameters(),
            self.grad_clip_norm
        )
        
        self.actor_optimizer.step()
        self.value_optimizer.step()
        
        return {
            'behavior/actor_loss': actor_loss.item(),
            'behavior/value_loss': value_loss.item(),
            'behavior/mean_return': returns.mean().item(),
            'behavior/mean_value': pred_values.mean().item()
        }
    
    def _compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float,
        lambda_: float
    ) -> torch.Tensor:
        """
        Compute λ-returns (GAE-style)
        
        R_t^λ = r_t + γ[(1-λ)V(s_{t+1}) + λR_{t+1}^λ]
        """
        B, H = rewards.shape
        returns = torch.zeros_like(rewards)
        
        # Bootstrap from last value
        next_return = values[:, -1]
        
        # Backward iteration
        for t in reversed(range(H)):
            returns[:, t] = rewards[:, t] + gamma * (
                (1 - lambda_) * values[:, t] + lambda_ * next_return
            )
            next_return = returns[:, t]
        
        return returns
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Complete training step (dynamics + behavior)
        """
        # Train dynamics (learn physics)
        dynamics_losses = self.train_dynamics(batch)
        
        # Train behavior (learn policy) 
        behavior_losses = self.train_behavior(batch)
        
        self._step_count += 1
        
        # Combine losses
        all_losses = {**dynamics_losses, **behavior_losses}
        all_losses['train_step'] = self._step_count
        
        return all_losses


if __name__ == "__main__":
    print("Testing World Model Trainer...")
    
    # Mock config
    config = {
        'encoder': {
            'input_shape': [3, 64, 64],
            'latent_dim': 32,
            'deterministic_dim': 200,
            'hidden_dims': [32, 64, 128, 256],
            'activation': 'relu'
        },
        'decoder': {
            'output_shape': [3, 64, 64],
            'hidden_dims': [256, 128, 64, 32]
        },
        'rssm': {
            'stochastic_size': 32,
            'deterministic_size': 200,
            'hidden_size': 200,
            'activation': 'elu',
            'free_nats': 3.0,
            'kl_balance_scale': 0.8,
            'imagination_horizon': 15
        },
        'actor': {
            'hidden_dims': [400, 400]
        },
        'reward_value': {
            'reward_hidden_dims': [400, 400],
            'value_hidden_dims': [400, 400],
            'gamma': 0.99,
            'lambda_': 0.95
        },
        'training': {
            'model_lr': 6e-4,
            'actor_lr': 8e-5,
            'value_lr': 8e-5,
            'adam_eps': 1e-5,
            'grad_clip_norm': 100.0,
            'reconstruction_weight': 1.0,
            'kl_weight': 1.0,
            'reward_weight': 1.0
        }
    }
    
    device = torch.device('cpu')
    
    # Create world model
    world_model = WorldModel(config).to(device)
    world_model.initialize_dynamics(action_dim=6)
    
    # Create trainer
    trainer = WorldModelTrainer(world_model, config, device)
    
    # Create dummy batch
    B, L = 4, 50
    batch = {
        'observations': torch.randint(0, 256, (B, L, 3, 64, 64)).float(),
        'actions': torch.randn(B, L, 6),
        'rewards': torch.randn(B, L),
        'dones': torch.zeros(B, L)
    }
    
    print("\nTraining dynamics...")
    dynamics_losses = trainer.train_dynamics(batch)
    for key, value in dynamics_losses.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nTraining behavior...")
    behavior_losses = trainer.train_behavior(batch)
    for key, value in behavior_losses.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ World Model Trainer Test Passed!")
    print("  系統現在可以：")
    print("  1. 學習物理動力學 (Dynamics)")
    print("  2. 在想像中規劃 (Behavior)")

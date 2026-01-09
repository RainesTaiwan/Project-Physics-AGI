"""
Module C & D: Actor-Critic Models
獎勵/價值評估器 + 動作控制器

這些模組定義了「什麼是好的」以及「如何做」
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from typing import Tuple, Dict


class DenseModel(nn.Module):
    """
    通用的全連接網絡
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [400, 400],
        activation: str = "elu",
        output_activation: str = None
    ):
        super().__init__()
        
        act_fn = nn.ELU if activation == "elu" else nn.ReLU
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(hidden_dims)):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                act_fn()
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class RewardModel(nn.Module):
    """
    Module C: Reward Model (獎勵模型)
    
    目的：預測即時獎勵 r_t = R(s_t)
    
    輸入: 潛在狀態 (h_t, z_t)
    輸出: 即時獎勵 r_t (標量)
    """
    
    def __init__(
        self,
        stochastic_size: int = 32,
        deterministic_size: int = 200,
        hidden_dims: list = [400, 400],
        activation: str = "elu"
    ):
        super().__init__()
        
        input_dim = stochastic_size + deterministic_size
        
        self.model = DenseModel(
            input_dim=input_dim,
            output_dim=1,  # Scalar reward
            hidden_dims=hidden_dims,
            activation=activation
        )
    
    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict reward from latent state
        
        Args:
            state: {'h': [B, D], 'z': [B, S]} or
                   {'h': [B, T, D], 'z': [B, T, S]} for sequences
                   
        Returns:
            reward: [B] or [B, T] predicted rewards
        """
        h = state['h']
        z = state['z']
        
        # Concatenate state components
        if h.dim() == 3:  # Sequence
            latent_state = torch.cat([h, z], dim=-1)
        else:  # Single step
            latent_state = torch.cat([h, z], dim=-1)
        
        reward = self.model(latent_state).squeeze(-1)
        return reward


class ValueModel(nn.Module):
    """
    Module C: Value Model (價值模型 / Critic)
    
    目的：評估狀態的長期價值 V(s_t) = E[Σ γ^k r_{t+k}]
    
    輸入: 潛在狀態 (h_t, z_t)
    輸出: 狀態價值 V(s_t)
    """
    
    def __init__(
        self,
        stochastic_size: int = 32,
        deterministic_size: int = 200,
        hidden_dims: list = [400, 400],
        activation: str = "elu"
    ):
        super().__init__()
        
        input_dim = stochastic_size + deterministic_size
        
        self.model = DenseModel(
            input_dim=input_dim,
            output_dim=1,  # Scalar value
            hidden_dims=hidden_dims,
            activation=activation
        )
    
    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict value from latent state
        
        Args:
            state: {'h': [B, D], 'z': [B, S]} or sequences
                   
        Returns:
            value: [B] or [B, T] predicted values
        """
        h = state['h']
        z = state['z']
        
        if h.dim() == 3:
            latent_state = torch.cat([h, z], dim=-1)
        else:
            latent_state = torch.cat([h, z], dim=-1)
        
        value = self.model(latent_state).squeeze(-1)
        return value


class Actor(nn.Module):
    """
    Module D: Actor (動作控制器 / Policy)
    
    目的：輸出動作策略 π(a|s)
    
    輸入: 潛在狀態 (h_t, z_t)
    輸出: 動作分佈 (連續控制用 Tanh Normal)
    """
    
    def __init__(
        self,
        action_dim: int,
        stochastic_size: int = 32,
        deterministic_size: int = 200,
        hidden_dims: list = [400, 400],
        activation: str = "elu",
        init_std: float = 5.0,
        min_std: float = 0.1,
        max_std: float = 10.0
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        
        input_dim = stochastic_size + deterministic_size
        
        # Shared trunk
        act_fn = nn.ELU if activation == "elu" else nn.ReLU
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(hidden_dims)):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                act_fn()
            ])
        
        self.trunk = nn.Sequential(*layers)
        
        # Action mean head
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # Action std head (learnable)
        self.std_head = nn.Linear(hidden_dims[-1], action_dim)
    
    def forward(
        self,
        state: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.distributions.Distribution]:
        """
        Compute action distribution and sample action
        
        Args:
            state: {'h': [B, D], 'z': [B, S]} or sequences
            deterministic: if True, return mean action (no sampling)
            
        Returns:
            action: [B, A] or [B, T, A] sampled/mean action
            dist: action distribution object
        """
        h = state['h']
        z = state['z']
        
        if h.dim() == 3:
            latent_state = torch.cat([h, z], dim=-1)
        else:
            latent_state = torch.cat([h, z], dim=-1)
        
        # Pass through trunk
        features = self.trunk(latent_state)
        
        # Compute mean and std
        mean = self.mean_head(features)
        std_logit = self.std_head(features)
        
        # Transform std to valid range using softplus
        std = F.softplus(std_logit) + self.min_std
        std = torch.clamp(std, self.min_std, self.max_std)
        
        # Create Tanh Normal distribution
        dist = self._create_tanh_normal(mean, std)
        
        if deterministic:
            # Use mean action (no exploration)
            action = torch.tanh(mean)
        else:
            # Sample from distribution
            action = dist.rsample()
        
        return action, dist
    
    def _create_tanh_normal(
        self,
        mean: torch.Tensor,
        std: torch.Tensor
    ) -> torch.distributions.Distribution:
        """
        Create Tanh Normal distribution
        
        This is crucial for continuous control:
        - Normal distribution in latent space
        - Tanh squashing to [-1, 1] action space
        """
        # Independent Normal (no correlation between action dimensions)
        normal_dist = Independent(Normal(mean, std), 1)
        
        return normal_dist
    
    def log_prob(
        self,
        dist: torch.distributions.Distribution,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of action under distribution
        
        Need to account for Tanh squashing:
        log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u))
        where u = atanh(a)
        """
        # Inverse tanh (atanh) with numerical stability
        action_clamped = torch.clamp(action, -0.999, 0.999)
        pre_tanh_action = torch.atanh(action_clamped)
        
        # Log prob in pre-tanh space
        log_prob = dist.log_prob(pre_tanh_action)
        
        # Correction for tanh squashing
        log_det_jacobian = torch.sum(
            torch.log(1 - action ** 2 + 1e-6),
            dim=-1
        )
        
        log_prob = log_prob - log_det_jacobian
        
        return log_prob


class ActorCritic(nn.Module):
    """
    完整的 Actor-Critic 系統
    結合 Actor, Critic (Value), 和 Reward Model
    """
    
    def __init__(
        self,
        action_dim: int,
        stochastic_size: int = 32,
        deterministic_size: int = 200,
        actor_hidden_dims: list = [400, 400],
        value_hidden_dims: list = [400, 400],
        reward_hidden_dims: list = [400, 400]
    ):
        super().__init__()
        
        self.actor = Actor(
            action_dim=action_dim,
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            hidden_dims=actor_hidden_dims
        )
        
        self.value = ValueModel(
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            hidden_dims=value_hidden_dims
        )
        
        self.reward = RewardModel(
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            hidden_dims=reward_hidden_dims
        )
    
    def act(
        self,
        state: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> torch.Tensor:
        """Simple action selection interface"""
        action, _ = self.actor(state, deterministic)
        return action
    
    def evaluate(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate state (for training)"""
        value = self.value(state)
        reward = self.reward(state)
        
        return {
            'value': value,
            'reward': reward
        }


if __name__ == "__main__":
    print("Testing Actor-Critic Models...")
    
    # Setup
    batch_size = 4
    sequence_length = 10
    action_dim = 6
    stochastic_size = 32
    deterministic_size = 200
    
    # Create dummy state
    state = {
        'h': torch.randn(batch_size, deterministic_size),
        'z': torch.randn(batch_size, stochastic_size)
    }
    
    # Test Reward Model
    print("\n[Test 1] Reward Model...")
    reward_model = RewardModel(stochastic_size, deterministic_size)
    predicted_reward = reward_model(state)
    print(f"  Predicted reward shape: {predicted_reward.shape}")
    print(f"  Sample reward: {predicted_reward[0].item():.4f}")
    
    # Test Value Model
    print("\n[Test 2] Value Model (Critic)...")
    value_model = ValueModel(stochastic_size, deterministic_size)
    predicted_value = value_model(state)
    print(f"  Predicted value shape: {predicted_value.shape}")
    print(f"  Sample value: {predicted_value[0].item():.4f}")
    
    # Test Actor
    print("\n[Test 3] Actor (Policy)...")
    actor = Actor(action_dim, stochastic_size, deterministic_size)
    
    # Stochastic action
    action_stochastic, dist = actor(state, deterministic=False)
    print(f"  Stochastic action shape: {action_stochastic.shape}")
    print(f"  Action range: [{action_stochastic.min():.2f}, {action_stochastic.max():.2f}]")
    
    # Deterministic action
    action_deterministic, _ = actor(state, deterministic=True)
    print(f"  Deterministic action shape: {action_deterministic.shape}")
    
    # Log probability
    log_prob = actor.log_prob(dist, action_stochastic)
    print(f"  Log prob shape: {log_prob.shape}")
    
    # Test with sequence
    print("\n[Test 4] Sequence processing...")
    state_seq = {
        'h': torch.randn(batch_size, sequence_length, deterministic_size),
        'z': torch.randn(batch_size, sequence_length, stochastic_size)
    }
    
    reward_seq = reward_model(state_seq)
    value_seq = value_model(state_seq)
    action_seq, _ = actor(state_seq)
    
    print(f"  Reward sequence: {reward_seq.shape}")
    print(f"  Value sequence: {value_seq.shape}")
    print(f"  Action sequence: {action_seq.shape}")
    
    # Test full ActorCritic
    print("\n[Test 5] Complete Actor-Critic System...")
    actor_critic = ActorCritic(
        action_dim=action_dim,
        stochastic_size=stochastic_size,
        deterministic_size=deterministic_size
    )
    
    action = actor_critic.act(state)
    evaluation = actor_critic.evaluate(state)
    
    print(f"  Action: {action.shape}")
    print(f"  Evaluation keys: {evaluation.keys()}")
    print(f"  Value: {evaluation['value'].shape}")
    print(f"  Reward: {evaluation['reward'].shape}")
    
    print("\n✓ Module C & D: Actor-Critic Test Passed!")
    print("  這些模組定義了：")
    print("  1. 什麼是好的狀態 (Value)")
    print("  2. 能獲得多少獎勵 (Reward)")
    print("  3. 應該採取什麼動作 (Actor)")

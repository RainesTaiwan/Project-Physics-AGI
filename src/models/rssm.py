"""
Module B: Recurrent State Space Model (RSSM)
循環狀態空間模型 - 系統的「物理引擎」

這是整個 World Model 的核心，學習環境的動力學方程 f(s_t, a_t) = s_{t+1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class RSSM(nn.Module):
    """
    Recurrent State Space Model (循環狀態空間模型)
    
    核心思想：將狀態分解為兩部分
    1. h_t: 確定性記憶 (Deterministic Hidden State) - RNN 狀態
    2. z_t: 隨機潛在變量 (Stochastic Latent Variable) - 從觀測推斷
    
    雙路徑設計：
    - Posterior Path (後驗): 根據真實觀測更新狀態 (用於訓練)
    - Prior Path (先驗): 純粹根據歷史預測未來 (用於想像)
    
    當 KL(Prior || Posterior) → 0 時，AI 理解了物理法則
    """
    
    def __init__(
        self,
        action_dim: int,
        stochastic_size: int = 32,
        deterministic_size: int = 200,
        hidden_size: int = 200,
        activation: str = "elu",
        free_nats: float = 3.0,
        kl_balance_scale: float = 0.8
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.hidden_size = hidden_size
        self.free_nats = free_nats
        self.kl_balance_scale = kl_balance_scale
        
        act_fn = nn.ELU if activation == "elu" else nn.ReLU
        
        # ===== Recurrent Core (GRU) =====
        # Input: [z_{t-1}, a_{t-1}]
        # Output: h_t (deterministic hidden state)
        self.rnn = nn.GRUCell(
            input_size=stochastic_size + action_dim,
            hidden_size=deterministic_size
        )
        
        # ===== Prior Network (想像路徑) =====
        # Predicts p(z_t | h_t) - 只用歷史，不用觀測
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_size, hidden_size),
            act_fn(),
            nn.Linear(hidden_size, hidden_size),
            act_fn()
        )
        self.prior_mean = nn.Linear(hidden_size, stochastic_size)
        self.prior_logstd = nn.Linear(hidden_size, stochastic_size)
        
        # ===== Posterior Network (真實路徑) =====
        # Predicts q(z_t | h_t, o_t) - 結合歷史和觀測
        self.posterior_net = nn.Sequential(
            nn.Linear(deterministic_size + hidden_size, hidden_size),
            act_fn(),
            nn.Linear(hidden_size, hidden_size),
            act_fn()
        )
        self.posterior_mean = nn.Linear(hidden_size, stochastic_size)
        self.posterior_logstd = nn.Linear(hidden_size, stochastic_size)
        
        # ===== Observation Embedding =====
        # 將編碼器的輸出投影到統一空間
        self.obs_embed = nn.Sequential(
            nn.Linear(stochastic_size, hidden_size),
            act_fn()
        )
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Initialize RSSM state at t=0
        
        Returns:
            state_dict with keys:
                - 'h': [batch_size, deterministic_size] deterministic state
                - 'z': [batch_size, stochastic_size] stochastic state
        """
        return {
            'h': torch.zeros(batch_size, self.deterministic_size, device=device),
            'z': torch.zeros(batch_size, self.stochastic_size, device=device)
        }
    
    def imagine_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Prior Path (想像路徑): 純粹根據動力學推演
        用於規劃和策略學習
        
        Args:
            prev_state: previous state {'h': h_{t-1}, 'z': z_{t-1}}
            action: a_{t-1}
            
        Returns:
            next_state: predicted state {'h': h_t, 'z': z_t}
            mean: predicted mean of z_t
            std: predicted std of z_t
        """
        prev_h = prev_state['h']
        prev_z = prev_state['z']
        
        # Step 1: Update deterministic state via RNN
        # h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        rnn_input = torch.cat([prev_z, action], dim=-1)
        h = self.rnn(rnn_input, prev_h)
        
        # Step 2: Predict prior distribution p(z_t | h_t)
        prior_features = self.prior_net(h)
        prior_mean = self.prior_mean(prior_features)
        prior_logstd = self.prior_logstd(prior_features)
        prior_logstd = torch.clamp(prior_logstd, -10, 2)
        prior_std = torch.exp(prior_logstd)
        
        # Step 3: Sample z_t from prior
        z = self.reparameterize(prior_mean, prior_std)
        
        next_state = {'h': h, 'z': z}
        return next_state, prior_mean, prior_std
    
    def observe_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        observation_latent: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Posterior Path (觀測路徑): 根據真實觀測更新
        用於訓練和狀態估計
        
        Args:
            prev_state: previous state {'h': h_{t-1}, 'z': z_{t-1}}
            action: a_{t-1}
            observation_latent: z^obs_t from encoder
            
        Returns:
            next_state: updated state {'h': h_t, 'z': z_t}
            prior_dist: {'mean': μ_prior, 'std': σ_prior}
            posterior_dist: {'mean': μ_post, 'std': σ_post}
        """
        prev_h = prev_state['h']
        prev_z = prev_state['z']
        
        # Step 1: Update deterministic state
        rnn_input = torch.cat([prev_z, action], dim=-1)
        h = self.rnn(rnn_input, prev_h)
        
        # Step 2: Compute prior p(z_t | h_t) (想像)
        prior_features = self.prior_net(h)
        prior_mean = self.prior_mean(prior_features)
        prior_logstd = self.prior_logstd(prior_features)
        prior_logstd = torch.clamp(prior_logstd, -10, 2)
        prior_std = torch.exp(prior_logstd)
        
        # Step 3: Embed observation
        obs_features = self.obs_embed(observation_latent)
        
        # Step 4: Compute posterior q(z_t | h_t, o_t) (真實)
        posterior_input = torch.cat([h, obs_features], dim=-1)
        posterior_features = self.posterior_net(posterior_input)
        posterior_mean = self.posterior_mean(posterior_features)
        posterior_logstd = self.posterior_logstd(posterior_features)
        posterior_logstd = torch.clamp(posterior_logstd, -10, 2)
        posterior_std = torch.exp(posterior_logstd)
        
        # Step 5: Sample from posterior (during training)
        z = self.reparameterize(posterior_mean, posterior_std)
        
        next_state = {'h': h, 'z': z}
        prior_dist = {'mean': prior_mean, 'std': prior_std}
        posterior_dist = {'mean': posterior_mean, 'std': posterior_std}
        
        return next_state, prior_dist, posterior_dist
    
    def imagine_rollout(
        self,
        initial_state: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        horizon: int
    ) -> Dict[str, torch.Tensor]:
        """
        展開想像鏈 (Imagination Rollout)
        
        這是 AI「做夢」的過程：
        給定初始狀態和動作序列，預測未來 H 步的狀態軌跡
        
        Args:
            initial_state: {'h': [B, D], 'z': [B, S]}
            actions: [B, H, A] action sequence
            horizon: H (number of steps to imagine)
            
        Returns:
            trajectory: {
                'h': [B, H, D] deterministic states
                'z': [B, H, S] stochastic states
                'mean': [B, H, S] predicted means
                'std': [B, H, S] predicted stds
            }
        """
        batch_size = initial_state['h'].size(0)
        device = initial_state['h'].device
        
        # Storage
        h_sequence = []
        z_sequence = []
        mean_sequence = []
        std_sequence = []
        
        state = initial_state
        
        for t in range(horizon):
            action = actions[:, t] if actions.dim() == 3 else actions
            state, mean, std = self.imagine_step(state, action)
            
            h_sequence.append(state['h'])
            z_sequence.append(state['z'])
            mean_sequence.append(mean)
            std_sequence.append(std)
        
        trajectory = {
            'h': torch.stack(h_sequence, dim=1),  # [B, H, D]
            'z': torch.stack(z_sequence, dim=1),   # [B, H, S]
            'mean': torch.stack(mean_sequence, dim=1),
            'std': torch.stack(std_sequence, dim=1)
        }
        
        return trajectory
    
    def reparameterize(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        eps = torch.randn_like(std)
        return mean + std * eps
    
    def kl_loss(
        self,
        prior_dist: Dict[str, torch.Tensor],
        posterior_dist: Dict[str, torch.Tensor],
        use_balancing: bool = True
    ) -> torch.Tensor:
        """
        計算 Prior 和 Posterior 之間的 KL 散度
        
        KL(Posterior || Prior) 衡量「想像」和「現實」的差距
        當這個值接近 0 時，意味著 AI 能準確預測物理現象
        
        Args:
            prior_dist: p(z_t | h_t)
            posterior_dist: q(z_t | h_t, o_t)
            use_balancing: use KL balancing trick from Dreamer
            
        Returns:
            kl_loss: scalar
        """
        prior_mean = prior_dist['mean']
        prior_std = prior_dist['std']
        post_mean = posterior_dist['mean']
        post_std = posterior_dist['std']
        
        if use_balancing:
            # KL Balancing: mix of forward and reverse KL
            # This prevents posterior collapse
            alpha = self.kl_balance_scale
            
            # Forward KL: KL(post || prior)
            kl_forward = self._gaussian_kl(post_mean, post_std, prior_mean, prior_std)
            
            # Reverse KL: KL(prior || post)  
            kl_reverse = self._gaussian_kl(
                prior_mean.detach(), prior_std.detach(),
                post_mean, post_std
            )
            
            kl = alpha * kl_forward + (1 - alpha) * kl_reverse
        else:
            kl = self._gaussian_kl(post_mean, post_std, prior_mean, prior_std)
        
        # Apply free nats
        kl = kl.sum(dim=-1)  # Sum over latent dimensions
        kl = torch.maximum(kl, torch.tensor(self.free_nats).to(kl.device))
        
        return kl.mean()
    
    def _gaussian_kl(
        self,
        mean1: torch.Tensor,
        std1: torch.Tensor,
        mean2: torch.Tensor,
        std2: torch.Tensor
    ) -> torch.Tensor:
        """
        KL divergence between two Gaussian distributions
        KL(N(μ1, σ1²) || N(μ2, σ2²))
        """
        var1 = std1 ** 2
        var2 = std2 ** 2
        
        kl = torch.log(std2 / (std1 + 1e-8) + 1e-8) + \
             (var1 + (mean1 - mean2) ** 2) / (2 * var2 + 1e-8) - 0.5
        
        return kl


if __name__ == "__main__":
    print("Testing RSSM (Recurrent State Space Model)...")
    
    # Hyperparameters
    batch_size = 4
    sequence_length = 10
    action_dim = 6
    stochastic_size = 32
    deterministic_size = 200
    device = torch.device("cpu")
    
    # Initialize RSSM
    rssm = RSSM(
        action_dim=action_dim,
        stochastic_size=stochastic_size,
        deterministic_size=deterministic_size
    )
    
    # Test 1: Initial state
    print("\n[Test 1] Initial state generation...")
    state = rssm.initial_state(batch_size, device)
    print(f"  h shape: {state['h'].shape}")
    print(f"  z shape: {state['z'].shape}")
    
    # Test 2: Imagine step (prior path)
    print("\n[Test 2] Imagine step (prior prediction)...")
    action = torch.randn(batch_size, action_dim)
    next_state, prior_mean, prior_std = rssm.imagine_step(state, action)
    print(f"  Next h shape: {next_state['h'].shape}")
    print(f"  Next z shape: {next_state['z'].shape}")
    print(f"  Prior mean shape: {prior_mean.shape}")
    
    # Test 3: Observe step (posterior path)
    print("\n[Test 3] Observe step (with real observation)...")
    obs_latent = torch.randn(batch_size, stochastic_size)
    next_state, prior_dist, post_dist = rssm.observe_step(state, action, obs_latent)
    print(f"  Updated state h: {next_state['h'].shape}")
    print(f"  Updated state z: {next_state['z'].shape}")
    
    # Test 4: KL divergence
    print("\n[Test 4] KL divergence (想像 vs 現實)...")
    kl_loss = rssm.kl_loss(prior_dist, post_dist)
    print(f"  KL divergence: {kl_loss.item():.4f}")
    print(f"  → 當這個值趨近0時，AI理解了物理法則")
    
    # Test 5: Imagine rollout (dreaming)
    print("\n[Test 5] Imagine rollout (做夢 - 預測未來15步)...")
    horizon = 15
    actions = torch.randn(batch_size, horizon, action_dim)
    trajectory = rssm.imagine_rollout(state, actions, horizon)
    print(f"  Trajectory h: {trajectory['h'].shape}")
    print(f"  Trajectory z: {trajectory['z'].shape}")
    print(f"  → AI 在腦海中模擬了未來 {horizon} 步的物理演化")
    
    print("\n✓ Module B: RSSM Test Passed!")
    print("  這個模組是整個系統的大腦，能夠：")
    print("  1. 記憶歷史 (h_t via GRU)")
    print("  2. 預測未來 (Prior path)")
    print("  3. 校準現實 (Posterior path)")
    print("  4. 想像規劃 (Rollout)")

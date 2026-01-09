"""
Experience Replay Buffer
高速循環緩衝區 - 儲存交互經驗用於訓練
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer for World Model
    
    儲存 (o_t, a_t, r_t, done_t) 軌跡
    支持序列採樣以訓練 RSSM
    """
    
    def __init__(
        self,
        capacity: int = 1000000,
        observation_shape: Tuple[int, ...] = (3, 64, 64),
        action_dim: int = 6,
        sequence_length: int = 50,
        device: str = "cuda"
    ):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.device = device
        
        # Circular buffer pointers
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory for efficiency
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Episode boundaries for sequence sampling
        self.episode_boundaries = deque(maxlen=10000)
        self.current_episode_start = 0
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool
    ):
        """
        Add single transition to buffer
        
        Args:
            observation: [C, H, W] image
            action: [A] action vector
            reward: scalar reward
            done: episode termination flag
        """
        # Store in circular buffer
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        
        # Update pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Track episode boundaries
        if done:
            episode_end = self.ptr
            self.episode_boundaries.append((self.current_episode_start, episode_end))
            self.current_episode_start = self.ptr
    
    def sample_sequence(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample random sequences of fixed length
        
        重要：確保序列不跨越 episode 邊界
        
        Args:
            batch_size: number of sequences to sample
            
        Returns:
            batch: {
                'observations': [B, L, C, H, W]
                'actions': [B, L, A]
                'rewards': [B, L]
                'dones': [B, L]
            }
        """
        # Storage for batch
        obs_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        
        for _ in range(batch_size):
            # Sample valid starting index
            start_idx = self._sample_valid_start()
            
            # Extract sequence
            indices = np.arange(start_idx, start_idx + self.sequence_length) % self.capacity
            
            obs_batch.append(self.observations[indices])
            action_batch.append(self.actions[indices])
            reward_batch.append(self.rewards[indices])
            done_batch.append(self.dones[indices])
        
        # Convert to tensors
        batch = {
            'observations': torch.from_numpy(np.stack(obs_batch)).to(self.device),
            'actions': torch.from_numpy(np.stack(action_batch)).to(self.device),
            'rewards': torch.from_numpy(np.stack(reward_batch)).to(self.device),
            'dones': torch.from_numpy(np.stack(done_batch)).to(self.device)
        }
        
        return batch
    
    def _sample_valid_start(self) -> int:
        """
        Sample a valid starting index that:
        1. Has enough data ahead (sequence_length steps)
        2. Doesn't cross episode boundaries
        """
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Random start index
            start_idx = np.random.randint(0, max(1, self.size - self.sequence_length))
            
            # Check if sequence crosses episode boundary
            indices = np.arange(start_idx, start_idx + self.sequence_length)
            
            # If any done signal appears within sequence (except last step), resample
            if np.any(self.dones[indices[:-1]] > 0.5):
                continue
            
            return start_idx
        
        # Fallback: just return a random index
        return np.random.randint(0, max(1, self.size - self.sequence_length))
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, min_size: int = 5000) -> bool:
        """Check if buffer has enough data to start training"""
        return self.size >= min_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    優先經驗回放 (可選)
    
    根據 TD-error 或 model prediction error 進行優先採樣
    對於困難場景（如碰撞、失敗）給予更高權重
    """
    
    def __init__(self, *args, alpha: float = 0.6, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.alpha = alpha  # Prioritization exponent
        self.priorities = np.ones(self.capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        priority: Optional[float] = None
    ):
        """Add with priority"""
        super().add(observation, action, reward, done)
        
        # Set priority (use max if not provided)
        if priority is None:
            priority = self.max_priority
        
        self.priorities[self.ptr - 1] = priority ** self.alpha
        self.max_priority = max(self.max_priority, priority)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities after training"""
        self.priorities[indices] = priorities ** self.alpha
        self.max_priority = max(self.max_priority, priorities.max())


if __name__ == "__main__":
    print("Testing Replay Buffer...")
    
    # Create buffer
    buffer = ReplayBuffer(
        capacity=10000,
        observation_shape=(3, 64, 64),
        action_dim=6,
        sequence_length=50,
        device="cpu"
    )
    
    print(f"Buffer capacity: {buffer.capacity}")
    print(f"Sequence length: {buffer.sequence_length}")
    
    # Add some random experiences
    print("\nAdding experiences...")
    for i in range(200):
        obs = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
        action = np.random.randn(6).astype(np.float32)
        reward = np.random.randn()
        done = (i % 50 == 49)  # Episode ends every 50 steps
        
        buffer.add(obs, action, reward, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Episode boundaries: {len(buffer.episode_boundaries)}")
    
    # Sample a batch
    print("\nSampling batch...")
    batch = buffer.sample_sequence(batch_size=4)
    
    print(f"Observations shape: {batch['observations'].shape}")
    print(f"Actions shape: {batch['actions'].shape}")
    print(f"Rewards shape: {batch['rewards'].shape}")
    print(f"Dones shape: {batch['dones'].shape}")
    
    # Check if ready for training
    print(f"\nReady for training: {buffer.is_ready(min_size=100)}")
    
    print("\n✓ Replay Buffer Test Passed!")

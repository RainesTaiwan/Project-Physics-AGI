"""
Environment Wrapper
環境包裝器 - 統一不同模擬器的接口
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict
import gymnasium as gym


class DMCWrapper:
    """
    DeepMind Control Suite 包裝器
    支持 MuJoCo 物理引擎
    """
    
    def __init__(
        self,
        domain_name: str = "walker",
        task_name: str = "walk",
        image_size: int = 64,
        action_repeat: int = 2,
        frame_stack: int = 1,
        seed: Optional[int] = None
    ):
        # Set up headless rendering
        import os
        os.environ.setdefault('MUJOCO_GL', 'egl')
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        
        try:
            from dm_control import suite
            from dm_control.suite.wrappers import pixels
        except ImportError:
            raise ImportError(
                "DeepMind Control Suite not installed. "
                "Install with: pip install dm-control"
            )
        
        self.domain_name = domain_name
        self.task_name = task_name
        self.action_repeat = action_repeat
        self.image_size = image_size
        
        # Create environment
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={'random': seed}
        )
        
        # Wrap to get pixel observations
        self._env = pixels.Wrapper(
            self._env,
            pixels_only=True,
            render_kwargs={'height': image_size, 'width': image_size, 'camera_id': 0}
        )
        
        # Get action spec
        action_spec = self._env.action_spec()
        self.action_dim = action_spec.shape[0]
        self.action_min = action_spec.minimum
        self.action_max = action_spec.maximum
        
        # Observation space
        self.observation_shape = (3, image_size, image_size)
        
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation"""
        time_step = self._env.reset()
        obs = self._process_observation(time_step.observation['pixels'])
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action with action repeat
        
        Returns:
            observation: [C, H, W]
            reward: scalar
            done: bool
            info: dict
        """
        total_reward = 0.0
        
        # Action repeat for frame skip
        for _ in range(self.action_repeat):
            time_step = self._env.step(action)
            total_reward += time_step.reward or 0.0
            
            if time_step.last():
                break
        
        obs = self._process_observation(time_step.observation['pixels'])
        done = time_step.last()
        info = {}
        
        return obs, total_reward, done, info
    
    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Process observation from [H, W, C] to [C, H, W]
        Normalize to [0, 255] uint8
        """
        # DMC returns [H, W, C] in [0, 255]
        obs = obs.transpose(2, 0, 1)  # [C, H, W]
        return obs.astype(np.uint8)
    
    def close(self):
        """Close environment"""
        self._env.close()


class GymWrapper:
    """
    OpenAI Gym / Gymnasium 包裝器
    """
    
    def __init__(
        self,
        env_name: str = "HalfCheetah-v4",
        image_size: int = 64,
        action_repeat: int = 2,
        render_mode: str = "rgb_array"
    ):
        self.env_name = env_name
        self.action_repeat = action_repeat
        self.image_size = image_size
        
        # Create environment
        self._env = gym.make(env_name, render_mode=render_mode)
        
        # Get action dimension
        if isinstance(self._env.action_space, gym.spaces.Box):
            self.action_dim = self._env.action_space.shape[0]
            self.action_min = self._env.action_space.low
            self.action_max = self._env.action_space.high
        else:
            raise ValueError("Only continuous action spaces supported")
        
        self.observation_shape = (3, image_size, image_size)
    
    def reset(self) -> np.ndarray:
        """Reset and return visual observation"""
        state, info = self._env.reset()
        obs = self._render_observation()
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action with repeat"""
        total_reward = 0.0
        
        for _ in range(self.action_repeat):
            state, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        obs = self._render_observation()
        done = terminated or truncated
        
        return obs, total_reward, done, info
    
    def _render_observation(self) -> np.ndarray:
        """Render environment to RGB image"""
        frame = self._env.render()
        
        # Resize if needed
        if frame.shape[:2] != (self.image_size, self.image_size):
            from PIL import Image
            frame = np.array(
                Image.fromarray(frame).resize(
                    (self.image_size, self.image_size)
                )
            )
        
        # [H, W, C] -> [C, H, W]
        frame = frame.transpose(2, 0, 1)
        return frame.astype(np.uint8)
    
    def close(self):
        self._env.close()


def make_env(config: Dict) -> DMCWrapper:
    """
    Factory function to create environment based on config
    
    Args:
        config: environment configuration dict
        
    Returns:
        env: wrapped environment
    """
    backend = config.get('backend', 'mujoco')
    env_name = config.get('name', 'DMC-walker-walk')
    
    if env_name.startswith('DMC-'):
        # DeepMind Control Suite
        parts = env_name.replace('DMC-', '').split('-')
        domain_name = parts[0]
        task_name = parts[1] if len(parts) > 1 else 'walk'
        
        env = DMCWrapper(
            domain_name=domain_name,
            task_name=task_name,
            image_size=config.get('image_size', 64),
            action_repeat=config.get('action_repeat', 2),
            frame_stack=config.get('frame_stack', 1),
            seed=config.get('seed', None)
        )
    else:
        # Gymnasium
        env = GymWrapper(
            env_name=env_name,
            image_size=config.get('image_size', 64),
            action_repeat=config.get('action_repeat', 2)
        )
    
    return env


if __name__ == "__main__":
    print("Testing Environment Wrappers...")
    
    # Test configuration
    config = {
        'name': 'DMC-cartpole-swingup',
        'backend': 'mujoco',
        'image_size': 64,
        'action_repeat': 2,
        'seed': 42
    }
    
    try:
        print(f"\nCreating environment: {config['name']}")
        env = make_env(config)
        
        print(f"Action dimension: {env.action_dim}")
        print(f"Observation shape: {env.observation_shape}")
        
        # Test reset
        print("\nTesting reset...")
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Observation dtype: {obs.dtype}")
        print(f"Observation range: [{obs.min()}, {obs.max()}]")
        
        # Test step
        print("\nTesting step...")
        action = np.random.uniform(
            env.action_min,
            env.action_max,
            size=env.action_dim
        )
        obs, reward, done, info = env.step(action)
        
        print(f"Next observation shape: {obs.shape}")
        print(f"Reward: {reward:.4f}")
        print(f"Done: {done}")
        
        env.close()
        print("\n✓ Environment Wrapper Test Passed!")
        
    except ImportError as e:
        print(f"\n⚠ Warning: {e}")
        print("  To test with real environments, install:")
        print("  pip install dm-control mujoco")

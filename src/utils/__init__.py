"""
Utilities package
"""

__all__ = [
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'DMCWrapper',
    'GymWrapper',
    'make_env'
]

def __getattr__(name):
    """Lazy import to avoid loading environment dependencies"""
    if name in ['ReplayBuffer', 'PrioritizedReplayBuffer']:
        from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
        return ReplayBuffer if name == 'ReplayBuffer' else PrioritizedReplayBuffer
    elif name in ['DMCWrapper', 'GymWrapper', 'make_env']:
        from .env_wrapper import DMCWrapper, GymWrapper, make_env
        if name == 'DMCWrapper':
            return DMCWrapper
        elif name == 'GymWrapper':
            return GymWrapper
        else:
            return make_env
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

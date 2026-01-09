"""
Physics-AGI Package Initialization
"""

__version__ = "1.0.0"
__author__ = "Project Physics-AGI Team"

# Lazy imports to avoid dependency errors during testing
__all__ = [
    'VariationalEncoder',
    'CNNDecoder',
    'RSSM',
    'Actor',
    'ValueModel',
    'RewardModel',
    'ActorCritic',
    'WorldModel',
    'WorldModelTrainer',
    'ReplayBuffer',
    'make_env'
]

def __getattr__(name):
    """Lazy import to avoid loading all dependencies at once"""
    if name in ['VariationalEncoder', 'CNNDecoder']:
        from src.models.encoder import VariationalEncoder, CNNDecoder
        return VariationalEncoder if name == 'VariationalEncoder' else CNNDecoder
    elif name == 'RSSM':
        from src.models.rssm import RSSM
        return RSSM
    elif name in ['Actor', 'ValueModel', 'RewardModel', 'ActorCritic']:
        from src.models.actor_critic import Actor, ValueModel, RewardModel, ActorCritic
        if name == 'Actor':
            return Actor
        elif name == 'ValueModel':
            return ValueModel
        elif name == 'RewardModel':
            return RewardModel
        else:
            return ActorCritic
    elif name in ['WorldModel', 'WorldModelTrainer']:
        from src.trainer import WorldModel, WorldModelTrainer
        return WorldModel if name == 'WorldModel' else WorldModelTrainer
    elif name == 'ReplayBuffer':
        from src.utils.replay_buffer import ReplayBuffer
        return ReplayBuffer
    elif name == 'make_env':
        from src.utils.env_wrapper import make_env
        return make_env
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

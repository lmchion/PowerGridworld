from gridworld.agents.pv import PVEnv
import numpy as np
import gym


class HSPVEnv(PVEnv):


    def __init__(
        self,
        name: str,
        profile_csv: str,
        profile_path: str = None,
        scaling_factor: float = 1.,
        rescale_spaces: bool = True,
        grid_aware: bool = False,
        max_episode_steps: int = None,
        **kwargs
    ):

        super().__init__(name=name, 
                         profile_csv=profile_csv, 
                         profile_path=profile_path,
                         scaling_factor=scaling_factor,
                         rescale_spaces=rescale_spaces,
                         grid_aware=grid_aware,
                         max_episode_steps=max_episode_steps,
                         **kwargs  )
        
        
        self._action_space = gym.spaces.Box(
            shape=(1,), low=0., high=1., dtype=np.float64)
        

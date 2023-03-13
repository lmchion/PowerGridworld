
import numpy as np

from gridworld.agents.pv import PVEnv
from gridworld.utils import to_raw


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
        super().__init__(name, profile_csv, profile_path, scaling_factor, rescale_spaces, grid_aware, max_episode_steps, **kwargs)


    def get_obs(self, **kwargs):
    
        obs, meta = super().get_obs(**kwargs)
        
        meta['pv_power']=meta['real_power']
        kwargs.update(meta)
        
        return obs,kwargs
    
    def reset(self, **kwargs):
        """Resetting consists of simply putting the index back to 0."""
        self.index = 0
        return self.get_obs(**kwargs)
    
    def is_terminal(self):
        """The episode is done when the end of the data is reached."""
        return self.index == (self.episode_length)
    
    def step(self, action, **kwargs):
        if self.rescale_spaces:
            action = to_raw(action, self._action_space.low, self._action_space.high)

        # We apply the control first, then step.  The correct thing to do here
        # could be subtle, but for now we're assuming the agent knows (based on
        # the last obs) what the max power output is, and can react accordingly.
        obs, obs_meta = self.get_obs(**kwargs)

        self._real_power = np.float64((action * obs_meta["real_power"]).squeeze())
        self.index += 1
        rew, rew_meta = self.step_reward(**kwargs)

        rew_meta['step_meta']['action'] = action.tolist()
        rew_meta['step_meta']['pv_power'] = obs_meta["real_power"]
        rew_meta['step_meta']['es_power'] = 0
        rew_meta['step_meta']['grid_power'] = 0
        rew_meta['step_meta']['device_custom_info'] = {'pv_actionable_power': self._real_power}
        obs_meta.update(rew_meta)
        return obs, rew, self.is_terminal(), obs_meta

    def step_reward(self, **kwargs):
        step_meta = {}

        reward = 0
        
        step_meta["device_id"] = self.name
        step_meta["timestamp"] = kwargs['timestamp']
        step_meta["cost"] = 0
        step_meta["reward"] = reward
        return reward, {"step_meta": step_meta}
    

    def is_terminal(self):
        """The episode is done when the end of the data is reached."""
        return self.index == self.episode_length
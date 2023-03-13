
import os

import numpy as np
import pandas as pd

import gym

from gridworld.log import logger
from gridworld import ComponentEnv
from gridworld.utils import maybe_rescale_box_space, to_raw, to_scaled

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROFILE_DIR = os.path.join(THIS_DIR, "profiles")

class HSPVEnv(ComponentEnv):

    def __init__(
        self,
        name: str,
        profile_csv: str,
        profile_path: str = None,
        profile_data: list= [],
        scaling_factor: float = 1.,
        rescale_spaces: bool = True,
        grid_aware: bool = False,
        max_episode_steps: int = None,
        **kwargs
    ):
        """
        Args:
            name:  Component name.

            profile_csv:  Relative path from ./profiles for CSV file containing 
                the maximum real power profile that the device can generate.
                We assume the first column of this file contains these values 
                and discard the rest.
                
            profile_path:  Full path to profile csv.  If provided, this overrides
                the profile_csv argument.

            scaling_factor:  Float between 0 and 1 to rescale the csv data by.

            rescale_spaces:  If True, rescale action/obs spaces to [-1, 1].

            grid_aware:  If True, add "min_voltage" to obs space (for examples
                where PV is rewarded for voltage support).
        """

        super().__init__(name=name, **kwargs)

        self.scaling_factor = scaling_factor
        self.rescale_spaces = rescale_spaces
        self.grid_aware = grid_aware


        if profile_data != []:
            self.data = np.array(profile_data)
        else:
            # Read csv file.  If a full path is provide, that overrides reference to 
            # names of csv files stored locally in the `profiles` directory.
            profile_csv = os.path.join(PROFILE_DIR, profile_csv)
            if profile_path is not None:
                profile_csv = profile_path
            self.profile_csv = profile_csv
            
            # Read the profile data, rescale it, and infer episode length.
            self.data = pd.read_csv(self.profile_csv).values[:, 0].squeeze()

        self.data *= self.scaling_factor
        self.episode_length = len(self.data)

        # Optionally shorted the episode.
        if max_episode_steps is not None:
            self.episode_length = min(max_episode_steps, self.episode_length)

        # Create the obs labels and bounds.
        self._obs_labels = ["real_power"]
        self._obs_labels += ["min_voltage"] if grid_aware else []

        obs_bounds = {
            "real_power": (-np.max(self.data), 0.),
            "min_voltage": (0.9, 1.1)
        }

        # Create the optionally rescaled gym spaces.
        self._observation_space = gym.spaces.Box(
            shape=(len(self.obs_labels),),
            low=np.array([v[0] for k, v in obs_bounds.items() if k in self.obs_labels]),
            high=np.array([v[1] for k, v in obs_bounds.items() if k in self.obs_labels]),
            dtype=np.float64)

        self.observation_space = maybe_rescale_box_space(
            self._observation_space, rescale=self.rescale_spaces)

        self._action_space = gym.spaces.Box(
            shape=(1,), low=0., high=1., dtype=np.float64)

        self.action_space = maybe_rescale_box_space(
            self._action_space, rescale=self.rescale_spaces)


    def get_obs(self, **kwargs):
    
        """Returns the maximum real power possible for the current row of csv 
        data."""
        raw_obs = [-self.data[self.index]]
        if self.grid_aware:
            raw_obs = raw_obs + [kwargs["min_voltage"]]
        raw_obs = np.array(raw_obs)
        if self.rescale_spaces:
            obs = to_scaled(raw_obs, self._observation_space.low, self._observation_space.high)
        else:
            obs = raw_obs
        
        meta= {"real_power": raw_obs[0]}
        meta.update(kwargs)
        meta['pv_power']=meta['real_power']
        
        return obs,meta
    
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
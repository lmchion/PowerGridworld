import os

import gymnasium as gym
import numpy as np
import pandas as pd

from gridworld import ComponentEnv
from gridworld.utils import maybe_rescale_box_space, to_raw, to_scaled

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROFILE_DIR = os.path.join(THIS_DIR, "data")


class HSDevicesEnv(ComponentEnv):

    index: int = None   # Data index

    def __init__(
        self,
        name: str,
        profile_csv: str,
        profile_path: str = None,
        profile_data : dict = {},
        scaling_factor: float = 1.,
        rescale_spaces: bool = True,
        max_episode_steps: int = None,
        minutes_per_step : int = 5,
        max_grid_power: float = None,
        max_pv_power: float = None,
        max_es_power: float = None,
        **kwargs
    ):

        """
        Args:
            name:  Component name.

            devices_profile_csv:  Relative path from ./data for CSV file containing 
                the demand each additional device used in the house
                
            profile_path:  Full path to profile csv.  If provided, this overrides
                the profile_csv argument.

            scaling_factor:  Float between 0 and 1 to rescale the csv data by.

            rescale_spaces:  If True, rescale action/obs spaces to [-1, 1].

        """

        super().__init__(name=name, **kwargs)

        self.scaling_factor = scaling_factor
        self.rescale_spaces = rescale_spaces
        self.minutes_per_step = minutes_per_step

        if profile_data != {}:
            data_np=np.array([ v for k,v in profile_data.items()] ).T
            self.data_pd=pd.DataFrame(data_np, columns=profile_data.keys())
        else:
            # Read csv file.  If a full path is provide, that overrides reference to 
            # names of csv files stored locally in the `profiles` directory.
            profile_csv = os.path.join(PROFILE_DIR, profile_csv)
            if profile_path is not None:
                profile_csv = profile_path
            self.profile_csv = profile_csv
            # Read the profile data, rescale it, and infer episode length.
            self.data_pd=pd.read_csv(self.profile_csv)

        
        
        self.data = self.data_pd.values[0:, :].squeeze()
        self.data *= self.scaling_factor
        self.episode_length = len(self.data)

        # Optionally shorted the episode.
        if max_episode_steps is not None:
            self.episode_length = min(max_episode_steps, self.episode_length)

        # Create the obs labels and bounds.
        self._obs_labels = list(self.data_pd.columns)
        self._oth_power_labels = list(self.data_pd.columns)

        obs_bounds={}
        for num,elem in enumerate(self._obs_labels ):
            obs_bounds[elem]=(0.0, max(list(self.data_pd[elem])))

        self._obs_labels.extend(["oth_grid_power_consumed"])

        obs_bounds["oth_grid_power_consumed"] = (0.0, max(list(self.data_pd[elem])))

        # Create the optionally rescaled gym spaces.
        self._observation_space = gym.spaces.Box(
            shape=(len(self._obs_labels),),
            low=np.array([v[0] for k, v in obs_bounds.items() if k in self._obs_labels]),
            high=np.array([v[1] for k, v in obs_bounds.items() if k in self._obs_labels]),
            dtype=np.float64)

        self.observation_space = maybe_rescale_box_space(
            self._observation_space, rescale=self.rescale_spaces)

        self._action_space = gym.spaces.Box(
            shape=(1,), low=0.99, high=1., dtype=np.float64)

        self.action_space = maybe_rescale_box_space(
            self._action_space, rescale=self.rescale_spaces)


    def get_obs(self, **kwargs):
        """Returns the maximum real power possible for the current row of csv 
        data."""
        raw_obs = np.array(self.data[self.index])

        # Update Observation space with availability and consumption information.
        # attach values to observation here
        raw_obs = np.append(raw_obs, 
                        [kwargs['grid_power_consumed']])
        
        if self.rescale_spaces:
            obs = to_scaled(raw_obs, self._observation_space.low, self._observation_space.high)
        else:
            obs = raw_obs
        
        meta=kwargs.copy()
        for elem in self._oth_power_labels:
            meta[elem]=self.data_pd.loc[self.index, elem]

        # attach values in observation meta as well
        meta["oth_pv_power_available"] = kwargs['pv_power']
        meta["oth_pv_power_consumed"] = kwargs['solar_power_consumed']
        meta["oth_es_power_available"] = kwargs['es_power']
        meta["oth_es_power_consumed"] = kwargs['es_power_consumed']
        meta["oth_grid_power_available"] = kwargs['grid_power']
        meta["oth_grid_power_consumed"] = kwargs['grid_power_consumed']
        

        return obs, meta


    def is_terminal(self):
        """The episode is done when the end of the data is reached."""
        return self.index == self.episode_length


    def step_reward(self, **kwargs):
        """Step reward is always zero."""
        step_cost = self.current_cost * self._real_power * (self.minutes_per_step/60.0)

        step_meta = {}
        reward = - np.exp(step_cost)
        #reward = - step_cost
        #reward = -(1+reward)**3

        # # This is the final sub-environment which gets to act in the system. On each step,
        # # if there is any solar or battery juice left which does not get used, penalize this.
        # if kwargs['oth_pv_power_available'] > 0.0:
        #     reward -= kwargs['oth_pv_power_available'] * kwargs['max_grid_cost'] * (self.minutes_per_step/60.0)

        # if kwargs['oth_es_power_available'] > 0.0:
        #     reward -= kwargs['oth_es_power_available'] * kwargs['max_grid_cost'] * (self.minutes_per_step/60.0)
        
        step_meta["device_id"] = self.name
        step_meta["timestamp"] = kwargs['timestamp']
        step_meta["cost"] = step_cost
        step_meta["reward"] = reward

        kwargs.update({"step_meta": step_meta})

        return reward, kwargs



    def reset(self, *, seed=None, options=None, **kwargs):
        """Resetting consists of simply putting the index back to 0."""
        self.index = 0

        return self.get_obs(**kwargs)


    def step(self, action, **kwargs):
        """Advance the data index, and apply the action of curtailing PV real power 
        injection by a factor of between 0 and 1."""

        if self.rescale_spaces:
            action = to_raw(action, self._action_space.low, self._action_space.high)

        # We apply the control first, then step.  The correct thing to do here
        # could be subtle, but for now we're assuming the agent knows (based on
        # the last obs) what the max power output is, and can react accordingly.
        #obs, obs_meta = self.get_obs(**kwargs)

        obs, kwargs = self.get_obs(**kwargs)

        obs_meta = kwargs.copy()

        sum_obs_meta=sum([kwargs[x] for x in self._oth_power_labels ])
        self._real_power = np.float64((action * sum_obs_meta).squeeze())

        solar_power_consumed = 0
        battery_power_consumed = 0
        grid_power_consumed = 0
        solar_capacity=kwargs['pv_power']
        battery_capacity = kwargs['es_power']
        grid_capacity=kwargs['grid_power']

        if round(self._real_power,3)==0.0:
            self.current_cost =0.0
        else:
            solar_cost=kwargs['pv_cost']

            battery_cost = kwargs['es_cost']

            grid_cost=kwargs['grid_cost']
            
            solar_power_consumed=min(self._real_power,solar_capacity)
            battery_power_consumed = min( battery_capacity, self._real_power - solar_power_consumed ) 
            grid_power_consumed=min( grid_capacity, self._real_power - solar_power_consumed - battery_power_consumed)

            #print("power ",self._real_power,solar_power, grid_power,battery_power,str(kwargs))

            self.current_cost = (solar_cost*solar_power_consumed + grid_cost*grid_power_consumed + battery_cost*battery_power_consumed ) / (solar_power_consumed+ grid_power_consumed+battery_power_consumed)

            kwargs['pv_power']=max(0.0, solar_capacity-solar_power_consumed)
            kwargs['es_power']=max(0.0, battery_capacity-battery_power_consumed)
            kwargs['grid_power']=max(0.0, grid_capacity-grid_power_consumed)

            kwargs['es_power_consumed']=battery_power_consumed
            kwargs['solar_power_consumed']=solar_power_consumed
            kwargs['grid_power_consumed']=grid_power_consumed

        # obs, obs_meta = self.get_obs(**kwargs)

        # obs_meta = kwargs.copy()

        rew, rewmeta = self.step_reward(**kwargs)

        rewmeta['step_meta']['action'] = action.tolist()
        rewmeta['step_meta']['solar_power_consumed'] = solar_power_consumed
        rewmeta['step_meta']['es_power_consumed'] = battery_power_consumed
        rewmeta['step_meta']['grid_power_consumed'] = grid_power_consumed
        rewmeta['step_meta']['device_custom_info'] = {'power_ask': self._real_power, 
                                                      'solar_power_available': kwargs['pv_power'], 
                                                      'es_power_available':kwargs['es_power'], 
                                                      'grid_power_available':kwargs['grid_power']}

        obs_meta.update(rewmeta)
        self.index += 1

        return obs, rew, self.is_terminal(), False, obs_meta

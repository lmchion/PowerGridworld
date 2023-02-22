import os
import pandas as pd

from abc import ABC, abstractmethod
from copy import deepcopy
import logging
from typing import Tuple, List, Dict

import numpy as np

import gym

from gridworld.log import logger
from gridworld import MultiComponentEnv
from gridworld.agents.pv import PVEnv
from gridworld.agents.energy_storage import EnergyStorageEnv
from gridworld.agents.vehicles import EVChargingEnv


class HSMultiComponentEnv(MultiComponentEnv):
    """
    Class representing a multicomponent enviroment of the Home energy steward.
    The action and observation spaces of the multi-component env
    are taken as the union over the components.
    
    """

    def __init__(
            self,
            #common_config: dict = {},
            #env_config: dict = {},
            name : str = None,
            components: dict = {},
            max_episode_steps: int = None,
            # rescale_spaces: bool = True,
            **kwargs
    ):

        #super().__init__(name=common_config.name, components=env_config.components, **kwargs)
        super().__init__(name=name, components=components, **kwargs)

        # get grid costs and find the maximum grid cost 
        self.grid_cost_data = kwargs['grid_cost']

        self.observation_space["grid_cost"] = gym.spaces.Box(shape=(1,), low=0.0, high=max(self.grid_cost_data), dtype=np.float64)
        self._obs_labels += ["grid_cost"]

        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else np.inf

        # Action spaces from the component envs are combined into the composite space in super.__init__


    def reset(self, **kwargs) -> dict:
        self.time_index = 0

        # reset the state of all subcomponents and collect the initialization state from each.
        obs, meta = super().reset(**kwargs)

        # Start an episode with grid cost of 0.
        obs["grid_cost"] = 0

        return obs,meta        

    def get_obs(self, **kwargs) -> Tuple[dict, dict]:
        """
        Get composite observation from base and update grid cost observation.
        """

        obs, meta = super().get_obs(**kwargs)
        # Start an episode with grid cost of 0.
        obs["grid_cost"] = self.grid_cost_data[self.time_index]

        return obs, meta

  
    def step(self, action: dict, **kwargs) -> Tuple[dict, float, bool, dict]:
        """Default step method composes the obs, reward, done, meta dictionaries
        from each component step."""

        self.time_index += 1

        # Initialize outputs.
        real_power = 0.
        obs = {}
        dones = []
        metas = {}

        kwargs_copy = {k: v for k,v in kwargs.items()}

        # Loop over envs and collect real power injection/consumption.
        for env in self.envs:
            env_kwargs = {k: v for k,
                          v in kwargs_copy.items() if k in env._obs_labels}
            
            subcomp_obs, _, subcomp_done, subcomp_meta = env.step(action[env.name], **env_kwargs)
            obs[env.name] = subcomp_obs.copy()
            dones.append(subcomp_done)
            metas[env.name] = subcomp_meta.copy()
            real_power += env.real_power

            # Intermediate state update to ensure there is no resource contention
            # after each component step the intermediate state is updated into the 
            # copy of kwargs and is provided to the next component in line.
            # The order of devices in the line is arbitrarily defined in `heterogenous_hs.py`
            # where the component list is built.
            
            for subcomp_obs_key, subcomp_obs_val in subcomp_obs.items():
                kwargs_copy[subcomp_obs_key] = subcomp_obs_val

        # Set real power attribute.  TODO:  Reactive power.
        self._real_power = real_power

        # Compute the step reward using user-implemented method.
        step_reward, _ = self.step_reward()

        return obs, step_reward, any(dones), metas
    
    # step reward from the base environment definition continues to apply to this env as well.
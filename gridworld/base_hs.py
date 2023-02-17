
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
from gridworld.agents.hvac import HVACEnv


class HSMultiComponentEnv(MultiComponentEnv):
    """Class for creating a single Gym environment from multiple component 
    environments.  The action and observation spaces of the multi-component env
    are taken as the union over the components.
    """

    def __init__(
            self, 
            name: str = None,
            components: List[dict] = None,
            **kwargs
        ):

        super().__init__(name=name, components=components, **kwargs)

        self.grid_cost=kwargs['grid_cost']

        # max_power={ 'grid': kwargs['max_grid_power'] }
        # max_grid_cost=max(kwargs['grid_cost'])

        # for env in super().envs:
        #     if env["cls"]==EnergyStorageEnv:
        #         max_power['es']=env.max_power

        #     if env["cls"]==PVEnv:
        #         max_power['pv']=env._observation_space.high[0]

        # obs_space=super().observation_space.spaces

        

        # self.add_obs_space= { 'grid_cost'  : gym.spaces.Box( shape=(1,), low=0.0, high=max_grid_cost, dtype=np.float64),
        #                       'pv_cost'    : gym.spaces.Box( shape=(1,), low=0.0, high=0.0, dtype=np.float64),
        #                       'es_cost'    : gym.spaces.Box( shape=(1,), low=0.0, high=max_grid_cost, dtype=np.float64),
        #                       'grid_power' : gym.spaces.Box( shape=(1,), low=0.0, high=max_power['grid'], dtype=np.float64),
        #                       'pv_power'   : gym.spaces.Box( shape=(1,), low=0.0, high=max_power['es'], dtype=np.float64),
        #                       'es_power'   : gym.spaces.Box( shape=(1,), low=0.0, high=max_power['pv'], dtype=np.float64) }
                             
                                                     
        # obs_space=obs_space.update(self.add_obs_space)                                         
        # self.observation_space = gym.spaces.Dict(obs_space)

        self.state = {'grid_cost'  : None,
                      'pv_cost'    : 0.0,
                      'es_cost'    : None,
                      'grid_power' : kwargs['max_grid_power'],
                      'pv_power'   : None,
                      'es_power'   : None }



    def reset(self, **kwargs) -> dict:
        self.time_index=0
        
        obs,meta=super().reset(**kwargs)

        self.state['grid_cost']=self.grid_cost[self.time_index]

        for env in self.envs:
            if type(env)==EnergyStorageEnv:
                self.state['es_power']=env.current_storage
                self.state['es_cost']=0.0  # need to complete the initial cost calculation

            if type(env)==PVEnv:
                  self.state['pv_power']=meta[env.name]['real_power']
                
        return obs,meta        

  
    def step(self, action: dict, **kwargs) -> Tuple[dict, float, bool, dict]:
        """Default step method composes the obs, reward, done, meta dictionaries
        from each component step."""

        self.time_index +=1
        self.state['grid_cost']=self.grid_cost[self.time_index]

        # Initialize outputs.
        real_power = 0.
        obs = {}
        dones = []
        metas = {}

        # Loop over envs and collect real power injection/consumption.
        for env in self.envs:
            env_kwargs = {k: v for k,v in kwargs.items() if k in env.obs_labels}
            env_kwargs.update(self.state)
            ob, _, done, meta = env.step(action[env.name], **env_kwargs)
            obs[env.name] = ob.copy()
            dones.append(done)
            metas[env.name] = meta.copy()
            real_power += env.real_power

        # Set real power attribute.  TODO:  Reactive power.
        self._real_power = real_power

        # Compute the step reward using user-implemented method.
        step_reward, _ = super().step_reward()
        
        return obs, step_reward, any(dones), metas
    




    def get_obs(self, **kwargs) -> Tuple[dict, dict]:
        """Default get obs composes a dictionary of observations from each 
        component env."""

        # Initialize outputs.
        obs = {}
        meta = {}

        # Loop over envs and create the observation dict (of dicts).
        for env in self.envs:
            env_kwargs = {k: v for k,v in kwargs.items() if k in env.obs_labels}
            obs[env.name], meta[env.name] = env.get_obs(**env_kwargs)

        return obs, meta

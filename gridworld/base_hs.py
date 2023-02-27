from collections import OrderedDict
from typing import Dict, List, Tuple

import gym
import numpy as np
import pandas as pd

from gridworld import MultiComponentEnv
from gridworld.agents.energy_storage import EnergyStorageEnv
from gridworld.agents.pv import PVEnv
from gridworld.agents.vehicles import EVChargingEnv
from gridworld.log import logger
from gridworld.utils import maybe_rescale_box_space, to_raw, to_scaled


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
            name: str = None,
            components: List[dict] = None,
            start_time: str = '',
            end_time: str = '',
            control_timedelta=pd.Timedelta(300, "s"),
            max_grid_power: float = 48,
            max_episode_steps: int = None,
            rescale_spaces: bool = True,
            **kwargs
    ):

        self.max_grid_power = max_grid_power

        #super().__init__(name=common_config.name, components=env_config.components, **kwargs)

        super().__init__(name=name, components=components, **kwargs)

        self.rescale_spaces = rescale_spaces

        # get grid costs and find the maximum grid cost
        self._grid_cost_data = kwargs['grid_cost']

        # self.observation_space["grid_cost"] = gym.spaces.Box(
        #     shape=(1,), low=0.0, high=max(self._grid_cost_data), dtype=np.float64)
        # self._obs_labels += ["grid_cost"]

        # self.observation_space["grid_cost"] = maybe_rescale_box_space(
        #     self.observation_space["grid_cost"], rescale=self.rescale_spaces)

        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else np.inf
        
        self.meta_state = {'grid_cost': None,
                           'es_cost': None,
                           'grid_power': self.max_grid_power,
                           'pv_power': None,
                           'es_power': None,
                           'pv_cost': 0.0,
                           'ev_step_cost': -1.0,
                           'es_step_cost': -1.0
                           }
        
        #self._obs_labels = self._obs_labels +['grid_cost']

        # Action spaces from the component envs are combined into the composite space in super.__init__

    def reset(self, **kwargs) -> Tuple[dict, dict]:
        self.time_index = 0

        self.meta_state['grid_cost']=self._grid_cost_data[self.time_index]
        self.meta_state['grid_power'] = self.max_grid_power
        # This internal state object will be used to pass around intermediate
        # state of the system during the course of a step. The lower level components
        # are expected to use the information that is present in this state as inputs to
        # their actions and also inject their states into this structure after each
        # action that they take.
        # The values in this state are also all to be rescaled to [-1, 1]

        kwargs.update(self.meta_state)

        # reset the state of all subcomponents and collect the initialization state from each.
        for e in self.envs:
            _, kwargs = e.reset(**kwargs)

        #_ = [e.reset(**kwargs) for e in self.envs]
        obs, meta = self.get_obs(**kwargs)

        return obs

    def get_obs(self, **kwargs) -> Tuple[dict, dict]:
        """
        Get composite observation from base and update grid cost observation.
        """

        # Initialize outputs.
        obs = {}
        meta = OrderedDict()

        # Loop over envs and create the observation dict (of dicts).
        for env in self.envs:
            env_kwargs = {k: v for k,
                          v in kwargs.items() if k in env.obs_labels}
            env_kwargs.update(kwargs)
            obs[env.name], meta[env.name] = env.get_obs(**env_kwargs)

        # grid_cost = np.array([meta[env.name]['grid_cost']])

        # if self.rescale_spaces:
        #     obs["grid_cost"] = to_scaled(
        #         grid_cost, self.observation_space["grid_cost"].low, self.observation_space["grid_cost"].high)
        # else:
        #     obs["grid_cost"] = grid_cost

        return obs, meta

    def step(self, action: dict, **kwargs) -> Tuple[dict, float, bool, dict]:
        """Default step method composes the obs, reward, done, meta dictionaries
        from each component step."""

        self.time_index += 1

        # Initialize outputs.
        real_power = 0
        obs = {}
        dones = []
        meta = {}

        self.meta_state['grid_cost'] = self._grid_cost_data[self.time_index]
        self.meta_state['grid_power'] = self.max_grid_power
        
        # Loop over envs and collect real power injection/consumption.
        for subcomp in self.envs:
            subcomp_kwargs = {k: v for k,
                              v in kwargs.items() if k in subcomp._obs_labels}
            subcomp_kwargs.update(self.meta_state)
            subcomp_obs, _, subcomp_done, subcomp_meta = subcomp.step(
                action[subcomp.name], **subcomp_kwargs)
            obs[subcomp.name] = subcomp_obs.copy()
            dones.append(subcomp_done)
            real_power += subcomp.real_power

            # Intermediate state update to ensure there is no resource contention
            # after each component step the intermediate state is updated into the
            # copy of kwargs and is provided to the next component in line.
            # The order of devices in the line is arbitrarily defined in `heterogenous_hs.py`
            # where the component list is built.
            #
            # expect that the subcomponent will update its subcomp_meta['meta_state'] with the
            # latest subcomponent meta_state after this step. Update this to the top level meta
            # for the next iteration so next component in the iteration gets state updates.
            # print('subcomp_meta',subcomp_meta)
            #meta[subcomp.name] = subcomp_meta.copy()[subcomp.name]
            # meta['meta_state'].update(subcomp_meta['meta_state'])

            #for k, v in self.meta_state.items():
            #    if k in subcomp_meta:
            #        self.meta_state[k] = subcomp_meta[k]
            self.meta_state.update(subcomp_meta)

        # if self.rescale_spaces:
        #     obs["grid_cost"] = to_scaled(
        #         self.meta_state['grid_cost'], self.observation_space["grid_cost"].low, self.observation_space["grid_cost"].high)
        # else:
        #     obs["grid_cost"] = self.meta_state['grid_cost']

        #logger.info(f"META: {meta}")
        # Set real power attribute.  TODO:  Reactive power.
        self._real_power = real_power

        # Compute the step reward using user-implemented method.
        step_reward, _ = self.step_reward(**self.meta_state)

        return obs, step_reward, any(dones), self.meta_state

    # step reward from the base environment definition continues to apply to this env as well.

    def step_reward(self, **kwargs) -> Tuple[float, dict]:
        """Default step reward simply sums those from the components.  Overwrite
        this method to customize how this is computed."""

        # Initialize outputs.
        reward = 0.
        meta = {}

        # Loop over envs and create the reward dict.
        for env in self.envs:
            r, m = env.step_reward(**kwargs)
            reward += r
            meta[env.name] = m.copy()

        return reward, meta

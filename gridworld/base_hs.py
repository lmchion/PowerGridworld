from typing import Tuple

import gym
import numpy as np

from gridworld import MultiComponentEnv
from gridworld.agents.pv import PVEnv
from gridworld.agents.energy_storage import EnergyStorageEnv
from gridworld.agents.vehicles import EVChargingEnv
from gridworld.utils import to_scaled

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
        self._grid_cost_data = kwargs['grid_cost']

        self.observation_space["grid_cost"] = gym.spaces.Box(shape=(1,), low=0.0, high=max(self._grid_cost_data), dtype=np.float64)
        self._obs_labels += ["grid_cost"]

        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else np.inf

        # Action spaces from the component envs are combined into the composite space in super.__init__

    def reset(self, **kwargs) -> dict:
        self.time_index = 0

        # reset the state of all subcomponents and collect the initialization state from each.
        super().reset(**kwargs)

        obs, meta = self.get_obs(**kwargs)

        # This internal state object will be used to pass around intermediate
        # state of the system during the course of a step. The lower level components
        # are expected to use the information that is present in this state as inputs to
        # their actions and also inject their states into this structure after each
        # action that they take.
        # The values in this state are also all to be rescaled to [-1, 1]
        self.meta_state = {'grid_cost': None,
                           'es_cost': None,
                           'grid_power': None,
                           'pv_power': None,
                           'es_power': None
                           }

        meta['meta_state'] = self.meta_state

        return obs, meta

    def get_obs(self, **kwargs) -> Tuple[dict, dict]:
        """
        Get composite observation from base and update grid cost observation.
        """

        obs, meta = super().get_obs(**kwargs)
        # Start an episode with grid cost of 0.
        obs["grid_cost"] = self._grid_cost_data[self.time_index]

        return obs, meta

    def step(self, action: dict, **kwargs) -> Tuple[dict, float, bool, dict]:
        """Default step method composes the obs, reward, done, meta dictionaries
        from each component step."""

        self.time_index += 1

        # Initialize outputs.
        real_power = 0.
        obs = {}
        dones = []
        meta = {}
        meta['meta_state'] = kwargs['meta_state']

        # Loop over envs and collect real power injection/consumption.
        for subcomp in self.envs:
            subcomp_kwargs = {k: v for k,
                              v in kwargs.items() if k in subcomp._obs_labels}
            subcomp_kwargs['meta_state'] = meta['meta_state']
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
            meta[subcomp.name] = subcomp_meta.copy()[subcomp.name]
            meta.update(subcomp_meta['meta_state'])

        # Set real power attribute.  TODO:  Reactive power.
        self._real_power = real_power

        # Compute the step reward using user-implemented method.
        step_reward, _ = self.step_reward()

        return obs, step_reward, any(dones), meta

    # step reward from the base environment definition continues to apply to this env as well.

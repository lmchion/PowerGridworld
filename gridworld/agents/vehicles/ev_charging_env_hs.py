

from gridworld.agents.vehicles import EVChargingEnv
from typing import Tuple

import numpy as np
from gridworld.utils import maybe_rescale_box_space, to_raw, to_scaled


class HSEVChargingEnv(EVChargingEnv):


    def reset(self, **kwargs) -> Tuple[dict, dict]:

        obs, meta = super().reset(**kwargs)
        self.current_cost = 0.0

        return obs, meta

    def get_obs(self, **kwargs) -> Tuple[dict, dict]:
        "Returns an observation dict and metadata dict."
        obs, meta = super().get_obs(**kwargs)

        meta.update(**kwargs)
        return obs.copy(), meta.copy()
    
    def step_reward(self, **kwargs) -> Tuple[float, dict]:
        """Return a non-zero reward here if you want to use RL."""

        reward = self.current_cost * self._real_power + kwargs['cost'][kwargs['labels'].index('grid')] * self.state["real_power_unserved"]   

        return reward, {}
   
    def step(self, action: np.ndarray = None, **kwargs) -> Tuple[np.ndarray, float, bool, dict]:

        obs, rew, done, meta = super().step(action, **kwargs)

        power=self._real_power* (60.0/self.minutes_per_step)

        solar_capacity =kwargs['power'][kwargs['labels'].index('pv')]
        solar_cost=kwargs['cost'][kwargs['labels'].index('pv')]
        
        battery_capacity = kwargs['power'][kwargs['labels'].index('es')]
        battery_cost = kwargs['cost'][kwargs['labels'].index('es')] 

        grid_cost=kwargs['cost'][kwargs['labels'].index('grid')]  

        solar_power=min(power,solar_capacity)
        battery_power = min( battery_capacity, power - solar_power ) 
        grid_power=min( 0.0, power - battery_power )

        self.current_cost = (solar_cost*solar_power + grid_cost*grid_power + battery_cost*battery_power ) / (solar_power+ grid_power+battery_power)

        kwargs['power'][kwargs['labels'].index('pv')]=solar_capacity-solar_power
        kwargs['power'][kwargs['labels'].index('es')]=battery_capacity-battery_power

        # Get the return values
        obs, meta = self.get_obs(**kwargs)
        rew, rew_meta = self.step_reward(**kwargs)
        done = self.is_terminal()

        meta.update(rew_meta)

        return obs, rew, done, meta
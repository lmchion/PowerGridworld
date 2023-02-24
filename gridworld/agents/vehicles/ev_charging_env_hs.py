

from gridworld.agents.vehicles import EVChargingEnv
from typing import Tuple

import numpy as np
from gridworld.utils import maybe_rescale_box_space, to_raw, to_scaled


class HSEVChargingEnv(EVChargingEnv):

    def __init__(
        self,
        num_vehicles: int = 100,
        minutes_per_step: int = 5,
        max_charge_rate_kw: float = 7.0,  # ~40. for fast charge
        max_episode_steps: int = None,
        unserved_penalty: float = 1.,
        peak_penalty: float = 1.,
        peak_threshold: float = 10.,
        reward_scale: float = 1e5,
        name: str = None,
        randomize: bool = False,
        vehicle_csv: str = None,
        vehicle_multiplier: int = 1,
        rescale_spaces: bool = True,
        **kwargs
    ):
        super().__init__(num_vehicles = num_vehicles,
                        minutes_per_step = minutes_per_step,
                        max_charge_rate_kw= max_charge_rate_kw,  # ~40. for fast charge
                        max_episode_steps = max_episode_steps,
                        unserved_penalty = unserved_penalty,
                        peak_penalty = peak_penalty,
                        peak_threshold = peak_threshold,
                        reward_scale = reward_scale,
                        name = name,
                        randomize = randomize,
                        vehicle_csv = vehicle_csv,
                        vehicle_multiplier = vehicle_multiplier,
                        rescale_spaces = rescale_spaces,
                        **kwargs)

        self.current_cost = 0.0

    def reset(self, **kwargs) -> Tuple[dict, dict]:

        super().reset(**kwargs)
        

        return self.get_obs(**kwargs)

    def get_obs(self, **kwargs) -> Tuple[dict, dict]:
        "Returns an observation dict and metadata dict."
        obs, meta = super().get_obs(**kwargs)

        meta.update(kwargs)
        return obs.copy(), meta.copy()
    
    def step_reward(self, **kwargs) -> Tuple[float, dict]:
        """Return a non-zero reward here if you want to use RL."""

        #reward = self.current_cost * self._real_power + kwargs['cost'][kwargs['labels'].index('grid')] * self.state["real_power_unserved"]   
        reward = self.current_cost * self._real_power + kwargs['grid_cost'] * self.state["real_power_unserved"]   

        return -reward, {}
   
    def step(self, action: np.ndarray = None, **kwargs) -> Tuple[np.ndarray, float, bool, dict]:

        obs, rew, done, meta = super().step(action, **kwargs)

        power=self._real_power* (60.0/self.minutes_per_step)

        # solar_capacity =kwargs['power'][kwargs['labels'].index('pv')]
        # solar_cost=kwargs['cost'][kwargs['labels'].index('pv')]
        
        # battery_capacity = kwargs['power'][kwargs['labels'].index('es')]
        # battery_cost = kwargs['cost'][kwargs['labels'].index('es')] 

        # grid_cost=kwargs['cost'][kwargs['labels'].index('grid')]  


        if power==0.0:
            self.current_cost =0.0
        else:
            solar_capacity=kwargs['pv_power']
            solar_cost=kwargs['pv_cost']

            battery_capacity = kwargs['es_power']
            battery_cost = kwargs['es_cost']

            grid_cost=kwargs['grid_cost']
            grid_capacity=kwargs['grid_power']

            solar_power=min(power,solar_capacity)
            battery_power = min( battery_capacity, power - solar_power ) 
            grid_power=min( grid_capacity, power - battery_power )

            self.current_cost = (solar_cost*solar_power + grid_cost*grid_power + battery_cost*battery_power ) / (solar_power+ grid_power+battery_power)

            # kwargs['power'][kwargs['labels'].index('pv')]=solar_capacity-solar_power
            # kwargs['power'][kwargs['labels'].index('es')]=battery_capacity-battery_power

            kwargs['pv_power']=min(0.0, solar_capacity-solar_power)
            kwargs['es_power']=min(0.0, battery_capacity-battery_power)

        # Get the return values
        obs, meta = self.get_obs(**kwargs)
        rew, rew_meta = self.step_reward(**kwargs)
        done = self.is_terminal()

        meta.update(rew_meta)

        return obs, rew, done, meta
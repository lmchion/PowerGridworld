import json
import os
from collections import OrderedDict
from typing import Tuple

import gymnasium as gym
import numpy as np
import pandas as pd

from gridworld import ComponentEnv
from gridworld.log import logger
from gridworld.utils import maybe_rescale_box_space, to_raw, to_scaled

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
class HSEVChargingEnv(ComponentEnv):

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
        max_charge_cost: float = 0.55,
        profile_data : dict = {},
        **kwargs
    ):
        super().__init__(name=name)

        self.num_vehicles = num_vehicles
        self.max_charge_rate_kw = max_charge_rate_kw
        self.minutes_per_step = minutes_per_step
        self.randomize = randomize
        self.vehicle_multiplier = vehicle_multiplier
        self.rescale_spaces = rescale_spaces

        # Reward parameters
        self.unserved_penalty = unserved_penalty
        self.peak_penalty = peak_penalty
        self.peak_threshold = peak_threshold
        self.reward_scale = reward_scale

        # By default, we simulate a whole day but allow user to specify
        # fewer steps if desired.
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else np.inf
        self.max_episode_steps = min(self.max_episode_steps, 24*60 / minutes_per_step)

        # Create an array of simulation times in minutes, in the interval
        # (0, max_episode_steps * minutes_per_step).
        self.simulation_times = np.arange(
            0, (self.max_episode_steps+1) * minutes_per_step, minutes_per_step)

        # Attributes that will be initialized in reset.
        self.time_index = None  # time index
        self.time = None  # time in minutes
        self.df = None    # episode vehicle dataframe
        self.charging_vehicles = None  # charging vehicle list
        self.departed_vehicles = None  # vehicle list departed in last time step

        if profile_data != {}:
            self._df=pd.read_json(json.dumps(profile_data), orient='split')
        # Read the source dataframe.
        else: 
            vehicle_csv = vehicle_csv if vehicle_csv else os.path.join(THIS_DIR, "vehicles_hs.csv")
            self._df = pd.read_csv(vehicle_csv)     # all vehicles
        self._df["energy_required_kwh"] *= self.vehicle_multiplier

        # Round the start/end times to the nearest step.
        self._df["start_time_min"] = self._round(self._df["start_time_min"])
        self._df["end_time_park_min"] = self._round(self._df["end_time_park_min"])


        # Bounds on the observation space variables.
        obs_bounds = OrderedDict({
            "time": (0, self.simulation_times[-1]),
            "num_active_vehicles": (
                0, self.num_vehicles),
            "real_power_consumed": (
                0, self.num_vehicles * self.max_charge_rate_kw),
            "real_power_demand": (
                0, self.num_vehicles * self._df["energy_required_kwh"].max()),
            "mean_charge_rate_deficit": (
                0, self._df["energy_required_kwh"].max() / (self.minutes_per_step / 60.)),
            "real_power_unserved": (
                0, self._df["energy_required_kwh"].max()),
            "current_cost" : (
                0, max_charge_cost),
            "ev_pv_power_available" : (
                0, 3.6), 
            "ev_pv_power_consumed" : (
                0, 3.6), 
            "ev_es_power_available" : (
                0, 12.0), 
            "ev_es_power_consumed" : (
                0, max_charge_rate_kw), 
            "ev_grid_power_available" : (
                0, 48.0), 
            "ev_grid_power_consumed" : (
                0, max_charge_rate_kw)
        })

        # Construct the gym spaces.
        self._observation_space = gym.spaces.Box(
            low=np.array([x[0] for x in obs_bounds.values()]),
            high=np.array([x[1] for x in obs_bounds.values()]),
            shape=(len(obs_bounds), ),
            dtype=np.float64)
        self.observation_space = maybe_rescale_box_space(
            self._observation_space, rescale=self.rescale_spaces)
        
                # Fraction between 0 and 1 of max charge rate for all charging vehicles.
        self._action_space = gym.spaces.Box(
            low=0.,
            high=1.,
            shape=(1, ),
            dtype=np.float64
        )
        self.action_space = maybe_rescale_box_space(
            self._action_space, rescale=self.rescale_spaces)
        
        # Use a dictionary to keep track of various state quantities.
        # Use the self._update(key, value) to ensure valid keys when updating state.
        self.state = OrderedDict({k: None for k in obs_bounds.keys()})

        # Use the state dict to create the observation labels.
        self._obs_labels = list(self.state.keys())


    def reset(self, *, seed=None, options=None, **kwargs) -> Tuple[dict, dict]:
       
        self.time_index = 0 
        self.time = self.simulation_times[self.time_index]
        self.charging_vehicles = []
        self.departed_vehicles = []

        # Select first N vehicles if not randomized, else shuffle rows of df.
        self.df = self._df
        self.df = self.df.reset_index()     # index is now 0 to N-1

        # Initialize real power.
        self._real_power = 0.
        
        # Step the simulator one time without a control action.
        self.step(**kwargs)

        # Get the observation needed to solve the first control step.
        obs, meta = self.get_obs(**kwargs)
        
        return obs, meta

    def get_obs(self, **kwargs) -> Tuple[dict, dict]:
        "Returns an observation dict and metadata dict."
        
        raw_obs = np.array(list(self.state.values()))

        if self.rescale_spaces:
            obs = to_scaled(raw_obs, self._observation_space.low, self._observation_space.high)
        else:
            obs = raw_obs

        meta=self.state.copy()        
        kwargs.update(meta)

        return obs.copy(), kwargs
    
    def is_terminal(self) -> bool:
        """Returns True if max episode steps have been reached."""
        return self.time_index == self.max_episode_steps
    
    def step_reward(self, **kwargs) -> Tuple[float, dict]:
        """Return a non-zero reward here if you want to use RL."""

        step_cost = self.current_cost * self._real_power

        step_meta = {}

        reward = -(step_cost + kwargs['max_grid_cost'] * self.state["real_power_unserved"])
        
        step_meta["device_id"] = self.name
        step_meta["timestamp"] = kwargs['timestamp']
        step_meta["cost"] = step_cost
        step_meta["reward"] = reward
        return reward, {"step_meta": step_meta}
   
    def step(self, action: np.ndarray = None, **kwargs) -> Tuple[np.ndarray, float, bool, dict]:

        logger.debug(f'Time index {self.time_index}/{self.max_episode_steps}')
        logger.debug(f'Action: {action}')

        # If no action is applied, use minimum.
        # TODO: Make sure you are scaling things correctly.
        action = action if action is not None else self._action_space.low
        if self.rescale_spaces:
            action = to_raw(action, self._action_space.low, self._action_space.high)

        action_kw = action[0] * self.max_charge_rate_kw
        action_kwh = action_kw * (self.minutes_per_step / 60.)

        # Get indexes of vehicles arriving and departing.
        start_idx = np.where(self.time >= np.floor(self.df["start_time_min"]))[0]
        end_idx = np.where(self.time <= np.floor(self.df["end_time_park_min"]))[0]

        # Get indexes of charging vehicles.
        charging_vehicles = list(set(start_idx.flatten()).intersection(set(end_idx.flatten())))

        charging_vehicles = [i for i in charging_vehicles if self.df.at[i, "energy_required_kwh"] > 0.]
        
        # Get vehicles that have left the station in the last time step.
        self.departed_vehicles = list(set(self.charging_vehicles) - set(charging_vehicles))
        logger.debug(f"STEP, {self.time}, {self.time_index}, {charging_vehicles}, {self.departed_vehicles}")

        # Aggregate quantities that are needed for obs space.
        real_power_consumed = 0.
        real_power_demand = 0.
        min_energy_required = 0.
        charge_rate_deficit = []  # charge rate missing to reach full charge

        for i in charging_vehicles:

            # Compute energy required to fully charge.
            energy_required_kwh = self.df["energy_required_kwh"][i]

            # Update the aggregate variables
            real_power_demand += energy_required_kwh
            min_energy_required = max(min_energy_required, energy_required_kwh)

            # If the vehicle does not require any more charging then skip it.
            if energy_required_kwh <= 0.:
                continue

            # What is the min energy this vehicle needs to reach full charge?
            time_left_h = (self.df["end_time_park_min"][i] - self.time) / 60.
            if time_left_h <= 0:
                continue
            deficit = max(
                0, self.max_charge_rate_kw - energy_required_kwh / time_left_h)
            charge_rate_deficit.append(deficit)

            # Apply action and update the vehicle data.
            charge_energy_kwh = min(action_kwh, energy_required_kwh)
            self.df.at[i, "energy_required_kwh"] -= charge_energy_kwh
            real_power_consumed += charge_energy_kwh

            # print(action_kwh, energy_required_kwh, real_power_consumed)

            logger.debug(f"{i}, {energy_required_kwh}, {action}")
            
        # Update time variables.
        
        self.time = self.simulation_times[self.time_index]
        self.charging_vehicles = charging_vehicles

        # Compute unmet charging demand for departed vehicles.
        unserved = 0.
        for i in self.departed_vehicles:
            unserved  += self.df["energy_required_kwh"][i]
        self._update("real_power_unserved", unserved)

        # Update the state dict.
        self._update("time", self.time)
        self._update("num_active_vehicles", self.vehicle_multiplier * len(charging_vehicles))
        self._update("real_power_consumed", self.vehicle_multiplier * real_power_consumed)
        self._update("real_power_demand", self.vehicle_multiplier * real_power_demand)
        self._update(
            "mean_charge_rate_deficit",
            0 if len(charge_rate_deficit) == 0 else np.mean(charge_rate_deficit))
        
        
        # Update the real power attribute needed for component envs.
        self._real_power = self.vehicle_multiplier * real_power_consumed
        "############################################################################################"

        power=self._real_power  * (60.0/self.minutes_per_step)
        solar_power_consumed = 0
        battery_power_consumed = 0
        grid_power_consumed = 0
        solar_capacity=kwargs['pv_power']
        battery_capacity = kwargs['es_power']
        grid_capacity=kwargs['grid_power']

        if power==0.0 or action==0.0:
            self.current_cost =0.0
        else:
            solar_cost=kwargs['pv_cost']

            battery_cost = kwargs['es_cost']

            grid_cost=kwargs['grid_cost']

            solar_power_consumed=min(power,solar_capacity)

            # if we want to consider the battery cost and conserve it for when 
            # grid is more expensive than battery, this below check is required.
            # but then the battery cost in the current_cost calculation can be ignored.
            if battery_cost < grid_cost:
                battery_power_consumed = min( battery_capacity, power - solar_power_consumed ) 
                grid_power_consumed=min( grid_capacity, power - solar_power_consumed - battery_power_consumed )
            elif battery_cost >= grid_cost:
                grid_power_consumed=min( grid_capacity, power - solar_power_consumed)
                battery_power_consumed = min( battery_capacity, power - solar_power_consumed - grid_power_consumed ) 

            # ignore battery cost here since it has already been counted when the battery was charged.
            if solar_power_consumed+grid_power_consumed+battery_power_consumed > 0:
                self.current_cost = (solar_cost*solar_power_consumed + grid_cost*grid_power_consumed+ battery_cost*battery_power_consumed) / (solar_power_consumed+ grid_power_consumed+battery_power_consumed)

            kwargs['pv_power']=max(0.0, solar_capacity-solar_power_consumed)
            kwargs['es_power']=max(0.0, battery_capacity-battery_power_consumed)
            kwargs['grid_power']=max(0.0, grid_capacity-grid_power_consumed)

            kwargs['es_power_consumed']=battery_power_consumed
            kwargs['solar_power_consumed']=solar_power_consumed
            kwargs['grid_power_consumed']=grid_power_consumed

        # Update Observation space with availability and consumption information.
        # attach values to observation here
        self._update("current_cost", self.current_cost)
        self._update("ev_pv_power_available", kwargs['pv_power'])
        self._update("ev_pv_power_consumed", solar_power_consumed)
        self._update("ev_es_power_available", kwargs['es_power'])
        self._update("ev_es_power_consumed", battery_power_consumed)
        self._update("ev_grid_power_available", kwargs['grid_power'])
        self._update("ev_grid_power_consumed", grid_power_consumed)

        # Get the return values
        obs, meta = self.get_obs(**kwargs)

        rew, rew_meta = self.step_reward(**kwargs)
        rew_meta['step_meta']['action'] = action.tolist()
        rew_meta['step_meta']['solar_power_consumed'] = solar_power_consumed
        rew_meta['step_meta']['es_power_consumed'] = battery_power_consumed
        rew_meta['step_meta']['grid_power_consumed'] = grid_power_consumed
        rew_meta['step_meta']['device_custom_info'] = {'power_ask': power , 'power_unserved': unserved, 'charging_vehicle':len(self.charging_vehicles), 'vehicle_charged': len(self.departed_vehicles), 'solar_power_available': kwargs['pv_power'], 'es_power_available':kwargs['es_power'], 'grid_power_available':kwargs['grid_power']}

        done = self.is_terminal()

        meta.update(rew_meta)
        self.time_index += 1
        return obs, rew, done, False, meta
    

    def _update(self, key, value):
        if key not in self.state:
            raise ValueError(f'Invalid state key {key}')
        self.state[key] = value


    def _round(self, x):
        """Round the value x down to the nearest time step interval."""
        return x - x % self.minutes_per_step
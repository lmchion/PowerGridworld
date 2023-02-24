import numpy as np
import pandas as pd

from scipy.stats import truncnorm

import gym

from gridworld import ComponentEnv
from gridworld.utils import maybe_rescale_box_space, to_raw, to_scaled


from gridworld.log import logger

class HSEnergyStorageEnv(ComponentEnv):
    """Simple model of energy storage device that has (separate) linear models 
    for charging and discharging.  Gym specs:
    
        - Observation space:  State of charge (energy stored).
        - Action space:  [-1, 1] for fully charging / discharging, resp.
        - Reward:  0. (reimplement to have a non-trivial reward).
    """

    def __init__(
        self,
        name: str = None,
        storage_range: tuple = (3.0, 50.0),
        initial_storage_mean: float = 30.0,
        initial_storage_std: float = 5.0,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.9,
        max_power: float = 15.0,
        max_episode_steps: int = 288,
        control_timedelta: pd.Timedelta = pd.Timedelta(300, "s"),
        rescale_spaces: bool = True,
        initial_storage_cost: float = 0.0,
        max_storage_cost: float = 0.55,
        **kwargs
    ):

        super().__init__(name=name)
        
        self.initial_storage_cost=initial_storage_cost
        self.current_cost = self.initial_storage_cost

        self.storage_range = storage_range
        self.initial_storage_mean = initial_storage_mean
        self.initial_storage_std = initial_storage_std
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.max_power = max_power
        self.current_storage = None
        self.rescale_spaces = rescale_spaces

        self.simulation_step = 0
        self.max_episode_steps = max_episode_steps

        self.control_interval_in_hr = control_timedelta.seconds / 3600.0

        self._obs_labels =["stage_of_charge","cost"]

        self._observation_space = gym.spaces.Box(
            shape=(2,),
            low=np.array([self.storage_range[0],0.00]),
            high=np.array([self.storage_range[1],max_storage_cost]),
            dtype=np.float64
        )
        self.observation_space = maybe_rescale_box_space(
            self._observation_space, rescale=self.rescale_spaces)

        self._action_space = gym.spaces.Box(
            shape=(1,),
            low=-1.0,
            high=1.0,
            dtype=np.float64
        )
        self.action_space = maybe_rescale_box_space(
            self._action_space, rescale=self.rescale_spaces)

    def reset(self, **kwargs):

        #super().reset(**kwargs)
        
        self.simulation_step = 0

        init_storage = kwargs['init_storage'] if 'init_storage' in kwargs.keys() else None

        if init_storage is None:
            # Initial battery storage is sampled from a truncated normal distribution.
            self.current_storage =\
                float(truncnorm(-1, 1).rvs() *\
                self.initial_storage_std + self.initial_storage_mean)
        else:
            try:
                init_storage = float(init_storage)
                init_storage = np.clip(
                    init_storage, self.storage_range[0], self.storage_range[1])
            except (TypeError, ValueError) as e:
                print(e)
                print("init_storage value needs to be a float, use default value instead")
                init_storage = self.initial_storage_mean

            self.current_storage = init_storage
        
       
        return self.get_obs(**kwargs)
    
    def validate_power(self, power):
        """ Sanity check if the battery can provide such power given its current 
            SOC, e.g., cannot discharge when SOC is at minimum.

        Args:
          power: A float, the controlled power to the storage. It discharges if 
            the value is positive, else it is negative.

        Return:
          power: A float, the feasible power of the energy storage.
        """

        if power > 0:
            # ensure the discharging power is within the range.
            if self.current_storage - \
                    power * self.control_interval_in_hr / self.discharge_efficiency <\
                    self.storage_range[0]:
                power = max(self.current_storage - self.storage_range[0], 0.0) /\
                    self.control_interval_in_hr

        elif power < 0:
            # ensure charging does not exceed the limit
            if self.current_storage - \
                    self.charge_efficiency * power * self.control_interval_in_hr >\
                    self.storage_range[1]:
                power = - max(self.storage_range[1] - self.current_storage, 0.0) /\
                    self.control_interval_in_hr

        return power

    def get_obs(self, **kwargs):


        raw_obs = np.array([self.current_storage,self.current_cost])


        if self.rescale_spaces:
            obs = to_scaled(raw_obs, self._observation_space.low, self._observation_space.high)
        else:
            obs = raw_obs

        meta ={'state_of_charge' : self.current_storage, 'es_cost' : self.current_cost, 'es_power' : self.current_storage }
        kwargs.update(meta)



        return obs, kwargs
    
    def step_reward(self,**kwargs):

        
        if self._real_power < 0 :  # discharging
            es_reward = 0.0  # when it is negative, the battery becomes a producer SO no reward
            reward_meta = {}
        else:  # charging
            # es_reward = cost of charging (solar + grid) $/Kwh * efficiency % * power (Kw) * time (h)
            es_reward = self.delta_cost* self.charge_efficiency * self._real_power * self.control_interval_in_hr
            reward_meta = {}

        #the reward has to be negative so higher reward for less cost
        return -es_reward, reward_meta

     
     
    def step(self, action: np.ndarray, **kwargs):
        """ Implement control to the storage.
        """

        if self.rescale_spaces:
            action = to_raw(action, self._action_space.low, self._action_space.high)

        power = action[0] * self.max_power
        power = self.validate_power(power)

        # solar_capacity =kwargs['power'][kwargs['labels'].index('pv')]
        # solar_cost=kwargs['cost'][kwargs['labels'].index('pv')]
        # grid_cost=kwargs['cost'][kwargs['labels'].index('grid')]                                    
            
        solar_capacity = kwargs['pv_power']
        solar_cost = kwargs['pv_cost']
        grid_cost = kwargs['grid_cost']
        grid_capacity=kwargs['grid_power']
        
        if power==0.0:
            self.delta_cost=0.0

        elif power < 0.0:  # power negative is charging
            delta_storage = self.charge_efficiency * power * self.control_interval_in_hr

            # first, take solar energy - the cheapest
            solar_power=min(-power,solar_capacity)
            
            # the rest, use the grid
            grid_power=min( grid_capacity, -power - solar_power  )

           
            # calculate the weighted average cost of charging for the time interval
            self.delta_cost = (solar_cost*solar_power + grid_cost*grid_power)  / (solar_power+ grid_power)

            # update the current cost
            self.current_cost = (self.current_storage  * self.current_cost - delta_storage * self.delta_cost)/ ( self.current_storage - delta_storage  )

            self.current_storage -= delta_storage
            # In case of small numerical error:
            self.current_storage = min(self.current_storage, self.storage_range[1])

            # kwargs['power'][kwargs['labels'].index('pv')]=solar_capacity-solar_power
            # kwargs['power'][kwargs['labels'].index('es')]=0.0

            kwargs['pv_power']=min(0.0, solar_capacity-solar_power)
            kwargs['es_power']=0.0


        elif power > 0.0:  # power positive is discharging
            self.current_storage -= power * self.control_interval_in_hr / self.discharge_efficiency
            self.current_storage = max(self.current_storage, self.storage_range[0])
            #kwargs['power'][kwargs['labels'].index('es')]=power 
            kwargs['es_power']=power

        #kwargs['cost'][kwargs['labels'].index('es')]=self.current_cost
        kwargs['es_cost'] = self.current_cost
        #  Convert to the positive for load and  negative for generation convention.
        self._real_power = -power

        obs, obs_meta = self.get_obs(**kwargs)
        rew, _ = self.step_reward()

        self.simulation_step += 1

        return obs, rew, self.is_terminal(), obs_meta
    

    def is_terminal(self):
        return self.simulation_step >= self.max_episode_steps

    

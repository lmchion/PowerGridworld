import gym
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from gridworld import ComponentEnv
from gridworld.utils import maybe_rescale_box_space, to_raw, to_scaled


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
        self.max_storage_cost = max_storage_cost

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
        st_min = self.storage_range[0]
        st_max = self.storage_range[1]
        
        if power > 0:
            # Discharging: ensure the discharging power is within the range.
            delta_storage = power * self.control_interval_in_hr / self.discharge_efficiency # kw to kwh conversion
            
            if self.current_storage <= st_min:
                power = 0.0
            elif self.current_storage - delta_storage < st_min:
                delta_storage = self.current_storage - st_min
                power = delta_storage / self.control_interval_in_hr
            
        elif power < 0:
            # Charging: ensure charging does not exceed the limit
            delta_storage = - (power * self.control_interval_in_hr * self.charge_efficiency) # kw to kwh conversion

            if self.current_storage>=self.storage_range[1]:
                power = 0.0
            elif self.current_storage + delta_storage > st_max:
                delta_storage = st_max - self.current_storage
                power = - (delta_storage / self.control_interval_in_hr)

        return power

    def get_obs(self, **kwargs):


        raw_obs = np.array([self.current_storage,self.current_cost])


        if self.rescale_spaces:
            obs = to_scaled(raw_obs, self._observation_space.low, self._observation_space.high)
        else:
            obs = raw_obs
        
        meta ={'state_of_charge' : self.current_storage}
        kwargs.update(meta)

        return obs, kwargs
    
    def step_reward(self,**kwargs):

        step_cost = 0.0
        
        if self._real_power < 0 :  # discharging
            step_cost = 0.0  # when it is negative, the battery becomes a producer SO no reward
        else:  # charging
            # es_cost = cost of charging (solar + grid) $/Kwh * efficiency % * power (Kw) * time (h)
            
            step_cost = self.delta_cost* self.charge_efficiency * self._real_power * self.control_interval_in_hr

        #the reward has to be negative so higher reward for less cost

        reward = -step_cost

        solar_capacity = kwargs['pv_power']
        battery_capacity = kwargs['es_power']
        if solar_capacity > 0.0 and battery_capacity > 0.0 and self.current_storage < max(self.storage_range):
            # When battery capacity is > 0 , it means the agent chose to discharge the battery.
            # this pseudo penalty is to encourage the agent to charge battery as well when solar is available
            # adding a simple maxcost penalty when the battery is below half charged and not charging 
            # when there is solar.
            reward -= self.max_storage_cost * (max(self.storage_range)-self.current_storage)
        
        step_meta = {}
        step_meta['device_id'] = self.name
        step_meta["timestamp"] = kwargs['timestamp']
        step_meta["cost"] = step_cost
        step_meta["reward"] = reward
        return reward, {"step_meta": step_meta}

     
     
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
        solar_power_consumed = 0
        grid_power_consumed = 0

        if power==0.0:
            self.delta_cost=0.0

        elif power < 0.0:  # power negative is charging
            delta_storage = self.charge_efficiency * power * self.control_interval_in_hr
            # first, take solar energy - the cheapest
            solar_power_consumed=min(-power,solar_capacity)
            
            # the rest, use the grid
            grid_power_consumed=min( grid_capacity, -power - solar_power_consumed  )

           
            # calculate the weighted average cost of charging for the time interval
            self.delta_cost = (solar_cost*solar_power_consumed + grid_cost*grid_power_consumed)  / (solar_power_consumed+ grid_power_consumed)

            # update the current cost
            self.current_cost = (self.current_storage  * self.current_cost - delta_storage * self.delta_cost)/ ( self.current_storage - delta_storage  )

            self.current_storage -= delta_storage
            # In case of small numerical error:
            self.current_storage = min(self.current_storage, self.storage_range[1])

            # kwargs['power'][kwargs['labels'].index('pv')]=solar_capacity-solar_power
            # kwargs['power'][kwargs['labels'].index('es')]=0.0

            kwargs['pv_power']=max(0.0, solar_capacity-solar_power_consumed)
            kwargs['grid_power']=max(0.0, grid_capacity-grid_power_consumed)
            kwargs['es_power']=0.0


        elif power > 0.0:  # power positive is discharging
            delta_storage = power * self.control_interval_in_hr / self.discharge_efficiency

            self.current_storage = max(self.current_storage-delta_storage, self.storage_range[0])
            #kwargs['power'][kwargs['labels'].index('es')]=power 
            kwargs['es_power'] = power

        #kwargs['cost'][kwargs['labels'].index('es')]=self.current_cost
        kwargs['es_cost'] = 0 #self.current_cost
        #  Convert to the positive for load and  negative for generation convention.
        self._real_power = -power
        obs, obs_meta = self.get_obs(**kwargs)

        rew, rew_meta = self.step_reward(**kwargs)


        rew_meta['step_meta']['action'] = action.tolist() 
        rew_meta['step_meta']['solar_power_consumed'] = solar_power_consumed
        rew_meta['step_meta']['es_power_consumed'] = 0
        rew_meta['step_meta']['grid_power_consumed'] = grid_power_consumed
        rew_meta['step_meta']['device_custom_info'] = {'current_storage': self.current_storage, 'power_ask': power, 'solar_power_available': solar_capacity-solar_power_consumed, 'grid_power_available':grid_capacity-grid_power_consumed, 'es_power_available':kwargs['es_power']}

        if power > 0.0: # discharging for setting the pv and grid power to 0.
            rew_meta['step_meta']['solar_power_consumed'] = 0.0
            rew_meta['step_meta']['grid_power_consumed'] = 0.0

        obs_meta.update(rew_meta)
        self.simulation_step += 1

        return obs, rew, self.is_terminal(), obs_meta
    

    def is_terminal(self):
        return self.simulation_step == self.max_episode_steps
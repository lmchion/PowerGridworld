import os
import time
from os import system

import pandas as pd

from gridworld import MultiAgentEnv, MultiComponentEnv
from gridworld.agents.devices import HSDevicesEnv
from gridworld.agents.energy_storage import HSEnergyStorageEnv
from gridworld.agents.pv import HSPVEnv, PVEnv
from gridworld.agents.vehicles import HSEVChargingEnv


def load_grid_cost(start_time: str = None, end_time: str = None):
    """Returns exogenous data dataframe, and state space model (per-zone) dict."""

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    df = pd.read_csv(os.path.join(THIS_DIR, "data/grid_cost.csv"), delimiter=',')
    time_col = df["time"]
    df = df.set_index("time")
    df.index = pd.DatetimeIndex(df.index)

    df['timestamp'] = time_col.to_list()
    start_time = pd.Timestamp(start_time) if start_time else df.index[0]
    end_time = pd.Timestamp(end_time) if end_time else df.index[-1]

    _df = df.loc[start_time:end_time]

    if _df is None or len(_df) == 0:
        raise ValueError(
            f"start and/or end times ({start_time}, {end_time}) " +
            "resulted in empty dataframe.  First and last indices are " +
            f"({df.index[0]}, {df.index[-1]}), choose values in this range.")

    return (_df['timestamp'].tolist(), _df['grid_cost'].tolist())


def make_env_config( rescale_spaces=True):

    start_time="08-31-2020 00:00:00"
    end_time="08-31-2020 23:55:00"
    # Make the multi-component building
    timestamps, grid_cost = load_grid_cost(start_time, end_time)
    pv = {
        "name": "pv",
        "cls": HSPVEnv,
        "config": {
            "profile_csv": "pv_profile_hs.csv",
            "scaling_factor": 40.,
            "rescale_spaces": rescale_spaces
        }
    }

    battery = {
        "name": "storage",
        "cls": HSEnergyStorageEnv,
        "config": {
            "max_power"                 : 10.0,
            "storage_range"             : (1., 28.),
            "initial_storage_mean"      : 14.0,  # 7*2
            "initial_storage_std"       : 2.828, # 2*sqrt(2)
            "charge_efficiency"         : 0.95,
            "discharge_efficiency"      : 0.95,
            "init_storage"              : 0.0,
            "rescale_spaces"            : rescale_spaces,
            'initial_storage_cost'      : min(grid_cost),
            'max_storage_cost'          : max(grid_cost)
        } 
    }

    ev = {
            "name": "ev-charging",
            "cls": HSEVChargingEnv,
            "config": {
                "num_vehicles"          : 1,
                "minutes_per_step"      : 5,
                "max_charge_rate_kw"    : 11.,
                "peak_threshold"        : 200.,
                #"vehicle_multiplier"    : 40.,
                "rescale_spaces"        : rescale_spaces,
                "max_charge_cost"       :  max(grid_cost)
            }
        }
    

    devs = {
            "name": "other-devices",
            "cls": HSDevicesEnv,
            "config": {
                'profile_csv' : 'devices_profile_hs.csv' ,
                'profile_path' : None,
                'scaling_factor' :  1.,
                'rescale_spaces':  True,
                'max_episode_steps':  None,

            }
        }





    # this defines the arbitrary order of devices and the action that they take in the composite environment.
    house_components = [pv, battery, ev, devs ]



    env_config={ "components"       : house_components,
                 "name"             : 'house',
                 "start_time"       : start_time,
                 "end_time"         : end_time,
                 "control_timedelta": pd.Timedelta(300, "s"),
                 'max_grid_power'   :  48,
                 'max_episode_steps' : 288
                 
                }

    env_config['grid_cost'] = grid_cost
    env_config['timestamps'] = timestamps

    return env_config





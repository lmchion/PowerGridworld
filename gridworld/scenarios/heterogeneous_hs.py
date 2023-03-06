import os,sys
from os import system

import pandas as pd
import json

from gridworld import MultiAgentEnv, MultiComponentEnv
from gridworld.agents.energy_storage import HSEnergyStorageEnv
from gridworld.agents.pv import HSPVEnv, PVEnv
from gridworld.agents.vehicles import HSEVChargingEnv
from gridworld.agents.devices import HSDevicesEnv

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_grid_cost(start_time: str = None, end_time: str = None) -> list:
    """Returns exogenous data dataframe, and state space model (per-zone) dict."""

    

    df = pd.read_csv(os.path.join(THIS_DIR, "data/grid_cost.csv"), index_col=0)
    df.index = pd.DatetimeIndex(df.index)

    start_time = pd.Timestamp(start_time) if start_time else df.index[0]
    end_time = pd.Timestamp(end_time) if end_time else df.index[-1]

    _df = df.loc[start_time:end_time]

    if _df is None or len(_df) == 0:
        raise ValueError(
            f"start and/or end times ({start_time}, {end_time}) " +
            "resulted in empty dataframe.  First and last indices are " +
            f"({df.index[0]}, {df.index[-1]}), choose values in this range.")

    return _df['grid_cost'].tolist()




def make_env_config():

    with open(os.path.join(THIS_DIR, "data/env_config.json"), 'r') as f:
        env_config = json.load(f)

    grid_cost = load_grid_cost(env_config['start_time'], env_config['end_time'])
    
    env_config['grid_cost'] = grid_cost

    for elem in env_config['components']:
        elem['cls']= getattr(sys.modules[__name__], elem['cls'])

    env_config['control_timedelta']  =  pd.Timedelta(env_config['control_timedelta'], env_config['control_time_delta_units'])
    del env_config['control_time_delta_units']

    return env_config

if __name__ == "__main__":
    
    test=make_env_config()
    print(test)
    r=1





